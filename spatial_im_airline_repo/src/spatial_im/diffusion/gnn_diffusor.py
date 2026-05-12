from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from spatial_im.evaluation.metrics import expected_coverage, intervention_cost
from spatial_im.policy.features import compute_dynamic_temporal_summaries

from .simulator import estimate_final_activation_prob, sample_training_batch


class EdgeGate(nn.Module):
    def __init__(self, edge_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, edge_feat: torch.Tensor) -> torch.Tensor:
        # Nonnegative gate; useful when the diffusor is used as a progressive surrogate.
        return F.softplus(self.net(edge_feat)).squeeze(-1)


class SpatialMessagePassing(nn.Module):
    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int):
        super().__init__()
        self.msg = nn.Linear(node_dim, hidden_dim)
        self.self_lin = nn.Linear(node_dim, hidden_dim)
        self.edge_gate = EdgeGate(edge_dim=edge_dim, hidden_dim=hidden_dim)
        self.out = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_feat: torch.Tensor) -> torch.Tensor:
        src, dst = edge_index
        gate = self.edge_gate(edge_feat)  # [E]
        msg_src = self.msg(x[src])  # [E, H]
        weighted_msg = gate.unsqueeze(-1) * msg_src
        agg = torch.zeros(x.size(0), weighted_msg.size(-1), device=x.device)
        agg.index_add_(0, dst, weighted_msg)
        h = F.relu(self.self_lin(x) + agg)
        return F.relu(self.out(h))


class SnapshotTemporalEncoder(nn.Module):
    def __init__(self, node_feat_dim: int, hidden_dim: int, layers: int):
        super().__init__()
        self.encoder = nn.Linear(node_feat_dim, hidden_dim)
        self.layers = nn.ModuleList([SpatialMessagePassing(hidden_dim, edge_dim=2, hidden_dim=hidden_dim) for _ in range(layers)])

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_feat: torch.Tensor,
    ) -> torch.Tensor:
        h = F.relu(self.encoder(node_features))
        for layer in self.layers:
            h = layer(h, edge_index, edge_feat)
        return h


class GNNDiffusor(nn.Module):
    def __init__(self, node_feat_dim: int, hidden_dim: int = 64, layers: int = 2, temporal_window_len: int = 4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.temporal_window_len = temporal_window_len
        self.dynamic_summary_dim = 2 * max(int(temporal_window_len), 1) + 6
        self.snapshot_encoder = SnapshotTemporalEncoder(node_feat_dim=node_feat_dim + 2, hidden_dim=hidden_dim, layers=layers)
        self.temporal_model = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.input_dim = node_feat_dim + hidden_dim + 2  # node/static features + temporal context + current activation + seed indicator
        self.encoder = nn.Linear(self.input_dim, hidden_dim)
        self.layers = nn.ModuleList([SpatialMessagePassing(hidden_dim, edge_dim=2, hidden_dim=hidden_dim) for _ in range(layers)])
        self.head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))
        self.seed_context_proj = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.Tanh(),
        )
        self.marginal_head = nn.Sequential(
            nn.Linear(hidden_dim + self.dynamic_summary_dim + 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        attn_heads = 4 if hidden_dim % 4 == 0 else 1
        self.candidate_query_proj = nn.Sequential(
            nn.Linear(hidden_dim + self.dynamic_summary_dim + 5, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.candidate_temporal_attention = nn.MultiheadAttention(hidden_dim, num_heads=attn_heads, batch_first=True)
        self.candidate_rerank_head = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    @staticmethod
    def _edge_features(edge_weight_t: torch.Tensor, distances_km: torch.Tensor) -> torch.Tensor:
        dist_penalty = 1.0 / (1.0 + distances_km / 1000.0)
        return torch.stack([edge_weight_t, dist_penalty], dim=-1)

    def encode_snapshot(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight_t: torch.Tensor,
        distances_km: torch.Tensor,
        seed_mask: torch.Tensor | None = None,
        current_active: torch.Tensor | None = None,
    ) -> torch.Tensor:
        edge_feat = self._edge_features(edge_weight_t, distances_km)
        if seed_mask is None:
            seed_mask = torch.zeros(node_features.size(0), dtype=node_features.dtype, device=node_features.device)
        if current_active is None:
            current_active = seed_mask
        node_feat = torch.cat([node_features, seed_mask.unsqueeze(-1), current_active.unsqueeze(-1)], dim=-1)
        return self.snapshot_encoder(node_features=node_feat, edge_index=edge_index, edge_feat=edge_feat)

    def encode_temporal_context(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight_seq: torch.Tensor,
        distances_km: torch.Tensor,
        seed_mask: torch.Tensor | None = None,
        current_active: torch.Tensor | None = None,
        window_len: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if edge_weight_seq.dim() == 1:
            edge_weight_seq = edge_weight_seq.unsqueeze(0)
        if seed_mask is None:
            seed_mask = torch.zeros(node_features.size(0), dtype=node_features.dtype, device=node_features.device)
        if current_active is None:
            current_active = seed_mask
        snapshot_embeddings = []
        for t in range(edge_weight_seq.size(0)):
            snapshot_embeddings.append(
                self.encode_snapshot(
                    node_features=node_features,
                    edge_index=edge_index,
                    edge_weight_t=edge_weight_seq[t],
                    distances_km=distances_km,
                    seed_mask=seed_mask,
                    current_active=current_active,
                )
            )
        stacked = torch.stack(snapshot_embeddings, dim=1)  # [N, T, H]
        T = stacked.size(1)
        if window_len is None or window_len <= 0:
            window_len = self.temporal_window_len if self.temporal_window_len > 0 else T
        context_seq = []
        for t in range(T):
            start = max(0, t - int(window_len) + 1)
            window = stacked[:, start:t + 1, :]
            _, hidden = self.temporal_model(window)
            context_seq.append(hidden[-1])
        context_seq = torch.stack(context_seq, dim=1)  # [N, T, H]
        temporal_embeddings = context_seq[:, -1, :]
        return temporal_embeddings, context_seq

    def build_temporal_cache(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight_seq: torch.Tensor,
        distances_km: torch.Tensor,
        window_len: int | None = None,
    ) -> torch.Tensor:
        zero = torch.zeros(node_features.size(0), dtype=node_features.dtype, device=node_features.device)
        _, context_seq = self.encode_temporal_context(
            node_features=node_features,
            edge_index=edge_index,
            edge_weight_seq=edge_weight_seq,
            distances_km=distances_km,
            seed_mask=zero,
            current_active=zero,
            window_len=window_len,
        )
        return context_seq

    def condition_temporal_cache(
        self,
        cached_context_seq: torch.Tensor,
        seed_mask: torch.Tensor,
        current_active: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        seed_state = torch.stack([seed_mask, current_active], dim=-1)
        cond = self.seed_context_proj(seed_state)
        conditioned_seq = cached_context_seq + cond.unsqueeze(1)
        temporal_embeddings = conditioned_seq[:, -1, :]
        return temporal_embeddings, conditioned_seq

    def forward_one_step(
        self,
        node_features: torch.Tensor,
        current_active: torch.Tensor,
        seed_mask: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight_t: torch.Tensor,
        distances_km: torch.Tensor,
        temporal_context: torch.Tensor,
    ) -> torch.Tensor:
        edge_feat = self._edge_features(edge_weight_t, distances_km)
        x = torch.cat([node_features, temporal_context, current_active.unsqueeze(-1), seed_mask.unsqueeze(-1)], dim=-1)
        h = F.relu(self.encoder(x))
        for layer in self.layers:
            h = layer(h, edge_index, edge_feat)
        logits = self.head(h).squeeze(-1)
        proposal = torch.sigmoid(logits)
        # progressive activation surrogate: once active, stay active.
        nxt = torch.maximum(current_active, proposal)
        return nxt

    def rollout(
        self,
        node_features: torch.Tensor,
        seed_mask: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight_seq: torch.Tensor,
        distances_km: torch.Tensor,
        temporal_context: torch.Tensor | None = None,
        temporal_context_seq: torch.Tensor | None = None,
        current_active_context: torch.Tensor | None = None,
        temporal_window_len: int | None = None,
    ) -> torch.Tensor:
        if current_active_context is None:
            current_active_context = seed_mask.float()
        if temporal_window_len is None:
            temporal_window_len = self.temporal_window_len
        if temporal_context is None and temporal_context_seq is None:
            temporal_context, temporal_context_seq = self.encode_temporal_context(
                node_features=node_features,
                edge_index=edge_index,
                edge_weight_seq=edge_weight_seq,
                distances_km=distances_km,
                seed_mask=seed_mask,
                current_active=current_active_context,
                window_len=temporal_window_len,
            )
        if temporal_context_seq is None:
            temporal_context_seq = temporal_context.unsqueeze(1).repeat(1, edge_weight_seq.size(0), 1)
        p = seed_mask.float().clone()
        for t in range(edge_weight_seq.size(0)):
            step_context = temporal_context_seq[:, min(t, temporal_context_seq.size(1) - 1), :]
            p = self.forward_one_step(
                node_features=node_features,
                current_active=p,
                seed_mask=seed_mask,
                edge_index=edge_index,
                edge_weight_t=edge_weight_seq[t],
                distances_km=distances_km,
                temporal_context=step_context,
            )
        return p

    def predict_marginals(
        self,
        temporal_embeddings: torch.Tensor,
        dynamic_summaries: torch.Tensor,
        seed_mask: torch.Tensor,
        remaining_budget_frac: float,
        step_frac: float,
    ) -> torch.Tensor:
        n = temporal_embeddings.size(0)
        global_feat = torch.tensor(
            [remaining_budget_frac, step_frac],
            dtype=temporal_embeddings.dtype,
            device=temporal_embeddings.device,
        ).repeat(n, 1)
        x = torch.cat(
            [
                temporal_embeddings,
                dynamic_summaries,
                seed_mask.unsqueeze(-1),
                global_feat,
            ],
            dim=-1,
        )
        return self.marginal_head(x).squeeze(-1)

    def predict_candidate_rerank_scores(
        self,
        temporal_context_seq: torch.Tensor,
        temporal_embeddings: torch.Tensor,
        dynamic_summaries: torch.Tensor,
        seed_mask: torch.Tensor,
        current_active: torch.Tensor,
        predicted_final: torch.Tensor,
        remaining_budget_frac: float,
        step_frac: float,
    ) -> torch.Tensor:
        n = temporal_embeddings.size(0)
        global_feat = torch.tensor(
            [remaining_budget_frac, step_frac],
            dtype=temporal_embeddings.dtype,
            device=temporal_embeddings.device,
        ).repeat(n, 1)
        query_input = torch.cat(
            [
                temporal_embeddings,
                dynamic_summaries,
                seed_mask.unsqueeze(-1),
                current_active.unsqueeze(-1),
                predicted_final.unsqueeze(-1),
                global_feat,
            ],
            dim=-1,
        )
        query = self.candidate_query_proj(query_input).unsqueeze(1)
        attn_out, _ = self.candidate_temporal_attention(query, temporal_context_seq, temporal_context_seq, need_weights=False)
        fused = torch.cat([query.squeeze(1), attn_out.squeeze(1)], dim=-1)
        return self.candidate_rerank_head(fused).squeeze(-1)


@dataclass
class DiffusorArtifacts:
    model: GNNDiffusor
    history: Dict[str, List[float]]


def train_diffusor(
    model: GNNDiffusor,
    node_features: torch.Tensor,
    edge_index: torch.Tensor,
    edge_prob_seq: np.ndarray,
    distances_km: torch.Tensor,
    epochs: int = 20,
    batch_size: int = 32,
    lr: float = 1e-3,
    seed: int = 7,
    max_horizon: int | None = None,
    coords: torch.Tensor | np.ndarray | None = None,
    node_costs: torch.Tensor | np.ndarray | None = None,
    regime: str = 'spread',
    beta_coverage: float = 0.0,
    lambda_cost: float = 0.0,
    cost_mode: str = 'geography',
    constant_cost: float = 1.0,
    distance_cost_scale: float = 0.0015,
    coverage_radius_km: float = 900.0,
    teacher_mc_rollouts: int = 24,
    marginal_loss_weight: float = 1.0,
    ranking_loss_weight: float = 0.2,
    selection_budget: int | None = None,
    teacher_refresh_interval: int = 1,
) -> DiffusorArtifacts:
    def _to_numpy(arr):
        if arr is None:
            return None
        if isinstance(arr, np.ndarray):
            return arr
        return arr.detach().cpu().numpy()

    def _raw_objective(final_active_probs: np.ndarray, selected_nodes: np.ndarray) -> float:
        mass = float(final_active_probs.sum())
        if regime == 'spread':
            return mass
        if coords_np is None or node_costs_np is None:
            return mass
        coverage = expected_coverage(coords_np, final_active_probs, coverage_radius_km)
        cost = intervention_cost(
            selected=selected_nodes,
            coords=coords_np,
            node_costs=node_costs_np,
            mode=cost_mode,
            constant_cost=constant_cost,
            distance_cost_scale=distance_cost_scale,
        )
        return float(mass + beta_coverage * coverage - lambda_cost * cost)

    device = node_features.device
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = {'loss': [], 'loss_final': [], 'loss_marginal': [], 'loss_rank': [], 'loss_temporal_rank': []}
    T = edge_prob_seq.shape[0]
    num_nodes = node_features.shape[0]
    horizon_cap = max_horizon or T
    edge_index_np = edge_index.detach().cpu().numpy().T
    distances_np = distances_km.detach().cpu().numpy()
    coords_np = _to_numpy(coords)
    node_costs_np = _to_numpy(node_costs)

    cached_batch = None

    def _refresh_teacher_batch(epoch_seed: int):
        rng_local = np.random.default_rng(epoch_seed)
        raw_batch = sample_training_batch(
            edge_index=edge_index_np,
            edge_prob_seq_all=edge_prob_seq,
            num_nodes=num_nodes,
            batch_size=batch_size,
            budget_range=(1, max(1, int(selection_budget or 3))),
            horizon_range=(1, max(1, horizon_cap)),
            seed=epoch_seed,
            mc_rollouts=teacher_mc_rollouts,
        )
        enriched = []
        for sample in raw_batch:
            seed_mask_np = sample['seed_mask']
            target_np = sample['teacher_final_probs']
            t = int(sample['start_t'])
            horizon = int(sample['horizon'])
            selected_nodes = np.flatnonzero(seed_mask_np > 0.5).astype(np.int64)
            candidate_nodes = np.flatnonzero(seed_mask_np < 0.5).astype(np.int64)
            pair = np.asarray([], dtype=np.int64)
            pair_deltas = np.asarray([], dtype=np.float32)
            if candidate_nodes.size > 0:
                pair_size = min(2, candidate_nodes.size)
                pair = rng_local.choice(candidate_nodes, size=pair_size, replace=False).astype(np.int64)
                base_objective = _raw_objective(target_np, selected_nodes)
                deltas = []
                for cand in np.atleast_1d(pair):
                    aug_seed_mask = seed_mask_np.copy()
                    aug_seed_mask[int(cand)] = 1.0
                    aug_probs = estimate_final_activation_prob(
                        edge_index=edge_index_np,
                        edge_prob_seq=edge_prob_seq[t:t + horizon],
                        seed_mask=aug_seed_mask,
                        mc_rollouts=teacher_mc_rollouts,
                        seed=int(rng_local.integers(1_000_000)),
                    )
                    aug_selected = np.flatnonzero(aug_seed_mask > 0.5).astype(np.int64)
                    deltas.append(float(_raw_objective(aug_probs, aug_selected) - base_objective))
                pair_deltas = np.asarray(deltas, dtype=np.float32)
            enriched.append({
                **sample,
                'pair_candidates': pair,
                'pair_teacher_deltas': pair_deltas,
            })
        return enriched

    for epoch in range(epochs):
        model.train()
        rng = np.random.default_rng(seed + epoch)
        if cached_batch is None or epoch % max(int(teacher_refresh_interval), 1) == 0:
            cached_batch = _refresh_teacher_batch(seed + epoch)
        batch = cached_batch
        losses = []
        losses_final = []
        losses_marginal = []
        losses_rank = []
        losses_temporal_rank = []
        for sample in batch:
            t = int(sample['start_t'])
            horizon = int(sample['horizon'])
            seed_mask_np = sample['seed_mask']
            target_np = sample['teacher_final_probs']

            seed_mask = torch.as_tensor(seed_mask_np, dtype=torch.float32, device=device)
            current_active = seed_mask.clone()
            target = torch.as_tensor(target_np, dtype=torch.float32, device=device)
            w_seq = torch.as_tensor(edge_prob_seq[t:t+horizon], dtype=torch.float32, device=device)
            dynamic_summaries_np = compute_dynamic_temporal_summaries(
                edge_index=edge_index_np.T,
                edge_weight_seq=edge_prob_seq[t:t+horizon],
                distances_km=distances_np,
                num_nodes=num_nodes,
                window_len=model.temporal_window_len,
                horizon_len=horizon,
            )
            dynamic_summaries = torch.as_tensor(dynamic_summaries_np, dtype=torch.float32, device=device)

            temporal_embeddings, temporal_context_seq = model.encode_temporal_context(
                node_features=node_features,
                edge_index=edge_index,
                edge_weight_seq=w_seq,
                distances_km=distances_km,
                seed_mask=seed_mask,
                current_active=current_active,
                window_len=model.temporal_window_len,
            )
            pred = model.rollout(
                node_features=node_features,
                seed_mask=seed_mask,
                edge_index=edge_index,
                edge_weight_seq=w_seq,
                distances_km=distances_km,
                temporal_context=temporal_embeddings,
                temporal_context_seq=temporal_context_seq,
                current_active_context=current_active,
                temporal_window_len=model.temporal_window_len,
            )

            loss_final = F.binary_cross_entropy(pred.clamp(1e-5, 1 - 1e-5), target)
            selected_nodes = np.flatnonzero(seed_mask_np > 0.5).astype(np.int64)
            budget_ref = max(int(selection_budget or 0), int(selected_nodes.size) + 1, 1)
            remaining_budget_frac = float(max(budget_ref - int(selected_nodes.size), 0) / budget_ref)
            step_frac = float(min(int(selected_nodes.size), budget_ref) / budget_ref)

            q_pred = model.predict_marginals(
                temporal_embeddings=temporal_embeddings,
                dynamic_summaries=dynamic_summaries,
                seed_mask=seed_mask,
                remaining_budget_frac=remaining_budget_frac,
                step_frac=step_frac,
            )
            temporal_rerank_scores = model.predict_candidate_rerank_scores(
                temporal_context_seq=temporal_context_seq,
                temporal_embeddings=temporal_embeddings,
                dynamic_summaries=dynamic_summaries,
                seed_mask=seed_mask,
                current_active=current_active,
                predicted_final=pred,
                remaining_budget_frac=remaining_budget_frac,
                step_frac=step_frac,
            )

            loss_marginal = torch.zeros((), dtype=pred.dtype, device=device)
            loss_rank = torch.zeros((), dtype=pred.dtype, device=device)
            loss_temporal_rank = torch.zeros((), dtype=pred.dtype, device=device)
            pair = np.asarray(sample.get('pair_candidates', []), dtype=np.int64)
            if pair.size > 0:
                q_teacher_values = sample.get('pair_teacher_deltas', np.asarray([], dtype=np.float32))
                q_pred_values = [q_pred[int(cand)] for cand in np.atleast_1d(pair)]
                q_teacher = torch.as_tensor(q_teacher_values, dtype=pred.dtype, device=device)
                q_pred_pair = torch.stack(q_pred_values)
                loss_marginal = F.mse_loss(q_pred_pair, q_teacher)
                if regime == 'dynamic':
                    rerank_pred_pair = torch.stack([temporal_rerank_scores[int(cand)] for cand in np.atleast_1d(pair)])
                    loss_temporal_rank = F.mse_loss(rerank_pred_pair, q_teacher)
                if q_teacher.numel() == 2:
                    teacher_diff = float(q_teacher_values[0] - q_teacher_values[1])
                    if abs(teacher_diff) > 1e-8:
                        sign = 1.0 if teacher_diff > 0.0 else -1.0
                        pred_diff = q_pred_pair[0] - q_pred_pair[1]
                        loss_rank = F.softplus(torch.tensor(-sign, dtype=pred.dtype, device=device) * pred_diff)
                        if regime == 'dynamic':
                            rerank_diff = rerank_pred_pair[0] - rerank_pred_pair[1]
                            loss_temporal_rank = loss_temporal_rank + F.softplus(
                                torch.tensor(-sign, dtype=pred.dtype, device=device) * rerank_diff
                            )

            loss = (
                loss_final
                + marginal_loss_weight * loss_marginal
                + ranking_loss_weight * loss_rank
                + loss_temporal_rank
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
            losses_final.append(float(loss_final.item()))
            losses_marginal.append(float(loss_marginal.item()))
            losses_rank.append(float(loss_rank.item()))
            losses_temporal_rank.append(float(loss_temporal_rank.item()))
        history['loss'].append(float(np.mean(losses)))
        history['loss_final'].append(float(np.mean(losses_final)))
        history['loss_marginal'].append(float(np.mean(losses_marginal)))
        history['loss_rank'].append(float(np.mean(losses_rank)))
        history['loss_temporal_rank'].append(float(np.mean(losses_temporal_rank)))
    return DiffusorArtifacts(model=model, history=history)
