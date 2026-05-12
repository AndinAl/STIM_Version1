from __future__ import annotations

from typing import Iterable
import numpy as np
import networkx as nx
import torch

from spatial_im.baselines.classical import degree_discount, degree_ranking, greedy_spread
from spatial_im.baselines.dynamic import (
    temporal_degree_discount,
    temporal_ris_proxy,
    temporal_weighted_degree,
    weighted_degree_per_slice,
)
from spatial_im.baselines.spatial import cost_aware_ranking, distance_strength, weighted_strength
from spatial_im.evaluation.metrics import expected_coverage, intervention_cost


def _model_device(diffusor) -> torch.device:
    return next(diffusor.parameters()).device


def _as_float_tensor(x, device: torch.device) -> torch.Tensor:
    if torch.is_tensor(x):
        return x.to(device=device, dtype=torch.float32)
    return torch.as_tensor(x, dtype=torch.float32, device=device)


def _as_long_tensor(x, device: torch.device) -> torch.Tensor:
    if torch.is_tensor(x):
        return x.to(device=device, dtype=torch.long)
    return torch.as_tensor(x, dtype=torch.long, device=device)


def infer_edge_gate_scores(diffusor, edge_weight_seq, distances_km, reduce: str = 'mean') -> np.ndarray:
    """
    Expose learned edge-transition scores from the diffusor.

    The current diffusor applies a learned nonnegative edge gate in every message-passing
    layer. This function aggregates those per-layer gates into one score per edge so
    non-RL baselines can consume the same learned edge signal as the environment.
    """
    device = _model_device(diffusor)
    edge_weight_seq_t = _as_float_tensor(edge_weight_seq, device)
    distances_t = _as_float_tensor(distances_km, device)
    if edge_weight_seq_t.dim() == 1:
        edge_weight_seq_t = edge_weight_seq_t.unsqueeze(0)
    if not diffusor.layers:
        return edge_weight_seq_t.detach().cpu().numpy()

    dist_penalty = 1.0 / (1.0 + distances_t / 1000.0)
    inferred = []
    with torch.no_grad():
        for t in range(edge_weight_seq_t.size(0)):
            edge_feat = torch.stack([edge_weight_seq_t[t], dist_penalty], dim=-1)
            layer_scores = torch.stack([layer.edge_gate(edge_feat) for layer in diffusor.layers], dim=0)
            if reduce == 'mean':
                score_t = layer_scores.mean(dim=0)
            elif reduce == 'prod':
                score_t = layer_scores.prod(dim=0)
            else:
                raise ValueError(f'Unsupported reduce mode: {reduce}')
            inferred.append(score_t.detach().cpu().numpy().astype(np.float32))
    return np.stack(inferred, axis=0)


def surrogate_final_activation(
    diffusor,
    node_features,
    edge_index,
    edge_weight_seq,
    distances_km,
    selected: Iterable[int],
) -> np.ndarray:
    device = _model_device(diffusor)
    node_features_t = _as_float_tensor(node_features, device)
    edge_index_t = _as_long_tensor(edge_index, device)
    edge_weight_seq_t = _as_float_tensor(edge_weight_seq, device)
    distances_t = _as_float_tensor(distances_km, device)
    if edge_index_t.dim() == 2 and edge_index_t.size(0) != 2:
        edge_index_t = edge_index_t.T
    if edge_weight_seq_t.dim() == 1:
        edge_weight_seq_t = edge_weight_seq_t.unsqueeze(0)

    seed_mask = torch.zeros(node_features_t.size(0), dtype=torch.float32, device=device)
    idx = list(selected)
    if idx:
        seed_mask[torch.as_tensor(idx, dtype=torch.long, device=device)] = 1.0

    with torch.no_grad():
        probs = diffusor.rollout(
            node_features=node_features_t,
            seed_mask=seed_mask,
            edge_index=edge_index_t,
            edge_weight_seq=edge_weight_seq_t,
            distances_km=distances_t,
        )
    return probs.detach().cpu().numpy()


def surrogate_objective(
    diffusor,
    node_features,
    edge_index,
    edge_weight_seq,
    distances_km,
    coords,
    node_costs,
    selected: Iterable[int],
    regime: str = 'dynamic',
    beta_coverage: float = 0.0,
    lambda_cost: float = 0.0,
    coverage_radius_km: float = 900.0,
    cost_mode: str = 'geography',
    constant_cost: float = 1.0,
    distance_cost_scale: float = 0.0015,
) -> tuple[float, np.ndarray]:
    selected = list(selected)
    final_active = surrogate_final_activation(
        diffusor=diffusor,
        node_features=node_features,
        edge_index=edge_index,
        edge_weight_seq=edge_weight_seq,
        distances_km=distances_km,
        selected=selected,
    )
    activated = float(final_active.sum())
    if regime == 'spread':
        return activated, final_active

    coverage = expected_coverage(np.asarray(coords), final_active, coverage_radius_km)
    cost = intervention_cost(
        selected=selected,
        coords=np.asarray(coords),
        node_costs=np.asarray(node_costs),
        mode=cost_mode,
        constant_cost=constant_cost,
        distance_cost_scale=distance_cost_scale,
    )
    objective = activated + beta_coverage * coverage - lambda_cost * cost
    return float(objective), final_active


def greedy_surrogate_objective(
    diffusor,
    node_features,
    edge_index,
    edge_weight_seq,
    distances_km,
    coords,
    node_costs,
    num_nodes: int,
    budget: int,
    regime: str = 'dynamic',
    beta_coverage: float = 0.0,
    lambda_cost: float = 0.0,
    coverage_radius_km: float = 900.0,
    cost_mode: str = 'geography',
    constant_cost: float = 1.0,
    distance_cost_scale: float = 0.0015,
) -> list[int]:
    selected: list[int] = []
    current = set()
    for _ in range(budget):
        base, _ = surrogate_objective(
            diffusor=diffusor,
            node_features=node_features,
            edge_index=edge_index,
            edge_weight_seq=edge_weight_seq,
            distances_km=distances_km,
            coords=coords,
            node_costs=node_costs,
            selected=current,
            regime=regime,
            beta_coverage=beta_coverage,
            lambda_cost=lambda_cost,
            coverage_radius_km=coverage_radius_km,
            cost_mode=cost_mode,
            constant_cost=constant_cost,
            distance_cost_scale=distance_cost_scale,
        )
        best_node = None
        best_gain = -1e18
        for v in range(num_nodes):
            if v in current:
                continue
            score, _ = surrogate_objective(
                diffusor=diffusor,
                node_features=node_features,
                edge_index=edge_index,
                edge_weight_seq=edge_weight_seq,
                distances_km=distances_km,
                coords=coords,
                node_costs=node_costs,
                selected=current | {v},
                regime=regime,
                beta_coverage=beta_coverage,
                lambda_cost=lambda_cost,
                coverage_radius_km=coverage_radius_km,
                cost_mode=cost_mode,
                constant_cost=constant_cost,
                distance_cost_scale=distance_cost_scale,
            )
            gain = score - base
            if gain > best_gain:
                best_gain = gain
                best_node = v
        current.add(best_node)
        selected.append(best_node)
    return selected


def greedy_surrogate_marginal(
    diffusor,
    node_features,
    edge_index,
    edge_weight_seq,
    distances_km,
    num_nodes: int,
    budget: int,
    dynamic_summaries: np.ndarray | None = None,
    prefilter_top_m: int | None = None,
) -> list[int]:
    """
    Fast learned-greedy selection using the diffusor marginal head directly.

    This avoids raw Monte Carlo greedy selection and also avoids repeatedly
    rerunning the full surrogate objective for every candidate.
    """
    device = _model_device(diffusor)
    node_features_t = _as_float_tensor(node_features, device)
    edge_index_t = _as_long_tensor(edge_index, device)
    edge_weight_seq_t = _as_float_tensor(edge_weight_seq, device)
    distances_t = _as_float_tensor(distances_km, device)
    if edge_index_t.dim() == 2 and edge_index_t.size(0) != 2:
        edge_index_t = edge_index_t.T
    if edge_weight_seq_t.dim() == 1:
        edge_weight_seq_t = edge_weight_seq_t.unsqueeze(0)
    if dynamic_summaries is None:
        dynamic_summaries = np.zeros((num_nodes, diffusor.dynamic_summary_dim), dtype=np.float32)
    dynamic_summaries_t = _as_float_tensor(dynamic_summaries, device)

    seed_mask = torch.zeros(num_nodes, dtype=torch.float32, device=device)
    current_active = torch.zeros_like(seed_mask)
    with torch.no_grad():
        cache_seq = diffusor.build_temporal_cache(
            node_features=node_features_t,
            edge_index=edge_index_t,
            edge_weight_seq=edge_weight_seq_t,
            distances_km=distances_t,
            window_len=diffusor.temporal_window_len,
        )

    selected: list[int] = []
    for step in range(int(budget)):
        with torch.no_grad():
            temporal_embeddings, _ = diffusor.condition_temporal_cache(cache_seq, seed_mask, current_active)
            remaining_budget_frac = float(max(int(budget) - len(selected), 0) / max(int(budget), 1))
            step_frac = float(step / max(int(budget), 1))
            q_pred = diffusor.predict_marginals(
                temporal_embeddings=temporal_embeddings,
                dynamic_summaries=dynamic_summaries_t,
                seed_mask=seed_mask,
                remaining_budget_frac=remaining_budget_frac,
                step_frac=step_frac,
            )
            scores = q_pred.clone()
            scores[seed_mask > 0.5] = -1e9
            if prefilter_top_m is not None:
                legal_idx = torch.where(seed_mask < 0.5)[0]
                if int(legal_idx.numel()) > int(prefilter_top_m):
                    top = torch.topk(scores[legal_idx], k=int(prefilter_top_m)).indices
                    filtered = legal_idx[top]
                    mask = torch.zeros_like(scores, dtype=torch.bool)
                    mask[filtered] = True
                    scores[~mask] = -1e9
            action = int(torch.argmax(scores).item())
        selected.append(action)
        seed_mask[action] = 1.0
        current_active[action] = 1.0
    return selected


def inferred_weighted_degree_per_slice(
    diffusor,
    edge_index: np.ndarray,
    edge_weight_t: np.ndarray,
    distances_km: np.ndarray,
    num_nodes: int,
    budget: int,
    reduce: str = 'mean',
):
    inferred_t = infer_edge_gate_scores(diffusor, edge_weight_t, distances_km, reduce=reduce)[0]
    return weighted_degree_per_slice(edge_index, inferred_t, num_nodes, budget)


def inferred_temporal_weighted_degree(
    diffusor,
    edge_index: np.ndarray,
    edge_weight_seq: np.ndarray,
    distances_km: np.ndarray,
    num_nodes: int,
    budget: int,
    decay: float = 0.9,
    reduce: str = 'mean',
):
    inferred_seq = infer_edge_gate_scores(diffusor, edge_weight_seq, distances_km, reduce=reduce)
    return temporal_weighted_degree(edge_index, inferred_seq, num_nodes, budget, decay=decay)


def inferred_temporal_degree_discount(
    diffusor,
    edge_index: np.ndarray,
    edge_weight_t: np.ndarray,
    distances_km: np.ndarray,
    num_nodes: int,
    budget: int,
    p: float = 0.1,
    reduce: str = 'mean',
):
    inferred_t = infer_edge_gate_scores(diffusor, edge_weight_t, distances_km, reduce=reduce)[0]
    return temporal_degree_discount(edge_index, inferred_t, num_nodes, budget, p=p)


def inferred_temporal_ris_proxy(
    diffusor,
    edge_index: np.ndarray,
    edge_weight_t: np.ndarray,
    distances_km: np.ndarray,
    num_nodes: int,
    budget: int,
    rr_sets: int = 200,
    seed: int = 7,
    reduce: str = 'mean',
):
    inferred_t = infer_edge_gate_scores(diffusor, edge_weight_t, distances_km, reduce=reduce)[0]
    return temporal_ris_proxy(edge_index, inferred_t, num_nodes, budget, rr_sets=rr_sets, seed=seed)


def surrogate_myopic_lookahead(
    diffusor,
    node_features,
    edge_index,
    edge_weight_seq,
    distances_km,
    coords,
    node_costs,
    num_nodes: int,
    budget: int,
    regime: str = 'dynamic',
    horizon: int = 2,
    beta_coverage: float = 0.0,
    lambda_cost: float = 0.0,
    coverage_radius_km: float = 900.0,
    cost_mode: str = 'geography',
    constant_cost: float = 1.0,
    distance_cost_scale: float = 0.0015,
) -> list[int]:
    edge_weight_seq = np.asarray(edge_weight_seq, dtype=np.float32)
    selected: list[int] = []
    remaining = set(range(num_nodes))
    for step in range(budget):
        start = min(step, max(edge_weight_seq.shape[0] - 1, 0))
        stop = min(edge_weight_seq.shape[0], start + horizon)
        window = edge_weight_seq[start:stop]
        best = None
        best_gain = -1e18
        base, _ = surrogate_objective(
            diffusor=diffusor,
            node_features=node_features,
            edge_index=edge_index,
            edge_weight_seq=window,
            distances_km=distances_km,
            coords=coords,
            node_costs=node_costs,
            selected=selected,
            regime=regime,
            beta_coverage=beta_coverage,
            lambda_cost=lambda_cost,
            coverage_radius_km=coverage_radius_km,
            cost_mode=cost_mode,
            constant_cost=constant_cost,
            distance_cost_scale=distance_cost_scale,
        )
        for v in list(remaining):
            score, _ = surrogate_objective(
                diffusor=diffusor,
                node_features=node_features,
                edge_index=edge_index,
                edge_weight_seq=window,
                distances_km=distances_km,
                coords=coords,
                node_costs=node_costs,
                selected=selected + [v],
                regime=regime,
                beta_coverage=beta_coverage,
                lambda_cost=lambda_cost,
                coverage_radius_km=coverage_radius_km,
                cost_mode=cost_mode,
                constant_cost=constant_cost,
                distance_cost_scale=distance_cost_scale,
            )
            gain = score - base
            if gain > best_gain:
                best_gain = gain
                best = v
        selected.append(best)
        remaining.remove(best)
    return selected


def inferred_weighted_strength(
    diffusor,
    edge_index: np.ndarray,
    edge_weight_t: np.ndarray,
    distances_km: np.ndarray,
    num_nodes: int,
    budget: int,
    reduce: str = 'mean',
):
    inferred_t = infer_edge_gate_scores(diffusor, edge_weight_t, distances_km, reduce=reduce)[0]
    return weighted_strength(edge_index, inferred_t, num_nodes, budget)


def inferred_distance_strength(
    diffusor,
    edge_index: np.ndarray,
    edge_weight_t: np.ndarray,
    distances_km: np.ndarray,
    num_nodes: int,
    budget: int,
    reduce: str = 'mean',
):
    inferred_t = infer_edge_gate_scores(diffusor, edge_weight_t, distances_km, reduce=reduce)[0]
    return distance_strength(edge_index, inferred_t, distances_km, num_nodes, budget)


def inferred_cost_aware_ranking(
    diffusor,
    edge_index: np.ndarray,
    edge_weight_t: np.ndarray,
    distances_km: np.ndarray,
    node_costs: np.ndarray,
    num_nodes: int,
    budget: int,
    reduce: str = 'mean',
):
    inferred_t = infer_edge_gate_scores(diffusor, edge_weight_t, distances_km, reduce=reduce)[0]
    return cost_aware_ranking(edge_index, inferred_t, distances_km, node_costs, num_nodes, budget)


def graph_with_edge_scores(
    graph: nx.DiGraph,
    edge_index: np.ndarray,
    edge_scores: np.ndarray,
    edge_cost_attr: str = 'edge_cost',
    edge_weight_attr: str = 'edge_weight',
) -> nx.DiGraph:
    g = graph.copy()
    for e, (u, v) in enumerate(edge_index):
        score = float(edge_scores[e])
        if g.has_edge(int(u), int(v)):
            g[int(u)][int(v)][edge_weight_attr] = score
            g[int(u)][int(v)][edge_cost_attr] = 1.0 / max(score, 1e-6)
    return g


def inferred_graph_degree_ranking(graph: nx.DiGraph, budget: int):
    return degree_ranking(graph, budget)


def inferred_graph_degree_discount(graph: nx.DiGraph, budget: int, p: float = 0.1):
    return degree_discount(graph, budget, p=p)


def inferred_greedy_spread(
    edge_index: np.ndarray,
    inferred_edge_prob_seq: np.ndarray,
    num_nodes: int,
    budget: int,
    mc_rollouts: int = 64,
):
    return greedy_spread(edge_index, inferred_edge_prob_seq, num_nodes, budget, mc_rollouts=mc_rollouts)


def inferred_weighted_betweenness(
    graph: nx.DiGraph,
    edge_index: np.ndarray,
    inferred_edge_weight_t: np.ndarray,
    budget: int,
):
    g = graph_with_edge_scores(graph, edge_index, inferred_edge_weight_t, edge_cost_attr='inferred_cost', edge_weight_attr='inferred_weight')
    und = g.to_undirected()
    bet = nx.betweenness_centrality(und, normalized=True, weight='inferred_cost')
    ranked = [v for v, _ in sorted(bet.items(), key=lambda kv: kv[1], reverse=True)[:budget]]
    return ranked, bet


def inferred_community_bridge_nodes(
    graph: nx.DiGraph,
    edge_index: np.ndarray,
    inferred_edge_weight_t: np.ndarray,
    budget: int,
):
    g = graph_with_edge_scores(graph, edge_index, inferred_edge_weight_t, edge_cost_attr='inferred_cost', edge_weight_attr='inferred_weight')
    und = g.to_undirected()
    communities = list(nx.community.greedy_modularity_communities(und, weight='inferred_weight'))
    node_to_comm = {}
    for i, comm in enumerate(communities):
        for v in comm:
            node_to_comm[v] = i
    bridge_score = {v: 0.0 for v in g.nodes()}
    for u, v, data in und.edges(data=True):
        if node_to_comm[u] != node_to_comm[v]:
            w = float(data.get('inferred_weight', 1.0))
            bridge_score[u] += w
            bridge_score[v] += w
    ranked = [v for v, _ in sorted(bridge_score.items(), key=lambda kv: kv[1], reverse=True)[:budget]]
    return ranked, bridge_score
