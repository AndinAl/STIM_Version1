from __future__ import annotations

import numpy as np
import torch
import networkx as nx


def compute_static_spatial_features(graph: nx.DiGraph, coords: np.ndarray, node_costs: np.ndarray):
    und = graph.to_undirected()
    n = graph.number_of_nodes()
    bet = nx.betweenness_centrality(und, normalized=True)
    deg = dict(graph.degree())
    out_deg = dict(graph.out_degree())
    in_deg = dict(graph.in_degree())
    features = np.zeros((n, 6), dtype=np.float32)
    for i in range(n):
        features[i] = [
            deg.get(i, 0.0),
            out_deg.get(i, 0.0),
            in_deg.get(i, 0.0),
            bet.get(i, 0.0),
            coords[i, 0],
            coords[i, 1],
        ]
    # normalize and append cost later in state build.
    features[:, :4] = (features[:, :4] - features[:, :4].mean(axis=0)) / (features[:, :4].std(axis=0) + 1e-6)
    features[:, 4:] = (features[:, 4:] - features[:, 4:].mean(axis=0)) / (features[:, 4:].std(axis=0) + 1e-6)
    node_costs_norm = (node_costs - node_costs.mean()) / (node_costs.std() + 1e-6)
    return np.concatenate([features, node_costs_norm[:, None]], axis=1).astype(np.float32)


def compute_dynamic_temporal_summaries(
    edge_index: np.ndarray,
    edge_weight_seq: np.ndarray,
    distances_km: np.ndarray,
    num_nodes: int,
    window_len: int = 4,
    horizon_len: int | None = None,
) -> np.ndarray:
    """
    Build explicit dynamic candidate summaries from raw temporal edge weights.

    The summaries use the visible temporal window starting at the current episode
    start snapshot. To keep the state dimension fixed for RL, the first
    `window_len` snapshots are used and zero-padded when fewer are available.
    """
    edge_weight_seq = np.asarray(edge_weight_seq, dtype=np.float32)
    if edge_weight_seq.ndim == 1:
        edge_weight_seq = edge_weight_seq[None, :]
    if edge_index.ndim == 2 and edge_index.shape[0] == 2:
        src = edge_index[0].astype(np.int64)
        dst = edge_index[1].astype(np.int64)
    else:
        src = edge_index[:, 0].astype(np.int64)
        dst = edge_index[:, 1].astype(np.int64)

    effective_T = edge_weight_seq.shape[0] if horizon_len is None else max(1, min(int(horizon_len), edge_weight_seq.shape[0]))
    visible_seq = edge_weight_seq[:effective_T]
    L = max(1, int(window_len))
    E = visible_seq.shape[1]
    window = np.zeros((L, E), dtype=np.float32)
    used = min(L, visible_seq.shape[0])
    window[:used] = visible_seq[:used]

    out_hist = np.zeros((num_nodes, L), dtype=np.float32)
    in_hist = np.zeros((num_nodes, L), dtype=np.float32)
    dyn_degree = np.zeros(num_nodes, dtype=np.float32)
    spatial_strength = np.zeros(num_nodes, dtype=np.float32)
    dist_penalty = 1.0 / (1.0 + np.asarray(distances_km, dtype=np.float32) / 1000.0)

    for k in range(L):
        weights = window[k]
        np.add.at(out_hist[:, k], src, weights)
        np.add.at(in_hist[:, k], dst, weights)

    if used > 1:
        for k in range(1, used):
            edge_gain = np.maximum(window[k] - window[k - 1], 0.0)
            np.add.at(dyn_degree, src, edge_gain)

    w_now = window[0]
    spatial_contrib = w_now * dist_penalty
    np.add.at(spatial_strength, src, spatial_contrib)
    np.add.at(spatial_strength, dst, spatial_contrib)

    wout_now = out_hist[:, 0]
    win_now = in_hist[:, 0]
    if used > 1:
        trend_out = out_hist[:, used - 1] - out_hist[:, 0]
        trend_in = in_hist[:, used - 1] - in_hist[:, 0]
    else:
        trend_out = np.zeros(num_nodes, dtype=np.float32)
        trend_in = np.zeros(num_nodes, dtype=np.float32)

    feats = np.concatenate(
        [
            wout_now[:, None],
            win_now[:, None],
            out_hist,
            in_hist,
            trend_out[:, None],
            trend_in[:, None],
            dyn_degree[:, None],
            spatial_strength[:, None],
        ],
        axis=1,
    ).astype(np.float32)
    feats = (feats - feats.mean(axis=0, keepdims=True)) / (feats.std(axis=0, keepdims=True) + 1e-6)
    return feats.astype(np.float32)


def build_candidate_features(
    temporal_embeddings: torch.Tensor,
    seed_mask: torch.Tensor,
    current_active: torch.Tensor,
    predicted_final: torch.Tensor,
    surrogate_marginals: torch.Tensor | None,
    candidate_rerank_scores: torch.Tensor | None,
    remaining_budget_frac: float,
    step_frac: float,
    static_summaries: torch.Tensor | None = None,
    dynamic_summaries: torch.Tensor | None = None,
    node_cost_feature: torch.Tensor | None = None,
) -> torch.Tensor:
    n = temporal_embeddings.size(0)
    global_feats = torch.tensor([remaining_budget_frac, step_frac], dtype=torch.float32, device=temporal_embeddings.device).repeat(n, 1)
    parts = [temporal_embeddings]
    if static_summaries is not None:
        parts.append(static_summaries)
    if dynamic_summaries is not None:
        parts.append(dynamic_summaries)
    parts.extend([
        seed_mask.unsqueeze(-1),
        current_active.unsqueeze(-1),
        predicted_final.unsqueeze(-1),
    ])
    if surrogate_marginals is not None:
        parts.append(surrogate_marginals.unsqueeze(-1))
    if candidate_rerank_scores is not None:
        parts.append(candidate_rerank_scores.unsqueeze(-1))
    parts.append(global_feats)
    if node_cost_feature is not None:
        parts.append(node_cost_feature.unsqueeze(-1))
    return torch.cat(parts, dim=-1)
