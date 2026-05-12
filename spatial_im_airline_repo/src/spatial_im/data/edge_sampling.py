from __future__ import annotations

import numpy as np
import torch


def topk_temporal_edge_neighborhood(
    edge_index: np.ndarray | torch.Tensor,
    edge_weight_seq: np.ndarray | torch.Tensor,
    distances_km: np.ndarray | torch.Tensor,
    num_nodes: int,
    top_k_out: int = 12,
    keep_best_incoming: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Keep a sparse, high-value temporal neighborhood for learned training/inference.

    The full raw graph is still available for Monte Carlo evaluation; this sampler is
    intended only for the learned diffusor / policy side to reduce runtime on larger
    graphs.
    """
    edge_index_np = edge_index.detach().cpu().numpy() if torch.is_tensor(edge_index) else np.asarray(edge_index)
    if edge_index_np.ndim == 2 and edge_index_np.shape[0] == 2:
        edge_index_np = edge_index_np.T
    W = edge_weight_seq.detach().cpu().numpy() if torch.is_tensor(edge_weight_seq) else np.asarray(edge_weight_seq)
    if W.ndim == 1:
        W = W[None, :]
    dist_np = distances_km.detach().cpu().numpy() if torch.is_tensor(distances_km) else np.asarray(distances_km)

    src = edge_index_np[:, 0].astype(np.int64)
    dst = edge_index_np[:, 1].astype(np.int64)
    mean_w = W.mean(axis=0).astype(np.float32)
    dist_pen = 1.0 / (1.0 + dist_np.astype(np.float32) / 1000.0)
    score = mean_w * dist_pen

    keep = np.zeros(edge_index_np.shape[0], dtype=bool)
    top_k_out = max(1, int(top_k_out))
    for u in range(int(num_nodes)):
        eids = np.flatnonzero(src == u)
        if eids.size == 0:
            continue
        chosen = eids[np.argsort(-score[eids])[:top_k_out]]
        keep[chosen] = True

    if keep_best_incoming:
        for v in range(int(num_nodes)):
            eids = np.flatnonzero(dst == v)
            if eids.size == 0:
                continue
            best = eids[int(np.argmax(score[eids]))]
            keep[best] = True

    kept_idx = np.flatnonzero(keep).astype(np.int64)
    return edge_index_np[kept_idx], W[:, kept_idx], dist_np[kept_idx], kept_idx
