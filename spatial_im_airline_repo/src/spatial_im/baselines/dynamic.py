from __future__ import annotations

import numpy as np

from spatial_im.baselines.classical import spread_score


def weighted_degree_per_slice(edge_index: np.ndarray, edge_weight_t: np.ndarray, num_nodes: int, budget: int):
    src = edge_index[:, 0]
    scores = np.zeros(num_nodes, dtype=np.float32)
    for e, u in enumerate(src):
        scores[u] += edge_weight_t[e]
    return np.argsort(-scores)[:budget].tolist(), scores


def temporal_weighted_degree(edge_index: np.ndarray, edge_weight_seq: np.ndarray, num_nodes: int, budget: int, decay: float = 0.9):
    T = edge_weight_seq.shape[0]
    agg = np.zeros(num_nodes, dtype=np.float32)
    src = edge_index[:, 0]
    for t in range(T):
        coeff = decay ** (T - 1 - t)
        for e, u in enumerate(src):
            agg[u] += coeff * edge_weight_seq[t, e]
    return np.argsort(-agg)[:budget].tolist(), agg


def temporal_degree_discount(edge_index: np.ndarray, edge_weight_t: np.ndarray, num_nodes: int, budget: int, p: float = 0.1):
    neighbors = {i: set() for i in range(num_nodes)}
    src, dst = edge_index[:, 0], edge_index[:, 1]
    deg_w = np.zeros(num_nodes, dtype=np.float32)
    for e, (u, v) in enumerate(zip(src, dst)):
        neighbors[u].add(v)
        deg_w[u] += edge_weight_t[e]
    t = np.zeros(num_nodes, dtype=np.float32)
    dd = deg_w.copy()
    S = []
    for _ in range(budget):
        mask = np.ones(num_nodes, dtype=bool)
        mask[S] = False
        u = int(np.argmax(np.where(mask, dd, -1e9)))
        S.append(u)
        for v in neighbors[u]:
            if v in S:
                continue
            t[v] += 1
            dd[v] = deg_w[v] - 2 * t[v] - (deg_w[v] - t[v]) * t[v] * p
    return S


def myopic_lookahead(edge_index: np.ndarray, edge_weight_seq: np.ndarray, num_nodes: int, budget: int, horizon: int = 2, mc_rollouts: int = 32):
    """
    Greedy but only over a short future window; still myopic compared to RL.
    """
    selected = []
    remaining = set(range(num_nodes))
    for step in range(budget):
        best = None
        best_gain = -1e18
        window = edge_weight_seq[step:step + horizon]
        base = spread_score(edge_index, window, num_nodes, selected, mc_rollouts=mc_rollouts)
        for v in list(remaining):
            gain = spread_score(edge_index, window, num_nodes, selected + [v], mc_rollouts=mc_rollouts) - base
            if gain > best_gain:
                best_gain = gain
                best = v
        selected.append(best)
        remaining.remove(best)
    return selected


def temporal_ris_proxy(edge_index: np.ndarray, edge_weight_t: np.ndarray, num_nodes: int, budget: int, rr_sets: int = 200, seed: int = 7):
    """Simple RR-style temporal heuristic for a single current weighted slice."""
    rng = np.random.default_rng(seed)
    src, dst = edge_index[:, 0], edge_index[:, 1]
    reverse_adj = {i: [] for i in range(num_nodes)}
    for e, (u, v) in enumerate(zip(src, dst)):
        reverse_adj[v].append((u, edge_weight_t[e]))

    rr_collection = []
    for _ in range(rr_sets):
        root = int(rng.integers(0, num_nodes))
        frontier = [root]
        seen = {root}
        while frontier:
            cur = frontier.pop()
            for u, p in reverse_adj[cur]:
                if u not in seen and rng.random() < p:
                    seen.add(u)
                    frontier.append(u)
        rr_collection.append(seen)

    covered = set()
    selected = []
    for _ in range(budget):
        best, best_cov = None, -1
        for v in range(num_nodes):
            if v in selected:
                continue
            cov = sum(1 for i, rr in enumerate(rr_collection) if i not in covered and v in rr)
            if cov > best_cov:
                best_cov = cov
                best = v
        selected.append(best)
        for i, rr in enumerate(rr_collection):
            if best in rr:
                covered.add(i)
    return selected
