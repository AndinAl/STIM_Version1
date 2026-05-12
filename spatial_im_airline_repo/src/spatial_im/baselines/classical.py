from __future__ import annotations

import numpy as np

from spatial_im.diffusion.simulator import estimate_final_activation_prob


def spread_score(edge_index, edge_prob_seq, num_nodes, seed_set, mc_rollouts=64):
    seed_mask = np.zeros(num_nodes, dtype=np.float32)
    seed_mask[list(seed_set)] = 1.0
    probs = estimate_final_activation_prob(edge_index, edge_prob_seq, seed_mask, mc_rollouts=mc_rollouts)
    return float(probs.sum())


def greedy_spread(edge_index, edge_prob_seq, num_nodes, budget: int, mc_rollouts: int = 64):
    """Classical spread-oriented greedy baseline on a frozen snapshot/horizon."""
    selected = []
    current = set()
    for _ in range(budget):
        best_node = None
        best_gain = -1e18
        base = spread_score(edge_index, edge_prob_seq, num_nodes, current, mc_rollouts=mc_rollouts)
        for v in range(num_nodes):
            if v in current:
                continue
            gain = spread_score(edge_index, edge_prob_seq, num_nodes, current | {v}, mc_rollouts=mc_rollouts) - base
            if gain > best_gain:
                best_gain = gain
                best_node = v
        current.add(best_node)
        selected.append(best_node)
    return selected


def degree_ranking(graph, budget: int):
    deg = dict(graph.degree())
    return [v for v, _ in sorted(deg.items(), key=lambda kv: kv[1], reverse=True)[:budget]]


def degree_discount(graph, budget: int, p: float = 0.1):
    """Chen et al.-style degree discount heuristic."""
    nodes = list(graph.nodes())
    d = {u: graph.degree(u) for u in nodes}
    t = {u: 0 for u in nodes}
    dd = d.copy()
    S = []
    for _ in range(budget):
        u = max((n for n in nodes if n not in S), key=lambda n: dd[n])
        S.append(u)
        for v in graph.neighbors(u):
            if v in S:
                continue
            t[v] += 1
            dd[v] = d[v] - 2 * t[v] - (d[v] - t[v]) * t[v] * p
    return S
