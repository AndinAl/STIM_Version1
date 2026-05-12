from __future__ import annotations

from typing import List
import numpy as np
import networkx as nx


def build_temporal_weighted_graph(
    graph: nx.DiGraph,
    edge_index: np.ndarray,
    edge_weight_t: np.ndarray,
    distances_km: np.ndarray,
    weight_attr: str = 'temporal_weight',
    length_attr: str = 'temporal_length',
):
    g = graph.copy()
    for e, (u, v) in enumerate(edge_index):
        if not g.has_edge(int(u), int(v)):
            continue
        w = float(edge_weight_t[e])
        d = float(distances_km[e])
        g[int(u)][int(v)][weight_attr] = w
        g[int(u)][int(v)][length_attr] = d / max(w, 1e-6)
    return g


def weighted_strength(edge_index: np.ndarray, edge_weight_t: np.ndarray, num_nodes: int, budget: int):
    scores = np.zeros(num_nodes, dtype=np.float32)
    for e, (u, v) in enumerate(edge_index):
        scores[u] += edge_weight_t[e]
        scores[v] += edge_weight_t[e]
    return np.argsort(-scores)[:budget].tolist(), scores


def distance_strength(edge_index: np.ndarray, edge_weight_t: np.ndarray, distances_km: np.ndarray, num_nodes: int, budget: int):
    scores = np.zeros(num_nodes, dtype=np.float32)
    for e, (u, v) in enumerate(edge_index):
        contrib = edge_weight_t[e] / (1.0 + distances_km[e] / 1000.0)
        scores[u] += contrib
        scores[v] += contrib
    return np.argsort(-scores)[:budget].tolist(), scores


def weighted_betweenness(graph: nx.DiGraph, budget: int, weight_attr: str = 'distance_km'):
    und = graph.to_undirected()
    bet = nx.betweenness_centrality(und, normalized=True, weight=weight_attr)
    return [v for v, _ in sorted(bet.items(), key=lambda kv: kv[1], reverse=True)[:budget]], bet


def community_bridge_nodes(graph: nx.DiGraph, budget: int, weight_attr: str | None = None):
    und = graph.to_undirected()
    communities = list(nx.community.greedy_modularity_communities(und, weight=weight_attr))
    node_to_comm = {}
    for i, comm in enumerate(communities):
        for v in comm:
            node_to_comm[v] = i
    bridge_score = {v: 0.0 for v in graph.nodes()}
    for u, v, data in und.edges(data=True):
        if node_to_comm[u] != node_to_comm[v]:
            contrib = float(data.get(weight_attr, 1.0)) if weight_attr else 1.0
            bridge_score[u] += contrib
            bridge_score[v] += contrib
    ranked = [v for v, _ in sorted(bridge_score.items(), key=lambda kv: kv[1], reverse=True)[:budget]]
    return ranked, bridge_score


def cost_aware_ranking(edge_index: np.ndarray, edge_weight_t: np.ndarray, distances_km: np.ndarray, node_costs: np.ndarray, num_nodes: int, budget: int):
    strength = np.zeros(num_nodes, dtype=np.float32)
    for e, (u, v) in enumerate(edge_index):
        contrib = edge_weight_t[e] / (1.0 + distances_km[e] / 1000.0)
        strength[u] += contrib
        strength[v] += contrib
    score = strength / np.maximum(node_costs, 1e-6)
    return np.argsort(-score)[:budget].tolist(), score
