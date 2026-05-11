#!/usr/bin/env python3
"""
generate_airline_topological_synthetic.py

Generate synthetic airline-style spatial-temporal weighted graph snapshots for
RL/GNN influence-maximization experiments.

Properties included:
1. Sparse graph with controlled average degree.
2. Right-skewed / truncated-heavy-tailed degree distribution.
3. Hub-and-spoke + regional connectivity.
4. Core / bridge / periphery airport hierarchy.
5. Spatial distance penalty: long links mostly connect to hubs.
6. Higher clustering than a random graph with the same N and E.
7. Weighted traffic concentration: high-degree hubs carry more traffic.
8. Time-varying route weights over fixed topology.
9. RL-ready diffusion probability p_ij(t) in [0, 1].

Example:
    python generate_airline_topological_synthetic.py --out data/generated --preset paper

For a smaller/sparser setup close to your current data:
    python generate_airline_topological_synthetic.py --out data/generated --preset sparse
"""

from __future__ import annotations

import argparse
import json
import math
from collections import deque
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


PAPER_ALIGNED_SPECS = [
    {"name": "source_100",  "n": 100,  "graphs": 3, "avg_degree": 5.8,  "snapshots": 120},
    {"name": "source_200",  "n": 200,  "graphs": 3, "avg_degree": 6.7,  "snapshots": 180},
    {"name": "target_300",  "n": 300,  "graphs": 1, "avg_degree": 8.4,  "snapshots": 220},
    {"name": "target_500",  "n": 500,  "graphs": 1, "avg_degree": 9.7,  "snapshots": 240},
    {"name": "target_1000", "n": 1000, "graphs": 1, "avg_degree": 12.0, "snapshots": 360},
]

SPARSE_TRANSFER_SPECS = [
    {"name": "source_100",  "n": 100,  "graphs": 3, "avg_degree": 5.2, "snapshots": 120},
    {"name": "source_200",  "n": 200,  "graphs": 3, "avg_degree": 5.3, "snapshots": 220},
    {"name": "target_300",  "n": 300,  "graphs": 1, "avg_degree": 5.3, "snapshots": 220},
    {"name": "target_500",  "n": 500,  "graphs": 1, "avg_degree": 5.3, "snapshots": 220},
    {"name": "target_1000", "n": 1000, "graphs": 1, "avg_degree": 5.5, "snapshots": 240},
]


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0088
    lat1 = np.radians(np.asarray(lat1, dtype=float))
    lon1 = np.radians(np.asarray(lon1, dtype=float))
    lat2 = np.radians(np.asarray(lat2, dtype=float))
    lon2 = np.radians(np.asarray(lon2, dtype=float))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def gini(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return 0.0
    if np.min(x) < 0:
        x = x - np.min(x)
    if np.sum(x) == 0:
        return 0.0
    x = np.sort(x)
    n = len(x)
    return float((2 * np.arange(1, n + 1).dot(x) / (n * x.sum())) - (n + 1) / n)


def fit_power_exponent(x: np.ndarray, y: np.ndarray) -> float | None:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = (x > 0) & (y > 0) & np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return None
    b, _a = np.polyfit(np.log(x[mask]), np.log(y[mask]), 1)
    return float(b)


def generate_airports(n: int, rng: np.random.Generator, core_frac: float = 0.08, bridge_frac: float = 0.27) -> pd.DataFrame:
    n_regions = int(np.clip(round(np.sqrt(n) / 2), 4, 14))
    center_lat = rng.uniform(26, 48, size=n_regions)
    center_lon = rng.uniform(-123, -68, size=n_regions)
    region_market = rng.lognormal(mean=0.0, sigma=0.55, size=n_regions)
    region_probs = rng.dirichlet(np.ones(n_regions) * 1.2)
    regions = rng.choice(np.arange(n_regions), size=n, p=region_probs)

    lats = np.empty(n)
    lons = np.empty(n)
    for r in range(n_regions):
        idx = np.where(regions == r)[0]
        if len(idx) == 0:
            continue
        lats[idx] = rng.normal(center_lat[r], 1.8, size=len(idx))
        lons[idx] = rng.normal(center_lon[r], 2.4, size=len(idx))
    lats = np.clip(lats, 24.5, 49.5)
    lons = np.clip(lons, -125.0, -66.0)

    local = rng.lognormal(mean=0.0, sigma=1.0, size=n)
    fitness = local * region_market[regions]
    fitness = fitness / fitness.max()

    order = np.argsort(-fitness)
    layer = np.array(["periphery"] * n, dtype=object)
    n_core = max(3, int(round(core_frac * n)))
    n_bridge = max(5, int(round(bridge_frac * n)))
    layer[order[:n_core]] = "core"
    layer[order[n_core:n_core + n_bridge]] = "bridge"

    demand = (50_000 + 20_000_000 * fitness ** 1.4 * rng.lognormal(0, 0.25, n)).astype(int)
    cost = 1.0 / (np.sqrt(fitness) + 0.05)
    cost = cost / cost.max()

    return pd.DataFrame({
        "node_id": np.arange(n, dtype=int),
        "airport_id": [f"A{i:04d}" for i in range(n)],
        "lat": lats,
        "lon": lons,
        "region_id": regions,
        "name": [f"Synthetic Airport {i:04d}" for i in range(n)],
        "city": [f"City_{int(regions[i]):02d}" for i in range(n)],
        "country": "SYN",
        "region": [f"SYN-R{int(r):02d}" for r in regions],
        "fitness": fitness,
        "demand": demand,
        "cost": cost,
        "layer": layer,
    })


def layer_pair_key(layer_i: str, layer_j: str) -> Tuple[str, str]:
    return tuple(sorted((str(layer_i), str(layer_j))))


LAYER_BIAS = {
    ("core", "core"): 4.50,
    ("bridge", "core"): 3.00,
    ("core", "periphery"): 1.55,
    ("bridge", "bridge"): 1.35,
    ("bridge", "periphery"): 0.65,
    ("periphery", "periphery"): 0.12,
}

LAYER_FLOW_BIAS = {
    ("core", "core"): 4.50,
    ("bridge", "core"): 3.20,
    ("core", "periphery"): 1.55,
    ("bridge", "bridge"): 1.25,
    ("bridge", "periphery"): 0.70,
    ("periphery", "periphery"): 0.18,
}


def build_pair_scores(airports: pd.DataFrame, rng: np.random.Generator, d0_topology_km: float = 1650.0, theta: float = 0.62) -> pd.DataFrame:
    n = len(airports)
    lat = airports["lat"].to_numpy()
    lon = airports["lon"].to_numpy()
    fitness = airports["fitness"].to_numpy()
    layers = airports["layer"].to_numpy()
    regions = airports["region_id"].to_numpy()

    rows = []
    for i in range(n):
        js = np.arange(i + 1, n)
        if len(js) == 0:
            continue
        dist = haversine_km(lat[i], lon[i], lat[js], lon[js])
        fprod = np.sqrt(fitness[i] * fitness[js])
        spatial_decay = np.exp(-dist / d0_topology_km)
        lb = np.array([LAYER_BIAS[layer_pair_key(layers[i], layers[j])] for j in js])
        same_region = np.where(regions[i] == regions[js], 1.65, 1.0)
        any_core = np.array([(layers[i] == "core") or (layers[j] == "core") for j in js])
        both_core = np.array([(layers[i] == "core") and (layers[j] == "core") for j in js])
        long_link = dist > 1800
        long_factor = np.ones_like(dist, dtype=float)
        long_factor[long_link & (~any_core)] *= 0.18
        long_factor[long_link & any_core] *= 1.25
        long_factor[long_link & both_core] *= 1.70
        jitter = rng.lognormal(mean=0.0, sigma=0.30, size=len(js))
        score = (fprod ** theta) * spatial_decay * lb * same_region * long_factor * jitter
        score = np.maximum(score, 1e-12)
        rows.extend(zip(np.repeat(i, len(js)), js, dist, score))
    return pd.DataFrame(rows, columns=["i", "j", "distance_km", "score"])


def add_edge(edge_set: set[Tuple[int, int]], i: int, j: int):
    if i == j:
        return
    if i > j:
        i, j = j, i
    edge_set.add((int(i), int(j)))


def generate_topology(airports: pd.DataFrame, target_avg_degree: float, rng: np.random.Generator, triadic_fraction: float = 0.12) -> pd.DataFrame:
    n = len(airports)
    target_edges = max(n - 1, int(round(target_avg_degree * n / 2)))
    pairs = build_pair_scores(airports, rng)
    pair_index = {(int(r.i), int(r.j)): idx for idx, r in pairs.iterrows()}

    layers = airports["layer"].to_numpy()
    regions = airports["region_id"].to_numpy()
    core_nodes = np.where(layers == "core")[0]
    bridge_nodes = np.where(layers == "bridge")[0]
    periph_nodes = np.where(layers == "periphery")[0]

    edge_set: set[Tuple[int, int]] = set()

    # Dense core backbone.
    core_pairs = []
    for a in range(len(core_nodes)):
        for b in range(a + 1, len(core_nodes)):
            i, j = int(core_nodes[a]), int(core_nodes[b])
            idx = pair_index.get((min(i, j), max(i, j)))
            if idx is not None:
                core_pairs.append((pairs.loc[idx, "score"], i, j))
    core_pairs.sort(reverse=True)
    max_core_edges = min(len(core_pairs), max(1, int(0.22 * target_edges)))
    for _s, i, j in core_pairs[:max_core_edges]:
        add_edge(edge_set, i, j)

    # Attach non-core airports to nearby/strong core nodes.
    for i in np.concatenate([bridge_nodes, periph_nodes]):
        cand = []
        for c in core_nodes:
            a, b = min(int(i), int(c)), max(int(i), int(c))
            idx = pair_index.get((a, b))
            if idx is not None:
                cand.append((pairs.loc[idx, "score"], int(c)))
        cand.sort(reverse=True)
        attach_count = 2 if layers[i] == "bridge" and len(cand) >= 2 else 1
        for _score, c in cand[:attach_count]:
            add_edge(edge_set, int(i), c)

    # Sample gravity-distance-layer candidate edges.
    pre_closure_target = max(len(edge_set), int(round((1.0 - triadic_fraction) * target_edges)))
    remaining_pairs = pairs.copy()
    remaining_pairs["edge"] = list(zip(remaining_pairs["i"].astype(int), remaining_pairs["j"].astype(int)))
    remaining_pairs = remaining_pairs[~remaining_pairs["edge"].isin(edge_set)]
    need = pre_closure_target - len(edge_set)
    if need > 0 and len(remaining_pairs) > 0:
        prob = remaining_pairs["score"].to_numpy()
        prob = prob / prob.sum()
        chosen_idx = rng.choice(remaining_pairs.index.to_numpy(), size=min(need, len(remaining_pairs)), replace=False, p=prob)
        for idx in chosen_idx:
            r = pairs.loc[idx]
            add_edge(edge_set, int(r.i), int(r.j))

    # Triadic closure.
    adj = [set() for _ in range(n)]
    for i, j in edge_set:
        adj[i].add(j)
        adj[j].add(i)

    closure_budget = target_edges - len(edge_set)
    closure_candidates = []
    high_nodes = np.argsort(-np.array([len(adj[i]) for i in range(n)]))[:max(5, int(0.15 * n))]
    for u in high_nodes:
        neigh = list(adj[u])
        if len(neigh) < 2:
            continue
        tries = min(300, len(neigh) * (len(neigh) - 1) // 2)
        for _ in range(tries):
            a, b = rng.choice(neigh, size=2, replace=False)
            i, j = min(int(a), int(b)), max(int(a), int(b))
            if i == j or (i, j) in edge_set:
                continue
            idx = pair_index.get((i, j))
            if idx is None:
                continue
            same_region_bonus = 1.6 if regions[i] == regions[j] else 1.0
            score = float(pairs.loc[idx, "score"]) * same_region_bonus
            closure_candidates.append((score, i, j))

    if closure_budget > 0 and closure_candidates:
        closure_candidates.sort(reverse=True)
        used = 0
        for _score, i, j in closure_candidates:
            if used >= closure_budget:
                break
            before = len(edge_set)
            add_edge(edge_set, i, j)
            if len(edge_set) > before:
                used += 1

    # Fill remaining target edges.
    if len(edge_set) < target_edges:
        remaining_pairs = pairs.copy()
        remaining_pairs["edge"] = list(zip(remaining_pairs["i"].astype(int), remaining_pairs["j"].astype(int)))
        remaining_pairs = remaining_pairs[~remaining_pairs["edge"].isin(edge_set)]
        need = target_edges - len(edge_set)
        if need > 0 and len(remaining_pairs) > 0:
            prob = remaining_pairs["score"].to_numpy()
            prob = prob / prob.sum()
            chosen_idx = rng.choice(remaining_pairs.index.to_numpy(), size=min(need, len(remaining_pairs)), replace=False, p=prob)
            for idx in chosen_idx:
                r = pairs.loc[idx]
                add_edge(edge_set, int(r.i), int(r.j))

    edges = []
    for i, j in sorted(edge_set):
        idx = pair_index[(i, j)]
        r = pairs.loc[idx]
        edges.append({
            "source_id": i,
            "target_id": j,
            "distance_km": float(r.distance_km),
            "topology_score": float(r.score),
        })
    return pd.DataFrame(edges)


def generate_temporal_routes(
    airports: pd.DataFrame,
    edges: pd.DataFrame,
    snapshots: int,
    rng: np.random.Generator,
    directed: bool = True,
    alpha: float = 0.6,
    d0_prob_km: float = 1200.0,
    p_max: float = 0.6,
    annual_growth_mean: float = 0.025,
    seasonal_amp: float = 0.18,
) -> pd.DataFrame:
    fitness = airports["fitness"].to_numpy()
    layers = airports["layer"].to_numpy()
    regions = airports["region_id"].to_numpy()
    m = len(edges)
    i_arr = edges["source_id"].to_numpy(dtype=int)
    j_arr = edges["target_id"].to_numpy(dtype=int)
    dist = edges["distance_km"].to_numpy(dtype=float)

    fprod = np.sqrt(fitness[i_arr] * fitness[j_arr])
    layer_flow = np.array([LAYER_FLOW_BIAS[layer_pair_key(layers[i], layers[j])] for i, j in zip(i_arr, j_arr)])
    same_region = np.where(regions[i_arr] == regions[j_arr], 1.25, 1.0)
    any_core = (layers[i_arr] == "core") | (layers[j_arr] == "core")
    long_factor = np.where((dist > 1800) & (~any_core), 0.35, 1.0)

    base_departures = 4.0 + 520.0 * (fprod ** 0.85) * np.exp(-dist / 4300.0) * layer_flow * same_region * long_factor * rng.lognormal(0, 0.35, size=m)
    growth = rng.normal(annual_growth_mean, 0.012, size=m)
    edge_noise = rng.normal(0.0, 0.12, size=m)
    macro_noise = 0.0

    if directed:
        directional_factor_forward = rng.lognormal(0, 0.08, size=m)
        directional_factor_backward = rng.lognormal(0, 0.08, size=m)
    else:
        directional_factor_forward = np.ones(m)
        directional_factor_backward = None

    all_rows = []
    for t in range(snapshots):
        seasonal = 1.0 + seasonal_amp * math.sin(2 * math.pi * t / 12.0 + 0.3) + 0.06 * math.sin(2 * math.pi * t / 6.0 + 1.7)
        seasonal = max(0.55, seasonal)
        macro_noise = 0.85 * macro_noise + rng.normal(0.0, 0.045)
        edge_noise = 0.82 * edge_noise + rng.normal(0.0, 0.10, size=m)
        disruption = rng.random(m) < 0.008
        disruption_factor = np.where(disruption, rng.uniform(0.0, 0.18, size=m), 1.0)
        trend = (1.0 + growth) ** (t / 12.0)
        mean_dep = base_departures * seasonal * trend * np.exp(macro_noise + edge_noise) * disruption_factor
        mean_dep = np.maximum(mean_dep, 0.0)
        departures = rng.poisson(mean_dep).astype(int)
        seats_per_departure = np.clip(rng.normal(145, 18, size=m), 70, 230)
        load_factor = np.clip(rng.normal(0.78, 0.08, size=m), 0.35, 0.98)
        seats = (departures * seats_per_departure).astype(int)
        passengers = np.minimum(seats, (seats * load_factor).astype(int))
        time_label = f"t{t:03d}"

        for idx in range(m):
            dep = int(round(departures[idx] * directional_factor_forward[idx]))
            st = int(round(seats[idx] * directional_factor_forward[idx]))
            pax = int(round(passengers[idx] * directional_factor_forward[idx]))
            all_rows.append({
                "time": time_label,
                "source": airports.loc[i_arr[idx], "airport_id"],
                "target": airports.loc[j_arr[idx], "airport_id"],
                "source_id": int(i_arr[idx]),
                "target_id": int(j_arr[idx]),
                "passengers": max(0, pax),
                "seats": max(0, st),
                "departures": max(0, dep),
                "distance_km": float(dist[idx]),
                "undirected_edge_id": int(idx),
            })
        if directed:
            for idx in range(m):
                dep = int(round(departures[idx] * directional_factor_backward[idx]))
                st = int(round(seats[idx] * directional_factor_backward[idx]))
                pax = int(round(passengers[idx] * directional_factor_backward[idx]))
                all_rows.append({
                    "time": time_label,
                    "source": airports.loc[j_arr[idx], "airport_id"],
                    "target": airports.loc[i_arr[idx], "airport_id"],
                    "source_id": int(j_arr[idx]),
                    "target_id": int(i_arr[idx]),
                    "passengers": max(0, pax),
                    "seats": max(0, st),
                    "departures": max(0, dep),
                    "distance_km": float(dist[idx]),
                    "undirected_edge_id": int(idx),
                })

    routes = pd.DataFrame(all_rows)
    max_log = np.log1p(routes["passengers"]).max()
    routes["weight"] = 0.0 if max_log <= 0 else np.log1p(routes["passengers"]) / max_log
    routes["prob"] = alpha * routes["weight"] * np.exp(-routes["distance_km"] / d0_prob_km)
    routes["prob"] = routes["prob"].clip(0.0, p_max)
    return routes


def build_adj(n: int, edges: pd.DataFrame) -> List[set[int]]:
    adj = [set() for _ in range(n)]
    for i, j in zip(edges["source_id"], edges["target_id"]):
        i = int(i)
        j = int(j)
        adj[i].add(j)
        adj[j].add(i)
    return adj


def connected_components(adj: List[set[int]]) -> List[List[int]]:
    n = len(adj)
    seen = np.zeros(n, dtype=bool)
    comps = []
    for s in range(n):
        if seen[s]:
            continue
        q = deque([s])
        seen[s] = True
        comp = []
        while q:
            u = q.popleft()
            comp.append(u)
            for v in adj[u]:
                if not seen[v]:
                    seen[v] = True
                    q.append(v)
        comps.append(comp)
    comps.sort(key=len, reverse=True)
    return comps


def average_clustering(adj: List[set[int]]) -> float:
    vals = []
    for neigh in adj:
        k = len(neigh)
        if k < 2:
            vals.append(0.0)
            continue
        neigh_list = list(neigh)
        links = 0
        for a_idx in range(k):
            a = neigh_list[a_idx]
            adj_a = adj[a]
            for b in neigh_list[a_idx + 1:]:
                if b in adj_a:
                    links += 1
        vals.append(2 * links / (k * (k - 1)))
    return float(np.mean(vals))


def shortest_path_stats(adj: List[set[int]]) -> Dict[str, float]:
    comps = connected_components(adj)
    if not comps:
        return {"largest_component_size": 0, "avg_shortest_path": None, "diameter": None, "components": 0}
    largest = comps[0]
    allowed = set(largest)
    n_lcc = len(largest)
    if n_lcc <= 1:
        return {"largest_component_size": n_lcc, "avg_shortest_path": 0.0, "diameter": 0, "components": len(comps)}
    total_dist = 0
    pair_count = 0
    diameter = 0
    for s in largest:
        dist = {s: 0}
        q = deque([s])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if v in allowed and v not in dist:
                    dist[v] = dist[u] + 1
                    q.append(v)
        for v, d in dist.items():
            if v > s:
                total_dist += d
                pair_count += 1
                diameter = max(diameter, d)
    return {
        "largest_component_size": int(n_lcc),
        "avg_shortest_path": float(total_dist / pair_count) if pair_count else 0.0,
        "diameter": int(diameter),
        "components": int(len(comps)),
    }


def compute_metrics(airports: pd.DataFrame, edges: pd.DataFrame, routes: pd.DataFrame) -> Dict:
    n = len(airports)
    e = len(edges)
    adj = build_adj(n, edges)
    degrees = np.array([len(a) for a in adj], dtype=float)
    clustering = average_clustering(adj)
    sp = shortest_path_stats(adj)

    strength = np.zeros(n)
    distance_strength = np.zeros(n)
    route_agg = routes.groupby(["source_id", "target_id"], as_index=False).agg(passengers=("passengers", "sum"), distance_km=("distance_km", "mean"))
    for _, r in route_agg.iterrows():
        i = int(r["source_id"])
        strength[i] += float(r["passengers"])
        distance_strength[i] += float(r["distance_km"])

    total_traffic = strength.sum()
    top_count = max(1, int(math.ceil(0.05 * n)))
    top_share = float(np.sort(strength)[-top_count:].sum() / total_traffic) if total_traffic > 0 else 0.0
    core_ids = set(airports.loc[airports["layer"] == "core", "node_id"].astype(int))
    core_route_mask = routes["source_id"].isin(core_ids) & routes["target_id"].isin(core_ids)
    core_traffic_share = float(routes.loc[core_route_mask, "passengers"].sum() / routes["passengers"].sum()) if routes["passengers"].sum() > 0 else 0.0

    return {
        "nodes": int(n),
        "undirected_edges": int(e),
        "directed_edges_per_snapshot": int(routes[["source_id", "target_id"]].drop_duplicates().shape[0]),
        "snapshots": int(routes["time"].nunique()),
        "avg_degree_undirected": float(2 * e / n),
        "max_degree": int(degrees.max()),
        "degree_gini": gini(degrees),
        "avg_clustering": float(clustering),
        "largest_component_size": sp["largest_component_size"],
        "connected_components": sp["components"],
        "avg_shortest_path_lcc": sp["avg_shortest_path"],
        "diameter_lcc": sp["diameter"],
        "traffic_gini_node_strength": gini(strength),
        "top_5pct_airport_traffic_share": top_share,
        "core_core_traffic_share": core_traffic_share,
        "beta_weight_strength_vs_degree": fit_power_exponent(degrees, strength),
        "beta_distance_strength_vs_degree": fit_power_exponent(degrees, distance_strength),
        "mean_edge_distance_km": float(edges["distance_km"].mean()),
        "median_edge_distance_km": float(edges["distance_km"].median()),
        "prob_min": float(routes["prob"].min()),
        "prob_max": float(routes["prob"].max()),
        "prob_mean": float(routes["prob"].mean()),
        "layer_counts": airports["layer"].value_counts().to_dict(),
    }


def generate_one_graph(out_dir: Path, graph_name: str, n: int, avg_degree: float, snapshots: int, seed: int, directed: bool = True, alpha: float = 0.6, d0_prob_km: float = 1200.0, p_max: float = 0.6) -> Dict:
    rng = np.random.default_rng(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    airports = generate_airports(n=n, rng=rng)
    edges = generate_topology(airports, target_avg_degree=avg_degree, rng=rng)
    routes = generate_temporal_routes(airports, edges, snapshots, rng, directed=directed, alpha=alpha, d0_prob_km=d0_prob_km, p_max=p_max)
    metrics = compute_metrics(airports, edges, routes)

    airports_out = airports[["node_id", "airport_id", "lat", "lon", "name", "city", "country", "region", "region_id", "fitness", "demand", "cost", "layer"]].copy()
    routes_out = routes[["time", "source", "target", "source_id", "target_id", "passengers", "seats", "departures", "distance_km", "weight", "prob", "undirected_edge_id"]].copy()
    edges_out = edges.copy()
    id_to_code = airports.set_index("node_id")["airport_id"]
    edges_out["source"] = edges_out["source_id"].map(id_to_code)
    edges_out["target"] = edges_out["target_id"].map(id_to_code)
    edges_out = edges_out[["source", "target", "source_id", "target_id", "distance_km", "topology_score"]]
    node_index = airports_out[["airport_id", "node_id"]].copy()

    airports_out.to_csv(out_dir / "airports.csv", index=False)
    routes_out.to_csv(out_dir / "routes.csv", index=False)
    edges_out.to_csv(out_dir / "edges_static.csv", index=False)
    node_index.to_csv(out_dir / "node_index.csv", index=False)

    metadata = {
        "graph_name": graph_name,
        "seed": seed,
        "directed_routes": directed,
        "target_avg_degree": avg_degree,
        "generation_assumptions": {
            "spatial_embedding": "clustered airport coordinates",
            "airport_fitness": "lognormal region-adjusted demand",
            "topology": "gravity + distance decay + layer bias + triadic closure",
            "layers": "core/bridge/periphery by fitness rank",
            "temporal_weights": "seasonality + growth trend + AR edge noise + rare disruptions",
            "diffusion_probability": "log-normalized passengers times exp(-distance/d0), clipped",
        },
        "diffusion_probability_params": {"alpha": alpha, "d0_prob_km": d0_prob_km, "p_max": p_max},
        "metrics": metrics,
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    return {"graph_name": graph_name, "path": str(out_dir), **metrics}


def generate_preset(out_root: str | Path, preset: str, base_seed: int = 123, directed: bool = True) -> List[Dict]:
    specs = PAPER_ALIGNED_SPECS if preset == "paper" else SPARSE_TRANSFER_SPECS
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    manifest = []
    counter = 0
    for spec in specs:
        for g in range(spec["graphs"]):
            graph_name = spec["name"] if spec["graphs"] == 1 else f"{spec['name']}_g{g}"
            graph_dir = out_root / graph_name
            seed = base_seed + counter * 9973
            counter += 1
            print(f"Generating {graph_name}: N={spec['n']}, avg_degree={spec['avg_degree']}, snapshots={spec['snapshots']}")
            summary = generate_one_graph(graph_dir, graph_name, spec["n"], spec["avg_degree"], spec["snapshots"], seed, directed=directed)
            manifest.append(summary)
            print(f"  edges={summary['undirected_edges']}, <k>={summary['avg_degree_undirected']:.2f}, C={summary['avg_clustering']:.3f}, L={summary['avg_shortest_path_lcc']:.2f}, diam={summary['diameter_lcc']}")
    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    pd.DataFrame(manifest).to_csv(out_root / "manifest.csv", index=False)
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/generated_airline_topological", help="Output root directory.")
    parser.add_argument("--preset", choices=["paper", "sparse"], default="paper")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--undirected-routes", action="store_true", help="Export one row per undirected edge per snapshot instead of both directions.")
    args = parser.parse_args()
    manifest = generate_preset(args.out, args.preset, args.seed, directed=not args.undirected_routes)
    print("\nDone.")
    print(f"Wrote dataset to: {args.out}")
    cols = ["graph_name", "nodes", "undirected_edges", "avg_degree_undirected", "snapshots", "avg_clustering", "avg_shortest_path_lcc", "diameter_lcc", "traffic_gini_node_strength", "top_5pct_airport_traffic_share"]
    print(pd.DataFrame(manifest)[cols].to_string(index=False))


if __name__ == "__main__":
    main()
