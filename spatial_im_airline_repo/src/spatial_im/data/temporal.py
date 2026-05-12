from __future__ import annotations

import numpy as np
from .graph_build import AirlineGraph


def generate_synthetic_temporal_weights(
    airline_graph: AirlineGraph,
    snapshots: int,
    seasonal_strength: float = 0.30,
    noise_std: float = 0.05,
    seed: int = 7,
) -> np.ndarray:
    """
    Returns W of shape [T, E].
    Each edge gets a base traffic term times seasonal airport activity and noise.
    Values are clipped into [0.01, 0.95] so they can be used as activation probabilities.
    """
    rng = np.random.default_rng(seed)
    E = airline_graph.edge_index.shape[0]
    T = snapshots
    weights = np.zeros((T, E), dtype=np.float32)

    src = airline_graph.edge_index[:, 0]
    dst = airline_graph.edge_index[:, 1]
    base = airline_graph.base_edge_weight
    coords = airline_graph.coords

    airport_phase = rng.uniform(0, 2 * np.pi, size=len(airline_graph.node_ids))
    airport_bias = 0.85 + 0.30 * rng.random(len(airline_graph.node_ids))

    # crude latitude-sensitive seasonality: northern airports get stronger seasonal effect.
    lat_scale = 0.5 + np.abs(coords[:, 0]) / max(np.abs(coords[:, 0]).max(), 1e-6)

    for t in range(T):
        seasonal = airport_bias * (1.0 + seasonal_strength * np.sin(2 * np.pi * t / T + airport_phase) * lat_scale)
        edge_seasonal = 0.5 * seasonal[src] + 0.5 * seasonal[dst]
        edge_noise = rng.normal(0.0, noise_std, size=E)
        w_t = base * edge_seasonal + edge_noise
        weights[t] = np.clip(w_t, 0.01, 0.95)
    return weights
