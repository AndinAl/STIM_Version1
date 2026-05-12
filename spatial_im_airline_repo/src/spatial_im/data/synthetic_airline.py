from __future__ import annotations

from dataclasses import dataclass
from math import radians, sin, cos, sqrt, atan2

import numpy as np
import pandas as pd

from .airline import AirlineTables

EARTH_RADIUS_KM = 6371.0


def _haversine_pairwise_km(coords: np.ndarray) -> np.ndarray:
    lat = np.radians(coords[:, 0]).reshape(-1, 1)
    lon = np.radians(coords[:, 1]).reshape(-1, 1)
    dlat = lat - lat.T
    dlon = lon - lon.T
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat) * np.cos(lat.T) * np.sin(dlon / 2.0) ** 2
    return 2.0 * EARTH_RADIUS_KM * np.arctan2(np.sqrt(a), np.sqrt(np.maximum(1.0 - a, 1e-12)))


def _clamp_coords(coords: np.ndarray) -> np.ndarray:
    out = coords.copy()
    out[:, 0] = np.clip(out[:, 0], -70.0, 70.0)
    out[:, 1] = ((out[:, 1] + 180.0) % 360.0) - 180.0
    return out


def generate_homogeneous_airline_tables(
    base_tables: AirlineTables,
    num_nodes: int,
    seed: int,
    avg_out_degree: float | None = None,
    coord_jitter_deg: float = 2.0,
    distance_scale_km: float = 1800.0,
    max_edge_prob: float = 0.75,
) -> AirlineTables:
    """
    Build a larger airline-like directed graph from the sample-airline prototype.

    The resulting graph is homogeneous in the sense that nodes are generated from
    the same prototype airport family and edges follow the same hub-and-distance
    process across all sizes.
    """
    rng = np.random.default_rng(seed)
    airports = base_tables.airports.reset_index(drop=True).copy()
    routes = base_tables.routes.reset_index(drop=True).copy()

    base_ids = airports["airport_id"].astype(int).tolist()
    out_deg = routes.groupby("source_airport_id").size().reindex(base_ids, fill_value=0).to_numpy(dtype=np.float32)
    in_deg = routes.groupby("target_airport_id").size().reindex(base_ids, fill_value=0).to_numpy(dtype=np.float32)
    total_deg = out_deg + in_deg
    proto_p = (total_deg + 1.0) / float(np.sum(total_deg + 1.0))

    chosen_proto_idx = rng.choice(len(airports), size=int(num_nodes), replace=True, p=proto_p)
    proto_rows = airports.iloc[chosen_proto_idx].reset_index(drop=True)

    lat = proto_rows["lat"].to_numpy(dtype=np.float32) + rng.normal(0.0, coord_jitter_deg, size=num_nodes).astype(np.float32)
    lon = proto_rows["lon"].to_numpy(dtype=np.float32) + rng.normal(0.0, coord_jitter_deg, size=num_nodes).astype(np.float32)
    coords = _clamp_coords(np.stack([lat, lon], axis=1).astype(np.float32))

    synth_airports = pd.DataFrame(
        {
            "airport_id": np.arange(1, int(num_nodes) + 1, dtype=np.int64),
            "name": [f"Synthetic Airport {i+1}" for i in range(int(num_nodes))],
            "iata": [f"A{i:03d}" for i in range(int(num_nodes))],
            "lat": coords[:, 0],
            "lon": coords[:, 1],
        }
    )

    proto_out = out_deg[chosen_proto_idx].astype(np.float32)
    proto_in = in_deg[chosen_proto_idx].astype(np.float32)
    out_strength = 0.6 + proto_out / max(float(out_deg.max()), 1.0)
    in_strength = 0.6 + proto_in / max(float(in_deg.max()), 1.0)

    pairwise_dist = _haversine_pairwise_km(coords)
    geo_decay = np.exp(-pairwise_dist / float(distance_scale_km)).astype(np.float32)
    hub_score = np.outer(out_strength, in_strength).astype(np.float32)
    score = hub_score * geo_decay
    np.fill_diagonal(score, 0.0)

    if avg_out_degree is None:
        avg_out_degree = max(2.0, float(len(routes)) / max(len(airports), 1))
    base_prob = float(avg_out_degree) / max(float(num_nodes - 1), 1.0)
    mean_score = float(score[score > 0].mean()) if np.any(score > 0) else 1.0
    edge_prob = np.clip(base_prob * score / max(mean_score, 1e-6), 0.0, float(max_edge_prob))
    sampled = rng.random(size=edge_prob.shape) < edge_prob
    np.fill_diagonal(sampled, False)

    out_count = sampled.sum(axis=1)
    in_count = sampled.sum(axis=0)
    for i in range(int(num_nodes)):
        if out_count[i] == 0:
            j = int(np.argmax(score[i]))
            if i != j:
                sampled[i, j] = True
        if in_count[i] == 0:
            j = int(np.argmax(score[:, i]))
            if i != j:
                sampled[j, i] = True

    edge_sources, edge_targets = np.where(sampled)
    synth_routes = pd.DataFrame(
        {
            "source_airport_id": edge_sources.astype(np.int64) + 1,
            "target_airport_id": edge_targets.astype(np.int64) + 1,
        }
    ).drop_duplicates(ignore_index=True)

    return AirlineTables(airports=synth_airports, routes=synth_routes)
