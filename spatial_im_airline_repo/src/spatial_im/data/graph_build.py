from __future__ import annotations

from dataclasses import dataclass
from math import radians, sin, cos, sqrt, atan2
from typing import Dict, List, Tuple
import numpy as np
import networkx as nx
import torch

from .airline import AirlineTables

EARTH_RADIUS_KM = 6371.0


def haversine_km(lat1, lon1, lat2, lon2):
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * atan2(sqrt(a), sqrt(1 - a))


@dataclass
class AirlineGraph:
    graph: nx.DiGraph
    node_ids: List[int]
    id_to_idx: Dict[int, int]
    idx_to_id: Dict[int, int]
    coords: np.ndarray
    node_costs: np.ndarray
    edge_index: np.ndarray
    distances_km: np.ndarray
    base_edge_weight: np.ndarray
    node_features: np.ndarray


def build_airline_graph(tables: AirlineTables) -> AirlineGraph:
    g = nx.DiGraph()
    id_to_idx = {aid: i for i, aid in enumerate(tables.airports['airport_id'].tolist())}
    idx_to_id = {i: aid for aid, i in id_to_idx.items()}

    coords = np.zeros((len(id_to_idx), 2), dtype=np.float32)
    node_costs = np.zeros(len(id_to_idx), dtype=np.float32)
    out_deg = np.zeros(len(id_to_idx), dtype=np.float32)
    in_deg = np.zeros(len(id_to_idx), dtype=np.float32)

    airport_lookup = tables.airports.set_index('airport_id')
    for aid, idx in id_to_idx.items():
        row = airport_lookup.loc[aid]
        g.add_node(idx, airport_id=aid, name=row['name'], iata=row['iata'], lat=float(row['lat']), lon=float(row['lon']))
        coords[idx] = [float(row['lat']), float(row['lon'])]

    edge_list = []
    distances = []
    base_weights = []
    for _, row in tables.routes.iterrows():
        s_id, t_id = int(row['source_airport_id']), int(row['target_airport_id'])
        if s_id not in id_to_idx or t_id not in id_to_idx:
            continue
        s, t = id_to_idx[s_id], id_to_idx[t_id]
        lat1, lon1 = coords[s]
        lat2, lon2 = coords[t]
        d = haversine_km(lat1, lon1, lat2, lon2)
        base = 1.0 / (1.0 + d / 1000.0)
        g.add_edge(s, t, distance_km=d, base_weight=base)
        edge_list.append((s, t))
        distances.append(d)
        base_weights.append(base)
        out_deg[s] += 1
        in_deg[t] += 1

    # Geography-aware cost: more remote airports cost more to seed.
    centroid = coords.mean(axis=0)
    remoteness = np.array([haversine_km(c[0], c[1], centroid[0], centroid[1]) for c in coords], dtype=np.float32)
    node_costs = 1.0 + remoteness / max(remoteness.max(), 1e-6)

    lat_norm = (coords[:, 0] - coords[:, 0].mean()) / (coords[:, 0].std() + 1e-6)
    lon_norm = (coords[:, 1] - coords[:, 1].mean()) / (coords[:, 1].std() + 1e-6)
    node_features = np.stack([lat_norm, lon_norm, out_deg, in_deg, node_costs], axis=1).astype(np.float32)

    return AirlineGraph(
        graph=g,
        node_ids=list(id_to_idx.keys()),
        id_to_idx=id_to_idx,
        idx_to_id=idx_to_id,
        coords=coords,
        node_costs=node_costs,
        edge_index=np.asarray(edge_list, dtype=np.int64),
        distances_km=np.asarray(distances, dtype=np.float32),
        base_edge_weight=np.asarray(base_weights, dtype=np.float32),
        node_features=node_features,
    )


def as_torch_tensors(airline_graph: AirlineGraph):
    edge_index = torch.as_tensor(airline_graph.edge_index.T, dtype=torch.long)
    node_features = torch.as_tensor(airline_graph.node_features, dtype=torch.float32)
    coords = torch.as_tensor(airline_graph.coords, dtype=torch.float32)
    base_edge_weight = torch.as_tensor(airline_graph.base_edge_weight, dtype=torch.float32)
    distances = torch.as_tensor(airline_graph.distances_km, dtype=torch.float32)
    node_costs = torch.as_tensor(airline_graph.node_costs, dtype=torch.float32)
    return edge_index, node_features, coords, base_edge_weight, distances, node_costs
