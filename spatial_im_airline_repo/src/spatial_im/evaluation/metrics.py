from __future__ import annotations

import numpy as np


def final_activated_mass(final_active_probs: np.ndarray, mode: str = 'expected', threshold: float = 0.5) -> float:
    if mode == 'threshold':
        return float((final_active_probs >= threshold).sum())
    return float(final_active_probs.sum())


def expected_coverage(coords: np.ndarray, active_probs: np.ndarray, radius_km: float) -> float:
    n = coords.shape[0]
    # approximate union coverage over node centers.
    cover_prob = np.zeros(n, dtype=np.float32)
    for i in range(n):
        lat1, lon1 = coords[i]
        for j in range(n):
            lat2, lon2 = coords[j]
            d = np.sqrt((lat1 - lat2) ** 2 + (lon1 - lon2) ** 2) * 111.0  # coarse km
            if d <= radius_km:
                cover_prob[j] = 1.0 - (1.0 - cover_prob[j]) * (1.0 - active_probs[i])
    return float(cover_prob.sum())


def intervention_cost(selected, coords: np.ndarray, node_costs: np.ndarray, mode: str = 'geography', constant_cost: float = 1.0, distance_cost_scale: float = 0.0015) -> float:
    selected = list(selected)
    if not selected:
        return 0.0
    if mode == 'constant':
        return float(len(selected) * constant_cost)
    # geography-aware cost: fixed cost plus remoteness penalty.
    centroid = coords.mean(axis=0)
    total = 0.0
    for idx in selected:
        d = np.sqrt(((coords[idx] - centroid) ** 2).sum()) * 111.0
        total += constant_cost + distance_cost_scale * d + 0.25 * float(node_costs[idx])
    return float(total)


def transfer_ratio(transfer_score: float, scratch_score: float, eps: float = 1e-8) -> float:
    return float(transfer_score / max(scratch_score, eps))


def adaptation_efficiency(learning_curve, threshold_fraction: float = 0.9):
    arr = np.asarray(learning_curve, dtype=np.float32)
    if arr.size == 0:
        return np.nan
    target = threshold_fraction * arr.max()
    hits = np.where(arr >= target)[0]
    if len(hits) == 0:
        return np.nan
    # larger is better: high final score reached quickly.
    return float(arr.max() / (1 + hits[0]))
