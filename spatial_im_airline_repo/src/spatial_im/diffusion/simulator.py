from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class SimulationResult:
    final_active: np.ndarray
    trajectory: np.ndarray


def simulate_progressive_activation(edge_index: np.ndarray, edge_prob_seq: np.ndarray, seed_mask: np.ndarray, rng=None) -> SimulationResult:
    """
    One Monte Carlo rollout of a progressive SI/IC-style process across a sequence of snapshots.

    edge_index: [E, 2]
    edge_prob_seq: [T, E]
    seed_mask: [N] in {0,1}
    """
    if rng is None:
        rng = np.random.default_rng()
    n = int(seed_mask.shape[0])
    T = int(edge_prob_seq.shape[0])
    active = seed_mask.astype(np.float32).copy()
    trajectory = [active.copy()]

    src = edge_index[:, 0]
    dst = edge_index[:, 1]

    for t in range(T):
        probs = edge_prob_seq[t]
        newly_active = active.copy()
        for e, (u, v) in enumerate(zip(src, dst)):
            if active[u] >= 0.5 and active[v] < 0.5:
                if rng.random() < probs[e]:
                    newly_active[v] = 1.0
        active = np.maximum(active, newly_active)
        trajectory.append(active.copy())
    return SimulationResult(final_active=active, trajectory=np.stack(trajectory, axis=0))


def estimate_final_activation_prob(edge_index, edge_prob_seq, seed_mask, mc_rollouts=64, seed=7):
    rng = np.random.default_rng(seed)
    acc = np.zeros_like(seed_mask, dtype=np.float32)
    for _ in range(mc_rollouts):
        res = simulate_progressive_activation(edge_index=edge_index, edge_prob_seq=edge_prob_seq, seed_mask=seed_mask, rng=rng)
        acc += res.final_active
    return acc / mc_rollouts


def sample_training_batch(edge_index, edge_prob_seq_all, num_nodes, batch_size, budget_range=(1, 3), horizon_range=(1, 1), seed=7, mc_rollouts=24):
    rng = np.random.default_rng(seed)
    T = edge_prob_seq_all.shape[0]
    batch = []
    for _ in range(batch_size):
        t = int(rng.integers(0, T))
        max_horizon = min(max(int(horizon_range[0]), 1), T - t)
        max_horizon = min(max_horizon, int(horizon_range[1]))
        min_horizon = min(int(horizon_range[0]), max_horizon)
        horizon = int(rng.integers(min_horizon, max_horizon + 1))
        budget = int(rng.integers(budget_range[0], budget_range[1] + 1))
        seeds = rng.choice(num_nodes, size=budget, replace=False)
        seed_mask = np.zeros(num_nodes, dtype=np.float32)
        seed_mask[seeds] = 1.0
        probs = estimate_final_activation_prob(
            edge_index=edge_index,
            edge_prob_seq=edge_prob_seq_all[t:t+horizon],
            seed_mask=seed_mask,
            mc_rollouts=mc_rollouts,
            seed=int(rng.integers(1_000_000)),
        )
        batch.append({
            'start_t': t,
            'horizon': horizon,
            'seed_mask': seed_mask,
            'teacher_final_probs': probs,
        })
    return batch
