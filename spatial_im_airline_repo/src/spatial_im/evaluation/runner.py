from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List
import numpy as np
import torch

from spatial_im.env.airline_env import AirlineSpatialTemporalEnv
from spatial_im.diffusion.simulator import estimate_final_activation_prob
from spatial_im.evaluation.metrics import (
    adaptation_efficiency,
    expected_coverage,
    final_activated_mass,
    intervention_cost,
    transfer_ratio,
)
from spatial_im.evaluation.submodularity import empirical_submodularity_violation_rate


@dataclass
class EvalResult:
    selected: List[int]
    objective: float
    final_activated_mass: float
    final_geographic_coverage: float
    total_intervention_cost: float


def infer_seed_set(env: AirlineSpatialTemporalEnv, agent) -> List[int]:
    state_feat, legal = env.reset()
    selected: List[int] = []
    for _ in range(env.budget):
        action_prior = env.action_prior_scores()
        action = agent.greedy_action(state_feat, legal, action_prior=action_prior)
        out = env.step(action)
        selected = list(out.info['selected'])
        if out.done:
            break
        state_feat = env.get_candidate_features()
        legal = env.scoring_mask()
    return selected


def rollout_policy(env: AirlineSpatialTemporalEnv, agent, epsilon: float = 0.0):
    if epsilon == 0.0 and hasattr(agent, 'greedy_action'):
        selected = infer_seed_set(env, agent)
        return evaluate_seed_set(env, selected)
    state_feat, legal = env.reset()
    while True:
        action_prior = env.action_prior_scores()
        action = agent.act(state_feat, legal, epsilon=epsilon, action_prior=action_prior)
        out = env.step(action)
        if out.done:
            final = out.info['predicted_final']
            mass = final_activated_mass(final)
            coverage = expected_coverage(env.coords.cpu().numpy(), final, env.coverage_radius_km)
            cost = intervention_cost(out.info['selected'], env.coords.cpu().numpy(), env.node_costs.cpu().numpy(), env.cost_mode, env.constant_cost, env.distance_cost_scale)
            return EvalResult(out.info['selected'], out.info['objective'], mass, coverage, cost)
        state_feat = env.get_candidate_features()
        legal = env.scoring_mask()


def evaluate_seed_set(env: AirlineSpatialTemporalEnv, selected: List[int]) -> EvalResult:
    env.reset()
    info = None
    if len(selected) == 0:
        final = env.predicted_final.detach().cpu().numpy()
        obj = env.prev_objective
        mass = final_activated_mass(final)
        coverage = expected_coverage(env.coords.cpu().numpy(), final, env.coverage_radius_km)
        cost = 0.0
        return EvalResult(selected, obj, mass, coverage, cost)
    for a in selected:
        out = env.step(a)
        info = out.info
    final = info['predicted_final']
    mass = final_activated_mass(final)
    coverage = expected_coverage(env.coords.cpu().numpy(), final, env.coverage_radius_km)
    cost = intervention_cost(selected, env.coords.cpu().numpy(), env.node_costs.cpu().numpy(), env.cost_mode, env.constant_cost, env.distance_cost_scale)
    return EvalResult(selected, info['objective'], mass, coverage, cost)


def evaluate_seed_set_raw(
    edge_index: np.ndarray,
    edge_prob_seq: np.ndarray,
    num_nodes: int,
    selected: List[int],
    coords: np.ndarray,
    node_costs: np.ndarray,
    coverage_radius_km: float,
    regime: str = 'spread',
    beta_coverage: float = 0.0,
    lambda_cost: float = 0.0,
    cost_mode: str = 'geography',
    constant_cost: float = 1.0,
    distance_cost_scale: float = 0.0015,
    mc_rollouts: int = 64,
) -> EvalResult:
    seed_mask = np.zeros(num_nodes, dtype=np.float32)
    if selected:
        seed_mask[np.asarray(selected, dtype=np.int64)] = 1.0
    final = estimate_final_activation_prob(
        edge_index=edge_index,
        edge_prob_seq=edge_prob_seq,
        seed_mask=seed_mask,
        mc_rollouts=mc_rollouts,
    )
    mass = final_activated_mass(final)
    coverage = expected_coverage(coords, final, coverage_radius_km)
    cost = intervention_cost(selected, coords, node_costs, cost_mode, constant_cost, distance_cost_scale)
    if regime == 'spread':
        objective = mass
    else:
        objective = mass + beta_coverage * coverage - lambda_cost * cost
    return EvalResult(list(selected), float(objective), mass, coverage, cost)


def compute_submodularity_metric(env: AirlineSpatialTemporalEnv, samples: int = 100):
    def eval_fn(S):
        return evaluate_seed_set(env, sorted(list(S))).objective
    max_set = max(1, env.budget - 1)
    rate, details = empirical_submodularity_violation_rate(range(env.num_nodes), eval_fn, samples=samples, max_set_size=max_set)
    return rate, details


def summarize_transfer(transfer_curve: List[float], scratch_curve: List[float], threshold_fraction: float = 0.9) -> Dict[str, float]:
    tr = transfer_ratio(max(transfer_curve), max(scratch_curve))
    ae_transfer = adaptation_efficiency(transfer_curve, threshold_fraction=threshold_fraction)
    ae_scratch = adaptation_efficiency(scratch_curve, threshold_fraction=threshold_fraction)
    return {
        'transfer_ratio': tr,
        'adaptation_efficiency_transfer': ae_transfer,
        'adaptation_efficiency_scratch': ae_scratch,
    }
