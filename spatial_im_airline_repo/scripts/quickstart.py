from __future__ import annotations

import argparse
import json
import numpy as np
import torch

from spatial_im.utils.io import load_yaml
from spatial_im.utils.seeds import set_seed
from spatial_im.data.airline import load_airline_tables
from spatial_im.data.graph_build import build_airline_graph, as_torch_tensors
from spatial_im.data.temporal import generate_synthetic_temporal_weights
from spatial_im.diffusion.gnn_diffusor import GNNDiffusor, train_diffusor
from spatial_im.policy.features import compute_static_spatial_features
from spatial_im.policy.dqn_agent import DQNAgent, ReplayBuffer
from spatial_im.env.airline_env import AirlineSpatialTemporalEnv
from spatial_im.evaluation.runner import evaluate_seed_set_raw, compute_submodularity_metric, summarize_transfer, rollout_policy
from spatial_im.baselines.classical import greedy_spread, degree_ranking, degree_discount
from spatial_im.baselines.dynamic import weighted_degree_per_slice, temporal_weighted_degree, temporal_degree_discount, myopic_lookahead, temporal_ris_proxy
from spatial_im.baselines.spatial import (
    build_temporal_weighted_graph,
    community_bridge_nodes,
    cost_aware_ranking,
    distance_strength,
    weighted_betweenness,
    weighted_strength,
)


def train_agent(env, cfg, W):
    sample_feat, _ = env.reset()
    agent = DQNAgent(
        feature_dim=sample_feat.shape[1],
        lr=cfg['rl']['lr'],
        gamma=cfg['rl']['gamma'],
        hidden_dim=cfg['rl'].get('hidden_dim', 64),
        beta_coverage=cfg['reward']['beta_coverage'],
        lambda_cost=cfg['reward']['lambda_cost'],
        regime=cfg['evaluation']['regime'],
    )
    buffer = ReplayBuffer(cfg['rl']['buffer_capacity'])
    epsilon = cfg['rl']['eps_start']
    curve = []
    for ep in range(cfg['rl']['train_episodes']):
        start_snapshot = 0 if cfg['evaluation']['regime'] == 'spread' else (ep % max(1, W.shape[0] - 1))
        state_feat, legal = env.reset(start_snapshot=start_snapshot)
        action_prior = env.action_prior_scores()
        ep_return = 0.0
        while True:
            action = agent.act(state_feat, legal, epsilon, action_prior=action_prior)
            out = env.step(action)
            next_state_feat = env.get_candidate_features() if not out.done else state_feat.copy()
            next_legal = env.scoring_mask() if not out.done else np.zeros_like(legal)
            next_action_prior = env.action_prior_scores() if not out.done else None
            buffer.push(state_feat, action, out.reward, next_state_feat, out.done, legal, next_legal, action_prior, next_action_prior)
            state_feat = next_state_feat
            legal = next_legal
            action_prior = next_action_prior
            ep_return += out.reward
            if out.done:
                break
        curve.append(ep_return)
        if len(buffer) >= cfg['rl']['batch_size'] and ep >= cfg['rl']['warmup_episodes']:
            agent.update(buffer.sample(cfg['rl']['batch_size']))
        if ep % cfg['rl']['target_update_every'] == 0:
            agent.sync_target()
        epsilon = max(cfg['rl']['eps_end'], epsilon * cfg['rl']['eps_decay'])
    return agent, curve


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    set_seed(cfg['seed'])

    tables = load_airline_tables(cfg['data']['airports_csv'], cfg['data']['routes_csv'])
    ag = build_airline_graph(tables)
    edge_index, node_features, coords, base_edge_weight, distances, node_costs = as_torch_tensors(ag)
    W = generate_synthetic_temporal_weights(ag, cfg['data']['snapshots'], cfg['data']['seasonal_strength'], cfg['data']['noise_std'], cfg['seed'])
    W_torch = torch.as_tensor(W, dtype=torch.float32)

    diffusor = GNNDiffusor(node_feat_dim=node_features.size(1), hidden_dim=cfg['model']['hidden_dim'], layers=cfg['model']['layers'])
    diff_art = train_diffusor(
        diffusor,
        node_features,
        edge_index,
        W,
        distances,
        cfg['model']['diffusor_epochs'],
        cfg['model']['batch_size'],
        cfg['model']['diffusor_lr'],
        cfg['seed'],
        coords=coords,
        node_costs=node_costs,
        regime=cfg['evaluation']['regime'],
        beta_coverage=cfg['reward']['beta_coverage'],
        lambda_cost=cfg['reward']['lambda_cost'],
        cost_mode=cfg['reward']['cost_mode'],
        constant_cost=cfg['reward']['constant_cost'],
        distance_cost_scale=cfg['reward']['distance_cost_scale'],
        coverage_radius_km=cfg['simulator']['coverage_radius_km'],
        teacher_mc_rollouts=cfg['model'].get('teacher_mc_rollouts', cfg['simulator']['mc_rollouts']),
        marginal_loss_weight=cfg['model'].get('marginal_loss_weight', 1.0),
        ranking_loss_weight=cfg['model'].get('ranking_loss_weight', 0.2),
        selection_budget=cfg['rl']['budget'],
    )

    policy_static = torch.as_tensor(compute_static_spatial_features(ag.graph, ag.coords, ag.node_costs), dtype=torch.float32)
    env = AirlineSpatialTemporalEnv(
        diffusor=diffusor,
        node_features=node_features,
        edge_index=edge_index,
        edge_weight_seq=W_torch,
        distances_km=distances,
        coords=coords,
        node_costs=node_costs,
        static_policy_features=policy_static,
        budget=cfg['rl']['budget'],
        beta_coverage=cfg['reward']['beta_coverage'],
        lambda_cost=cfg['reward']['lambda_cost'],
        coverage_radius_km=cfg['simulator']['coverage_radius_km'],
        regime=cfg['evaluation']['regime'],
        cost_mode=cfg['reward']['cost_mode'],
        constant_cost=cfg['reward']['constant_cost'],
        distance_cost_scale=cfg['reward']['distance_cost_scale'],
        use_shaping=cfg['reward'].get('use_shaping', False),
        alpha_spread=cfg['reward'].get('alpha_spread', 0.0),
        alpha_cover=cfg['reward'].get('alpha_cover', 0.0),
        alpha_cost=cfg['reward'].get('alpha_cost', 0.0),
    )

    agent, curve = train_agent(env, cfg, W)
    rl_eval = rollout_policy(env, agent, epsilon=0.0)

    budget = cfg['rl']['budget']
    current_t = 0
    one_slice = W[current_t:current_t+1]
    weighted_graph = build_temporal_weighted_graph(ag.graph, ag.edge_index, W[current_t], ag.distances_km)
    baseline_seed_sets = {
        'greedy_spread': greedy_spread(ag.edge_index, one_slice, len(ag.node_ids), budget, cfg['simulator']['mc_rollouts']),
        'degree': degree_ranking(ag.graph, budget),
        'degree_discount': degree_discount(ag.graph, budget),
        'weighted_degree_slice': weighted_degree_per_slice(ag.edge_index, W[current_t], len(ag.node_ids), budget)[0],
        'temporal_weighted_degree': temporal_weighted_degree(ag.edge_index, W, len(ag.node_ids), budget)[0],
        'temporal_degree_discount': temporal_degree_discount(ag.edge_index, W[current_t], len(ag.node_ids), budget),
        'myopic_lookahead': myopic_lookahead(ag.edge_index, W, len(ag.node_ids), budget),
        'temporal_ris_proxy': temporal_ris_proxy(ag.edge_index, W[current_t], len(ag.node_ids), budget),
        'weighted_strength': weighted_strength(ag.edge_index, W[current_t], len(ag.node_ids), budget)[0],
        'distance_strength': distance_strength(ag.edge_index, W[current_t], ag.distances_km, len(ag.node_ids), budget)[0],
        'weighted_betweenness': weighted_betweenness(weighted_graph, budget, weight_attr='temporal_length')[0],
        'community_bridge': community_bridge_nodes(weighted_graph, budget, weight_attr='temporal_weight')[0],
        'cost_aware': cost_aware_ranking(ag.edge_index, W[current_t], ag.distances_km, ag.node_costs, len(ag.node_ids), budget)[0],
    }

    report = {
        'diffusor_last_loss': diff_art.history['loss'][-1],
        'rl': {
            'selected': rl_eval.selected,
            'objective': rl_eval.objective,
            'final_activated_mass': rl_eval.final_activated_mass,
            'final_geographic_coverage': rl_eval.final_geographic_coverage,
            'total_intervention_cost': rl_eval.total_intervention_cost,
        },
        'baselines': {},
    }
    baseline_edge_prob_seq = one_slice if cfg['evaluation']['regime'] in {'spread', 'spatial'} else W
    for name, seeds in baseline_seed_sets.items():
        ev = evaluate_seed_set_raw(
            edge_index=ag.edge_index,
            edge_prob_seq=baseline_edge_prob_seq,
            num_nodes=len(ag.node_ids),
            selected=seeds,
            coords=ag.coords,
            node_costs=ag.node_costs,
            coverage_radius_km=cfg['simulator']['coverage_radius_km'],
            regime=cfg['evaluation']['regime'],
            beta_coverage=cfg['reward']['beta_coverage'],
            lambda_cost=cfg['reward']['lambda_cost'],
            cost_mode=cfg['reward']['cost_mode'],
            constant_cost=cfg['reward']['constant_cost'],
            distance_cost_scale=cfg['reward']['distance_cost_scale'],
            mc_rollouts=cfg['simulator']['mc_rollouts'],
        )
        report['baselines'][name] = {
            'selected': seeds,
            'objective': ev.objective,
            'final_activated_mass': ev.final_activated_mass,
            'final_geographic_coverage': ev.final_geographic_coverage,
            'total_intervention_cost': ev.total_intervention_cost,
        }

    svr, _ = compute_submodularity_metric(env, cfg['evaluation']['submodularity_samples'])
    report['empirical_submodularity_violation_rate'] = svr
    report['adaptation_efficiency'] = summarize_transfer(curve, curve, cfg['evaluation']['threshold_fraction'])

    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
