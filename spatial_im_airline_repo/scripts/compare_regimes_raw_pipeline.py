from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch

from spatial_im.baselines.classical import degree_discount, degree_ranking, greedy_spread
from spatial_im.baselines.dynamic import (
    myopic_lookahead,
    temporal_degree_discount,
    temporal_ris_proxy,
    temporal_weighted_degree,
    weighted_degree_per_slice,
)
from spatial_im.baselines.spatial import (
    build_temporal_weighted_graph,
    community_bridge_nodes,
    cost_aware_ranking,
    distance_strength,
    weighted_betweenness,
    weighted_strength,
)
from spatial_im.data.airline import load_airline_tables
from spatial_im.data.graph_build import as_torch_tensors, build_airline_graph
from spatial_im.data.temporal import generate_synthetic_temporal_weights
from spatial_im.diffusion.gnn_diffusor import GNNDiffusor, train_diffusor
from spatial_im.env.airline_env import AirlineSpatialTemporalEnv
from spatial_im.evaluation.runner import (
    compute_submodularity_metric,
    evaluate_seed_set_raw,
    rollout_policy,
    summarize_transfer,
)
from spatial_im.policy.dqn_agent import DQNAgent, ReplayBuffer
from spatial_im.policy.features import compute_static_spatial_features
from spatial_im.utils.io import load_yaml
from spatial_im.utils.seeds import set_seed


def make_env(diffusor, node_features, edge_index, W_torch, distances, coords, node_costs, policy_static, cfg, regime: str):
    return AirlineSpatialTemporalEnv(
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
        regime=regime,
        cost_mode=cfg['reward']['cost_mode'],
        constant_cost=cfg['reward']['constant_cost'],
        distance_cost_scale=cfg['reward']['distance_cost_scale'],
        use_shaping=cfg['reward'].get('use_shaping', False),
        alpha_spread=cfg['reward'].get('alpha_spread', 0.0),
        alpha_cover=cfg['reward'].get('alpha_cover', 0.0),
        alpha_cost=cfg['reward'].get('alpha_cost', 0.0),
    )


def apply_artifact_cfg(base_cfg: dict, artifact_path: str | None) -> dict:
    cfg = copy.deepcopy(base_cfg)
    if not artifact_path:
        return cfg
    path = Path(artifact_path)
    if not path.exists():
        return cfg
    artifact = json.loads(path.read_text(encoding='utf-8'))
    best = artifact.get('best', {})
    if 'model' in best:
        cfg['model'].update(best['model'])
    if 'rl' in best:
        cfg['rl'].update(best['rl'])
    return cfg


def train_agent(env, cfg: dict, W: np.ndarray, regime: str, start_choices: list[int] | None = None):
    sample_feat, _ = env.reset()
    agent = DQNAgent(
        feature_dim=sample_feat.shape[1],
        lr=cfg['rl']['lr'],
        gamma=cfg['rl']['gamma'],
        hidden_dim=cfg['rl'].get('hidden_dim', 64),
        beta_coverage=cfg['reward']['beta_coverage'],
        lambda_cost=cfg['reward']['lambda_cost'],
        regime=regime,
    )
    buffer = ReplayBuffer(cfg['rl']['buffer_capacity'])
    epsilon = cfg['rl']['eps_start']
    curve: list[float] = []
    for ep in range(cfg['rl']['train_episodes']):
        if start_choices:
            start_snapshot = int(start_choices[ep % len(start_choices)])
        else:
            start_snapshot = 0
        state_feat, legal = env.reset(start_snapshot=start_snapshot)
        action_prior = env.action_prior_scores()
        ep_return = 0.0
        while True:
            action = agent.act(state_feat, legal, epsilon=epsilon, action_prior=action_prior)
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


def baseline_seed_sets(ag, W: np.ndarray, cfg: dict, regime: str):
    budget = cfg['rl']['budget']
    t0 = 0
    one_slice = W[t0:t0 + 1]
    weighted_graph = build_temporal_weighted_graph(ag.graph, ag.edge_index, W[t0], ag.distances_km)
    if regime == 'spread':
        return {
            'greedy_spread': greedy_spread(ag.edge_index, one_slice, len(ag.node_ids), budget, cfg['simulator']['mc_rollouts']),
            'degree': degree_ranking(ag.graph, budget),
            'degree_discount': degree_discount(ag.graph, budget),
        }
    if regime == 'dynamic':
        return {
            'weighted_degree_slice': weighted_degree_per_slice(ag.edge_index, W[t0], len(ag.node_ids), budget)[0],
            'temporal_weighted_degree': temporal_weighted_degree(ag.edge_index, W, len(ag.node_ids), budget)[0],
            'temporal_degree_discount': temporal_degree_discount(ag.edge_index, W[t0], len(ag.node_ids), budget),
            'myopic_lookahead': myopic_lookahead(ag.edge_index, W, len(ag.node_ids), budget, mc_rollouts=cfg['simulator']['mc_rollouts']),
            'temporal_ris_proxy': temporal_ris_proxy(ag.edge_index, W[t0], len(ag.node_ids), budget, seed=cfg['seed']),
        }
    return {
        'weighted_strength': weighted_strength(ag.edge_index, W[t0], len(ag.node_ids), budget)[0],
        'distance_strength': distance_strength(ag.edge_index, W[t0], ag.distances_km, len(ag.node_ids), budget)[0],
        'weighted_betweenness': weighted_betweenness(weighted_graph, budget, weight_attr='temporal_length')[0],
        'community_bridge': community_bridge_nodes(weighted_graph, budget, weight_attr='temporal_weight')[0],
        'cost_aware': cost_aware_ranking(ag.edge_index, W[t0], ag.distances_km, ag.node_costs, len(ag.node_ids), budget)[0],
    }


def eval_report_raw(edge_index, edge_prob_seq, num_nodes, coords, node_costs, cfg: dict, regime: str, seeds, name_of):
    ev = evaluate_seed_set_raw(
        edge_index=edge_index,
        edge_prob_seq=edge_prob_seq,
        num_nodes=num_nodes,
        selected=seeds,
        coords=coords,
        node_costs=node_costs,
        coverage_radius_km=cfg['simulator']['coverage_radius_km'],
        regime=regime,
        beta_coverage=cfg['reward']['beta_coverage'],
        lambda_cost=cfg['reward']['lambda_cost'],
        cost_mode=cfg['reward']['cost_mode'],
        constant_cost=cfg['reward']['constant_cost'],
        distance_cost_scale=cfg['reward']['distance_cost_scale'],
        mc_rollouts=cfg['simulator']['mc_rollouts'],
    )
    return {
        'seeds_idx': seeds,
        'seeds_iata': [name_of[i] for i in seeds],
        'objective': ev.objective,
        'final_activated_mass': ev.final_activated_mass,
        'expected_coverage': ev.final_geographic_coverage,
        'intervention_cost': ev.total_intervention_cost,
    }


def transfer_report(env, cfg: dict, W: np.ndarray, regime: str):
    transfer_split = cfg['evaluation']['transfer_split']
    if regime in {'spread', 'spatial'}:
        transfer_starts = [0]
        scratch_starts = [0]
    else:
        transfer_starts = list(range(0, max(1, transfer_split)))
        scratch_starts = list(range(transfer_split, max(transfer_split + 1, W.shape[0] - 1)))
    transfer_agent, transfer_curve = train_agent(env, cfg, W, regime, start_choices=transfer_starts)
    scratch_agent, scratch_curve = train_agent(env, cfg, W, regime, start_choices=scratch_starts)
    transfer_eval = rollout_policy(env, transfer_agent, epsilon=0.0)
    scratch_eval = rollout_policy(env, scratch_agent, epsilon=0.0)
    summary = summarize_transfer(transfer_curve, scratch_curve, cfg['evaluation']['threshold_fraction'])
    return {
        'transfer_eval_objective': transfer_eval.objective,
        'scratch_eval_objective': scratch_eval.objective,
        'transfer_ratio': summary['transfer_ratio'],
        'adaptation_efficiency_transfer': summary['adaptation_efficiency_transfer'],
        'adaptation_efficiency_scratch': summary['adaptation_efficiency_scratch'],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--spread-artifact', default=None)
    parser.add_argument('--dynamic-artifact', default=None)
    parser.add_argument('--spatial-artifact', default=None)
    parser.add_argument('--out', default='artifacts/compare_regimes_raw_pipeline.json')
    args = parser.parse_args()

    base_cfg = load_yaml(args.config)
    set_seed(base_cfg['seed'])

    tables = load_airline_tables(base_cfg['data']['airports_csv'], base_cfg['data']['routes_csv'])
    ag = build_airline_graph(tables)
    edge_index, node_features, coords, _, distances, node_costs = as_torch_tensors(ag)
    W = generate_synthetic_temporal_weights(
        ag,
        base_cfg['data']['snapshots'],
        base_cfg['data']['seasonal_strength'],
        base_cfg['data']['noise_std'],
        base_cfg['seed'],
    )
    W_torch = torch.as_tensor(W, dtype=torch.float32)
    policy_static = torch.as_tensor(compute_static_spatial_features(ag.graph, ag.coords, ag.node_costs), dtype=torch.float32)
    name_of = {i: tables.airports[tables.airports['airport_id'] == aid].iloc[0]['iata'] for i, aid in ag.idx_to_id.items()}
    artifact_map = {
        'spread': args.spread_artifact,
        'dynamic': args.dynamic_artifact,
        'spatial': args.spatial_artifact,
    }

    report = {'regimes': {}}

    for regime in ['spread', 'dynamic', 'spatial']:
        cfg = apply_artifact_cfg(base_cfg, artifact_map[regime])
        cfg['evaluation']['regime'] = regime
        set_seed(cfg['seed'])
        diffusor = GNNDiffusor(
            node_feat_dim=node_features.size(1),
            hidden_dim=cfg['model']['hidden_dim'],
            layers=cfg['model']['layers'],
        )
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
            regime=regime,
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
        env = make_env(diffusor, node_features, edge_index, W_torch, distances, coords, node_costs, policy_static, cfg, regime)
        agent, curve = train_agent(env, cfg, W, regime)
        baseline_edge_prob_seq = W[0:1] if regime in {'spread', 'spatial'} else W
        rl_env_eval = rollout_policy(env, agent, epsilon=0.0)
        rl_eval = evaluate_seed_set_raw(
            edge_index=ag.edge_index,
            edge_prob_seq=baseline_edge_prob_seq,
            num_nodes=len(ag.node_ids),
            selected=rl_env_eval.selected,
            coords=ag.coords,
            node_costs=ag.node_costs,
            coverage_radius_km=cfg['simulator']['coverage_radius_km'],
            regime=regime,
            beta_coverage=cfg['reward']['beta_coverage'],
            lambda_cost=cfg['reward']['lambda_cost'],
            cost_mode=cfg['reward']['cost_mode'],
            constant_cost=cfg['reward']['constant_cost'],
            distance_cost_scale=cfg['reward']['distance_cost_scale'],
            mc_rollouts=cfg['simulator']['mc_rollouts'],
        )
        baselines = baseline_seed_sets(ag, W, cfg, regime)
        baseline_reports = {
            name: eval_report_raw(
                edge_index=ag.edge_index,
                edge_prob_seq=baseline_edge_prob_seq,
                num_nodes=len(ag.node_ids),
                coords=ag.coords,
                node_costs=ag.node_costs,
                cfg=cfg,
                regime=regime,
                seeds=seeds,
                name_of=name_of,
            )
            for name, seeds in baselines.items()
        }
        best_name, best_report = max(baseline_reports.items(), key=lambda kv: kv[1]['objective'])
        transfer = transfer_report(env, cfg, W, regime)
        svr, _ = compute_submodularity_metric(env, cfg['evaluation']['submodularity_samples'])
        report['regimes'][regime] = {
            'train_start_mode': 'fixed_snapshot_0' if regime in {'spread', 'spatial'} else 'fixed_window_start_0',
            'diffusor_last_loss': diff_art.history['loss'][-1],
            'rl': {
                'seeds_idx': rl_eval.selected,
                'seeds_iata': [name_of[i] for i in rl_eval.selected],
                'objective': rl_eval.objective,
                'final_activated_mass': rl_eval.final_activated_mass,
                'expected_coverage': rl_eval.final_geographic_coverage,
                'intervention_cost': rl_eval.total_intervention_cost,
                'training_curve_last10_mean': float(np.mean(curve[-10:])),
            },
            'rl_learned_env': {
                'objective': rl_env_eval.objective,
                'final_activated_mass': rl_env_eval.final_activated_mass,
                'expected_coverage': rl_env_eval.final_geographic_coverage,
                'intervention_cost': rl_env_eval.total_intervention_cost,
            },
            'baselines': baseline_reports,
            'best_baseline_name': best_name,
            'best_baseline': best_report,
            'transfer': transfer,
            'submodularity_violation_rate': svr,
        }

    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
