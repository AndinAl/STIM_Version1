from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch

from spatial_im.baselines.dynamic import (
    myopic_lookahead,
    temporal_degree_discount,
    temporal_ris_proxy,
    temporal_weighted_degree,
    weighted_degree_per_slice,
)
from spatial_im.baselines.learned import (
    graph_with_edge_scores,
    infer_edge_gate_scores,
    inferred_community_bridge_nodes,
    inferred_graph_degree_discount,
    inferred_graph_degree_ranking,
    inferred_greedy_spread,
    inferred_weighted_betweenness,
)
from spatial_im.baselines.spatial import cost_aware_ranking, distance_strength, weighted_strength
from spatial_im.data.airline import load_airline_tables
from spatial_im.data.graph_build import as_torch_tensors, build_airline_graph
from spatial_im.data.temporal import generate_synthetic_temporal_weights
from spatial_im.diffusion.gnn_diffusor import GNNDiffusor, train_diffusor
from spatial_im.env.airline_env import AirlineSpatialTemporalEnv
from spatial_im.evaluation.runner import (
    compute_submodularity_metric,
    evaluate_seed_set,
    rollout_policy,
    summarize_transfer,
)
from spatial_im.policy.dqn_agent import DQNAgent, ReplayBuffer
from spatial_im.policy.features import compute_static_spatial_features
from spatial_im.utils.io import load_yaml
from spatial_im.utils.seeds import set_seed


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


def _episode_start_snapshot(regime: str, ep: int, W: np.ndarray, start_choices: list[int] | None = None) -> int:
    if start_choices is not None and len(start_choices) > 0:
        return int(start_choices[ep % len(start_choices)])
    if regime == 'spread':
        # Match the main evaluation rollout, which always starts from snapshot t=0.
        return 0
    return int(ep % max(1, W.shape[0] - 1))


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
        start_snapshot = _episode_start_snapshot(regime, ep, W, start_choices=start_choices)
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


def transfer_report(env, cfg: dict, W: np.ndarray, regime: str) -> dict:
    transfer_split = cfg['evaluation']['transfer_split']
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


def eval_report(env, seeds, name_of):
    ev = evaluate_seed_set(env, seeds)
    return {
        'seeds_idx': seeds,
        'seeds_iata': [name_of[i] for i in seeds],
        'objective': ev.objective,
        'final_activated_mass': ev.final_activated_mass,
        'expected_coverage': ev.final_geographic_coverage,
        'intervention_cost': ev.total_intervention_cost,
    }


def regime_baselines(regime: str, ag, env, inferred_seq: np.ndarray, cfg: dict, name_of: dict[int, str]):
    budget = cfg['rl']['budget']
    inferred_t0 = inferred_seq[0]
    inferred_graph = graph_with_edge_scores(ag.graph, ag.edge_index, inferred_t0, edge_cost_attr='inferred_cost', edge_weight_attr='inferred_weight')

    baselines: dict[str, list[int]] = {}
    if regime == 'spread':
        baselines['greedy_spread'] = inferred_greedy_spread(
            ag.edge_index,
            inferred_seq[:1],
            len(ag.node_ids),
            budget,
            mc_rollouts=cfg['simulator']['mc_rollouts'],
        )
        # These heuristics are topology-only; the inferred graph carries the same adjacency plus edge attributes.
        baselines['degree'] = inferred_graph_degree_ranking(inferred_graph, budget)
        baselines['degree_discount'] = inferred_graph_degree_discount(inferred_graph, budget)
    elif regime == 'dynamic':
        baselines['weighted_degree_slice'] = weighted_degree_per_slice(ag.edge_index, inferred_t0, len(ag.node_ids), budget)[0]
        baselines['temporal_weighted_degree'] = temporal_weighted_degree(ag.edge_index, inferred_seq, len(ag.node_ids), budget)[0]
        baselines['temporal_degree_discount'] = temporal_degree_discount(ag.edge_index, inferred_t0, len(ag.node_ids), budget)
        baselines['myopic_lookahead'] = myopic_lookahead(
            ag.edge_index,
            inferred_seq,
            len(ag.node_ids),
            budget,
            mc_rollouts=cfg['simulator']['mc_rollouts'],
        )
        baselines['temporal_ris_proxy'] = temporal_ris_proxy(
            ag.edge_index,
            inferred_t0,
            len(ag.node_ids),
            budget,
            seed=cfg['seed'],
        )
    else:
        baselines['weighted_strength'] = weighted_strength(ag.edge_index, inferred_t0, len(ag.node_ids), budget)[0]
        baselines['distance_strength'] = distance_strength(ag.edge_index, inferred_t0, ag.distances_km, len(ag.node_ids), budget)[0]
        baselines['weighted_betweenness'] = inferred_weighted_betweenness(inferred_graph, ag.edge_index, inferred_t0, budget)[0]
        baselines['community_bridge'] = inferred_community_bridge_nodes(inferred_graph, ag.edge_index, inferred_t0, budget)[0]
        baselines['cost_aware'] = cost_aware_ranking(ag.edge_index, inferred_t0, ag.distances_km, ag.node_costs, len(ag.node_ids), budget)[0]

    reports = {name: eval_report(env, seeds, name_of) for name, seeds in baselines.items()}
    best_name, best_report = max(reports.items(), key=lambda kv: kv[1]['objective'])
    return reports, best_name, best_report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--spread-artifact', default='artifacts/optuna_spread.json')
    parser.add_argument('--dynamic-artifact', default='artifacts/optuna_dynamic.json')
    parser.add_argument('--spatial-artifact', default='artifacts/optuna_spatial.json')
    parser.add_argument('--out', default='artifacts/compare_regimes_learned_env.json')
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
        regime_cfg = apply_artifact_cfg(base_cfg, artifact_map[regime])
        regime_cfg['evaluation']['regime'] = regime

        set_seed(regime_cfg['seed'])
        diffusor = GNNDiffusor(
            node_feat_dim=node_features.size(1),
            hidden_dim=regime_cfg['model']['hidden_dim'],
            layers=regime_cfg['model']['layers'],
        )
        diff_art = train_diffusor(
            diffusor,
            node_features,
            edge_index,
            W,
            distances,
            regime_cfg['model']['diffusor_epochs'],
            regime_cfg['model']['batch_size'],
            regime_cfg['model']['diffusor_lr'],
            regime_cfg['seed'],
            coords=coords,
            node_costs=node_costs,
            regime=regime,
            beta_coverage=regime_cfg['reward']['beta_coverage'],
            lambda_cost=regime_cfg['reward']['lambda_cost'],
            cost_mode=regime_cfg['reward']['cost_mode'],
            constant_cost=regime_cfg['reward']['constant_cost'],
            distance_cost_scale=regime_cfg['reward']['distance_cost_scale'],
            coverage_radius_km=regime_cfg['simulator']['coverage_radius_km'],
            teacher_mc_rollouts=regime_cfg['model'].get('teacher_mc_rollouts', regime_cfg['simulator']['mc_rollouts']),
            marginal_loss_weight=regime_cfg['model'].get('marginal_loss_weight', 1.0),
            ranking_loss_weight=regime_cfg['model'].get('ranking_loss_weight', 0.2),
            selection_budget=regime_cfg['rl']['budget'],
        )
        env = make_env(
            diffusor,
            node_features,
            edge_index,
            W_torch,
            distances,
            coords,
            node_costs,
            policy_static,
            regime_cfg,
            regime,
        )

        agent, curve = train_agent(env, regime_cfg, W, regime)
        rl_eval = rollout_policy(env, agent, epsilon=0.0)
        transfer = transfer_report(env, regime_cfg, W, regime)
        svr, _ = compute_submodularity_metric(env, regime_cfg['evaluation']['submodularity_samples'])
        raw_window = W if regime == 'dynamic' else W[0:1]
        inferred_seq = infer_edge_gate_scores(diffusor, raw_window, ag.distances_km)
        baseline_reports, best_name, best_report = regime_baselines(regime, ag, env, inferred_seq, regime_cfg, name_of)
        inferred_W = inferred_seq

        report['regimes'][regime] = {
            'train_start_mode': 'fixed_snapshot_0' if regime == 'spread' else 'cycled_snapshots',
            'diffusor_last_loss': diff_art.history['loss'][-1],
            'inferred_weight_summary': {
                'raw_min': float(raw_window.min()),
                'raw_max': float(raw_window.max()),
                'inferred_min': float(inferred_W.min()),
                'inferred_max': float(inferred_W.max()),
            },
            'rl': {
                'seeds_idx': rl_eval.selected,
                'seeds_iata': [name_of[i] for i in rl_eval.selected],
                'objective': rl_eval.objective,
                'final_activated_mass': rl_eval.final_activated_mass,
                'expected_coverage': rl_eval.final_geographic_coverage,
                'intervention_cost': rl_eval.total_intervention_cost,
                'training_curve_last10_mean': float(np.mean(curve[-10:])),
            },
            'baselines': baseline_reports,
            'best_baseline_name': best_name,
            'best_baseline': best_report,
            'transfer': transfer,
            'submodularity_violation_rate': svr,
        }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
