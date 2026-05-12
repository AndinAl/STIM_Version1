from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

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
from spatial_im.evaluation.metrics import transfer_ratio
from spatial_im.evaluation.runner import evaluate_seed_set_raw, summarize_transfer
from spatial_im.training import (
    adapt_from_pretrained_artifacts,
    adapt_from_scratch_policy,
    prepare_target_graph,
    pretrain_reusable_policy,
    reuse_policy_zero_shot_on_prepared_graph,
)
from spatial_im.utils.io import load_yaml
from spatial_im.utils.seeds import set_seed


def apply_artifact_cfg(base_cfg: dict, artifact_path: str | None) -> dict:
    cfg = copy.deepcopy(base_cfg)
    # Keep the transfer-evaluation budget from the passed config.
    # Artifact files can contain a source-tuning budget (e.g., 3) that should
    # not override target transfer budgets (e.g., 5/10/15).
    target_budget = cfg.get('rl', {}).get('budget')
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
    if target_budget is not None:
        cfg['rl']['budget'] = int(target_budget)
    return cfg


def baseline_seed_sets(target_graph, cfg: dict, regime: str, start_snapshot: int):
    budget = cfg['rl']['budget']
    W = target_graph.temporal_weights
    t0 = int(start_snapshot)
    one_slice = W[t0:t0 + 1]
    weighted_graph = build_temporal_weighted_graph(target_graph.graph.graph, target_graph.graph.edge_index, W[t0], target_graph.graph.distances_km)
    if regime == 'spread':
        return {
            'greedy_spread': greedy_spread(target_graph.graph.edge_index, one_slice, len(target_graph.graph.node_ids), budget, cfg['simulator']['mc_rollouts']),
            'degree': degree_ranking(target_graph.graph.graph, budget),
            'degree_discount': degree_discount(target_graph.graph.graph, budget),
        }
    if regime == 'dynamic':
        return {
            'weighted_degree_slice': weighted_degree_per_slice(target_graph.graph.edge_index, W[t0], len(target_graph.graph.node_ids), budget)[0],
            'temporal_weighted_degree': temporal_weighted_degree(target_graph.graph.edge_index, W, len(target_graph.graph.node_ids), budget)[0],
            'temporal_degree_discount': temporal_degree_discount(target_graph.graph.edge_index, W[t0], len(target_graph.graph.node_ids), budget),
            'myopic_lookahead': myopic_lookahead(target_graph.graph.edge_index, W, len(target_graph.graph.node_ids), budget, mc_rollouts=cfg['simulator']['mc_rollouts']),
            'temporal_ris_proxy': temporal_ris_proxy(target_graph.graph.edge_index, W[t0], len(target_graph.graph.node_ids), budget, seed=cfg['seed']),
        }
    return {
        'weighted_strength': weighted_strength(target_graph.graph.edge_index, W[t0], len(target_graph.graph.node_ids), budget)[0],
        'distance_strength': distance_strength(target_graph.graph.edge_index, W[t0], target_graph.graph.distances_km, len(target_graph.graph.node_ids), budget)[0],
        'weighted_betweenness': weighted_betweenness(weighted_graph, budget, weight_attr='temporal_length')[0],
        'community_bridge': community_bridge_nodes(weighted_graph, budget, weight_attr='temporal_weight')[0],
        'cost_aware': cost_aware_ranking(target_graph.graph.edge_index, W[t0], target_graph.graph.distances_km, target_graph.graph.node_costs, len(target_graph.graph.node_ids), budget)[0],
    }


def eval_raw_target(target_graph, cfg: dict, regime: str, start_snapshot: int, seeds):
    W = target_graph.temporal_weights
    if regime == 'dynamic':
        edge_prob_seq = W[int(start_snapshot):]
    else:
        edge_prob_seq = W[int(start_snapshot):int(start_snapshot) + 1]
    return evaluate_seed_set_raw(
        edge_index=target_graph.graph.edge_index,
        edge_prob_seq=edge_prob_seq,
        num_nodes=len(target_graph.graph.node_ids),
        selected=seeds,
        coords=target_graph.graph.coords,
        node_costs=target_graph.graph.node_costs,
        coverage_radius_km=cfg['simulator']['coverage_radius_km'],
        regime=regime,
        beta_coverage=cfg['reward']['beta_coverage'],
        lambda_cost=cfg['reward']['lambda_cost'],
        cost_mode=cfg['reward']['cost_mode'],
        constant_cost=cfg['reward']['constant_cost'],
        distance_cost_scale=cfg['reward']['distance_cost_scale'],
        mc_rollouts=cfg['simulator']['mc_rollouts'],
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--spread-artifact', default=None)
    parser.add_argument('--dynamic-artifact', default=None)
    parser.add_argument('--spatial-artifact', default=None)
    parser.add_argument('--out', default='artifacts/transfer_homogeneous_report.json')
    args = parser.parse_args()

    base_cfg = load_yaml(args.config)
    artifact_map = {
        'spread': args.spread_artifact,
        'dynamic': args.dynamic_artifact,
        'spatial': args.spatial_artifact,
    }
    report = {'regimes': {}}

    for regime in ['spread', 'dynamic', 'spatial']:
        cfg = apply_artifact_cfg(base_cfg, artifact_map[regime])
        cfg['evaluation']['regime'] = regime
        cfg.setdefault('pretraining', {})
        cfg['pretraining']['regimes'] = [regime]
        cfg['pretraining']['budgets'] = [cfg['rl']['budget']]
        set_seed(cfg['seed'])

        start_snapshot = int(cfg.get('transfer_experiment', {}).get('start_snapshot', 0))
        adapt_iterations = int(cfg.get('transfer_experiment', {}).get('adapt_iterations', cfg.get('adaptation', {}).get('iterations', 4)))

        pretrained = pretrain_reusable_policy(cfg)
        target_graph = prepare_target_graph(cfg)

        zero_shot = reuse_policy_zero_shot_on_prepared_graph(
            cfg=cfg,
            target_graph=target_graph,
            pretrained=pretrained,
            regime=regime,
            budget=cfg['rl']['budget'],
            start_snapshot=start_snapshot,
        )
        transfer_adapt = adapt_from_pretrained_artifacts(
            cfg=cfg,
            pretrained=pretrained,
            target_graph=target_graph,
            regime=regime,
            budget=cfg['rl']['budget'],
            start_snapshot=start_snapshot,
            n_adapt=adapt_iterations,
        )
        scratch_adapt = adapt_from_scratch_policy(
            cfg=cfg,
            regime=regime,
            budget=cfg['rl']['budget'],
            start_snapshot=start_snapshot,
            n_adapt=adapt_iterations,
        )

        transfer_summary = summarize_transfer(
            transfer_adapt.history['episode_return'],
            scratch_adapt.history['episode_return'],
            cfg['evaluation']['threshold_fraction'],
        )
        raw_transfer_ratio_warmstart_vs_scratch = transfer_ratio(
            transfer_adapt.result.raw_objective,
            scratch_adapt.result.raw_objective,
        )
        raw_transfer_ratio_zero_shot_vs_scratch = transfer_ratio(
            zero_shot.raw_objective,
            scratch_adapt.result.raw_objective,
        )
        raw_objective_gain_warmstart_vs_scratch = (
            transfer_adapt.result.raw_objective - scratch_adapt.result.raw_objective
        )
        raw_objective_gain_zero_shot_vs_scratch = (
            zero_shot.raw_objective - scratch_adapt.result.raw_objective
        )

        baselines = baseline_seed_sets(target_graph, cfg, regime, start_snapshot)
        baseline_reports = {}
        for name, seeds in baselines.items():
            ev = eval_raw_target(target_graph, cfg, regime, start_snapshot, seeds)
            baseline_reports[name] = {
                'seeds_idx': list(seeds),
                'seeds_iata': [target_graph.graph.graph.nodes[i].get('iata', str(i)) for i in seeds],
                'objective': ev.objective,
                'final_activated_mass': ev.final_activated_mass,
                'expected_coverage': ev.final_geographic_coverage,
                'intervention_cost': ev.total_intervention_cost,
            }
        best_name, best_report = max(baseline_reports.items(), key=lambda kv: kv[1]['objective'])

        report['regimes'][regime] = {
            'target_graph': target_graph.name,
            'target_start_snapshot': start_snapshot,
            'source_graphs': pretrained.source_graphs,
            'zero_shot': {
                'seeds_idx': zero_shot.selected,
                'seeds_iata': zero_shot.selected_iata,
                'learned_env_objective': zero_shot.learned_env_objective,
                'raw_objective': zero_shot.raw_objective,
                'raw_final_activated_mass': zero_shot.raw_final_activated_mass,
                'raw_expected_coverage': zero_shot.raw_expected_coverage,
                'raw_intervention_cost': zero_shot.raw_intervention_cost,
            },
            'transfer_adapt': {
                'seeds_idx': transfer_adapt.result.selected,
                'seeds_iata': transfer_adapt.result.selected_iata,
                'learned_env_objective': transfer_adapt.result.learned_env_objective,
                'raw_objective': transfer_adapt.result.raw_objective,
                'raw_final_activated_mass': transfer_adapt.result.raw_final_activated_mass,
                'raw_expected_coverage': transfer_adapt.result.raw_expected_coverage,
                'raw_intervention_cost': transfer_adapt.result.raw_intervention_cost,
                'adapt_episode_returns': transfer_adapt.history['episode_return'],
            },
            'scratch_adapt': {
                'seeds_idx': scratch_adapt.result.selected,
                'seeds_iata': scratch_adapt.result.selected_iata,
                'learned_env_objective': scratch_adapt.result.learned_env_objective,
                'raw_objective': scratch_adapt.result.raw_objective,
                'raw_final_activated_mass': scratch_adapt.result.raw_final_activated_mass,
                'raw_expected_coverage': scratch_adapt.result.raw_expected_coverage,
                'raw_intervention_cost': scratch_adapt.result.raw_intervention_cost,
                'adapt_episode_returns': scratch_adapt.history['episode_return'],
            },
            'transfer_metrics': {
                'curve_transfer_ratio': transfer_summary['transfer_ratio'],
                'adaptation_efficiency_transfer': transfer_summary['adaptation_efficiency_transfer'],
                'adaptation_efficiency_scratch': transfer_summary['adaptation_efficiency_scratch'],
                'raw_transfer_ratio_warmstart_vs_scratch': raw_transfer_ratio_warmstart_vs_scratch,
                'raw_transfer_ratio_zero_shot_vs_scratch': raw_transfer_ratio_zero_shot_vs_scratch,
                'raw_objective_gain_warmstart_vs_scratch': raw_objective_gain_warmstart_vs_scratch,
                'raw_objective_gain_zero_shot_vs_scratch': raw_objective_gain_zero_shot_vs_scratch,
            },
            'baselines': baseline_reports,
            'best_baseline_name': best_name,
            'best_baseline': best_report,
        }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
