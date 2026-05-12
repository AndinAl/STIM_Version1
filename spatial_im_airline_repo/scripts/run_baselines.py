from __future__ import annotations

import argparse
import json
import torch

from spatial_im.utils.io import load_yaml
from spatial_im.utils.seeds import set_seed
from spatial_im.data.airline import load_airline_tables
from spatial_im.data.graph_build import build_airline_graph, as_torch_tensors
from spatial_im.data.temporal import generate_synthetic_temporal_weights
from spatial_im.diffusion.gnn_diffusor import GNNDiffusor, train_diffusor
from spatial_im.policy.features import compute_static_spatial_features
from spatial_im.env.airline_env import AirlineSpatialTemporalEnv
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
from spatial_im.evaluation.runner import evaluate_seed_set_raw


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
    train_diffusor(
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

    budget = cfg['rl']['budget']
    current_t = 0
    one_slice = W[current_t:current_t+1]
    weighted_graph = build_temporal_weighted_graph(ag.graph, ag.edge_index, W[current_t], ag.distances_km)
    results = {}

    baselines = {}
    baselines['greedy_spread'] = greedy_spread(ag.edge_index, one_slice, len(ag.node_ids), budget, cfg['simulator']['mc_rollouts'])
    baselines['degree'] = degree_ranking(ag.graph, budget)
    baselines['degree_discount'] = degree_discount(ag.graph, budget)
    baselines['weighted_degree_slice'] = weighted_degree_per_slice(ag.edge_index, W[current_t], len(ag.node_ids), budget)[0]
    baselines['temporal_weighted_degree'] = temporal_weighted_degree(ag.edge_index, W, len(ag.node_ids), budget)[0]
    baselines['temporal_degree_discount'] = temporal_degree_discount(ag.edge_index, W[current_t], len(ag.node_ids), budget)
    baselines['myopic_lookahead'] = myopic_lookahead(ag.edge_index, W, len(ag.node_ids), budget)
    baselines['temporal_ris_proxy'] = temporal_ris_proxy(ag.edge_index, W[current_t], len(ag.node_ids), budget)
    baselines['weighted_strength'] = weighted_strength(ag.edge_index, W[current_t], len(ag.node_ids), budget)[0]
    baselines['distance_strength'] = distance_strength(ag.edge_index, W[current_t], ag.distances_km, len(ag.node_ids), budget)[0]
    baselines['weighted_betweenness'] = weighted_betweenness(weighted_graph, budget, weight_attr='temporal_length')[0]
    baselines['community_bridge'] = community_bridge_nodes(weighted_graph, budget, weight_attr='temporal_weight')[0]
    baselines['cost_aware'] = cost_aware_ranking(ag.edge_index, W[current_t], ag.distances_km, ag.node_costs, len(ag.node_ids), budget)[0]

    edge_prob_seq = one_slice if cfg['evaluation']['regime'] in {'spread', 'spatial'} else W
    for name, seeds in baselines.items():
        res = evaluate_seed_set_raw(
            edge_index=ag.edge_index,
            edge_prob_seq=edge_prob_seq,
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
        results[name] = {
            'seeds': seeds,
            'objective': res.objective,
            'final_activated_mass': res.final_activated_mass,
            'final_geographic_coverage': res.final_geographic_coverage,
            'total_intervention_cost': res.total_intervention_cost,
        }
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
