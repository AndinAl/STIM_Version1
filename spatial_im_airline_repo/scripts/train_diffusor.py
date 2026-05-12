from __future__ import annotations

import argparse
from pathlib import Path
import torch

from spatial_im.utils.io import load_yaml
from spatial_im.utils.seeds import set_seed
from spatial_im.data.airline import load_airline_tables
from spatial_im.data.graph_build import build_airline_graph, as_torch_tensors
from spatial_im.data.temporal import generate_synthetic_temporal_weights
from spatial_im.diffusion.gnn_diffusor import GNNDiffusor, train_diffusor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--out', default='artifacts/diffusor.pt')
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    set_seed(cfg['seed'])
    tables = load_airline_tables(cfg['data']['airports_csv'], cfg['data']['routes_csv'])
    ag = build_airline_graph(tables)
    edge_index, node_features, coords, base_edge_weight, distances, node_costs = as_torch_tensors(ag)
    W = generate_synthetic_temporal_weights(
        ag,
        snapshots=cfg['data']['snapshots'],
        seasonal_strength=cfg['data']['seasonal_strength'],
        noise_std=cfg['data']['noise_std'],
        seed=cfg['seed'],
    )
    model = GNNDiffusor(node_feat_dim=node_features.size(1), hidden_dim=cfg['model']['hidden_dim'], layers=cfg['model']['layers'])
    artifacts = train_diffusor(
        model=model,
        node_features=node_features,
        edge_index=edge_index,
        edge_prob_seq=W,
        distances_km=distances,
        epochs=cfg['model']['diffusor_epochs'],
        batch_size=cfg['model']['batch_size'],
        lr=cfg['model']['diffusor_lr'],
        seed=cfg['seed'],
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
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'state_dict': model.state_dict(), 'history': artifacts.history, 'temporal_weights': W}, out)
    print('Saved diffusor to', out)
    print('Final training loss:', artifacts.history['loss'][-1])


if __name__ == '__main__':
    main()
