from __future__ import annotations

import argparse
from pathlib import Path
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--out', default='artifacts/rl_agent.pt')
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
    learning_curve = []

    for ep in range(cfg['rl']['train_episodes']):
        start_snapshot = 0 if cfg['evaluation']['regime'] == 'spread' else (ep % max(1, W.shape[0] - 1))
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
        learning_curve.append(ep_return)
        if len(buffer) >= cfg['rl']['batch_size'] and ep >= cfg['rl']['warmup_episodes']:
            batch = buffer.sample(cfg['rl']['batch_size'])
            _ = agent.update(batch)
        if ep % cfg['rl']['target_update_every'] == 0:
            agent.sync_target()
        epsilon = max(cfg['rl']['eps_end'], epsilon * cfg['rl']['eps_decay'])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'agent_state_dict': agent.q.state_dict(), 'learning_curve': learning_curve, 'temporal_weights': W}, out)
    print('Saved RL agent to', out)
    print('Last 10 episode mean return:', float(np.mean(learning_curve[-10:])))


if __name__ == '__main__':
    main()
