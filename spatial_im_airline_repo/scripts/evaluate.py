from __future__ import annotations

import argparse
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
from spatial_im.evaluation.runner import compute_submodularity_metric, rollout_policy, summarize_transfer


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

    # Transfer setup: first train on earlier windows, then fine-tune/test on later windows.
    transfer_split = cfg['evaluation']['transfer_split']
    sample_feat, _ = env.reset()
    feat_dim = sample_feat.shape[1]

    def train_agent(window_start, window_end, episodes):
        agent = DQNAgent(
            feature_dim=feat_dim,
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
        if cfg['evaluation']['regime'] == 'spread':
            valid_starts = [0]
        else:
            valid_starts = list(range(window_start, max(window_start + 1, window_end)))
        for ep in range(episodes):
            state_feat, legal = env.reset(start_snapshot=valid_starts[ep % len(valid_starts)])
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
                agent.update(batch)
            if ep % cfg['rl']['target_update_every'] == 0:
                agent.sync_target()
            epsilon = max(cfg['rl']['eps_end'], epsilon * cfg['rl']['eps_decay'])
        return agent, learning_curve

    transfer_agent, transfer_curve = train_agent(0, transfer_split, cfg['rl']['train_episodes'] // 2)
    scratch_agent, scratch_curve = train_agent(transfer_split, W.shape[0] - 1, cfg['rl']['train_episodes'] // 2)

    transfer_eval = rollout_policy(env, transfer_agent, epsilon=0.0)
    scratch_eval = rollout_policy(env, scratch_agent, epsilon=0.0)
    transfer_summary = summarize_transfer(transfer_curve, scratch_curve, cfg['evaluation']['threshold_fraction'])
    svr, details = compute_submodularity_metric(env, cfg['evaluation']['submodularity_samples'])

    print('Transfer evaluation objective:', transfer_eval.objective)
    print('Scratch evaluation objective:', scratch_eval.objective)
    print('Transfer summary:', transfer_summary)
    print('Empirical submodularity-violation rate:', svr)


if __name__ == '__main__':
    main()
