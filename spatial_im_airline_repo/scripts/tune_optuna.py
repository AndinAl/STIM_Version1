from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch

try:
    import optuna
except ImportError as exc:
    raise SystemExit(
        "optuna is not installed. Install dependencies first, for example: pip install -r requirements.txt"
    ) from exc

from spatial_im.data.airline import load_airline_tables
from spatial_im.data.graph_build import as_torch_tensors, build_airline_graph
from spatial_im.data.temporal import generate_synthetic_temporal_weights
from spatial_im.diffusion.gnn_diffusor import GNNDiffusor, train_diffusor
from spatial_im.env.airline_env import AirlineSpatialTemporalEnv
from spatial_im.evaluation.runner import rollout_policy
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


def train_agent(env, rl_cfg: dict, W: np.ndarray, regime: str) -> tuple[DQNAgent, list[float]]:
    sample_feat, _ = env.reset()
    agent = DQNAgent(
        feature_dim=sample_feat.shape[1],
        lr=rl_cfg['lr'],
        gamma=rl_cfg['gamma'],
        hidden_dim=rl_cfg.get('hidden_dim', 64),
        beta_coverage=env.beta_coverage,
        lambda_cost=env.lambda_cost,
        regime=regime,
    )
    buffer = ReplayBuffer(rl_cfg['buffer_capacity'])
    epsilon = rl_cfg['eps_start']
    curve: list[float] = []

    for ep in range(rl_cfg['train_episodes']):
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
        if len(buffer) >= rl_cfg['batch_size'] and ep >= rl_cfg['warmup_episodes']:
            agent.update(buffer.sample(rl_cfg['batch_size']))
        if ep % rl_cfg['target_update_every'] == 0:
            agent.sync_target()
        epsilon = max(rl_cfg['eps_end'], epsilon * rl_cfg['eps_decay'])
    return agent, curve


def apply_params(base_cfg: dict, params: dict) -> dict:
    cfg = copy.deepcopy(base_cfg)

    cfg['model']['hidden_dim'] = params['model.hidden_dim']
    cfg['model']['layers'] = params['model.layers']
    cfg['model']['diffusor_lr'] = params['model.diffusor_lr']
    cfg['model']['diffusor_epochs'] = params['model.diffusor_epochs']
    cfg['model']['batch_size'] = params['model.batch_size']

    cfg['rl']['hidden_dim'] = params['rl.hidden_dim']
    cfg['rl']['lr'] = params['rl.lr']
    cfg['rl']['gamma'] = params['rl.gamma']
    cfg['rl']['buffer_capacity'] = params['rl.buffer_capacity']
    cfg['rl']['batch_size'] = params['rl.batch_size']
    cfg['rl']['train_episodes'] = params['rl.train_episodes']
    cfg['rl']['warmup_episodes'] = params['rl.warmup_episodes']
    cfg['rl']['target_update_every'] = params['rl.target_update_every']
    cfg['rl']['eps_decay'] = params['rl.eps_decay']
    return cfg


def apply_trial_params(base_cfg: dict, trial: optuna.Trial) -> dict:
    train_episodes = trial.suggest_int('rl.train_episodes', 40, 160, step=20)
    rl_batch = trial.suggest_categorical('rl.batch_size', [8, 16, 32])
    warmup_upper = max(10, min(train_episodes - 5, 40))
    params = {
        'model.hidden_dim': trial.suggest_categorical('model.hidden_dim', [32, 64, 96, 128]),
        'model.layers': trial.suggest_int('model.layers', 1, 3),
        'model.diffusor_lr': trial.suggest_float('model.diffusor_lr', 5e-4, 5e-3, log=True),
        'model.diffusor_epochs': trial.suggest_int('model.diffusor_epochs', 5, 25, step=5),
        'model.batch_size': trial.suggest_categorical('model.batch_size', [8, 16, 32]),
        'rl.hidden_dim': trial.suggest_categorical('rl.hidden_dim', [32, 64, 96, 128]),
        'rl.lr': trial.suggest_float('rl.lr', 1e-4, 5e-3, log=True),
        'rl.gamma': trial.suggest_float('rl.gamma', 0.85, 0.99),
        'rl.buffer_capacity': trial.suggest_categorical('rl.buffer_capacity', [1000, 2000, 4000, 8000]),
        'rl.batch_size': rl_batch,
        'rl.train_episodes': train_episodes,
        'rl.warmup_episodes': trial.suggest_int('rl.warmup_episodes', 5, warmup_upper, step=5),
        'rl.target_update_every': trial.suggest_categorical('rl.target_update_every', [5, 10, 20]),
        'rl.eps_decay': trial.suggest_float('rl.eps_decay', 0.95, 0.995),
    }
    return apply_params(base_cfg, params)


def build_static_data(cfg: dict):
    tables = load_airline_tables(cfg['data']['airports_csv'], cfg['data']['routes_csv'])
    ag = build_airline_graph(tables)
    edge_index, node_features, coords, _, distances, node_costs = as_torch_tensors(ag)
    W = generate_synthetic_temporal_weights(
        ag,
        cfg['data']['snapshots'],
        cfg['data']['seasonal_strength'],
        cfg['data']['noise_std'],
        cfg['seed'],
    )
    W_torch = torch.as_tensor(W, dtype=torch.float32)
    policy_static = torch.as_tensor(compute_static_spatial_features(ag.graph, ag.coords, ag.node_costs), dtype=torch.float32)
    return ag, edge_index, node_features, coords, distances, node_costs, W, W_torch, policy_static


def evaluate_configuration(cfg: dict, regime: str, repeats: int, static_data, seed_offset: int = 0):
    _, edge_index, node_features, coords, distances, node_costs, W, W_torch, policy_static = static_data
    scores = []
    details = []

    for repeat_idx in range(repeats):
        train_seed = cfg['seed'] + seed_offset + repeat_idx
        set_seed(train_seed)
        diffusor = GNNDiffusor(
            node_feat_dim=node_features.size(1),
            hidden_dim=cfg['model']['hidden_dim'],
            layers=cfg['model']['layers'],
        )
        diff_art = train_diffusor(
            model=diffusor,
            node_features=node_features,
            edge_index=edge_index,
            edge_prob_seq=W,
            distances_km=distances,
            epochs=cfg['model']['diffusor_epochs'],
            batch_size=cfg['model']['batch_size'],
            lr=cfg['model']['diffusor_lr'],
            seed=train_seed,
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
        agent, curve = train_agent(env, cfg['rl'], W, regime)
        eval_res = rollout_policy(env, agent, epsilon=0.0)
        scores.append(eval_res.objective)
        details.append({
            'objective': eval_res.objective,
            'mass': eval_res.final_activated_mass,
            'coverage': eval_res.final_geographic_coverage,
            'cost': eval_res.total_intervention_cost,
            'selected': eval_res.selected,
            'diffusor_last_loss': diff_art.history['loss'][-1],
            'learning_curve_last10_mean': float(np.mean(curve[-10:])),
        })
    return float(np.mean(scores)), details


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--trials', type=int, default=20)
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--regime', default=None)
    parser.add_argument('--study-name', default='spatial-im-optuna')
    parser.add_argument('--storage', default=None, help='Optional Optuna storage URL, e.g. sqlite:///artifacts/optuna.db')
    parser.add_argument('--out', default='artifacts/optuna_best.json')
    args = parser.parse_args()

    base_cfg = load_yaml(args.config)
    regime = args.regime or base_cfg['evaluation']['regime']
    static_data = build_static_data(base_cfg)

    default_score, default_details = evaluate_configuration(base_cfg, regime, args.repeats, static_data, seed_offset=0)

    sampler = optuna.samplers.TPESampler(seed=base_cfg['seed'])
    study = optuna.create_study(
        study_name=args.study_name,
        direction='maximize',
        sampler=sampler,
        storage=args.storage,
        load_if_exists=bool(args.storage),
    )

    def objective(trial: optuna.Trial) -> float:
        trial_cfg = apply_trial_params(base_cfg, trial)
        score, details = evaluate_configuration(
            trial_cfg,
            regime,
            args.repeats,
            static_data,
            seed_offset=10_000 * (trial.number + 1),
        )
        trial.set_user_attr('details', details)
        trial.set_user_attr('params_full', {
            'model': trial_cfg['model'],
            'rl': trial_cfg['rl'],
        })
        return score

    study.optimize(objective, n_trials=args.trials)

    if args.regime is not None and args.regime not in {'spread', 'dynamic', 'spatial'}:
        raise ValueError(f"Unsupported regime: {args.regime}")

    best_cfg = apply_params(base_cfg, study.best_trial.params)
    best_score, best_details = evaluate_configuration(
        best_cfg,
        regime,
        args.repeats,
        static_data,
        seed_offset=200_000,
    )

    report = {
        'regime': regime,
        'trials': args.trials,
        'repeats': args.repeats,
        'default': {
            'score': default_score,
            'details': default_details,
            'model': base_cfg['model'],
            'rl': base_cfg['rl'],
        },
        'best': {
            'score': best_score,
            'details': best_details,
            'params': study.best_params,
            'model': best_cfg['model'],
            'rl': best_cfg['rl'],
            'improvement': best_score - default_score,
        },
        'study': {
            'best_value': study.best_value,
            'best_trial_number': study.best_trial.number,
            'n_trials': len(study.trials),
        },
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding='utf-8')

    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
