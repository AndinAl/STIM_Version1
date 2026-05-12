from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List
import copy
import numpy as np
import torch

from spatial_im.data.airline import AirlineTables, load_airline_tables
from spatial_im.data.graph_build import AirlineGraph, as_torch_tensors, build_airline_graph
from spatial_im.data.temporal import generate_synthetic_temporal_weights
from spatial_im.diffusion.gnn_diffusor import DiffusorArtifacts, GNNDiffusor, train_diffusor
from spatial_im.env.airline_env import AirlineSpatialTemporalEnv
from spatial_im.evaluation.runner import evaluate_seed_set_raw
from spatial_im.policy.dqn_agent import DQNAgent, ReplayBuffer
from spatial_im.policy.features import compute_static_spatial_features


@dataclass
class PreparedSourceGraph:
    name: str
    graph: AirlineGraph
    edge_index: torch.Tensor
    node_features: torch.Tensor
    coords: torch.Tensor
    distances: torch.Tensor
    node_costs: torch.Tensor
    temporal_weights: np.ndarray
    temporal_weights_torch: torch.Tensor
    static_policy_features: torch.Tensor


@dataclass
class PretrainArtifacts:
    diffusor: GNNDiffusor
    policy: DQNAgent
    history: Dict[str, List[Any]]
    source_graphs: List[str]
    feature_dim: int
    node_feat_dim: int

    def checkpoint(self) -> dict:
        return {
            'diffusor_state_dict': self.diffusor.state_dict(),
            'policy_state_dict': self.policy.state_dict(),
            'history': self.history,
            'source_graphs': self.source_graphs,
            'feature_dim': self.feature_dim,
            'node_feat_dim': self.node_feat_dim,
        }


@dataclass
class ZeroShotReuseResult:
    selected: List[int]
    selected_iata: List[str]
    learned_env_objective: float
    raw_objective: float
    raw_final_activated_mass: float
    raw_expected_coverage: float
    raw_intervention_cost: float


@dataclass
class WarmStartAdaptationArtifacts:
    diffusor: GNNDiffusor
    policy: DQNAgent
    result: ZeroShotReuseResult
    history: Dict[str, List[Any]]
    source_graphs: List[str]
    feature_dim: int
    node_feat_dim: int
    source_checkpoint: str

    def checkpoint(self, cfg: dict) -> dict:
        return {
            'diffusor_state_dict': self.diffusor.state_dict(),
            'policy_state_dict': self.policy.state_dict(),
            'history': self.history,
            'source_graphs': self.source_graphs,
            'feature_dim': self.feature_dim,
            'node_feat_dim': self.node_feat_dim,
            'source_checkpoint': self.source_checkpoint,
            'config': cfg,
        }


def _merged_source_spec(cfg: dict, source_spec: dict | None, idx: int) -> dict:
    source_spec = source_spec or {}
    data_cfg = copy.deepcopy(cfg['data'])
    for key in ['airports_csv', 'routes_csv', 'snapshots', 'seasonal_strength', 'noise_std', 'self_loop']:
        if key in source_spec:
            data_cfg[key] = source_spec[key]
    spec_seed = int(source_spec.get('seed', cfg['seed'] + int(source_spec.get('seed_offset', idx))))
    return {
        'name': source_spec.get('name', f'source_graph_{idx}'),
        'seed': spec_seed,
        **data_cfg,
    }


def _prepare_graph_from_tables(
    name: str,
    tables: AirlineTables,
    snapshots: int,
    seasonal_strength: float,
    noise_std: float,
    seed: int,
) -> PreparedSourceGraph:
    ag = build_airline_graph(tables)
    edge_index, node_features, coords, _, distances, node_costs = as_torch_tensors(ag)
    W = generate_synthetic_temporal_weights(
        ag,
        snapshots=int(snapshots),
        seasonal_strength=float(seasonal_strength),
        noise_std=float(noise_std),
        seed=int(seed),
    )
    static_policy = torch.as_tensor(
        compute_static_spatial_features(ag.graph, ag.coords, ag.node_costs),
        dtype=torch.float32,
    )
    return PreparedSourceGraph(
        name=str(name),
        graph=ag,
        edge_index=edge_index,
        node_features=node_features,
        coords=coords,
        distances=distances,
        node_costs=node_costs,
        temporal_weights=W,
        temporal_weights_torch=torch.as_tensor(W, dtype=torch.float32),
        static_policy_features=static_policy,
    )


def _prepare_graph_from_spec(spec: dict) -> PreparedSourceGraph:
    tables = load_airline_tables(spec['airports_csv'], spec['routes_csv'])
    return _prepare_graph_from_tables(
        name=str(spec['name']),
        tables=tables,
        snapshots=int(spec['snapshots']),
        seasonal_strength=float(spec['seasonal_strength']),
        noise_std=float(spec['noise_std']),
        seed=int(spec['seed']),
    )


def _clone_prepared_graph_with_weights(
    base_graph: PreparedSourceGraph,
    name: str,
    temporal_weights: np.ndarray,
) -> PreparedSourceGraph:
    W = np.asarray(temporal_weights, dtype=np.float32).copy()
    return PreparedSourceGraph(
        name=name,
        graph=base_graph.graph,
        edge_index=base_graph.edge_index,
        node_features=base_graph.node_features,
        coords=base_graph.coords,
        distances=base_graph.distances,
        node_costs=base_graph.node_costs,
        temporal_weights=W,
        temporal_weights_torch=torch.as_tensor(W, dtype=torch.float32),
        static_policy_features=base_graph.static_policy_features,
    )


def _copy_cfg_with_fixed_regime_budget(cfg: dict, regime: str, budget: int) -> dict:
    local_cfg = copy.deepcopy(cfg)
    local_cfg['evaluation']['regime'] = regime
    local_cfg['pretraining']['regimes'] = [regime]
    local_cfg['pretraining']['budgets'] = [int(budget)]
    return local_cfg


def prepare_source_graph_family(cfg: dict) -> List[PreparedSourceGraph]:
    pre_cfg = cfg.get('pretraining', {})
    source_specs = pre_cfg.get('source_graphs') or [{}]
    prepared: List[PreparedSourceGraph] = []
    expected_node_feat_dim: int | None = None
    for idx, raw_spec in enumerate(source_specs):
        spec = _merged_source_spec(cfg, raw_spec, idx)
        prepared_graph = _prepare_graph_from_spec(spec)
        node_features = prepared_graph.node_features
        if expected_node_feat_dim is None:
            expected_node_feat_dim = int(node_features.size(1))
        elif int(node_features.size(1)) != expected_node_feat_dim:
            raise ValueError(
                f"Source graph '{spec['name']}' has node feature dim {int(node_features.size(1))}, "
                f"expected {expected_node_feat_dim} for shared-policy pretraining."
            )
        prepared.append(prepared_graph)
    return prepared


def prepare_target_graph(cfg: dict) -> PreparedSourceGraph:
    target_spec = cfg.get('zero_shot_target', {})
    spec = _merged_source_spec(cfg, target_spec, idx=10_000)
    spec['name'] = target_spec.get('name', 'target_graph')
    return _prepare_graph_from_spec(spec)


def _tables_for_target_graph(cfg: dict) -> AirlineTables:
    target_spec = cfg.get('zero_shot_target', {})
    airports_csv = target_spec.get('airports_csv', cfg['data']['airports_csv'])
    routes_csv = target_spec.get('routes_csv', cfg['data']['routes_csv'])
    return load_airline_tables(airports_csv, routes_csv)


def _build_snapshot_source_family(
    target_graph: PreparedSourceGraph,
    train_snapshot_count: int,
) -> List[PreparedSourceGraph]:
    T = int(target_graph.temporal_weights.shape[0])
    train_snapshot_count = max(1, min(int(train_snapshot_count), T))
    family: List[PreparedSourceGraph] = []
    for start in range(train_snapshot_count):
        window = target_graph.temporal_weights[start:train_snapshot_count]
        if window.shape[0] == 0:
            continue
        family.append(
            _clone_prepared_graph_with_weights(
                base_graph=target_graph,
                name=f'{target_graph.name}_train_window_{start}_{train_snapshot_count}',
                temporal_weights=window,
            )
        )
    return family


def _sample_subgraph_node_ids(
    base_graph: PreparedSourceGraph,
    sample_size: int,
    method: str,
    seed: int,
    ego_hops: int = 2,
) -> List[int]:
    rng = np.random.default_rng(seed)
    n = int(base_graph.node_features.size(0))
    sample_size = max(2, min(int(sample_size), n))
    all_nodes = list(base_graph.graph.graph.nodes())
    if sample_size >= n:
        return all_nodes

    und = base_graph.graph.graph.to_undirected()
    method = str(method).lower()
    if method == 'random_nodes':
        chosen = rng.choice(np.asarray(all_nodes, dtype=np.int64), size=sample_size, replace=False)
        return chosen.astype(np.int64).tolist()

    start = int(rng.choice(np.asarray(all_nodes, dtype=np.int64)))
    if method == 'ego':
        selected = {start}
        frontier = {start}
        for _ in range(max(1, int(ego_hops))):
            nxt = set()
            for node in frontier:
                nxt.update(list(und.neighbors(node)))
            selected.update(nxt)
            frontier = nxt
            if len(selected) >= sample_size:
                break
        selected = list(selected)
        if len(selected) > sample_size:
            chosen = rng.choice(np.asarray(selected, dtype=np.int64), size=sample_size, replace=False)
            return chosen.astype(np.int64).tolist()
        if len(selected) < sample_size:
            remaining = [node for node in all_nodes if node not in selected]
            extra = rng.choice(np.asarray(remaining, dtype=np.int64), size=sample_size - len(selected), replace=False)
            selected.extend(extra.astype(np.int64).tolist())
        return selected

    # bfs is the default structural sampler.
    queue = [start]
    selected: List[int] = []
    seen = {start}
    while queue and len(selected) < sample_size:
        node = queue.pop(0)
        selected.append(node)
        neigh = list(und.neighbors(node))
        rng.shuffle(neigh)
        for nxt in neigh:
            if nxt not in seen:
                seen.add(nxt)
                queue.append(int(nxt))
    if len(selected) < sample_size:
        remaining = [node for node in all_nodes if node not in selected]
        extra = rng.choice(np.asarray(remaining, dtype=np.int64), size=sample_size - len(selected), replace=False)
        selected.extend(extra.astype(np.int64).tolist())
    return selected[:sample_size]


def _filter_tables_to_airport_ids(tables: AirlineTables, airport_ids: List[int]) -> AirlineTables:
    airport_ids = list({int(aid) for aid in airport_ids})
    airports = tables.airports[tables.airports['airport_id'].isin(airport_ids)].copy().reset_index(drop=True)
    routes = tables.routes[
        tables.routes['source_airport_id'].isin(airport_ids) &
        tables.routes['target_airport_id'].isin(airport_ids)
    ].copy().reset_index(drop=True)
    return AirlineTables(airports=airports, routes=routes)


def _build_sampled_subgraph_family(cfg: dict, num_subgraphs: int, sample_size: int, sample_method: str, ego_hops: int = 2) -> tuple[List[PreparedSourceGraph], PreparedSourceGraph]:
    target_graph = prepare_target_graph(cfg)
    base_tables = _tables_for_target_graph(cfg)
    source_graphs: List[PreparedSourceGraph] = []
    for j in range(max(1, int(num_subgraphs))):
        sampled_node_idx = _sample_subgraph_node_ids(
            base_graph=target_graph,
            sample_size=sample_size,
            method=sample_method,
            seed=cfg['seed'] + j,
            ego_hops=ego_hops,
        )
        airport_ids = [target_graph.graph.idx_to_id[idx] for idx in sampled_node_idx]
        sub_tables = _filter_tables_to_airport_ids(base_tables, airport_ids)
        if len(sub_tables.airports) < 2 or len(sub_tables.routes) == 0:
            continue
        source_graphs.append(
            _prepare_graph_from_tables(
                name=f'{sample_method}_subgraph_{j}',
                tables=sub_tables,
                snapshots=int(cfg['data']['snapshots']),
                seasonal_strength=float(cfg['data']['seasonal_strength']),
                noise_std=float(cfg['data']['noise_std']),
                seed=int(cfg['seed'] + j),
            )
        )
    if not source_graphs:
        raise ValueError('Subgraph sampling produced no valid source graphs.')
    return source_graphs, target_graph


def make_env_for_source(
    source_graph: PreparedSourceGraph,
    diffusor: GNNDiffusor,
    cfg: dict,
    regime: str,
    budget: int,
) -> AirlineSpatialTemporalEnv:
    return AirlineSpatialTemporalEnv(
        diffusor=diffusor,
        node_features=source_graph.node_features,
        edge_index=source_graph.edge_index,
        edge_weight_seq=source_graph.temporal_weights_torch,
        distances_km=source_graph.distances,
        coords=source_graph.coords,
        node_costs=source_graph.node_costs,
        static_policy_features=source_graph.static_policy_features,
        budget=budget,
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


def _sample_start_snapshot(rng: np.random.Generator, source_graph: PreparedSourceGraph) -> int:
    T = int(source_graph.temporal_weights.shape[0])
    if T <= 1:
        return 0
    return int(rng.integers(0, T))


def _diffusor_update(
    model: GNNDiffusor,
    source_graph: PreparedSourceGraph,
    cfg: dict,
    regime: str,
    budget: int,
    epochs: int,
    batch_size: int,
    seed: int,
) -> DiffusorArtifacts:
    return train_diffusor(
        model=model,
        node_features=source_graph.node_features,
        edge_index=source_graph.edge_index,
        edge_prob_seq=source_graph.temporal_weights,
        distances_km=source_graph.distances,
        epochs=epochs,
        batch_size=batch_size,
        lr=cfg['model']['diffusor_lr'],
        seed=seed,
        coords=source_graph.coords,
        node_costs=source_graph.node_costs,
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
        selection_budget=budget,
    )


def pretrain_reusable_policy_from_graphs(cfg: dict, prepared_graphs: List[PreparedSourceGraph]) -> PretrainArtifacts:
    if not prepared_graphs:
        raise ValueError('Need at least one source graph for reusable pretraining.')
    pre_cfg = cfg.get('pretraining', {})
    regimes = [str(r) for r in pre_cfg.get('regimes', ['spread', 'dynamic', 'spatial'])]
    budgets = [int(b) for b in pre_cfg.get('budgets', [cfg['rl']['budget']])]
    iterations = int(pre_cfg.get('iterations', 60))
    episodes_per_iteration = int(pre_cfg.get('episodes_per_iteration', 1))
    diffusor_update_epochs = int(pre_cfg.get('diffusor_update_epochs', 1))
    diffusor_update_batch = int(pre_cfg.get('diffusor_update_batch_size', cfg['model']['batch_size']))

    first_graph = prepared_graphs[0]
    first_regime = regimes[0]
    first_budget = max(1, min(int(budgets[0]), int(first_graph.node_features.size(0))))
    diffusor = GNNDiffusor(
        node_feat_dim=first_graph.node_features.size(1),
        hidden_dim=cfg['model']['hidden_dim'],
        layers=cfg['model']['layers'],
    )
    warm_env = make_env_for_source(first_graph, diffusor, cfg, first_regime, first_budget)
    sample_feat, _ = warm_env.reset(start_snapshot=0)
    agent = DQNAgent(
        feature_dim=sample_feat.shape[1],
        lr=cfg['rl']['lr'],
        gamma=cfg['rl']['gamma'],
        hidden_dim=cfg['rl'].get('hidden_dim', 64),
        beta_coverage=cfg['reward']['beta_coverage'],
        lambda_cost=cfg['reward']['lambda_cost'],
        regime=first_regime,
    )
    replay = ReplayBuffer(cfg['rl']['buffer_capacity'])
    epsilon = float(cfg['rl']['eps_start'])
    rng = np.random.default_rng(cfg['seed'])
    episode_count = 0
    update_count = 0
    history: Dict[str, List[Any]] = {
        'sampled_graph': [],
        'sampled_regime': [],
        'sampled_budget': [],
        'sampled_start_snapshot': [],
        'diffusor_loss': [],
        'episode_return': [],
        'epsilon': [],
    }

    for iteration in range(iterations):
        graph_idx = int(rng.integers(0, len(prepared_graphs)))
        source_graph = prepared_graphs[graph_idx]
        regime = regimes[int(rng.integers(0, len(regimes)))]
        sampled_budget = budgets[int(rng.integers(0, len(budgets)))]
        budget = max(1, min(int(sampled_budget), int(source_graph.node_features.size(0))))
        diff_artifacts = _diffusor_update(
            model=diffusor,
            source_graph=source_graph,
            cfg=cfg,
            regime=regime,
            budget=budget,
            epochs=diffusor_update_epochs,
            batch_size=diffusor_update_batch,
            seed=cfg['seed'] + iteration,
        )
        env = make_env_for_source(source_graph, diffusor, cfg, regime, budget)
        for _ in range(episodes_per_iteration):
            start_snapshot = _sample_start_snapshot(rng, source_graph)
            state_feat, legal = env.reset(start_snapshot=start_snapshot)
            action_prior = env.action_prior_scores()
            ep_return = 0.0
            while True:
                action = agent.act(state_feat, legal, epsilon=epsilon, action_prior=action_prior)
                out = env.step(action)
                next_state_feat = env.get_candidate_features() if not out.done else state_feat.copy()
                next_legal = env.scoring_mask() if not out.done else np.zeros_like(legal)
                next_action_prior = env.action_prior_scores() if not out.done else None
                replay.push(state_feat, action, out.reward, next_state_feat, out.done, legal, next_legal, action_prior, next_action_prior)
                state_feat = next_state_feat
                legal = next_legal
                action_prior = next_action_prior
                ep_return += out.reward
                if len(replay) >= cfg['rl']['batch_size'] and episode_count >= cfg['rl']['warmup_episodes']:
                    agent.update(replay.sample(cfg['rl']['batch_size']))
                    update_count += 1
                    if update_count % max(1, int(cfg['rl']['target_update_every'])) == 0:
                        agent.sync_target()
                if out.done:
                    break
            history['sampled_graph'].append(source_graph.name)
            history['sampled_regime'].append(regime)
            history['sampled_budget'].append(budget)
            history['sampled_start_snapshot'].append(start_snapshot)
            history['diffusor_loss'].append(diff_artifacts.history['loss'][-1])
            history['episode_return'].append(float(ep_return))
            history['epsilon'].append(float(epsilon))
            episode_count += 1
            epsilon = max(float(cfg['rl']['eps_end']), epsilon * float(cfg['rl']['eps_decay']))

    agent.sync_target()
    return PretrainArtifacts(
        diffusor=diffusor,
        policy=agent,
        history=history,
        source_graphs=[g.name for g in prepared_graphs],
        feature_dim=int(sample_feat.shape[1]),
        node_feat_dim=int(first_graph.node_features.size(1)),
    )


def pretrain_reusable_policy(cfg: dict) -> PretrainArtifacts:
    prepared_graphs = prepare_source_graph_family(cfg)
    return pretrain_reusable_policy_from_graphs(cfg, prepared_graphs)


def _raw_window_for_regime(target_graph: PreparedSourceGraph, regime: str, start_snapshot: int) -> np.ndarray:
    if regime == 'dynamic':
        return target_graph.temporal_weights[int(start_snapshot):]
    return target_graph.temporal_weights[int(start_snapshot):int(start_snapshot) + 1]


def _evaluate_selected_on_target(
    target_graph: PreparedSourceGraph,
    cfg: dict,
    regime: str,
    start_snapshot: int,
    selected: List[int],
) -> ZeroShotReuseResult:
    raw_eval = evaluate_seed_set_raw(
        edge_index=target_graph.graph.edge_index,
        edge_prob_seq=_raw_window_for_regime(target_graph, regime, start_snapshot),
        num_nodes=len(target_graph.graph.node_ids),
        selected=selected,
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
    selected_iata = [target_graph.graph.graph.nodes[idx].get('iata', str(idx)) for idx in selected]
    return ZeroShotReuseResult(
        selected=selected,
        selected_iata=selected_iata,
        learned_env_objective=float('nan'),
        raw_objective=float(raw_eval.objective),
        raw_final_activated_mass=float(raw_eval.final_activated_mass),
        raw_expected_coverage=float(raw_eval.final_geographic_coverage),
        raw_intervention_cost=float(raw_eval.total_intervention_cost),
    )


def _run_greedy_selection(
    env: AirlineSpatialTemporalEnv,
    agent: DQNAgent,
    start_snapshot: int,
) -> tuple[List[int], float]:
    state_feat, legal = env.reset(start_snapshot=int(start_snapshot))
    selected: List[int] = []
    learned_env_objective = float(env.prev_objective)
    for _ in range(env.budget):
        action_prior = env.action_prior_scores()
        action = agent.greedy_action(state_feat, legal, action_prior=action_prior)
        out = env.step(action)
        selected = list(out.info['selected'])
        learned_env_objective = float(out.info['objective'])
        if out.done:
            break
        state_feat = env.get_candidate_features()
        legal = env.scoring_mask()
    return selected, learned_env_objective


def reuse_policy_zero_shot_on_prepared_graph(
    cfg: dict,
    target_graph: PreparedSourceGraph,
    pretrained: PretrainArtifacts,
    regime: str,
    budget: int,
    start_snapshot: int = 0,
) -> ZeroShotReuseResult:
    budget = max(1, min(int(budget), int(target_graph.node_features.size(0))))
    max_start = max(0, int(target_graph.temporal_weights.shape[0]) - 1)
    start_snapshot = max(0, min(int(start_snapshot), max_start))
    env = make_env_for_source(target_graph, pretrained.diffusor, cfg, regime, budget)
    selected, learned_env_objective = _run_greedy_selection(env, pretrained.policy, start_snapshot)
    result = _evaluate_selected_on_target(target_graph, cfg, regime, start_snapshot, selected)
    result.learned_env_objective = learned_env_objective
    return result


def _fresh_target_components(
    cfg: dict,
    target_graph: PreparedSourceGraph,
    regime: str,
    budget: int,
) -> tuple[GNNDiffusor, DQNAgent]:
    diffusor = GNNDiffusor(
        node_feat_dim=target_graph.node_features.size(1),
        hidden_dim=cfg['model']['hidden_dim'],
        layers=cfg['model']['layers'],
    )
    warm_env = make_env_for_source(target_graph, diffusor, cfg, regime, budget)
    sample_feat, _ = warm_env.reset(start_snapshot=0)
    agent = DQNAgent(
        feature_dim=sample_feat.shape[1],
        lr=cfg['rl']['lr'],
        gamma=cfg['rl']['gamma'],
        hidden_dim=cfg['rl'].get('hidden_dim', 64),
        beta_coverage=cfg['reward']['beta_coverage'],
        lambda_cost=cfg['reward']['lambda_cost'],
        regime=regime,
    )
    return diffusor, agent


def _pretrained_target_components(
    cfg: dict,
    target_graph: PreparedSourceGraph,
    pretrained: PretrainArtifacts,
    regime: str,
    budget: int,
) -> tuple[GNNDiffusor, DQNAgent]:
    diffusor = GNNDiffusor(
        node_feat_dim=target_graph.node_features.size(1),
        hidden_dim=cfg['model']['hidden_dim'],
        layers=cfg['model']['layers'],
    )
    diffusor.load_state_dict(pretrained.diffusor.state_dict())
    warm_env = make_env_for_source(target_graph, diffusor, cfg, regime, budget)
    sample_feat, _ = warm_env.reset(start_snapshot=0)
    agent = DQNAgent(
        feature_dim=sample_feat.shape[1],
        lr=cfg['rl']['lr'],
        gamma=cfg['rl']['gamma'],
        hidden_dim=cfg['rl'].get('hidden_dim', 64),
        beta_coverage=cfg['reward']['beta_coverage'],
        lambda_cost=cfg['reward']['lambda_cost'],
        regime=regime,
    )
    agent.load_state_dict(pretrained.policy.state_dict())
    return diffusor, agent


def load_pretrained_components(
    checkpoint_path: str | Path,
    target_graph: PreparedSourceGraph,
    cfg: dict,
    regime: str,
    budget: int,
) -> tuple[GNNDiffusor, DQNAgent, AirlineSpatialTemporalEnv, dict]:
    checkpoint = torch.load(Path(checkpoint_path), map_location='cpu')
    saved_cfg = checkpoint.get('config', cfg)
    expected_node_feat_dim = int(checkpoint['node_feat_dim'])
    if int(target_graph.node_features.size(1)) != expected_node_feat_dim:
        raise ValueError(
            f"Target graph node feature dim {int(target_graph.node_features.size(1))} does not match "
            f"pretrained checkpoint node feature dim {expected_node_feat_dim}."
        )

    diffusor = GNNDiffusor(
        node_feat_dim=target_graph.node_features.size(1),
        hidden_dim=saved_cfg['model']['hidden_dim'],
        layers=saved_cfg['model']['layers'],
    )
    diffusor.load_state_dict(checkpoint['diffusor_state_dict'])
    diffusor.eval()

    env = make_env_for_source(target_graph, diffusor, cfg, regime, budget)
    sample_feat, _ = env.reset(start_snapshot=0)
    expected_feature_dim = int(checkpoint['feature_dim'])
    if int(sample_feat.shape[1]) != expected_feature_dim:
        raise ValueError(
            f"Target graph policy feature dim {int(sample_feat.shape[1])} does not match "
            f"pretrained checkpoint feature dim {expected_feature_dim}."
        )

    policy_state = checkpoint['policy_state_dict']
    agent = DQNAgent(
        feature_dim=sample_feat.shape[1],
        lr=saved_cfg['rl']['lr'],
        gamma=float(policy_state.get('gamma', saved_cfg['rl']['gamma'])),
        hidden_dim=saved_cfg['rl'].get('hidden_dim', 64),
        beta_coverage=float(policy_state.get('beta_coverage', saved_cfg['reward']['beta_coverage'])),
        lambda_cost=float(policy_state.get('lambda_cost', saved_cfg['reward']['lambda_cost'])),
        regime=regime,
    )
    agent.load_state_dict(policy_state)
    return diffusor, agent, env, checkpoint


def reuse_policy_zero_shot(
    cfg: dict,
    checkpoint_path: str | Path,
    regime: str,
    budget: int,
    start_snapshot: int = 0,
) -> ZeroShotReuseResult:
    target_graph = prepare_target_graph(cfg)
    budget = max(1, min(int(budget), int(target_graph.node_features.size(0))))
    max_start = max(0, int(target_graph.temporal_weights.shape[0]) - 1)
    start_snapshot = max(0, min(int(start_snapshot), max_start))
    _, agent, env, _ = load_pretrained_components(
        checkpoint_path=checkpoint_path,
        target_graph=target_graph,
        cfg=cfg,
        regime=regime,
        budget=budget,
    )
    selected, learned_env_objective = _run_greedy_selection(env, agent, start_snapshot)
    result = _evaluate_selected_on_target(target_graph, cfg, regime, start_snapshot, selected)
    result.learned_env_objective = learned_env_objective
    return result


def reuse_across_future_snapshots(
    cfg: dict,
    regime: str,
    budget: int,
    train_snapshot_count: int,
    test_start_snapshot: int,
) -> tuple[PretrainArtifacts, ZeroShotReuseResult]:
    target_graph = prepare_target_graph(cfg)
    source_graphs = _build_snapshot_source_family(target_graph, train_snapshot_count=train_snapshot_count)
    local_cfg = _copy_cfg_with_fixed_regime_budget(cfg, regime=regime, budget=budget)
    pretrained = pretrain_reusable_policy_from_graphs(local_cfg, source_graphs)
    result = reuse_policy_zero_shot_on_prepared_graph(
        cfg=cfg,
        target_graph=target_graph,
        pretrained=pretrained,
        regime=regime,
        budget=budget,
        start_snapshot=test_start_snapshot,
    )
    return pretrained, result


def reuse_from_sampled_subgraphs(
    cfg: dict,
    regime: str,
    budget: int,
    num_subgraphs: int,
    sample_size: int,
    sample_method: str = 'bfs',
    start_snapshot: int = 0,
    ego_hops: int = 2,
) -> tuple[PretrainArtifacts, ZeroShotReuseResult]:
    source_graphs, target_graph = _build_sampled_subgraph_family(
        cfg=cfg,
        num_subgraphs=num_subgraphs,
        sample_size=sample_size,
        sample_method=sample_method,
        ego_hops=ego_hops,
    )
    local_cfg = _copy_cfg_with_fixed_regime_budget(cfg, regime=regime, budget=budget)
    pretrained = pretrain_reusable_policy_from_graphs(local_cfg, source_graphs)
    result = reuse_policy_zero_shot_on_prepared_graph(
        cfg=cfg,
        target_graph=target_graph,
        pretrained=pretrained,
        regime=regime,
        budget=budget,
        start_snapshot=start_snapshot,
    )
    return pretrained, result


def _adapt_target_components(
    cfg: dict,
    target_graph: PreparedSourceGraph,
    regime: str,
    budget: int,
    start_snapshot: int,
    n_adapt: int | None,
    diffusor: GNNDiffusor,
    agent: DQNAgent,
    source_graphs: List[str],
    source_checkpoint: str,
) -> WarmStartAdaptationArtifacts:
    adapt_cfg = cfg.get('adaptation', {})
    adapt_iters = int(n_adapt if n_adapt is not None else adapt_cfg.get('iterations', 5))
    diffusor_refresh = bool(adapt_cfg.get('refresh_diffusor', True))
    diffusor_update_epochs = int(adapt_cfg.get('diffusor_update_epochs', 1))
    diffusor_update_batch_size = int(adapt_cfg.get('diffusor_update_batch_size', cfg['model']['batch_size']))
    policy_refine = bool(adapt_cfg.get('refine_policy', True))
    policy_episodes_per_iter = int(adapt_cfg.get('policy_episodes_per_iteration', 1))
    replay_capacity = int(adapt_cfg.get('buffer_capacity', cfg['rl']['buffer_capacity']))
    warmup_episodes = int(adapt_cfg.get('warmup_episodes', 0))
    target_update_every = int(adapt_cfg.get('target_update_every', cfg['rl']['target_update_every']))
    epsilon = float(adapt_cfg.get('eps_start', cfg['rl']['eps_start']))
    eps_end = float(adapt_cfg.get('eps_end', cfg['rl']['eps_end']))
    eps_decay = float(adapt_cfg.get('eps_decay', cfg['rl']['eps_decay']))
    fixed_start = bool(adapt_cfg.get('fixed_start_snapshot', True))

    replay = ReplayBuffer(replay_capacity)
    episode_count = 0
    update_count = 0
    history: Dict[str, List[Any]] = {
        'adapt_iteration': [],
        'diffusor_loss': [],
        'episode_return': [],
        'epsilon': [],
        'start_snapshot': [],
    }

    for adapt_iter in range(adapt_iters):
        if diffusor_refresh:
            diff_artifacts = _diffusor_update(
                model=diffusor,
                source_graph=target_graph,
                cfg=cfg,
                regime=regime,
                budget=budget,
                epochs=diffusor_update_epochs,
                batch_size=diffusor_update_batch_size,
                seed=cfg['seed'] + 50_000 + adapt_iter,
            )
            history['diffusor_loss'].append(float(diff_artifacts.history['loss'][-1]))
        else:
            history['diffusor_loss'].append(float('nan'))

        env = make_env_for_source(target_graph, diffusor, cfg, regime, budget)
        if policy_refine:
            for local_ep in range(policy_episodes_per_iter):
                episode_start = start_snapshot if fixed_start else _sample_start_snapshot(
                    np.random.default_rng(cfg['seed'] + 70_000 + adapt_iter * 100 + local_ep),
                    target_graph,
                )
                state_feat, legal = env.reset(start_snapshot=episode_start)
                action_prior = env.action_prior_scores()
                ep_return = 0.0
                while True:
                    action = agent.act(state_feat, legal, epsilon=epsilon, action_prior=action_prior)
                    out = env.step(action)
                    next_state_feat = env.get_candidate_features() if not out.done else state_feat.copy()
                    next_legal = env.scoring_mask() if not out.done else np.zeros_like(legal)
                    next_action_prior = env.action_prior_scores() if not out.done else None
                    replay.push(state_feat, action, out.reward, next_state_feat, out.done, legal, next_legal, action_prior, next_action_prior)
                    state_feat = next_state_feat
                    legal = next_legal
                    action_prior = next_action_prior
                    ep_return += out.reward
                    if len(replay) >= cfg['rl']['batch_size'] and episode_count >= warmup_episodes:
                        agent.update(replay.sample(cfg['rl']['batch_size']))
                        update_count += 1
                        if update_count % max(1, target_update_every) == 0:
                            agent.sync_target()
                    if out.done:
                        break
                history['adapt_iteration'].append(adapt_iter)
                history['episode_return'].append(float(ep_return))
                history['epsilon'].append(float(epsilon))
                history['start_snapshot'].append(int(episode_start))
                episode_count += 1
                epsilon = max(eps_end, epsilon * eps_decay)
        else:
            history['adapt_iteration'].append(adapt_iter)
            history['episode_return'].append(float('nan'))
            history['epsilon'].append(float(epsilon))
            history['start_snapshot'].append(int(start_snapshot))

    agent.sync_target()
    final_env = make_env_for_source(target_graph, diffusor, cfg, regime, budget)
    selected, learned_env_objective = _run_greedy_selection(final_env, agent, start_snapshot)
    result = _evaluate_selected_on_target(target_graph, cfg, regime, start_snapshot, selected)
    result.learned_env_objective = learned_env_objective

    return WarmStartAdaptationArtifacts(
        diffusor=diffusor,
        policy=agent,
        result=result,
        history=history,
        source_graphs=list(source_graphs),
        feature_dim=int(agent.q.trunk[0].in_features),
        node_feat_dim=int(target_graph.node_features.size(1)),
        source_checkpoint=source_checkpoint,
    )


def adapt_and_reuse_policy(
    cfg: dict,
    checkpoint_path: str | Path,
    regime: str,
    budget: int,
    start_snapshot: int = 0,
    n_adapt: int | None = None,
) -> WarmStartAdaptationArtifacts:
    target_graph = prepare_target_graph(cfg)
    budget = max(1, min(int(budget), int(target_graph.node_features.size(0))))
    max_start = max(0, int(target_graph.temporal_weights.shape[0]) - 1)
    start_snapshot = max(0, min(int(start_snapshot), max_start))
    diffusor, agent, _, checkpoint = load_pretrained_components(
        checkpoint_path=checkpoint_path,
        target_graph=target_graph,
        cfg=cfg,
        regime=regime,
        budget=budget,
    )
    return _adapt_target_components(
        cfg=cfg,
        target_graph=target_graph,
        regime=regime,
        budget=budget,
        start_snapshot=start_snapshot,
        n_adapt=n_adapt,
        diffusor=diffusor,
        agent=agent,
        source_graphs=list(checkpoint.get('source_graphs', [])),
        source_checkpoint=str(checkpoint_path),
    )


def adapt_from_pretrained_artifacts(
    cfg: dict,
    pretrained: PretrainArtifacts,
    target_graph: PreparedSourceGraph,
    regime: str,
    budget: int,
    start_snapshot: int = 0,
    n_adapt: int | None = None,
) -> WarmStartAdaptationArtifacts:
    budget = max(1, min(int(budget), int(target_graph.node_features.size(0))))
    max_start = max(0, int(target_graph.temporal_weights.shape[0]) - 1)
    start_snapshot = max(0, min(int(start_snapshot), max_start))
    diffusor, agent = _pretrained_target_components(cfg, target_graph, pretrained, regime, budget)
    return _adapt_target_components(
        cfg=cfg,
        target_graph=target_graph,
        regime=regime,
        budget=budget,
        start_snapshot=start_snapshot,
        n_adapt=n_adapt,
        diffusor=diffusor,
        agent=agent,
        source_graphs=list(pretrained.source_graphs),
        source_checkpoint='in_memory_pretraining',
    )


def adapt_from_scratch_policy(
    cfg: dict,
    regime: str,
    budget: int,
    start_snapshot: int = 0,
    n_adapt: int | None = None,
) -> WarmStartAdaptationArtifacts:
    target_graph = prepare_target_graph(cfg)
    budget = max(1, min(int(budget), int(target_graph.node_features.size(0))))
    max_start = max(0, int(target_graph.temporal_weights.shape[0]) - 1)
    start_snapshot = max(0, min(int(start_snapshot), max_start))
    diffusor, agent = _fresh_target_components(cfg, target_graph, regime, budget)
    return _adapt_target_components(
        cfg=cfg,
        target_graph=target_graph,
        regime=regime,
        budget=budget,
        start_snapshot=start_snapshot,
        n_adapt=n_adapt,
        diffusor=diffusor,
        agent=agent,
        source_graphs=[],
        source_checkpoint='scratch',
    )
