from __future__ import annotations

import argparse
import copy
import csv
import json
from collections import Counter, defaultdict
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
from spatial_im.baselines.learned import greedy_surrogate_marginal
from spatial_im.baselines.spatial import (
    build_temporal_weighted_graph,
    community_bridge_nodes,
    cost_aware_ranking,
    distance_strength,
    weighted_betweenness,
    weighted_strength,
)
from spatial_im.data.airline import load_airline_tables
from spatial_im.data.edge_sampling import topk_temporal_edge_neighborhood
from spatial_im.data.graph_build import as_torch_tensors, build_airline_graph
from spatial_im.data.synthetic_airline import generate_homogeneous_airline_tables
from spatial_im.data.temporal import generate_synthetic_temporal_weights
from spatial_im.diffusion.gnn_diffusor import GNNDiffusor, train_diffusor
from spatial_im.env.airline_env import AirlineSpatialTemporalEnv
from spatial_im.evaluation.runner import evaluate_seed_set, evaluate_seed_set_raw, rollout_policy
from spatial_im.policy.dqn_agent import DQNAgent, ReplayBuffer
from spatial_im.policy.features import compute_dynamic_temporal_summaries, compute_static_spatial_features
from spatial_im.utils.io import load_yaml
from spatial_im.utils.seeds import set_seed


def parse_int_list(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def apply_artifact_cfg(base_cfg: dict, artifact_path: str | None) -> dict:
    cfg = copy.deepcopy(base_cfg)
    if not artifact_path:
        return cfg
    path = Path(artifact_path)
    if not path.exists():
        return cfg
    artifact = json.loads(path.read_text(encoding="utf-8"))
    best = artifact.get("best", {})
    if "model" in best:
        cfg["model"].update(best["model"])
    if "rl" in best:
        cfg["rl"].update(best["rl"])
    if "performance" in best:
        cfg.setdefault("performance", {})
        cfg["performance"].update(best["performance"])
    return cfg


def make_env(
    diffusor,
    node_features,
    edge_index,
    W_torch,
    distances,
    coords,
    node_costs,
    policy_static,
    cfg,
    regime: str,
    *,
    prefilter_top_m: int | None,
):
    return AirlineSpatialTemporalEnv(
        diffusor=diffusor,
        node_features=node_features,
        edge_index=edge_index,
        edge_weight_seq=W_torch,
        distances_km=distances,
        coords=coords,
        node_costs=node_costs,
        static_policy_features=policy_static,
        budget=cfg["rl"]["budget"],
        beta_coverage=cfg["reward"]["beta_coverage"],
        lambda_cost=cfg["reward"]["lambda_cost"],
        coverage_radius_km=cfg["simulator"]["coverage_radius_km"],
        regime=regime,
        cost_mode=cfg["reward"]["cost_mode"],
        constant_cost=cfg["reward"]["constant_cost"],
        distance_cost_scale=cfg["reward"]["distance_cost_scale"],
        use_shaping=cfg["reward"].get("use_shaping", False),
        alpha_spread=cfg["reward"].get("alpha_spread", 0.0),
        alpha_cover=cfg["reward"].get("alpha_cover", 0.0),
        alpha_cost=cfg["reward"].get("alpha_cost", 0.0),
        use_temporal_cache=cfg.get("performance", {}).get("use_temporal_cache", True),
        prefilter_top_m=prefilter_top_m,
    )


def train_agent(env, rl_cfg: dict) -> tuple[DQNAgent, list[float]]:
    sample_feat, _ = env.reset()
    agent = DQNAgent(
        feature_dim=sample_feat.shape[1],
        lr=rl_cfg["lr"],
        gamma=rl_cfg["gamma"],
        hidden_dim=rl_cfg.get("hidden_dim", 64),
        beta_coverage=env.beta_coverage,
        lambda_cost=env.lambda_cost,
        regime=env.regime,
    )
    buffer = ReplayBuffer(rl_cfg["buffer_capacity"])
    epsilon = rl_cfg["eps_start"]
    curve: list[float] = []
    for ep in range(rl_cfg["train_episodes"]):
        state_feat, legal = env.reset(start_snapshot=0)
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
        if len(buffer) >= rl_cfg["batch_size"] and ep >= rl_cfg["warmup_episodes"]:
            agent.update(buffer.sample(rl_cfg["batch_size"]))
        if ep % rl_cfg["target_update_every"] == 0:
            agent.sync_target()
        epsilon = max(rl_cfg["eps_end"], epsilon * rl_cfg["eps_decay"])
    return agent, curve


def build_learned_graph_views(
    ag,
    W: np.ndarray,
    *,
    top_k_edges: int | None,
    sampling_min_nodes: int,
):
    edge_index_full, node_features, coords, _, distances_full, node_costs = as_torch_tensors(ag)
    policy_static = torch.as_tensor(compute_static_spatial_features(ag.graph, ag.coords, ag.node_costs), dtype=torch.float32)
    if top_k_edges is None or len(ag.node_ids) < int(sampling_min_nodes):
        return {
            "edge_index": edge_index_full,
            "node_features": node_features,
            "coords": coords,
            "distances": distances_full,
            "node_costs": node_costs,
            "W": W,
            "W_torch": torch.as_tensor(W, dtype=torch.float32),
            "policy_static": policy_static,
        }
    edge_idx_np, W_sampled, dist_np, _ = topk_temporal_edge_neighborhood(
        edge_index=ag.edge_index,
        edge_weight_seq=W,
        distances_km=ag.distances_km,
        num_nodes=len(ag.node_ids),
        top_k_out=int(top_k_edges),
    )
    return {
        "edge_index": torch.as_tensor(edge_idx_np.T, dtype=torch.long),
        "node_features": node_features,
        "coords": coords,
        "distances": torch.as_tensor(dist_np, dtype=torch.float32),
        "node_costs": node_costs,
        "W": W_sampled,
        "W_torch": torch.as_tensor(W_sampled, dtype=torch.float32),
        "policy_static": policy_static,
    }


def spread_baseline_seed_set(
    mode: str,
    ag,
    W: np.ndarray,
    learned_view: dict,
    diffusor,
    cfg: dict,
    prefilter_top_m: int | None,
) -> dict[str, list[int]]:
    budget = cfg["rl"]["budget"]
    one_slice = W[0:1]
    baselines = {
        "degree": degree_ranking(ag.graph, budget),
        "degree_discount": degree_discount(ag.graph, budget),
    }
    if mode == "surrogate":
        dynamic_summaries = compute_dynamic_temporal_summaries(
            edge_index=learned_view["edge_index"].detach().cpu().numpy(),
            edge_weight_seq=learned_view["W"][:1],
            distances_km=learned_view["distances"].detach().cpu().numpy(),
            num_nodes=len(ag.node_ids),
            window_len=diffusor.temporal_window_len,
            horizon_len=1,
        )
        baselines["learned_greedy_spread"] = greedy_surrogate_marginal(
            diffusor=diffusor,
            node_features=learned_view["node_features"],
            edge_index=learned_view["edge_index"],
            edge_weight_seq=learned_view["W"][:1],
            distances_km=learned_view["distances"],
            num_nodes=len(ag.node_ids),
            budget=budget,
            dynamic_summaries=dynamic_summaries,
            prefilter_top_m=prefilter_top_m,
        )
    else:
        baselines["greedy_spread"] = greedy_spread(
            ag.edge_index,
            one_slice,
            len(ag.node_ids),
            budget,
            cfg["simulator"]["mc_rollouts"],
        )
    return baselines


def baseline_seed_sets(
    ag,
    W: np.ndarray,
    cfg: dict,
    regime: str,
    diffusor=None,
    learned_view: dict | None = None,
    spread_baseline_mode: str = "raw",
    prefilter_top_m: int | None = None,
):
    budget = cfg["rl"]["budget"]
    t0 = 0
    weighted_graph = build_temporal_weighted_graph(ag.graph, ag.edge_index, W[t0], ag.distances_km)
    if regime == "spread":
        return spread_baseline_seed_set(spread_baseline_mode, ag, W, learned_view, diffusor, cfg, prefilter_top_m)
    if regime == "dynamic":
        return {
            "weighted_degree_slice": weighted_degree_per_slice(ag.edge_index, W[t0], len(ag.node_ids), budget)[0],
            "temporal_weighted_degree": temporal_weighted_degree(ag.edge_index, W, len(ag.node_ids), budget)[0],
            "temporal_degree_discount": temporal_degree_discount(ag.edge_index, W[t0], len(ag.node_ids), budget),
            "myopic_lookahead": myopic_lookahead(ag.edge_index, W, len(ag.node_ids), budget, mc_rollouts=cfg["simulator"]["mc_rollouts"]),
            "temporal_ris_proxy": temporal_ris_proxy(ag.edge_index, W[t0], len(ag.node_ids), budget, seed=cfg["seed"]),
        }
    return {
        "weighted_strength": weighted_strength(ag.edge_index, W[t0], len(ag.node_ids), budget)[0],
        "distance_strength": distance_strength(ag.edge_index, W[t0], ag.distances_km, len(ag.node_ids), budget)[0],
        "weighted_betweenness": weighted_betweenness(weighted_graph, budget, weight_attr="temporal_length")[0],
        "community_bridge": community_bridge_nodes(weighted_graph, budget, weight_attr="temporal_weight")[0],
        "cost_aware": cost_aware_ranking(ag.edge_index, W[t0], ag.distances_km, ag.node_costs, len(ag.node_ids), budget)[0],
    }


def raw_eval(ag, W: np.ndarray, cfg: dict, regime: str, selected: list[int]) -> dict:
    raw_window = W[0:1] if regime in {"spread", "spatial"} else W
    ev = evaluate_seed_set_raw(
        edge_index=ag.edge_index,
        edge_prob_seq=raw_window,
        num_nodes=len(ag.node_ids),
        selected=selected,
        coords=ag.coords,
        node_costs=ag.node_costs,
        coverage_radius_km=cfg["simulator"]["coverage_radius_km"],
        regime=regime,
        beta_coverage=cfg["reward"]["beta_coverage"],
        lambda_cost=cfg["reward"]["lambda_cost"],
        cost_mode=cfg["reward"]["cost_mode"],
        constant_cost=cfg["reward"]["constant_cost"],
        distance_cost_scale=cfg["reward"]["distance_cost_scale"],
        mc_rollouts=cfg["simulator"]["mc_rollouts"],
    )
    return {
        "objective": float(ev.objective),
        "mass": float(ev.final_activated_mass),
        "coverage": float(ev.final_geographic_coverage),
        "cost": float(ev.total_intervention_cost),
        "selected": list(selected),
    }


def run_single_cell(
    base_cfg: dict,
    artifact_map: dict[str, str | None],
    size: int,
    snapshots: int,
    budget: int,
    regime: str,
    repeat_idx: int,
    base_tables,
    *,
    diffusor_epochs_override: int | None = None,
    rl_episodes_override: int | None = None,
    mc_rollouts_override: int | None = None,
    teacher_mc_rollouts_override: int | None = None,
    teacher_refresh_interval: int | None = None,
    prefilter_top_m: int | None = None,
    learned_top_k_edges: int | None = None,
    sampling_min_nodes: int = 80,
    avg_out_degree_override: float | None = None,
    spread_baseline_mode: str = "surrogate",
    spread_use_surrogate_policy: bool = True,
):
    seed = int(base_cfg["seed"] + 10_000 * repeat_idx + 100 * size + 3 * snapshots + budget)
    set_seed(seed)

    synth_tables = generate_homogeneous_airline_tables(
        base_tables,
        num_nodes=size,
        seed=seed,
        avg_out_degree=avg_out_degree_override,
    )
    ag = build_airline_graph(synth_tables)
    W = generate_synthetic_temporal_weights(
        ag,
        snapshots=snapshots,
        seasonal_strength=base_cfg["data"]["seasonal_strength"],
        noise_std=base_cfg["data"]["noise_std"],
        seed=seed,
    )
    learned_view = build_learned_graph_views(
        ag,
        W,
        top_k_edges=learned_top_k_edges,
        sampling_min_nodes=sampling_min_nodes,
    )

    cfg = apply_artifact_cfg(base_cfg, artifact_map.get(regime))
    cfg["seed"] = seed
    cfg["data"]["snapshots"] = snapshots
    cfg["rl"]["budget"] = int(budget)
    cfg["evaluation"]["regime"] = regime
    if diffusor_epochs_override is not None:
        cfg["model"]["diffusor_epochs"] = int(diffusor_epochs_override)
    if rl_episodes_override is not None:
        cfg["rl"]["train_episodes"] = int(rl_episodes_override)
        cfg["rl"]["warmup_episodes"] = min(int(cfg["rl"]["warmup_episodes"]), max(int(rl_episodes_override) // 4, 1))
    if mc_rollouts_override is not None:
        cfg["simulator"]["mc_rollouts"] = int(mc_rollouts_override)
    if teacher_mc_rollouts_override is not None:
        cfg["model"]["teacher_mc_rollouts"] = int(teacher_mc_rollouts_override)
    perf_cfg = cfg.setdefault("performance", {})
    if teacher_refresh_interval is None:
        teacher_refresh_interval = int(perf_cfg.get("teacher_refresh_interval", 1))
    if prefilter_top_m is None:
        prefilter_top_m = perf_cfg.get("prefilter_top_m")
    if learned_top_k_edges is None:
        learned_top_k_edges = perf_cfg.get("learned_top_k_edges")

    diffusor = GNNDiffusor(
        node_feat_dim=learned_view["node_features"].size(1),
        hidden_dim=cfg["model"]["hidden_dim"],
        layers=cfg["model"]["layers"],
    )
    diff_art = train_diffusor(
        model=diffusor,
        node_features=learned_view["node_features"],
        edge_index=learned_view["edge_index"],
        edge_prob_seq=learned_view["W"],
        distances_km=learned_view["distances"],
        epochs=cfg["model"]["diffusor_epochs"],
        batch_size=cfg["model"]["batch_size"],
        lr=cfg["model"]["diffusor_lr"],
        seed=seed,
        coords=learned_view["coords"],
        node_costs=learned_view["node_costs"],
        regime=regime,
        beta_coverage=cfg["reward"]["beta_coverage"],
        lambda_cost=cfg["reward"]["lambda_cost"],
        cost_mode=cfg["reward"]["cost_mode"],
        constant_cost=cfg["reward"]["constant_cost"],
        distance_cost_scale=cfg["reward"]["distance_cost_scale"],
        coverage_radius_km=cfg["simulator"]["coverage_radius_km"],
        teacher_mc_rollouts=cfg["model"].get("teacher_mc_rollouts", cfg["simulator"]["mc_rollouts"]),
        marginal_loss_weight=cfg["model"].get("marginal_loss_weight", 1.0),
        ranking_loss_weight=cfg["model"].get("ranking_loss_weight", 0.2),
        selection_budget=cfg["rl"]["budget"],
        teacher_refresh_interval=teacher_refresh_interval,
    )

    env = make_env(
        diffusor,
        learned_view["node_features"],
        learned_view["edge_index"],
        learned_view["W_torch"],
        learned_view["distances"],
        learned_view["coords"],
        learned_view["node_costs"],
        learned_view["policy_static"],
        cfg,
        regime,
        prefilter_top_m=prefilter_top_m,
    )

    if regime == "spread" and spread_use_surrogate_policy:
        dynamic_summaries = compute_dynamic_temporal_summaries(
            edge_index=learned_view["edge_index"].detach().cpu().numpy(),
            edge_weight_seq=learned_view["W"][:1],
            distances_km=learned_view["distances"].detach().cpu().numpy(),
            num_nodes=len(ag.node_ids),
            window_len=diffusor.temporal_window_len,
            horizon_len=1,
        )
        rl_selected = greedy_surrogate_marginal(
            diffusor=diffusor,
            node_features=learned_view["node_features"],
            edge_index=learned_view["edge_index"],
            edge_weight_seq=learned_view["W"][:1],
            distances_km=learned_view["distances"],
            num_nodes=len(ag.node_ids),
            budget=cfg["rl"]["budget"],
            dynamic_summaries=dynamic_summaries,
            prefilter_top_m=prefilter_top_m,
        )
        rl_env_eval = evaluate_seed_set(env, rl_selected)
        curve = []
    else:
        agent, curve = train_agent(env, cfg["rl"])
        rl_env_eval = rollout_policy(env, agent, epsilon=0.0)
        rl_selected = rl_env_eval.selected

    rl_eval = raw_eval(ag, W, cfg, regime, rl_selected)

    baselines = baseline_seed_sets(
        ag,
        W,
        cfg,
        regime,
        diffusor=diffusor,
        learned_view=learned_view,
        spread_baseline_mode=spread_baseline_mode,
        prefilter_top_m=prefilter_top_m,
    )
    baseline_reports = {name: raw_eval(ag, W, cfg, regime, seeds) for name, seeds in baselines.items()}
    best_baseline_name, best_baseline = max(baseline_reports.items(), key=lambda kv: kv[1]["objective"])

    return {
        "size": int(size),
        "snapshots": int(snapshots),
        "budget": int(budget),
        "regime": regime,
        "repeat": int(repeat_idx),
        "seed": int(seed),
        "num_edges": int(ag.edge_index.shape[0]),
        "num_learned_edges": int(learned_view["edge_index"].size(1)),
        "diffusor_last_loss": float(diff_art.history["loss"][-1]),
        "rl_objective": rl_eval["objective"],
        "rl_mass": rl_eval["mass"],
        "rl_coverage": rl_eval["coverage"],
        "rl_cost": rl_eval["cost"],
        "rl_seeds": rl_eval["selected"],
        "rl_learned_env_objective": float(rl_env_eval.objective),
        "rl_training_curve_last10_mean": float(np.mean(curve[-10:])) if curve else None,
        "best_baseline_name": best_baseline_name,
        "baseline_objective": float(best_baseline["objective"]),
        "baseline_mass": float(best_baseline["mass"]),
        "baseline_coverage": float(best_baseline["coverage"]),
        "baseline_cost": float(best_baseline["cost"]),
        "baseline_seeds": best_baseline["selected"],
        "delta_objective": float(rl_eval["objective"] - best_baseline["objective"]),
        "rl_beats_baseline": float(rl_eval["objective"] > best_baseline["objective"]),
    }


def aggregate_rows(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        key = (row["size"], row["snapshots"], row["budget"], row["regime"])
        grouped[key].append(row)

    summary: list[dict] = []
    for key in sorted(grouped):
        grp = grouped[key]
        baseline_counts = Counter(r["best_baseline_name"] for r in grp)
        summary.append(
            {
                "size": key[0],
                "snapshots": key[1],
                "budget": key[2],
                "regime": key[3],
                "repeats": len(grp),
                "avg_edges": float(np.mean([r["num_edges"] for r in grp])),
                "avg_learned_edges": float(np.mean([r["num_learned_edges"] for r in grp])),
                "rl_objective_mean": float(np.mean([r["rl_objective"] for r in grp])),
                "rl_objective_std": float(np.std([r["rl_objective"] for r in grp])),
                "baseline_objective_mean": float(np.mean([r["baseline_objective"] for r in grp])),
                "baseline_objective_std": float(np.std([r["baseline_objective"] for r in grp])),
                "delta_objective_mean": float(np.mean([r["delta_objective"] for r in grp])),
                "delta_objective_std": float(np.std([r["delta_objective"] for r in grp])),
                "win_rate": float(np.mean([r["rl_beats_baseline"] for r in grp])),
                "most_common_best_baseline": baseline_counts.most_common(1)[0][0],
            }
        )
    return summary


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a start experiment ladder on homogeneous synthetic airline graphs.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--spread-artifact", default="artifacts/optuna_spread_current.json")
    parser.add_argument("--dynamic-artifact", default="artifacts/optuna_dynamic_current.json")
    parser.add_argument("--spatial-artifact", default="artifacts/optuna_spatial_current.json")
    parser.add_argument("--sizes", default="50,100,200")
    parser.add_argument("--snapshots", default="70,120")
    parser.add_argument("--budgets", default="3,5,10,15")
    parser.add_argument("--regimes", default="spread,dynamic,spatial")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--repeat-offset", type=int, default=0)
    parser.add_argument("--diffusor-epochs", type=int, default=None)
    parser.add_argument("--train-episodes", type=int, default=None)
    parser.add_argument("--mc-rollouts", type=int, default=None)
    parser.add_argument("--teacher-mc-rollouts", type=int, default=None)
    parser.add_argument("--teacher-refresh-interval", type=int, default=1)
    parser.add_argument("--prefilter-top-m", type=int, default=None)
    parser.add_argument("--learned-top-k-edges", type=int, default=None)
    parser.add_argument("--sampling-min-nodes", type=int, default=80)
    parser.add_argument("--spread-baseline-mode", choices=["raw", "surrogate"], default="surrogate")
    parser.add_argument("--spread-use-surrogate-policy", action="store_true")
    parser.add_argument("--out", default="artifacts/experiment_ladder_start.json")
    args = parser.parse_args()

    base_cfg = load_yaml(args.config)
    artifact_map = {
        "spread": args.spread_artifact,
        "dynamic": args.dynamic_artifact,
        "spatial": args.spatial_artifact,
    }
    sizes = parse_int_list(args.sizes)
    snapshots_list = parse_int_list(args.snapshots)
    budgets = parse_int_list(args.budgets)
    regimes = [r.strip() for r in args.regimes.split(",") if r.strip()]

    base_tables = load_airline_tables(base_cfg["data"]["airports_csv"], base_cfg["data"]["routes_csv"])
    rows: list[dict] = []
    total = len(sizes) * len(snapshots_list) * len(budgets) * len(regimes) * int(args.repeats)
    step = 0
    for local_repeat_idx in range(int(args.repeats)):
        repeat_idx = int(args.repeat_offset) + local_repeat_idx
        for size in sizes:
            for snapshots in snapshots_list:
                for budget in budgets:
                    for regime in regimes:
                        step += 1
                        print(
                            f"[{step}/{total}] size={size} snapshots={snapshots} budget={budget} regime={regime} repeat={repeat_idx}",
                            flush=True,
                        )
                        rows.append(
                            run_single_cell(
                                base_cfg=base_cfg,
                                artifact_map=artifact_map,
                                size=size,
                                snapshots=snapshots,
                                budget=budget,
                                regime=regime,
                                repeat_idx=repeat_idx,
                                base_tables=base_tables,
                                diffusor_epochs_override=args.diffusor_epochs,
                                rl_episodes_override=args.train_episodes,
                                mc_rollouts_override=args.mc_rollouts,
                                teacher_mc_rollouts_override=args.teacher_mc_rollouts,
                                teacher_refresh_interval=args.teacher_refresh_interval,
                                prefilter_top_m=args.prefilter_top_m,
                                learned_top_k_edges=args.learned_top_k_edges,
                                sampling_min_nodes=args.sampling_min_nodes,
                                spread_baseline_mode=args.spread_baseline_mode,
                                spread_use_surrogate_policy=args.spread_use_surrogate_policy,
                            )
                        )

    summary = aggregate_rows(rows)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": {
            "config": args.config,
            "artifact_map": artifact_map,
            "sizes": sizes,
            "snapshots": snapshots_list,
            "budgets": budgets,
            "regimes": regimes,
            "repeats": int(args.repeats),
            "repeat_offset": int(args.repeat_offset),
            "diffusor_epochs_override": args.diffusor_epochs,
            "train_episodes_override": args.train_episodes,
            "mc_rollouts_override": args.mc_rollouts,
            "teacher_mc_rollouts_override": args.teacher_mc_rollouts,
            "teacher_refresh_interval": args.teacher_refresh_interval,
            "prefilter_top_m": args.prefilter_top_m,
            "learned_top_k_edges": args.learned_top_k_edges,
            "sampling_min_nodes": args.sampling_min_nodes,
            "spread_baseline_mode": args.spread_baseline_mode,
            "spread_use_surrogate_policy": bool(args.spread_use_surrogate_policy),
            "num_cells": len(summary),
            "num_runs": len(rows),
        },
        "rows": rows,
        "summary": summary,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_csv(out_path.with_suffix(".detailed.csv"), rows)
    write_csv(out_path.with_suffix(".summary.csv"), summary)
    print(f"Saved: {out_path}")
    print(f"Saved: {out_path.with_suffix('.detailed.csv')}")
    print(f"Saved: {out_path.with_suffix('.summary.csv')}")


if __name__ == "__main__":
    main()
