from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import optuna

from run_experiment_ladder import parse_int_list, run_single_cell
from spatial_im.data.airline import load_airline_tables
from spatial_im.utils.io import load_yaml


def with_default_performance(cfg: dict) -> dict:
    local = copy.deepcopy(cfg)
    perf = local.setdefault("performance", {})
    perf.setdefault("teacher_refresh_interval", 2)
    perf.setdefault("prefilter_top_m", 32)
    perf.setdefault("learned_top_k_edges", 12)
    return local


def parse_float_list(raw: str) -> list[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def apply_trial_cfg(base_cfg: dict, trial: optuna.trial.FrozenTrial) -> dict:
    cfg = with_default_performance(base_cfg)
    full = trial.user_attrs.get("params_full", {})
    if isinstance(full, dict):
        if "model" in full and isinstance(full["model"], dict):
            cfg["model"].update(full["model"])
        if "rl" in full and isinstance(full["rl"], dict):
            cfg["rl"].update(full["rl"])
        if "performance" in full and isinstance(full["performance"], dict):
            cfg["performance"].update(full["performance"])
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="Raw-MC rerank top Optuna trials on held-out source-family graphs.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--regime", required=True, choices=["spread", "dynamic", "spatial"])
    parser.add_argument("--storage", required=True, help="Optuna storage URL, e.g. sqlite:///artifacts/optuna.db")
    parser.add_argument("--study-name", required=True)
    parser.add_argument("--sizes", default="100,200")
    parser.add_argument("--snapshots", default="120,220")
    parser.add_argument("--budgets", default="5,10")
    parser.add_argument("--avg-degrees", default="", help="Optional comma-separated avg out-degree per size.")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--graphs-per-cell", type=int, default=3)
    parser.add_argument("--holdout-repeat-offset", type=int, default=1000)
    parser.add_argument("--mc-rollouts", type=int, default=None)
    parser.add_argument("--teacher-mc-rollouts", type=int, default=None)
    parser.add_argument("--diffusor-epochs", type=int, default=None)
    parser.add_argument("--train-episodes", type=int, default=None)
    parser.add_argument("--sampling-min-nodes", type=int, default=80)
    parser.add_argument("--out", required=True)
    parser.add_argument("--winner-out", required=True)
    args = parser.parse_args()

    base_cfg = with_default_performance(load_yaml(args.config))
    base_cfg["evaluation"]["regime"] = args.regime
    if args.mc_rollouts is not None:
        base_cfg["simulator"]["mc_rollouts"] = int(args.mc_rollouts)
    if args.teacher_mc_rollouts is not None:
        base_cfg["model"]["teacher_mc_rollouts"] = int(args.teacher_mc_rollouts)

    sizes = parse_int_list(args.sizes)
    snapshots = parse_int_list(args.snapshots)
    budgets = parse_int_list(args.budgets)
    avg_degrees = parse_float_list(args.avg_degrees) if args.avg_degrees else []
    size_to_avg_degree = {
        int(size): float(avg_degrees[min(idx, len(avg_degrees) - 1)])
        for idx, size in enumerate(sizes)
    } if avg_degrees else None
    base_tables = load_airline_tables(base_cfg["data"]["airports_csv"], base_cfg["data"]["routes_csv"])

    study = optuna.load_study(study_name=args.study_name, storage=args.storage)
    candidates = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None]
    candidates.sort(key=lambda t: float(t.value), reverse=True)
    selected = candidates[: max(1, int(args.top_k))]

    rows: list[dict] = []
    trial_summaries: list[dict] = []

    for rank, trial in enumerate(selected, start=1):
        cfg = apply_trial_cfg(base_cfg, trial)
        cfg["evaluation"]["regime"] = args.regime

        trial_rows: list[dict] = []
        for gidx in range(int(args.graphs_per_cell)):
            repeat_idx = int(args.holdout_repeat_offset) + gidx
            for size in sizes:
                for snap in snapshots:
                    for budget in budgets:
                        trial_rows.append(
                            run_single_cell(
                                base_cfg=cfg,
                                artifact_map={},
                                size=size,
                                snapshots=snap,
                                budget=budget,
                                regime=args.regime,
                                repeat_idx=repeat_idx,
                                base_tables=base_tables,
                                diffusor_epochs_override=args.diffusor_epochs,
                                rl_episodes_override=args.train_episodes,
                                mc_rollouts_override=cfg["simulator"]["mc_rollouts"],
                                teacher_mc_rollouts_override=cfg["model"].get(
                                    "teacher_mc_rollouts",
                                    cfg["simulator"]["mc_rollouts"],
                                ),
                                teacher_refresh_interval=cfg.get("performance", {}).get("teacher_refresh_interval", 1),
                                prefilter_top_m=cfg.get("performance", {}).get("prefilter_top_m"),
                                learned_top_k_edges=cfg.get("performance", {}).get("learned_top_k_edges"),
                                sampling_min_nodes=int(args.sampling_min_nodes),
                                avg_out_degree_override=(
                                    float(size_to_avg_degree[size])
                                    if size_to_avg_degree is not None and size in size_to_avg_degree
                                    else None
                                ),
                                spread_baseline_mode="surrogate",
                                spread_use_surrogate_policy=True,
                            )
                        )
        mean_raw_objective = float(np.mean([r["rl_objective"] for r in trial_rows]))
        mean_delta = float(np.mean([r["delta_objective"] for r in trial_rows]))
        win_rate = float(np.mean([r["rl_beats_baseline"] for r in trial_rows]))
        summary = {
            "rank_by_optuna": rank,
            "trial_number": int(trial.number),
            "optuna_value": float(trial.value),
            "mean_raw_objective": mean_raw_objective,
            "mean_delta_objective": mean_delta,
            "win_rate": win_rate,
            "params_full": trial.user_attrs.get("params_full", {}),
        }
        rows.extend(
            [
                {
                    "trial_number": int(trial.number),
                    "rank_by_optuna": rank,
                    **r,
                }
                for r in trial_rows
            ]
        )
        trial_summaries.append(summary)
        print(
            f"[{args.regime}] trial={trial.number} rank={rank} "
            f"raw_mean={mean_raw_objective:.4f} delta_mean={mean_delta:.4f} win_rate={win_rate:.3f}",
            flush=True,
        )

    if not trial_summaries:
        raise RuntimeError("No completed trials found in study.")

    winner = max(trial_summaries, key=lambda x: x["mean_raw_objective"])
    winner_cfg = with_default_performance(load_yaml(args.config))
    winner_cfg["evaluation"]["regime"] = args.regime
    params_full = winner.get("params_full", {})
    if isinstance(params_full, dict):
        if "model" in params_full and isinstance(params_full["model"], dict):
            winner_cfg["model"].update(params_full["model"])
        if "rl" in params_full and isinstance(params_full["rl"], dict):
            winner_cfg["rl"].update(params_full["rl"])
        if "performance" in params_full and isinstance(params_full["performance"], dict):
            winner_cfg.setdefault("performance", {})
            winner_cfg["performance"].update(params_full["performance"])

    report = {
        "regime": args.regime,
        "study_name": args.study_name,
        "storage": args.storage,
        "sizes": sizes,
        "snapshots": snapshots,
        "budgets": budgets,
        "avg_degrees": size_to_avg_degree or {},
        "graphs_per_cell": int(args.graphs_per_cell),
        "holdout_repeat_offset": int(args.holdout_repeat_offset),
        "top_k": int(args.top_k),
        "winner": winner,
        "trial_summaries": trial_summaries,
        "rows": rows,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    winner_path = Path(args.winner_out)
    winner_path.parent.mkdir(parents=True, exist_ok=True)
    winner_payload = {
        "regime": args.regime,
        "winner_trial_number": int(winner["trial_number"]),
        "winner_mean_raw_objective": float(winner["mean_raw_objective"]),
        "winner_mean_delta_objective": float(winner["mean_delta_objective"]),
        "winner_win_rate": float(winner["win_rate"]),
        "best": {
            "model": winner_cfg["model"],
            "rl": winner_cfg["rl"],
            "performance": winner_cfg.get("performance", {}),
        },
    }
    winner_path.write_text(json.dumps(winner_payload, indent=2), encoding="utf-8")

    print(json.dumps({"winner": winner_payload, "report": str(out_path), "winner_out": str(winner_path)}, indent=2))


if __name__ == "__main__":
    main()
