from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np

try:
    import optuna
except ImportError as exc:
    raise SystemExit("optuna is not installed.") from exc

from run_experiment_ladder import apply_artifact_cfg, parse_int_list, run_single_cell
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


def apply_trial_params(base_cfg: dict, regime: str, trial: optuna.Trial) -> tuple[dict, dict]:
    cfg = with_default_performance(base_cfg)
    cfg["model"]["hidden_dim"] = trial.suggest_categorical("model.hidden_dim", [64, 96, 128])
    cfg["model"]["layers"] = trial.suggest_int("model.layers", 1, 3)
    cfg["model"]["diffusor_lr"] = trial.suggest_float("model.diffusor_lr", 5e-4, 5e-3, log=True)
    cfg["model"]["diffusor_epochs"] = trial.suggest_int("model.diffusor_epochs", 4, 12, step=2)
    cfg["model"]["batch_size"] = trial.suggest_categorical("model.batch_size", [16, 32])
    cfg["model"]["teacher_mc_rollouts"] = trial.suggest_categorical("model.teacher_mc_rollouts", [8, 12, 16])
    cfg["performance"]["teacher_refresh_interval"] = trial.suggest_int("performance.teacher_refresh_interval", 1, 4)
    cfg["performance"]["prefilter_top_m"] = trial.suggest_categorical("performance.prefilter_top_m", [16, 24, 32, 48])
    cfg["performance"]["learned_top_k_edges"] = trial.suggest_categorical("performance.learned_top_k_edges", [8, 12, 16])

    if regime != "spread":
        train_episodes = trial.suggest_int("rl.train_episodes", 16, 60, step=4)
        cfg["rl"]["hidden_dim"] = trial.suggest_categorical("rl.hidden_dim", [64, 96, 128])
        cfg["rl"]["lr"] = trial.suggest_float("rl.lr", 1e-4, 5e-3, log=True)
        cfg["rl"]["gamma"] = trial.suggest_float("rl.gamma", 0.85, 0.99)
        cfg["rl"]["buffer_capacity"] = trial.suggest_categorical("rl.buffer_capacity", [1000, 2000, 4000])
        cfg["rl"]["batch_size"] = trial.suggest_categorical("rl.batch_size", [16, 32])
        cfg["rl"]["train_episodes"] = train_episodes
        cfg["rl"]["warmup_episodes"] = trial.suggest_int("rl.warmup_episodes", 4, max(4, min(train_episodes - 4, 16)), step=4)
        cfg["rl"]["target_update_every"] = trial.suggest_categorical("rl.target_update_every", [5, 10, 20])
        cfg["rl"]["eps_decay"] = trial.suggest_float("rl.eps_decay", 0.95, 0.99)

    return cfg, {
        "model": cfg["model"],
        "rl": cfg["rl"],
        "performance": cfg["performance"],
    }


def evaluate_subset(
    cfg: dict,
    regime: str,
    sizes: list[int],
    snapshots: list[int],
    budgets: list[int],
    repeats: int,
    base_tables,
    size_to_avg_degree: dict[int, float] | None = None,
    *,
    spread_baseline_mode: str,
    spread_use_surrogate_policy: bool,
) -> tuple[float, float, list[dict]]:
    rows: list[dict] = []
    for repeat_idx in range(int(repeats)):
        for size in sizes:
            for snapshot in snapshots:
                for budget in budgets:
                    rows.append(
                        run_single_cell(
                            base_cfg=cfg,
                            artifact_map={},
                            size=size,
                            snapshots=snapshot,
                            budget=budget,
                            regime=regime,
                            repeat_idx=repeat_idx,
                            base_tables=base_tables,
                            diffusor_epochs_override=None,
                            rl_episodes_override=None,
                            mc_rollouts_override=cfg["simulator"]["mc_rollouts"],
                            teacher_mc_rollouts_override=cfg["model"].get("teacher_mc_rollouts", cfg["simulator"]["mc_rollouts"]),
                            teacher_refresh_interval=None,
                            prefilter_top_m=None,
                                learned_top_k_edges=None,
                                sampling_min_nodes=80,
                                avg_out_degree_override=(
                                    float(size_to_avg_degree[size])
                                    if size_to_avg_degree is not None and size in size_to_avg_degree
                                    else None
                                ),
                                spread_baseline_mode=spread_baseline_mode,
                                spread_use_surrogate_policy=spread_use_surrogate_policy,
                            )
                    )
    mean_delta = float(np.mean([r["delta_objective"] for r in rows]))
    mean_rl_objective = float(np.mean([r["rl_objective"] for r in rows]))
    return mean_delta, mean_rl_objective, rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Optuna tuning on a ladder subset.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--regime", required=True, choices=["spread", "dynamic", "spatial"])
    parser.add_argument("--artifact", default=None, help="Optional current tuned artifact to start from.")
    parser.add_argument("--sizes", default="100,200")
    parser.add_argument("--snapshots", default="120")
    parser.add_argument("--budgets", default="10,15")
    parser.add_argument("--avg-degrees", default="", help="Optional comma-separated avg out-degree per size.")
    parser.add_argument("--repeats", type=int, default=1, help="Repeats per trial objective.")
    parser.add_argument("--trials", type=int, default=4)
    parser.add_argument("--study-name", default="spatial-im-ladder-optuna")
    parser.add_argument("--storage", default=None)
    parser.add_argument("--mc-rollouts", type=int, default=None)
    parser.add_argument("--spread-baseline-mode", choices=["raw", "surrogate"], default="surrogate")
    parser.add_argument("--spread-use-surrogate-policy", action="store_true")
    parser.add_argument("--out", default="artifacts/optuna_ladder_best.json")
    args = parser.parse_args()

    base_cfg = load_yaml(args.config)
    base_cfg = apply_artifact_cfg(base_cfg, args.artifact)
    base_cfg = with_default_performance(base_cfg)
    if args.mc_rollouts is not None:
        base_cfg["simulator"]["mc_rollouts"] = int(args.mc_rollouts)
    base_cfg["evaluation"]["regime"] = args.regime
    sizes = parse_int_list(args.sizes)
    snapshots = parse_int_list(args.snapshots)
    budgets = parse_int_list(args.budgets)
    avg_degrees = parse_float_list(args.avg_degrees) if args.avg_degrees else []
    size_to_avg_degree = {
        int(size): float(avg_degrees[min(idx, len(avg_degrees) - 1)])
        for idx, size in enumerate(sizes)
    } if avg_degrees else None
    base_tables = load_airline_tables(base_cfg["data"]["airports_csv"], base_cfg["data"]["routes_csv"])

    default_score, default_rl_obj, default_rows = evaluate_subset(
        base_cfg,
        args.regime,
        sizes,
        snapshots,
        budgets,
        args.repeats,
        base_tables,
        size_to_avg_degree=size_to_avg_degree,
        spread_baseline_mode=args.spread_baseline_mode,
        spread_use_surrogate_policy=args.spread_use_surrogate_policy,
    )

    sampler = optuna.samplers.TPESampler(seed=base_cfg["seed"])
    study = optuna.create_study(
        study_name=args.study_name,
        direction="maximize",
        sampler=sampler,
        storage=args.storage,
        load_if_exists=bool(args.storage),
    )

    def objective(trial: optuna.Trial) -> float:
        trial_cfg, cfg_dump = apply_trial_params(base_cfg, args.regime, trial)
        score, mean_rl_obj, rows = evaluate_subset(
            trial_cfg,
            args.regime,
            sizes,
            snapshots,
            budgets,
            args.repeats,
            base_tables,
            size_to_avg_degree=size_to_avg_degree,
            spread_baseline_mode=args.spread_baseline_mode,
            spread_use_surrogate_policy=args.spread_use_surrogate_policy,
        )
        trial.set_user_attr("mean_rl_objective", mean_rl_obj)
        trial.set_user_attr("details", rows)
        trial.set_user_attr("params_full", cfg_dump)
        return score

    study.optimize(objective, n_trials=args.trials)

    best_cfg, best_cfg_dump = apply_trial_params(base_cfg, args.regime, study.best_trial)
    best_score, best_rl_obj, best_rows = evaluate_subset(
        best_cfg,
        args.regime,
        sizes,
        snapshots,
        budgets,
        args.repeats,
        base_tables,
        size_to_avg_degree=size_to_avg_degree,
        spread_baseline_mode=args.spread_baseline_mode,
        spread_use_surrogate_policy=args.spread_use_surrogate_policy,
    )

    report = {
        "regime": args.regime,
        "sizes": sizes,
        "snapshots": snapshots,
        "budgets": budgets,
        "avg_degrees": size_to_avg_degree or {},
        "repeats": args.repeats,
        "trials": args.trials,
        "objective": "mean_delta_objective_vs_best_baseline",
        "default": {
            "score": default_score,
            "mean_rl_objective": default_rl_obj,
            "model": base_cfg["model"],
            "rl": base_cfg["rl"],
            "performance": base_cfg["performance"],
            "details": default_rows,
        },
        "best": {
            "score": best_score,
            "mean_rl_objective": best_rl_obj,
            "params": study.best_trial.params,
            "model": best_cfg["model"],
            "rl": best_cfg["rl"],
            "performance": best_cfg["performance"],
            "details": best_rows,
            "improvement": best_score - default_score,
        },
        "study": {
            "best_value": study.best_value,
            "best_trial_number": study.best_trial.number,
            "n_trials": len(study.trials),
        },
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
