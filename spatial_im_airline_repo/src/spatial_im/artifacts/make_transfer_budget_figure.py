from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REGIMES = ("spread", "dynamic", "spatial")
REGIME_TITLES = {
    "spread": "Spread",
    "dynamic": "Dynamic",
    "spatial": "Spatial",
}
MODE_ORDER = ("zero_shot", "transfer_adapt", "scratch_adapt", "baseline")
MODE_LABELS = {
    "zero_shot": "Zero-shot",
    "transfer_adapt": "Warm-start",
    "scratch_adapt": "Scratch",
    "baseline": "Best baseline",
}
MODE_COLORS = {
    "zero_shot": "#2ca02c",
    "transfer_adapt": "#17becf",
    "scratch_adapt": "#7f7f7f",
    "baseline": "#d62728",
}


def _parse_target_budget(path: Path) -> tuple[int, int]:
    match = re.search(r"transfer_target(\d+)_budget(\d+)(?:_[^.]+)?\.json$", path.name)
    if not match:
        raise ValueError(f"Unexpected transfer filename: {path}")
    return int(match.group(1)), int(match.group(2))


def load_transfer_grid(
    transfer_dir: Path,
    glob_pattern: str,
    name_filter: str,
) -> tuple[list[int], list[int], dict[str, dict[tuple[int, int], dict[str, float]]]]:
    rows: dict[str, dict[tuple[int, int], dict[str, float]]] = {regime: {} for regime in REGIMES}
    targets: set[int] = set()
    budgets: set[int] = set()

    paths = sorted(transfer_dir.glob(glob_pattern))
    if name_filter:
        paths = [p for p in paths if name_filter in p.name]
    if not paths:
        raise FileNotFoundError(
            f"No transfer artifacts found in {transfer_dir} matching glob='{glob_pattern}' filter='{name_filter}'"
        )

    for path in paths:
        target, budget = _parse_target_budget(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        targets.add(target)
        budgets.add(budget)
        for regime in REGIMES:
            entry = payload["regimes"][regime]
            rows[regime][(target, budget)] = {
                "zero_shot": float(entry["zero_shot"]["raw_objective"]),
                "transfer_adapt": float(entry["transfer_adapt"]["raw_objective"]),
                "scratch_adapt": float(entry["scratch_adapt"]["raw_objective"]),
                "baseline": float(entry["best_baseline"]["objective"]),
            }

    return sorted(targets), sorted(budgets), rows


def make_transfer_budget_figure(
    targets: list[int],
    budgets: list[int],
    rows: dict[str, dict[tuple[int, int], dict[str, float]]],
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(len(targets), len(REGIMES), figsize=(14, 7.2), constrained_layout=True)
    if len(targets) == 1:
        axes = np.expand_dims(axes, axis=0)

    for row_idx, target in enumerate(targets):
        for col_idx, regime in enumerate(REGIMES):
            ax = axes[row_idx, col_idx]
            for mode in MODE_ORDER:
                y = [rows[regime][(target, budget)][mode] for budget in budgets]
                ax.plot(
                    budgets,
                    y,
                    marker="o",
                    linewidth=2.0,
                    color=MODE_COLORS[mode],
                    label=MODE_LABELS[mode],
                )
            ax.set_title(f"{REGIME_TITLES[regime]} | N={target}", fontsize=10.5, fontweight="bold")
            ax.set_xlabel("Budget k")
            if col_idx == 0:
                ax.set_ylabel("Raw objective")
            ax.grid(axis="y", alpha=0.25)
            ax.set_axisbelow(True)
            ax.set_xticks(budgets)

    axes[0, 0].legend(frameon=False, loc="best")
    fig.suptitle("Transfer Objective vs Budget (Corrected k=5,10,15)", fontsize=13, fontweight="bold")
    fig.savefig(out_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create transfer-budget comparison figure from transfer JSON artifacts.")
    parser.add_argument("--transfer-dir", default="artifacts")
    parser.add_argument("--glob", default="transfer_target*_budget*.json")
    parser.add_argument("--name-filter", default="", help="Optional substring filter for filenames.")
    parser.add_argument("--outdir", default="artifacts/experiment_ladder_figures")
    parser.add_argument("--outfile", default="fig_transfer_budget_sweep.png")
    args = parser.parse_args()

    transfer_dir = Path(args.transfer_dir)
    outdir = Path(args.outdir)
    out_path = outdir / args.outfile

    targets, budgets, rows = load_transfer_grid(transfer_dir, args.glob, args.name_filter)
    make_transfer_budget_figure(targets, budgets, rows, out_path)
    print("Saved:", out_path)


if __name__ == "__main__":
    main()
