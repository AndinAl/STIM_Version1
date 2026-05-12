from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_ladder_summary(path: str | Path) -> tuple[dict, list[dict]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return payload["meta"], payload["summary"]


def _group_rows(rows: list[dict]) -> dict[tuple[int, int, str], dict[int, dict]]:
    grouped: dict[tuple[int, int, str], dict[int, dict]] = defaultdict(dict)
    for row in rows:
        key = (int(row["size"]), int(row["snapshots"]), str(row["regime"]))
        grouped[key][int(row["budget"])] = row
    return grouped


def make_snapshot_heatmap_figure(rows: list[dict], snapshot: int, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sizes = sorted({int(r["size"]) for r in rows})
    budgets = sorted({int(r["budget"]) for r in rows})
    regimes = ["spread", "dynamic", "spatial"]
    grouped = _group_rows(rows)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), constrained_layout=True)
    vmin = min(float(r["delta_objective_mean"]) for r in rows if int(r["snapshots"]) == snapshot)
    vmax = max(float(r["delta_objective_mean"]) for r in rows if int(r["snapshots"]) == snapshot)
    lim = max(abs(vmin), abs(vmax))

    for ax, regime in zip(axes, regimes):
        mat = np.zeros((len(sizes), len(budgets)), dtype=np.float32)
        for i, size in enumerate(sizes):
            for j, budget in enumerate(budgets):
                row = grouped[(size, snapshot, regime)][budget]
                mat[i, j] = float(row["delta_objective_mean"])
        im = ax.imshow(mat, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
        ax.set_title(f"{regime.capitalize()} | snapshots={snapshot}", fontsize=11, fontweight="bold")
        ax.set_xticks(np.arange(len(budgets)))
        ax.set_xticklabels(budgets)
        ax.set_yticks(np.arange(len(sizes)))
        ax.set_yticklabels(sizes)
        ax.set_xlabel("Budget")
        if ax is axes[0]:
            ax.set_ylabel("Graph size")
        for i in range(len(sizes)):
            for j in range(len(budgets)):
                ax.text(j, i, f"{mat[i, j]:.1f}", ha="center", va="center", fontsize=8, color="black")
    cbar = fig.colorbar(im, ax=axes, shrink=0.95, pad=0.02)
    cbar.set_label("RL objective - best baseline objective")
    fig.suptitle("Experiment Ladder: Objective Gap Heatmap", fontsize=13, fontweight="bold")
    fig.savefig(out_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def make_budget_delta_figure(rows: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sizes = sorted({int(r["size"]) for r in rows})
    budgets = sorted({int(r["budget"]) for r in rows})
    regimes = ["spread", "dynamic", "spatial"]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    colors = {50: "#1f77b4", 100: "#ff7f0e", 200: "#2ca02c"}
    for ax, regime in zip(axes, regimes):
        for size in sizes:
            y = []
            for budget in budgets:
                subset = [
                    float(r["delta_objective_mean"])
                    for r in rows
                    if str(r["regime"]) == regime and int(r["size"]) == size and int(r["budget"]) == budget
                ]
                y.append(float(np.mean(subset)))
            ax.plot(budgets, y, marker="o", linewidth=2.0, label=f"N={size}", color=colors.get(size))
        ax.axhline(0.0, color="#666666", linewidth=1.0, linestyle="--")
        ax.set_title(regime.capitalize(), fontsize=11, fontweight="bold")
        ax.set_xlabel("Budget")
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Mean objective gap (RL - best baseline)\naveraged over snapshots")
    axes[0].legend(frameon=False, loc="best")
    fig.suptitle("Experiment Ladder: Objective Gap vs Budget", fontsize=13, fontweight="bold")
    fig.savefig(out_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def make_regime_summary_figure(rows: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    regimes = ["spread", "dynamic", "spatial"]
    rl_means = []
    base_means = []
    deltas = []
    for regime in regimes:
        grp = [r for r in rows if str(r["regime"]) == regime]
        rl_means.append(float(np.mean([float(r["rl_objective_mean"]) for r in grp])))
        base_means.append(float(np.mean([float(r["baseline_objective_mean"]) for r in grp])))
        deltas.append(float(np.mean([float(r["delta_objective_mean"]) for r in grp])))

    x = np.arange(len(regimes))
    width = 0.34
    fig, ax = plt.subplots(figsize=(7.5, 4.5), constrained_layout=True)
    ax.bar(x - width / 2, rl_means, width=width, label="RL", color="#1f77b4")
    ax.bar(x + width / 2, base_means, width=width, label="Best baseline", color="#ff7f0e")
    ax.set_xticks(x)
    ax.set_xticklabels([r.capitalize() for r in regimes])
    ax.set_ylabel("Mean raw objective across all ladder cells")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, loc="upper left")
    for xi, delta in zip(x, deltas):
        ax.text(
            xi,
            max(rl_means[xi], base_means[xi]) * 1.01,
            f"Δ={delta:.1f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_title("Experiment Ladder: Regime-Level Summary", fontsize=13, fontweight="bold")
    fig.savefig(out_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create figures for the experiment ladder run.")
    parser.add_argument("--ladder-json", default="artifacts/experiment_ladder_start_light.json")
    parser.add_argument("--outdir", default="artifacts/experiment_ladder_figures")
    args = parser.parse_args()

    meta, rows = load_ladder_summary(args.ladder_json)
    outdir = Path(args.outdir)
    snapshots = [int(s) for s in meta["snapshots"]]

    for snapshot in snapshots:
        make_snapshot_heatmap_figure(rows, snapshot, outdir / f"fig_ladder_delta_heatmap_snap{snapshot}.png")
    make_budget_delta_figure(rows, outdir / "fig_ladder_delta_vs_budget.png")
    make_regime_summary_figure(rows, outdir / "fig_ladder_regime_summary.png")

    print(f"Saved figures in {outdir}")


if __name__ == "__main__":
    main()
