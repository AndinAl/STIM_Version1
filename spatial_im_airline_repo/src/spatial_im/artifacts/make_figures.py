from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from spatial_im.artifacts.common import PaperArtifactPaths, load_main_results, load_transfer_results


REGIME_TITLES = {
    "spread": "Spread Regime",
    "dynamic": "Dynamic Regime",
    "spatial": "Spatial Regime",
}


def _setup_axes(title: str, ylabel: str | None = None):
    ax = plt.gca()
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.grid(axis="y", alpha=0.25, linewidth=0.8)
    ax.set_axisbelow(True)
    if ylabel:
        ax.set_ylabel(ylabel)
    return ax


def make_main_objective_figure(main_results, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), constrained_layout=True)
    colors = ["#1f77b4", "#ff7f0e"]
    for ax, regime in zip(axes, ("spread", "dynamic", "spatial")):
        res = main_results[regime]
        values = [res.rl_objective, res.baseline_objective]
        labels = ["RL", "Best baseline"]
        bars = ax.bar(labels, values, color=colors, width=0.58)
        ax.set_title(REGIME_TITLES[regime], fontsize=11, fontweight="bold")
        ax.set_ylabel("Raw final objective")
        ax.grid(axis="y", alpha=0.25, linewidth=0.8)
        ax.set_axisbelow(True)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, val, f"{val:.3f}", ha="center", va="bottom", fontsize=9)
        ax.text(
            0.5,
            0.92,
            f"Δ RL-baseline = {res.delta_objective:+.3f}",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "#f6f6f6", "edgecolor": "#cccccc"},
        )
    fig.suptitle("Main Raw Monte Carlo Comparison", fontsize=13, fontweight="bold")
    fig.savefig(out_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def make_decomposition_figure(main_results, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    metric_names = ["Mass", "Coverage", "Cost"]
    for ax, regime in zip(axes, ("spread", "dynamic", "spatial")):
        res = main_results[regime]
        x = np.arange(len(metric_names))
        width = 0.34
        rl_vals = [res.rl_mass, res.rl_coverage, res.rl_cost]
        base_vals = [res.baseline_mass, res.baseline_coverage, res.baseline_cost]
        ax.bar(x - width / 2, rl_vals, width=width, label="RL", color="#1f77b4")
        ax.bar(x + width / 2, base_vals, width=width, label="Best baseline", color="#ff7f0e")
        ax.set_xticks(x)
        ax.set_xticklabels(metric_names)
        ax.set_title(REGIME_TITLES[regime], fontsize=11, fontweight="bold")
        ax.grid(axis="y", alpha=0.25, linewidth=0.8)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Metric value")
    axes[0].legend(frameon=False, loc="upper left")
    fig.suptitle("Metric Decomposition Under Raw Evaluation", fontsize=13, fontweight="bold")
    fig.savefig(out_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def make_transfer_objective_figure(transfer_results, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4), constrained_layout=True)
    labels = ["Zero-shot", "Warm-start", "Scratch", "Best baseline"]
    colors = ["#2ca02c", "#17becf", "#7f7f7f", "#d62728"]
    for ax, regime in zip(axes, ("spread", "dynamic", "spatial")):
        res = transfer_results[regime]
        values = [
            res.zero_shot_raw_objective,
            res.warm_start_raw_objective,
            res.scratch_raw_objective,
            res.baseline_raw_objective,
        ]
        bars = ax.bar(labels, values, color=colors, width=0.66)
        ax.set_title(REGIME_TITLES[regime], fontsize=11, fontweight="bold")
        ax.set_ylabel("Raw final objective")
        ax.grid(axis="y", alpha=0.25, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(axis="x", rotation=18)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, val, f"{val:.3f}", ha="center", va="bottom", fontsize=8)
        ax.text(
            0.5,
            0.96,
            f"ZS/SC={res.raw_transfer_ratio_zero_shot_vs_scratch:.2f} | WS/SC={res.raw_transfer_ratio_warmstart_vs_scratch:.2f}",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=8.5,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "#f6f6f6", "edgecolor": "#cccccc"},
        )
    fig.suptitle("Homogeneous-Target Transfer: Raw Final Objective", fontsize=13, fontweight="bold")
    fig.savefig(out_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def write_manifest(paths: PaperArtifactPaths, main_path: Path, transfer_path: Path) -> None:
    payload = {
        "inputs": {
            "main_results_json": str(main_path),
            "transfer_results_json": str(transfer_path),
        },
        "figures": {
            "main_objective": str(paths.main_objective_figure),
            "metric_decomposition": str(paths.decomposition_figure),
            "transfer_objectives": str(paths.transfer_figure),
        },
    }
    paths.manifest_json.parent.mkdir(parents=True, exist_ok=True)
    paths.manifest_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the minimal paper figure set from current JSON artifacts.")
    parser.add_argument("--main-results", default="artifacts/compare_regimes_current_tuned.json")
    parser.add_argument("--transfer-results", default="artifacts/transfer_homogeneous_tuned.json")
    parser.add_argument("--outdir", default="artifacts/paper_minimal")
    args = parser.parse_args()

    main_path = Path(args.main_results)
    transfer_path = Path(args.transfer_results)
    paths = PaperArtifactPaths(Path(args.outdir))

    main_results = load_main_results(main_path)
    transfer_results = load_transfer_results(transfer_path)

    make_main_objective_figure(main_results, paths.main_objective_figure)
    make_decomposition_figure(main_results, paths.decomposition_figure)
    make_transfer_objective_figure(transfer_results, paths.transfer_figure)
    write_manifest(paths, main_path, transfer_path)

    print("Saved:", paths.main_objective_figure)
    print("Saved:", paths.decomposition_figure)
    print("Saved:", paths.transfer_figure)
    print("Saved:", paths.manifest_json)


if __name__ == "__main__":
    main()
