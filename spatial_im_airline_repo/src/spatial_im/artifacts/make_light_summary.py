from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REGIMES = ("spread", "dynamic", "spatial")


def _fmt(x: float) -> str:
    return f"{x:.3f}"


def load_winners(artifacts_dir: Path) -> list[dict]:
    winners: list[dict] = []
    for regime in REGIMES:
        path = artifacts_dir / f"rerank_light_{regime}_source_subset.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        winner = payload["winner"]
        winners.append(
            {
                "regime": regime,
                "trial_number": int(winner["trial_number"]),
                "rank_by_optuna": int(winner["rank_by_optuna"]),
                "optuna_value": float(winner["optuna_value"]),
                "mean_raw_objective": float(winner["mean_raw_objective"]),
                "mean_delta_objective": float(winner["mean_delta_objective"]),
                "win_rate": float(winner["win_rate"]),
            }
        )
    return winners


def build_winner_md(rows: list[dict]) -> str:
    lines = [
        "| Regime | Winner trial | Optuna rank | Optuna value | Mean raw objective | Mean delta objective | Win rate |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| `{r['regime']}` | {r['trial_number']} | {r['rank_by_optuna']} | {_fmt(r['optuna_value'])} | "
            f"{_fmt(r['mean_raw_objective'])} | {_fmt(r['mean_delta_objective'])} | {_fmt(r['win_rate'])} |"
        )
    return "\n".join(lines)


def build_winner_tex(rows: list[dict]) -> str:
    lines = [
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Regime & Winner trial & Optuna rank & Optuna value & Mean raw obj. & Mean delta obj. & Win rate \\",
        r"\midrule",
    ]
    for r in rows:
        lines.append(
            f"{r['regime']} & {r['trial_number']} & {r['rank_by_optuna']} & {_fmt(r['optuna_value'])} & "
            f"{_fmt(r['mean_raw_objective'])} & {_fmt(r['mean_delta_objective'])} & {_fmt(r['win_rate'])} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines)


def make_winner_figure(rows: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    x = np.arange(len(rows))
    labels = [r["regime"].capitalize() for r in rows]
    mean_raw = [r["mean_raw_objective"] for r in rows]
    mean_delta = [r["mean_delta_objective"] for r in rows]
    win_rate = [r["win_rate"] for r in rows]

    fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.4), constrained_layout=True)

    axes[0].bar(x, mean_raw, color="#1f77b4")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].set_title("Winner Mean Raw Objective", fontsize=10.5, fontweight="bold")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].set_axisbelow(True)

    axes[1].bar(x, mean_delta, color="#ff7f0e")
    axes[1].axhline(0.0, color="#666666", linewidth=1.0, linestyle="--")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].set_title("Winner Mean Delta vs Baseline", fontsize=10.5, fontweight="bold")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].set_axisbelow(True)

    axes[2].bar(x, win_rate, color="#2ca02c")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels)
    axes[2].set_ylim(0.0, 1.0)
    axes[2].set_title("Winner Win Rate", fontsize=10.5, fontweight="bold")
    axes[2].grid(axis="y", alpha=0.25)
    axes[2].set_axisbelow(True)

    fig.suptitle("Light Protocol: Source-Side Rerank Winners", fontsize=13, fontweight="bold")
    fig.savefig(out_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build light-protocol winner summary figure and tables.")
    parser.add_argument("--artifacts-dir", default="artifacts")
    parser.add_argument("--outdir", default="artifacts/paper_light")
    args = parser.parse_args()

    artifacts_dir = Path(args.artifacts_dir)
    outdir = Path(args.outdir)
    tables_dir = outdir / "tables"
    figures_dir = outdir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    winners = load_winners(artifacts_dir)

    md_path = tables_dir / "table_light_rerank_winners.md"
    tex_path = tables_dir / "table_light_rerank_winners.tex"
    json_path = tables_dir / "table_light_rerank_winners.json"
    fig_path = figures_dir / "fig_light_rerank_winners.png"

    md_path.write_text(build_winner_md(winners), encoding="utf-8")
    tex_path.write_text(build_winner_tex(winners), encoding="utf-8")
    json_path.write_text(json.dumps({"rows": winners}, indent=2), encoding="utf-8")
    make_winner_figure(winners, fig_path)

    print("Saved:", md_path)
    print("Saved:", tex_path)
    print("Saved:", json_path)
    print("Saved:", fig_path)


if __name__ == "__main__":
    main()
