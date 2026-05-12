from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from spatial_im.data.airline import load_airline_tables
from spatial_im.data.graph_build import build_airline_graph
from spatial_im.data.temporal import generate_synthetic_temporal_weights
from spatial_im.utils.io import load_yaml


def compute_node_out_strength(edge_index: np.ndarray, weights: np.ndarray, num_nodes: int) -> np.ndarray:
    # weights: [T, E], returns [T, N]
    out_strength = np.zeros((weights.shape[0], num_nodes), dtype=np.float32)
    src = edge_index[:, 0]
    for e in range(weights.shape[1]):
        out_strength[:, src[e]] += weights[:, e]
    return out_strength


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot temporal edge-weight dynamics for top-k nodes.")
    parser.add_argument("--config", default="configs/airline_synth.yaml")
    parser.add_argument("--airports-csv", required=True)
    parser.add_argument("--routes-csv", required=True)
    parser.add_argument("--snapshots", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--out-figure", required=True)
    parser.add_argument("--out-summary", default=None)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    tables = load_airline_tables(args.airports_csv, args.routes_csv)
    graph = build_airline_graph(tables)

    weights = generate_synthetic_temporal_weights(
        airline_graph=graph,
        snapshots=int(args.snapshots),
        seasonal_strength=float(cfg["data"]["seasonal_strength"]),
        noise_std=float(cfg["data"]["noise_std"]),
        seed=int(args.seed),
    )
    out_strength = compute_node_out_strength(graph.edge_index, weights, len(graph.node_ids))

    mean_strength = out_strength.mean(axis=0)
    top_k = max(1, min(int(args.top_k), len(graph.node_ids)))
    top_idx = np.argsort(-mean_strength)[:top_k]

    x = np.arange(int(args.snapshots))
    out_path = Path(args.out_figure)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True, sharex=True)
    cmap = plt.get_cmap("tab10")

    summary_rows = []
    for i, node_idx in enumerate(top_idx):
        series = out_strength[:, node_idx]
        iata = graph.graph.nodes[int(node_idx)].get("iata", f"idx{int(node_idx)}")
        color = cmap(i % 10)

        axes[0].plot(x, series, linewidth=1.8, color=color, label=iata)

        baseline = float(series.mean()) if float(series.mean()) != 0.0 else 1.0
        rel = (series - baseline) / baseline * 100.0
        axes[1].plot(x, rel, linewidth=1.6, color=color, label=iata)

        summary_rows.append(
            {
                "node_idx": int(node_idx),
                "iata": str(iata),
                "mean_out_strength": float(series.mean()),
                "min_out_strength": float(series.min()),
                "max_out_strength": float(series.max()),
                "std_out_strength": float(series.std()),
                "relative_swing_pct": float((series.max() - series.min()) / max(baseline, 1e-8) * 100.0),
            }
        )

    axes[0].set_title("Top-10 Nodes: Outgoing Edge-Weight Strength Over Time", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Outgoing weight sum")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].set_axisbelow(True)

    axes[1].set_title("Relative Change vs Node Mean (%)", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Snapshot")
    axes[1].set_ylabel("Change (%)")
    axes[1].axhline(0.0, color="#666666", linewidth=1.0, linestyle="--")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].set_axisbelow(True)

    axes[0].legend(ncol=2, frameon=False, fontsize=8.5)
    fig.savefig(out_path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", out_path)

    if args.out_summary:
        summary_path = Path(args.out_summary)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "airports_csv": str(args.airports_csv),
            "routes_csv": str(args.routes_csv),
            "snapshots": int(args.snapshots),
            "seed": int(args.seed),
            "top_k": int(top_k),
            "rows": summary_rows,
        }
        summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print("Saved:", summary_path)


if __name__ == "__main__":
    main()
