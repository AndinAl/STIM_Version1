from __future__ import annotations

import argparse
import json
from pathlib import Path

from spatial_im.data.airline import load_airline_tables
from spatial_im.data.synthetic_airline import generate_homogeneous_airline_tables
from spatial_im.utils.io import load_yaml


def _parse_int_list(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def _parse_float_list(raw: str) -> list[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def _write_tables(tables, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    airports_path = out_dir / "airports.csv"
    routes_path = out_dir / "routes.csv"
    tables.airports.to_csv(airports_path, index=False)
    tables.routes.to_csv(routes_path, index=False)
    return {
        "airports_csv": str(airports_path),
        "routes_csv": str(routes_path),
        "num_airports": int(len(tables.airports)),
        "num_routes": int(len(tables.routes)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate topological synthetic airline graph families "
            "(source + target) for transfer experiments."
        )
    )
    parser.add_argument("--config", default="configs/airline_synth.yaml")
    parser.add_argument("--source-sizes", default="100,200")
    parser.add_argument("--source-graphs-per-size", type=int, default=3)
    parser.add_argument("--source-avg-degrees", default="", help="Optional avg out-degree per source size.")
    parser.add_argument("--source-snapshots", default="120,220", help="Snapshots per source size.")
    parser.add_argument("--target-sizes", default="300,500")
    parser.add_argument("--target-avg-degrees", default="", help="Optional avg out-degree per target size.")
    parser.add_argument("--target-snapshots", default="220", help="Snapshots per target size.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out-dir", default="data/homogeneous_family_topological")
    parser.add_argument("--manifest-out", default="artifacts/homogeneous_family_topological_manifest.json")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    base_tables = load_airline_tables(cfg["data"]["airports_csv"], cfg["data"]["routes_csv"])
    source_sizes = _parse_int_list(args.source_sizes)
    target_sizes = _parse_int_list(args.target_sizes)
    source_avg_degrees = _parse_float_list(args.source_avg_degrees) if args.source_avg_degrees else []
    target_avg_degrees = _parse_float_list(args.target_avg_degrees) if args.target_avg_degrees else []
    source_snapshots = _parse_int_list(args.source_snapshots)
    target_snapshots = _parse_int_list(args.target_snapshots)

    source_size_to_avg = {
        int(size): float(source_avg_degrees[min(idx, len(source_avg_degrees) - 1)])
        for idx, size in enumerate(source_sizes)
    } if source_avg_degrees else {}
    target_size_to_avg = {
        int(size): float(target_avg_degrees[min(idx, len(target_avg_degrees) - 1)])
        for idx, size in enumerate(target_sizes)
    } if target_avg_degrees else {}
    source_size_to_snapshots = {
        int(size): int(source_snapshots[min(idx, len(source_snapshots) - 1)])
        for idx, size in enumerate(source_sizes)
    }
    target_size_to_snapshots = {
        int(size): int(target_snapshots[min(idx, len(target_snapshots) - 1)])
        for idx, size in enumerate(target_sizes)
    }

    out_root = Path(args.out_dir)
    manifest = {
        "generator": "generate_airline_topological_synthetic.py",
        "source": [],
        "target": [],
        "seed": int(args.seed),
        "source_sizes": source_sizes,
        "source_graphs_per_size": int(args.source_graphs_per_size),
        "source_avg_degrees": source_size_to_avg,
        "source_snapshots": source_size_to_snapshots,
        "target_sizes": target_sizes,
        "target_avg_degrees": target_size_to_avg,
        "target_snapshots": target_size_to_snapshots,
    }

    seed_cursor = int(args.seed)
    for size in source_sizes:
        for graph_idx in range(int(args.source_graphs_per_size)):
            tables = generate_homogeneous_airline_tables(
                base_tables=base_tables,
                num_nodes=int(size),
                seed=seed_cursor,
                avg_out_degree=source_size_to_avg.get(int(size)),
            )
            out_dir = out_root / "source" / f"size_{size}" / f"graph_{graph_idx}"
            info = _write_tables(tables, out_dir)
            info.update(
                {
                    "name": f"source_{int(size)}_g{int(graph_idx)}",
                    "size": int(size),
                    "graph_idx": int(graph_idx),
                    "seed": int(seed_cursor),
                    "avg_degree": float(source_size_to_avg.get(int(size), 0.0)),
                    "snapshots": int(source_size_to_snapshots[int(size)]),
                }
            )
            manifest["source"].append(info)
            seed_cursor += 1

    for size in target_sizes:
        tables = generate_homogeneous_airline_tables(
            base_tables=base_tables,
            num_nodes=int(size),
            seed=seed_cursor,
            avg_out_degree=target_size_to_avg.get(int(size)),
        )
        out_dir = out_root / "target" / f"size_{size}"
        info = _write_tables(tables, out_dir)
        info.update(
            {
                "name": f"target_{int(size)}",
                "size": int(size),
                "seed": int(seed_cursor),
                "avg_degree": float(target_size_to_avg.get(int(size), 0.0)),
                "snapshots": int(target_size_to_snapshots[int(size)]),
            }
        )
        manifest["target"].append(info)
        seed_cursor += 1

    manifest_path = Path(args.manifest_out)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    summary = {
        "manifest": str(manifest_path),
        "out_dir": str(out_root),
        "source_graphs": len(manifest["source"]),
        "targets": len(manifest["target"]),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
