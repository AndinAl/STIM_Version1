from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from spatial_im.data.airline import load_airline_tables
from spatial_im.data.synthetic_airline import generate_homogeneous_airline_tables
from spatial_im.utils.io import load_yaml


def parse_int_list(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def write_tables(tables, out_dir: Path) -> dict:
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
    parser = argparse.ArgumentParser(description="Generate homogeneous synthetic airline CSV families for source/target transfer experiments.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--source-sizes", default="100,200")
    parser.add_argument("--source-graphs-per-size", type=int, default=3)
    parser.add_argument("--target-sizes", default="300,500")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out-dir", default="data/homogeneous_family")
    parser.add_argument("--manifest-out", default="artifacts/homogeneous_family_manifest.json")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    base_tables = load_airline_tables(cfg["data"]["airports_csv"], cfg["data"]["routes_csv"])
    source_sizes = parse_int_list(args.source_sizes)
    target_sizes = parse_int_list(args.target_sizes)

    out_root = Path(args.out_dir)
    manifest = {
        "source": [],
        "target": [],
        "seed": int(args.seed),
        "source_sizes": source_sizes,
        "source_graphs_per_size": int(args.source_graphs_per_size),
        "target_sizes": target_sizes,
    }

    seed_cursor = int(args.seed)
    for size in source_sizes:
        for graph_idx in range(int(args.source_graphs_per_size)):
            tables = generate_homogeneous_airline_tables(
                base_tables=base_tables,
                num_nodes=int(size),
                seed=seed_cursor,
            )
            out_dir = out_root / "source" / f"size_{size}" / f"graph_{graph_idx}"
            info = write_tables(tables, out_dir)
            info.update({"size": int(size), "graph_idx": int(graph_idx), "seed": int(seed_cursor)})
            manifest["source"].append(info)
            seed_cursor += 1

    for size in target_sizes:
        tables = generate_homogeneous_airline_tables(
            base_tables=base_tables,
            num_nodes=int(size),
            seed=seed_cursor,
        )
        out_dir = out_root / "target" / f"size_{size}"
        info = write_tables(tables, out_dir)
        info.update({"size": int(size), "seed": int(seed_cursor)})
        manifest["target"].append(info)
        seed_cursor += 1

    manifest_path = Path(args.manifest_out)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"manifest": str(manifest_path), "source_graphs": len(manifest["source"]), "targets": len(manifest["target"])}, indent=2))


if __name__ == "__main__":
    main()
