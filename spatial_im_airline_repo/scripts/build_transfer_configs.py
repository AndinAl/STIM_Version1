from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import yaml

from spatial_im.utils.io import load_yaml


def parse_int_list(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build transfer config YAMLs from a homogeneous-family manifest.")
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--budgets", default="5,10,15")
    parser.add_argument("--source-snapshots", default="120,220")
    parser.add_argument("--target-snapshots", default="220")
    parser.add_argument("--out-dir", default="configs/generated_transfer")
    args = parser.parse_args()

    base_cfg = load_yaml(args.base_config)
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))

    budgets = parse_int_list(args.budgets)
    source_snapshots = parse_int_list(args.source_snapshots)
    target_snapshots = parse_int_list(args.target_snapshots)
    source_sizes = sorted({int(s["size"]) for s in manifest.get("source", [])})
    target_sizes = sorted({int(t["size"]) for t in manifest.get("target", [])})
    size_to_snapshot = {}
    for idx, size in enumerate(source_sizes):
        size_to_snapshot[size] = int(source_snapshots[min(idx, len(source_snapshots) - 1)])
    target_size_to_snapshot = {}
    for idx, size in enumerate(target_sizes):
        target_size_to_snapshot[size] = int(target_snapshots[min(idx, len(target_snapshots) - 1)])

    source_specs = []
    for idx, src in enumerate(manifest.get("source", [])):
        src_size = int(src["size"])
        source_specs.append(
            {
                "name": str(src.get("name", f"source_size{src_size}_g{int(src['graph_idx'])}")),
                "airports_csv": src["airports_csv"],
                "routes_csv": src["routes_csv"],
                "snapshots": int(src.get("snapshots", size_to_snapshot[src_size])),
                "seasonal_strength": float(base_cfg["data"]["seasonal_strength"]),
                "noise_std": float(base_cfg["data"]["noise_std"]),
                "seed_offset": int(idx),
            }
        )

    targets = manifest.get("target", [])
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    generated: list[str] = []

    for target in targets:
        target_size = int(target["size"])
        target_snapshot = int(target.get("snapshots", target_size_to_snapshot[target_size]))
        for budget in budgets:
            cfg = copy.deepcopy(base_cfg)
            cfg["rl"]["budget"] = int(budget)
            cfg["data"]["snapshots"] = int(max(source_snapshots + target_snapshots))
            cfg.setdefault("pretraining", {})
            cfg["pretraining"]["source_graphs"] = source_specs
            cfg["pretraining"]["budgets"] = [int(budget)]
            cfg["pretraining"]["regimes"] = ["spread", "dynamic", "spatial"]
            cfg.setdefault("zero_shot_target", {})
            cfg["zero_shot_target"].update(
                {
                    "name": str(target.get("name", f"target_size{target_size}")),
                    "airports_csv": target["airports_csv"],
                    "routes_csv": target["routes_csv"],
                    "snapshots": int(target_snapshot),
                    "seasonal_strength": float(base_cfg["data"]["seasonal_strength"]),
                    "noise_std": float(base_cfg["data"]["noise_std"]),
                    "seed_offset": int(target.get("seed", base_cfg["seed"])),
                }
            )
            out_path = out_dir / f"transfer_target{target_size}_budget{budget}.yaml"
            out_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
            generated.append(str(out_path))

    print(json.dumps({"generated_configs": generated, "count": len(generated)}, indent=2))


if __name__ == "__main__":
    main()
