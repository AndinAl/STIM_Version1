from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


REGIMES = ("spread", "dynamic", "spatial")
MODE_MAP = (
    ("zero_shot", "Zero-shot"),
    ("transfer_adapt", "Warm-start"),
    ("scratch_adapt", "Scratch"),
)


def _parse_target_budget(path: Path) -> tuple[int, int]:
    match = re.search(r"transfer_target(\d+)_budget(\d+)\.json$", path.name)
    if not match:
        raise ValueError(f"Unexpected transfer filename: {path}")
    return int(match.group(1)), int(match.group(2))


def _fmt(x: float) -> str:
    return f"{x:.3f}"


def load_rows(transfer_dir: Path) -> list[dict]:
    rows: list[dict] = []
    for path in sorted(transfer_dir.glob("transfer_target*_budget*.json")):
        target, budget = _parse_target_budget(path)
        payload = json.loads(path.read_text(encoding="utf-8"))["regimes"]
        for regime in REGIMES:
            entry = payload[regime]
            for mode, mode_label in MODE_MAP:
                rows.append(
                    {
                        "target_size": target,
                        "budget": budget,
                        "regime": regime,
                        "mode": mode_label,
                        "raw_objective": float(entry[mode]["raw_objective"]),
                        "best_baseline_objective": float(entry["best_baseline"]["objective"]),
                        "gap_vs_baseline": float(entry[mode]["raw_objective"]) - float(entry["best_baseline"]["objective"]),
                    }
                )
    rows.sort(key=lambda r: (r["target_size"], r["budget"], r["regime"], r["mode"]))
    return rows


def build_markdown(rows: list[dict]) -> str:
    lines = [
        "| Target size | Budget k | Regime | Mode | Raw objective | Best baseline objective | Gap vs baseline |",
        "|---:|---:|---|---|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['target_size']} | {r['budget']} | `{r['regime']}` | {r['mode']} | "
            f"{_fmt(r['raw_objective'])} | {_fmt(r['best_baseline_objective'])} | {_fmt(r['gap_vs_baseline'])} |"
        )
    return "\n".join(lines)


def build_latex(rows: list[dict]) -> str:
    lines = [
        r"\begin{tabular}{rrllrrr}",
        r"\toprule",
        r"Target & k & Regime & Mode & Raw obj. & Best baseline & Gap \\",
        r"\midrule",
    ]
    for r in rows:
        lines.append(
            f"{r['target_size']} & {r['budget']} & {r['regime']} & {r['mode']} & "
            f"{_fmt(r['raw_objective'])} & {_fmt(r['best_baseline_objective'])} & {_fmt(r['gap_vs_baseline'])} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export corrected transfer budget-sweep tables (md/tex).")
    parser.add_argument("--transfer-dir", default="artifacts")
    parser.add_argument("--outdir", default="artifacts/paper_minimal/tables")
    args = parser.parse_args()

    transfer_dir = Path(args.transfer_dir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    rows = load_rows(transfer_dir)

    md_path = outdir / "table_transfer_budget_sweep.md"
    tex_path = outdir / "table_transfer_budget_sweep.tex"
    json_path = outdir / "table_transfer_budget_sweep.json"

    md_path.write_text(build_markdown(rows), encoding="utf-8")
    tex_path.write_text(build_latex(rows), encoding="utf-8")
    json_path.write_text(json.dumps({"rows": rows}, indent=2), encoding="utf-8")

    print("Saved:", md_path)
    print("Saved:", tex_path)
    print("Saved:", json_path)


if __name__ == "__main__":
    main()
