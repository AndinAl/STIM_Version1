from __future__ import annotations

import argparse
from pathlib import Path

from spatial_im.artifacts.common import PaperArtifactPaths, load_main_results, load_transfer_results


def _fmt(x: float) -> str:
    return f"{x:.3f}"


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def build_main_results_markdown(main_results) -> str:
    lines = [
        "| Regime | RL seeds | RL objective | RL mass | RL coverage | RL cost | Best baseline | Baseline seeds | Baseline objective | Delta RL-baseline |",
        "|---|---|---:|---:|---:|---:|---|---|---:|---:|",
    ]
    for regime in ("spread", "dynamic", "spatial"):
        res = main_results[regime]
        lines.append(
            f"| `{regime}` | {res.rl_seeds} | {_fmt(res.rl_objective)} | {_fmt(res.rl_mass)} | {_fmt(res.rl_coverage)} | {_fmt(res.rl_cost)} | "
            f"`{res.best_baseline_name}` | {res.baseline_seeds} | {_fmt(res.baseline_objective)} | {_fmt(res.delta_objective)} |"
        )
    return "\n".join(lines)


def build_main_results_latex(main_results) -> str:
    lines = [
        r"\begin{tabular}{llrrrrlr}",
        r"\toprule",
        r"Regime & RL seeds & RL obj. & RL mass & RL cov. & RL cost & Best baseline & Base obj. \\",
        r"\midrule",
    ]
    for regime in ("spread", "dynamic", "spatial"):
        res = main_results[regime]
        lines.append(
            f"{regime} & {res.rl_seeds} & {_fmt(res.rl_objective)} & {_fmt(res.rl_mass)} & {_fmt(res.rl_coverage)} & "
            f"{_fmt(res.rl_cost)} & {res.best_baseline_name} & {_fmt(res.baseline_objective)} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines)


def build_transfer_markdown(transfer_results) -> str:
    lines = [
        "| Regime | Zero-shot | Warm-start | Scratch | Best baseline | Raw ZS/SC | Raw WS/SC | Curve TR |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for regime in ("spread", "dynamic", "spatial"):
        res = transfer_results[regime]
        lines.append(
            f"| `{regime}` | {_fmt(res.zero_shot_raw_objective)} | {_fmt(res.warm_start_raw_objective)} | {_fmt(res.scratch_raw_objective)} | "
            f"{_fmt(res.baseline_raw_objective)} | {_fmt(res.raw_transfer_ratio_zero_shot_vs_scratch)} | "
            f"{_fmt(res.raw_transfer_ratio_warmstart_vs_scratch)} | {_fmt(res.curve_transfer_ratio)} |"
        )
    return "\n".join(lines)


def build_transfer_latex(transfer_results) -> str:
    lines = [
        r"\begin{tabular}{lrrrrrrr}",
        r"\toprule",
        r"Regime & Zero-shot & Warm-start & Scratch & Best baseline & Raw ZS/SC & Raw WS/SC & Curve TR \\",
        r"\midrule",
    ]
    for regime in ("spread", "dynamic", "spatial"):
        res = transfer_results[regime]
        lines.append(
            f"{regime} & {_fmt(res.zero_shot_raw_objective)} & {_fmt(res.warm_start_raw_objective)} & {_fmt(res.scratch_raw_objective)} & "
            f"{_fmt(res.baseline_raw_objective)} & {_fmt(res.raw_transfer_ratio_zero_shot_vs_scratch)} & "
            f"{_fmt(res.raw_transfer_ratio_warmstart_vs_scratch)} & {_fmt(res.curve_transfer_ratio)} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the minimal paper tables from current JSON artifacts.")
    parser.add_argument("--main-results", default="artifacts/compare_regimes_current_tuned.json")
    parser.add_argument("--transfer-results", default="artifacts/transfer_homogeneous_tuned.json")
    parser.add_argument("--outdir", default="artifacts/paper_minimal")
    args = parser.parse_args()

    main_results = load_main_results(args.main_results)
    transfer_results = load_transfer_results(args.transfer_results)
    paths = PaperArtifactPaths(Path(args.outdir))

    _write(paths.main_results_md, build_main_results_markdown(main_results))
    _write(paths.main_results_tex, build_main_results_latex(main_results))
    _write(paths.transfer_results_md, build_transfer_markdown(transfer_results))
    _write(paths.transfer_results_tex, build_transfer_latex(transfer_results))

    print("Saved:", paths.main_results_md)
    print("Saved:", paths.main_results_tex)
    print("Saved:", paths.transfer_results_md)
    print("Saved:", paths.transfer_results_tex)


if __name__ == "__main__":
    main()
