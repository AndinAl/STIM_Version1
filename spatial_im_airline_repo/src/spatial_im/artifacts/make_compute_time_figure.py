from __future__ import annotations

import argparse
import json
import re
import sqlite3
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REGIMES = ("spread", "dynamic", "spatial")


@dataclass(frozen=True)
class RegimeTiming:
    regime: str
    tuning_seconds: float
    avg_trial_seconds: float
    rerank_seconds: float


def _study_timing_seconds(db_path: Path) -> tuple[float, float]:
    if not db_path.exists():
        return 0.0, 0.0
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    rows = cur.execute(
        "select julianday(datetime_start), julianday(datetime_complete) "
        "from trials where state='COMPLETE' and datetime_start is not null and datetime_complete is not null"
    ).fetchall()
    con.close()
    if not rows:
        return 0.0, 0.0
    starts = [r[0] for r in rows]
    ends = [r[1] for r in rows]
    durations = [(ed - st) * 86400.0 for st, ed in rows]
    total = max(0.0, (max(ends) - min(starts)) * 86400.0)
    avg_trial = float(np.mean(durations)) if durations else 0.0
    return total, avg_trial


def _log_duration_seconds(path: Path) -> float:
    if not path.exists():
        return 0.0
    # On Linux, ctime is inode-change time and can equal mtime; prefer birth time from `stat`.
    try:
        out = subprocess.check_output(["stat", "-c", "%W %Y", str(path)], text=True).strip()
        born_raw, mtime_raw = out.split()
        born = int(born_raw)
        mtime = int(mtime_raw)
        if born > 0:
            return max(0.0, float(mtime - born))
    except Exception:
        pass
    st = path.stat()
    return max(0.0, float(st.st_mtime - st.st_ctime))


def load_regime_timings(
    optuna_dir: Path,
    log_dir: Path,
    study_prefix: str,
    rerank_prefix: str,
) -> list[RegimeTiming]:
    timings: list[RegimeTiming] = []
    for regime in REGIMES:
        tune_db = optuna_dir / f"{study_prefix}_{regime}.db"
        tune_total, tune_avg = _study_timing_seconds(tune_db)
        rerank_log = log_dir / f"{rerank_prefix}_{regime}.log"
        rerank_total = _log_duration_seconds(rerank_log)
        timings.append(
            RegimeTiming(
                regime=regime,
                tuning_seconds=float(tune_total),
                avg_trial_seconds=float(tune_avg),
                rerank_seconds=float(rerank_total),
            )
        )
    return timings


def load_transfer_timings(log_dir: Path, transfer_glob: str, name_filter: str) -> list[tuple[str, float]]:
    rows: list[tuple[str, float]] = []
    pattern = re.compile(r"transfer_target(\d+)_budget(\d+)\.log$")
    for path in sorted(log_dir.glob(transfer_glob)):
        if name_filter and name_filter not in path.name:
            continue
        m = pattern.search(path.name)
        if not m:
            continue
        target, budget = m.group(1), m.group(2)
        label = f"N={target}, k={budget}"
        rows.append((label, _log_duration_seconds(path)))
    return rows


def make_figure(regime_timings: list[RegimeTiming], transfer_timings: list[tuple[str, float]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), constrained_layout=True)

    # Left panel: tuning vs rerank per regime.
    ax = axes[0]
    x = np.arange(len(regime_timings))
    width = 0.35
    tune_hours = np.asarray([t.tuning_seconds / 3600.0 for t in regime_timings], dtype=np.float32)
    rerank_hours = np.asarray([t.rerank_seconds / 3600.0 for t in regime_timings], dtype=np.float32)
    ax.bar(x - width / 2, tune_hours, width=width, label="Optuna tuning", color="#1f77b4")
    ax.bar(x + width / 2, rerank_hours, width=width, label="Top-5 raw rerank", color="#ff7f0e")
    ax.set_xticks(x)
    ax.set_xticklabels([t.regime.capitalize() for t in regime_timings])
    ax.set_ylabel("Wall-clock time (hours)")
    ax.set_title("Tuning And Reranking Time By Regime", fontsize=11, fontweight="bold")
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper left")

    # Right panel: transfer runtime by target/budget run.
    ax = axes[1]
    if transfer_timings:
        labels = [l for l, _ in transfer_timings]
        mins = np.asarray([v / 60.0 for _, v in transfer_timings], dtype=np.float32)
        ax.bar(np.arange(len(labels)), mins, color="#2ca02c")
        ax.set_xticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=30, ha="right")
        for i, val in enumerate(mins):
            ax.text(i, val, f"{val:.1f}", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("Wall-clock time (minutes)")
    ax.set_title("Transfer Runtime By Target Cell", fontsize=11, fontweight="bold")
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)

    fig.suptitle("Computational Time Comparison", fontsize=13, fontweight="bold")
    fig.savefig(out_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create computational-time comparison figure.")
    parser.add_argument("--optuna-dir", default="artifacts/optuna")
    parser.add_argument("--log-dir", default="artifacts/logs")
    parser.add_argument("--outdir", default="artifacts/paper_minimal")
    parser.add_argument("--study-prefix", default="ladder_subset")
    parser.add_argument("--rerank-prefix", default="rerank")
    parser.add_argument("--transfer-glob", default="transfer_target*_budget*.log")
    parser.add_argument("--name-filter", default="", help="Optional substring filter for transfer logs.")
    parser.add_argument("--outfile", default="fig_compute_time_comparison.png")
    args = parser.parse_args()

    optuna_dir = Path(args.optuna_dir)
    log_dir = Path(args.log_dir)
    outdir = Path(args.outdir)

    regime_timings = load_regime_timings(
        optuna_dir,
        log_dir,
        study_prefix=str(args.study_prefix),
        rerank_prefix=str(args.rerank_prefix),
    )
    transfer_timings = load_transfer_timings(
        log_dir,
        transfer_glob=str(args.transfer_glob),
        name_filter=str(args.name_filter),
    )

    fig_path = outdir / "figures" / str(args.outfile)
    make_figure(regime_timings, transfer_timings, fig_path)

    summary = {
        "regime_timings": [asdict(t) for t in regime_timings],
        "transfer_timings": [{"label": label, "seconds": sec} for label, sec in transfer_timings],
    }
    summary_path = outdir / "compute_time_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Saved:", fig_path)
    print("Saved:", summary_path)


if __name__ == "__main__":
    main()
