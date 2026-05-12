#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$ROOT_DIR/.venv/bin/python"
LOG_PATH="$ROOT_DIR/artifacts/logs/eta_hourly.log"
INTERVAL_SEC="${1:-3600}"

mkdir -p "$ROOT_DIR/artifacts/logs"

if [[ ! -x "$PY" ]]; then
  echo "Missing python at $PY"
  exit 1
fi

while true; do
  TS="$(date '+%Y-%m-%d %H:%M:%S %Z')"
  {
    echo "============================================================"
    echo "ETA Snapshot @ $TS"
    ROOT_DIR="$ROOT_DIR" "$PY" - <<'PY'
import datetime as dt
import os
import re
import sqlite3
import statistics
import subprocess
from pathlib import Path

root = Path(os.environ["ROOT_DIR"])
now = dt.datetime.now()

def human_sec(sec: float | None) -> str:
    if sec is None:
        return "unknown"
    sec = max(0, int(sec))
    h = sec // 3600
    m = (sec % 3600) // 60
    s = sec % 60
    return f"{h:02d}h {m:02d}m {s:02d}s"

def study_eta(regime: str, target_trials: int):
    db = root / "artifacts" / "optuna" / f"ladder_subset_{regime}.db"
    if not db.exists():
        return {"regime": regime, "status": "missing", "eta_sec": None}
    con = sqlite3.connect(db)
    cur = con.cursor()
    rows = cur.execute(
        "select number, state, julianday(datetime_start), julianday(datetime_complete) from trials order by number"
    ).fetchall()
    con.close()

    completed = 0
    running = 0
    durations = []
    for _, state, st, ed in rows:
        if state == "COMPLETE":
            completed += 1
            if st and ed:
                durations.append((ed - st) * 86400.0)
        elif state == "RUNNING":
            running += 1

    avg_dur = statistics.mean(durations) if durations else None
    remaining = max(int(target_trials) - completed, 0)
    eta_sec = (remaining * avg_dur) if avg_dur is not None else None
    status = "done" if remaining == 0 and running == 0 else "running"
    return {
        "regime": regime,
        "status": status,
        "completed": completed,
        "target": int(target_trials),
        "running": running,
        "avg_trial_sec": avg_dur,
        "eta_sec": eta_sec,
    }

def ps_lines():
    out = subprocess.check_output(["ps", "-eo", "pid,etimes,cmd"], text=True)
    return out.splitlines()

ps = ps_lines()

def find_process(substr: str):
    hits = []
    for line in ps:
        if substr in line:
            m = re.match(r"\s*(\d+)\s+(\d+)\s+(.*)$", line)
            if m:
                hits.append({"pid": int(m.group(1)), "etimes": int(m.group(2)), "cmd": m.group(3)})
    return hits

def rerank_eta(regime: str, top_k: int = 5):
    winner = root / "artifacts" / f"best_config_{regime}_source_subset.json"
    if winner.exists():
        return {"regime": regime, "status": "done", "eta_sec": 0, "done_trials": top_k, "top_k": top_k}

    log = root / "artifacts" / "logs" / f"rerank_{regime}.log"
    done_trials = 0
    if log.exists():
        txt = log.read_text(errors="ignore")
        done_trials = len(re.findall(rf"^\[{re.escape(regime)}\]\s+trial=", txt, flags=re.MULTILINE))

    proc = find_process(f"rerank_optuna_top_trials.py --regime {regime}")
    if not proc:
        return {
            "regime": regime,
            "status": "pending" if done_trials == 0 else "unknown",
            "eta_sec": None,
            "done_trials": done_trials,
            "top_k": top_k,
        }

    etimes = max(p["etimes"] for p in proc)
    if done_trials > 0:
        per_trial = etimes / done_trials
        remaining = max(top_k - done_trials, 0)
        eta_sec = per_trial * remaining
    else:
        eta_sec = None
    return {
        "regime": regime,
        "status": "running",
        "eta_sec": eta_sec,
        "elapsed_sec": etimes,
        "done_trials": done_trials,
        "top_k": top_k,
    }

studies = [
    study_eta("spread", 20),
    study_eta("spatial", 20),
    study_eta("dynamic", 30),
]
reranks = [
    rerank_eta("spread", 5),
    rerank_eta("spatial", 5),
    rerank_eta("dynamic", 5),
]

print("Tuning progress:")
for s in studies:
    if s["status"] == "missing":
        print(f"  - {s['regime']}: missing db")
        continue
    eta_txt = human_sec(s["eta_sec"])
    avg_txt = human_sec(s["avg_trial_sec"])
    print(
        f"  - {s['regime']}: {s['completed']}/{s['target']} complete, running={s['running']}, "
        f"avg_trial={avg_txt}, eta={eta_txt}, status={s['status']}"
    )

print("Rerank progress:")
for r in reranks:
    extra = f"{r.get('done_trials', 0)}/{r.get('top_k', 5)}"
    print(f"  - {r['regime']}: {extra}, status={r['status']}, eta={human_sec(r.get('eta_sec'))}")

active_etas = []
for s in studies:
    if s["status"] == "running" and s["eta_sec"] is not None:
        active_etas.append(float(s["eta_sec"]))
for r in reranks:
    if r["status"] == "running" and r.get("eta_sec") is not None:
        active_etas.append(float(r["eta_sec"]))

if active_etas:
    total = sum(active_etas)
    eta_clock = now + dt.timedelta(seconds=total)
    print(f"Estimated remaining active time: {human_sec(total)}")
    print(f"Estimated finish (active stages): {eta_clock.strftime('%Y-%m-%d %H:%M:%S')}")
else:
    print("Estimated remaining active time: unknown (insufficient data)")

transfer_proc = find_process("evaluate_transfer_homogeneous.py")
if transfer_proc:
    print(f"Transfer stage: running ({len(transfer_proc)} proc)")
else:
    print("Transfer stage: not running yet (or finished)")
PY
    echo
  } | tee -a "$LOG_PATH"
  sleep "$INTERVAL_SEC"
done

