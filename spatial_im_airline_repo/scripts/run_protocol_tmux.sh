#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$ROOT_DIR/.venv/bin/python"
LOG_DIR="$ROOT_DIR/artifacts/logs"
OUT_DIR="$ROOT_DIR/artifacts"
OPTUNA_DIR="$ROOT_DIR/artifacts/optuna"

mkdir -p "$LOG_DIR" "$OUT_DIR" "$OPTUNA_DIR"

if [[ ! -x "$PY" ]]; then
  echo "Missing virtualenv python at $PY"
  exit 1
fi

run_tmux_session() {
  local session_name="$1"
  local command_body="$2"
  if tmux has-session -t "$session_name" 2>/dev/null; then
    echo "Session already exists: $session_name (skipping)"
    return
  fi
  local script_path="$LOG_DIR/.tmux_${session_name}.sh"
  cat >"$script_path" <<EOF
#!/usr/bin/env bash
$command_body
EOF
  chmod +x "$script_path"
  tmux new-session -d -s "$session_name" "bash \"$script_path\""
  echo "Started tmux session: $session_name"
}

BASE_CMD="
cd '$ROOT_DIR'
set -euo pipefail
export PYTHONPATH='$ROOT_DIR/src:\${PYTHONPATH:-}'
echo '[base] started at ' \$(date)
$PY scripts/tune_ladder_optuna.py \
  --config configs/airline_synth.yaml \
  --regime spread \
  --sizes 100 \
  --snapshots 120 \
  --budgets 5 \
  --repeats 1 \
  --trials 4 \
  --study-name base_spread \
  --storage sqlite:///artifacts/optuna/base_spread.db \
  --out artifacts/optuna_spread_base.json \
  2>&1 | tee artifacts/logs/base_spread.log
$PY scripts/tune_ladder_optuna.py \
  --config configs/airline_synth.yaml \
  --regime dynamic \
  --sizes 100 \
  --snapshots 120 \
  --budgets 5 \
  --repeats 1 \
  --trials 4 \
  --study-name base_dynamic \
  --storage sqlite:///artifacts/optuna/base_dynamic.db \
  --out artifacts/optuna_dynamic_base.json \
  2>&1 | tee artifacts/logs/base_dynamic.log
$PY scripts/tune_ladder_optuna.py \
  --config configs/airline_synth.yaml \
  --regime spatial \
  --sizes 100 \
  --snapshots 120 \
  --budgets 5 \
  --repeats 1 \
  --trials 4 \
  --study-name base_spatial \
  --storage sqlite:///artifacts/optuna/base_spatial.db \
  --out artifacts/optuna_spatial_base.json \
  2>&1 | tee artifacts/logs/base_spatial.log
echo '[base] finished at ' \$(date)
"

SPREAD_CMD="
cd '$ROOT_DIR'
set -euo pipefail
export PYTHONPATH='$ROOT_DIR/src:\${PYTHONPATH:-}'
echo '[spread] tuning started at ' \$(date)
$PY scripts/tune_ladder_optuna.py \
  --config configs/airline_synth.yaml \
  --regime spread \
  --sizes 100,200 \
  --snapshots 120,220 \
  --budgets 5,10 \
  --repeats 1 \
  --trials 20 \
  --study-name ladder_subset_spread \
  --storage sqlite:///artifacts/optuna/ladder_subset_spread.db \
  --spread-baseline-mode surrogate \
  --spread-use-surrogate-policy \
  --out artifacts/optuna_ladder_spread_100_200_120_220_5_10.json \
  2>&1 | tee artifacts/logs/tune_ladder_spread.log
echo '[spread] rerank started at ' \$(date)
$PY scripts/rerank_optuna_top_trials.py \
  --config configs/airline_synth.yaml \
  --regime spread \
  --storage sqlite:///artifacts/optuna/ladder_subset_spread.db \
  --study-name ladder_subset_spread \
  --sizes 100,200 \
  --snapshots 120,220 \
  --budgets 5,10 \
  --top-k 5 \
  --graphs-per-cell 3 \
  --out artifacts/rerank_spread_source_subset.json \
  --winner-out artifacts/best_config_spread_source_subset.json \
  2>&1 | tee artifacts/logs/rerank_spread.log
echo '[spread] finished at ' \$(date)
"

DYNAMIC_CMD="
cd '$ROOT_DIR'
set -euo pipefail
export PYTHONPATH='$ROOT_DIR/src:\${PYTHONPATH:-}'
echo '[dynamic] tuning started at ' \$(date)
$PY scripts/tune_ladder_optuna.py \
  --config configs/airline_synth.yaml \
  --regime dynamic \
  --sizes 100,200 \
  --snapshots 120,220 \
  --budgets 5,10 \
  --repeats 1 \
  --trials 30 \
  --study-name ladder_subset_dynamic \
  --storage sqlite:///artifacts/optuna/ladder_subset_dynamic.db \
  --out artifacts/optuna_ladder_dynamic_100_200_120_220_5_10.json \
  2>&1 | tee artifacts/logs/tune_ladder_dynamic.log
echo '[dynamic] rerank started at ' \$(date)
$PY scripts/rerank_optuna_top_trials.py \
  --config configs/airline_synth.yaml \
  --regime dynamic \
  --storage sqlite:///artifacts/optuna/ladder_subset_dynamic.db \
  --study-name ladder_subset_dynamic \
  --sizes 100,200 \
  --snapshots 120,220 \
  --budgets 5,10 \
  --top-k 5 \
  --graphs-per-cell 3 \
  --out artifacts/rerank_dynamic_source_subset.json \
  --winner-out artifacts/best_config_dynamic_source_subset.json \
  2>&1 | tee artifacts/logs/rerank_dynamic.log
echo '[dynamic] finished at ' \$(date)
"

SPATIAL_CMD="
cd '$ROOT_DIR'
set -euo pipefail
export PYTHONPATH='$ROOT_DIR/src:\${PYTHONPATH:-}'
echo '[spatial] tuning started at ' \$(date)
$PY scripts/tune_ladder_optuna.py \
  --config configs/airline_synth.yaml \
  --regime spatial \
  --sizes 100,200 \
  --snapshots 120,220 \
  --budgets 5,10 \
  --repeats 1 \
  --trials 20 \
  --study-name ladder_subset_spatial \
  --storage sqlite:///artifacts/optuna/ladder_subset_spatial.db \
  --out artifacts/optuna_ladder_spatial_100_200_120_220_5_10.json \
  2>&1 | tee artifacts/logs/tune_ladder_spatial.log
echo '[spatial] rerank started at ' \$(date)
$PY scripts/rerank_optuna_top_trials.py \
  --config configs/airline_synth.yaml \
  --regime spatial \
  --storage sqlite:///artifacts/optuna/ladder_subset_spatial.db \
  --study-name ladder_subset_spatial \
  --sizes 100,200 \
  --snapshots 120,220 \
  --budgets 5,10 \
  --top-k 5 \
  --graphs-per-cell 3 \
  --out artifacts/rerank_spatial_source_subset.json \
  --winner-out artifacts/best_config_spatial_source_subset.json \
  2>&1 | tee artifacts/logs/rerank_spatial.log
echo '[spatial] finished at ' \$(date)
"

TRANSFER_CMD="
cd '$ROOT_DIR'
set -euo pipefail
export PYTHONPATH='$ROOT_DIR/src:\${PYTHONPATH:-}'
echo '[transfer] waiting for winner configs at ' \$(date)
until [[ -f artifacts/best_config_spread_source_subset.json && -f artifacts/best_config_dynamic_source_subset.json && -f artifacts/best_config_spatial_source_subset.json ]]; do
  sleep 60
done
echo '[transfer] winners found at ' \$(date)
$PY scripts/generate_homogeneous_family_csv.py \
  --config configs/airline_synth.yaml \
  --source-sizes 100,200 \
  --source-graphs-per-size 3 \
  --target-sizes 300,500 \
  --out-dir data/homogeneous_family \
  --manifest-out artifacts/homogeneous_family_manifest.json \
  2>&1 | tee artifacts/logs/transfer_generate_family.log
$PY scripts/build_transfer_configs.py \
  --base-config configs/airline_pretrain_reusable.yaml \
  --manifest artifacts/homogeneous_family_manifest.json \
  --budgets 5,10,15 \
  --source-snapshots 120,220 \
  --target-snapshots 220 \
  --out-dir configs/generated_transfer \
  2>&1 | tee artifacts/logs/transfer_build_configs.log
for cfg in configs/generated_transfer/transfer_target*_budget*.yaml; do
  name=\$(basename \"\$cfg\" .yaml)
  $PY scripts/evaluate_transfer_homogeneous.py \
    --config \"\$cfg\" \
    --spread-artifact artifacts/best_config_spread_source_subset.json \
    --dynamic-artifact artifacts/best_config_dynamic_source_subset.json \
    --spatial-artifact artifacts/best_config_spatial_source_subset.json \
    --out \"artifacts/\${name}.json\" \
    2>&1 | tee \"artifacts/logs/\${name}.log\"
done
echo '[transfer] finished at ' \$(date)
"

run_tmux_session "stim_base" "$BASE_CMD"
run_tmux_session "stim_tune_spread" "$SPREAD_CMD"
run_tmux_session "stim_tune_dynamic" "$DYNAMIC_CMD"
run_tmux_session "stim_tune_spatial" "$SPATIAL_CMD"
run_tmux_session "stim_transfer" "$TRANSFER_CMD"

echo "tmux sessions created."
echo "Use: tmux ls"
echo "Attach: tmux attach -t stim_tune_dynamic"
