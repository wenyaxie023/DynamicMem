#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="python3"
PROJECT_ROOT="/users/4/xie00470/mem_bench/behavior_and_conversation"
OUTPUT_ROOT="$PROJECT_ROOT/data_construction/generated_outputs/gemini_3_flash_preview"

# supported user ids: "1" or "001_user_001"
USERS=("001_user_001")

normalize_user_dir() {
  local x="$1"
  if [[ "$x" =~ ^[0-9]+$ ]]; then
    printf "%03d_user_%03d" "$x" "$x"
  else
    printf "%s" "$x"
  fi
}

for user in "${USERS[@]}"; do
  user_dir="$(normalize_user_dir "$user")"
  app_logs_final="$OUTPUT_ROOT/$user_dir/app_logs_final.json"

  if [[ ! -f "$app_logs_final" ]]; then
    echo "[DSP-BENCH] SKIP $user_dir: missing $app_logs_final"
    continue
  fi

  echo "[DSP-BENCH] Building benchmark for $user_dir"
  (
    cd "$PROJECT_ROOT"
    "$PYTHON_BIN" data_construction/build_dynamic_state_prediction_benchmark.py \
      --app-logs-final "$app_logs_final"
  )
done
