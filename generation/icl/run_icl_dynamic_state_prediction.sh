#!/usr/bin/env bash
set -euo pipefail

# Batch runner for ICL dynamic state prediction generation.
# Produces: generation/icl/results/<user_id>/prediction/dynamic_state_prediction_results.json

PYTHON_BIN="python"
PROJECT_ROOT="/users/4/xie00470/mem_bench/behavior_and_conversation"
BENCHMARK_ROOT="$PROJECT_ROOT/data_construction/generated_outputs/gemini_3_flash_preview"
OUTPUT_ROOT="$PROJECT_ROOT/generation/icl/results"

# Accept either numeric IDs (1,2,3) or full user dirs (001_user_001)
USERS=("001_user_001")
# Optional truncation for debugging.
# Leave empty to use full history until each checkpoint.
MAX_VISIBLE_LOGS=""
LLM_PROVIDER="openai"
LLM_MODEL="gpt-5-mini"
LLM_MAX_WORKERS="1"
RESUME="true"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

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

  benchmark_path="$BENCHMARK_ROOT/$user_dir/dynamic_state_prediction_benchmark.json"

  app_logs_path="$BENCHMARK_ROOT/$user_dir/app_log_large.json"
  output_path="$OUTPUT_ROOT/$user_dir/prediction/dynamic_state_prediction_results.json"

  if [[ ! -f "$benchmark_path" ]]; then
    echo "[ICL-DSP] SKIP $user_dir: benchmark not found"
    continue
  fi

  if [[ ! -f "$app_logs_path" ]]; then
    echo "[ICL-DSP] SKIP $user_dir: app_log_large not found: $app_logs_path"
    continue
  fi

  echo "[ICL-DSP] user=$user_dir model=$LLM_MODEL max_visible_logs=${MAX_VISIBLE_LOGS:-full}"

  cmd=(
    "$PYTHON_BIN" -m generation.icl.dynamic_state_prediction
    --benchmark "$benchmark_path"
    --app-logs-path "$app_logs_path"
    --output "$output_path"
    --llm-provider "$LLM_PROVIDER"
    --llm-model "$LLM_MODEL"
    --llm-max-workers "$LLM_MAX_WORKERS"
    --max-checkpoints 20
  )

  if [[ -n "$MAX_VISIBLE_LOGS" ]]; then
    cmd+=(--max-visible-logs "$MAX_VISIBLE_LOGS")
  fi

  if [[ "$RESUME" == "true" ]]; then
    cmd+=(--resume)
  fi

  "${cmd[@]}"
done
