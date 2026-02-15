#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="python"
PROJECT_ROOT="/users/4/xie00470/mem_bench/behavior_and_conversation"
BENCHMARK_ROOT="$PROJECT_ROOT/data_construction/generated_outputs/gemini_3_flash_preview"
OUTPUT_ROOT="$PROJECT_ROOT/generation/letta/results"

USERS=("001_user_001")
MAX_VISIBLE_LOGS=""
RETRIEVAL_TOP_K="10"
LLM_PROVIDER="openai"
LLM_MODEL="gpt-5-mini"
LLM_MAX_WORKERS="1"

LETTA_MODE="sdk" # sdk | local
ALLOW_LOCAL_FALLBACK="true"
RESUME="true"
SAVE_PROMPT_AND_RAW="true"
PERSONA=""
HUMAN=""

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
    echo "[LETTA-DSP] SKIP $user_dir: benchmark not found"
    continue
  fi

  if [[ ! -f "$app_logs_path" ]]; then
    echo "[LETTA-DSP] SKIP $user_dir: app_log_large not found: $app_logs_path"
    continue
  fi

  cmd=(
    "$PYTHON_BIN" -m generation.letta.dynamic_state_prediction
    --benchmark "$benchmark_path"
    --app-logs-path "$app_logs_path"
    --output "$output_path"
    --retrieval-top-k "$RETRIEVAL_TOP_K"
    --llm-provider "$LLM_PROVIDER"
    --llm-model "$LLM_MODEL"
    --llm-max-workers "$LLM_MAX_WORKERS"
    --letta-mode "$LETTA_MODE"
  )

  if [[ -n "$MAX_VISIBLE_LOGS" ]]; then
    cmd+=(--max-visible-logs "$MAX_VISIBLE_LOGS")
  fi

  if [[ "$ALLOW_LOCAL_FALLBACK" != "true" ]]; then
    cmd+=(--no-local-fallback)
  fi

  if [[ "$RESUME" == "true" ]]; then
    cmd+=(--resume)
  fi

  if [[ "$SAVE_PROMPT_AND_RAW" == "true" ]]; then
    cmd+=(--save-prompt-and-raw)
  fi
  if [[ -n "$PERSONA" ]]; then
    cmd+=(--persona "$PERSONA")
  fi
  if [[ -n "$HUMAN" ]]; then
    cmd+=(--human "$HUMAN")
  fi

  "${cmd[@]}"
done
