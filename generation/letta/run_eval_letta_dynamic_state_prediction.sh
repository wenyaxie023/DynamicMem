#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="python"
PROJECT_ROOT="/users/4/xie00470/mem_bench/behavior_and_conversation"
BENCHMARK_ROOT="$PROJECT_ROOT/data_construction/generated_outputs/gemini_3_flash_preview"
PREDICTION_ROOT="$PROJECT_ROOT/generation/letta/results"

USERS=("001_user_001")
ENABLE_LLM_JUDGE="${ENABLE_LLM_JUDGE:-true}"
SAVE_EYEBALL="${SAVE_EYEBALL:-true}"
LLM_PROVIDER="${LLM_PROVIDER:-openai}"
LLM_MODEL="${LLM_MODEL:-gpt-5-mini}"
LLM_MAX_WORKERS="${LLM_MAX_WORKERS:-1}"

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
  prediction_path="$PREDICTION_ROOT/$user_dir/prediction/dynamic_state_prediction_results.json"
  output_path="$PREDICTION_ROOT/$user_dir/eval/dynamic_state_prediction_eval.json"
  mkdir -p "$(dirname "$output_path")"

  if [[ ! -f "$benchmark_path" ]]; then
    echo "[LETTA-DSP-EVAL] SKIP $user_dir: benchmark not found: $benchmark_path"
    continue
  fi
  if [[ ! -f "$prediction_path" ]]; then
    echo "[LETTA-DSP-EVAL] SKIP $user_dir: prediction not found: $prediction_path"
    continue
  fi

  cmd=(
    "$PYTHON_BIN" -m eval.eval_dynamic_state_prediction
    --benchmark "$benchmark_path"
    --prediction "$prediction_path"
    --output "$output_path"
  )

  if [[ "$SAVE_EYEBALL" == "true" ]]; then
    cmd+=(--save-eyeball)
  fi
  if [[ "$ENABLE_LLM_JUDGE" == "true" ]]; then
    cmd+=(
      --enable-llm-judge
      --llm-provider "$LLM_PROVIDER"
      --llm-model "$LLM_MODEL"
      --llm-max-workers "$LLM_MAX_WORKERS"
    )
  fi

  "${cmd[@]}"
done

