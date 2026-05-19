#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="python3"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
BENCHMARK_ROOT="$PROJECT_ROOT/outputs/gemini_3_flash_preview"
PREDICTION_ROOT="$PROJECT_ROOT/baseline_prediction/oracle/results"

# Accept either numeric IDs (1,2,3) or full user dirs (001_user_001)
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

resolve_latest_named_file() {
  local dir="$1"
  local pattern="$2"
  find "$dir" -maxdepth 1 -type f -name "$pattern" \
    ! -name '*final_qa*' \
    ! -name '*run_settings*' \
    -print 2>/dev/null | LC_ALL=C sort | tail -n 1
}

resolve_existing_file() {
  local candidate
  for candidate in "$@"; do
    if [[ -n "$candidate" && -f "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  return 1
}

for user in "${USERS[@]}"; do
  user_dir="$(normalize_user_dir "$user")"
  benchmark_dir="$BENCHMARK_ROOT/$user_dir"
  prediction_dir="$PREDICTION_ROOT/$user_dir/prediction"
  eval_dir="$PREDICTION_ROOT/$user_dir/eval"

  if [[ -n "${BENCHMARK_PATH:-}" ]]; then
    benchmark_path="$BENCHMARK_PATH"
  else
    benchmark_path="$(resolve_latest_named_file "$benchmark_dir" 'tce_benchmark_vnext_*task_packs*.json' || true)"
    if [[ -z "$benchmark_path" ]]; then
      benchmark_path="$(resolve_latest_named_file "$benchmark_dir" 'tce_benchmark*task_packs*.json' || true)"
    fi
    if [[ -z "$benchmark_path" ]]; then
      benchmark_path="$(resolve_existing_file "$benchmark_dir/tce_benchmark.json" || true)"
    fi
  fi

  if [[ -n "${PREDICTION_PATH:-}" ]]; then
    prediction_path="$PREDICTION_PATH"
  else
    prediction_path="$(resolve_latest_named_file "$prediction_dir" 'tce_results_v14_taskabc*.json' || true)"
    if [[ -z "$prediction_path" ]]; then
      prediction_path="$(resolve_latest_named_file "$prediction_dir" 'tce_results*taskabc*.json' || true)"
    fi
    if [[ -z "$prediction_path" ]]; then
      prediction_path="$(resolve_latest_named_file "$prediction_dir" 'tce_results*.json' || true)"
    fi
  fi

  if [[ -n "${OUTPUT_PATH:-}" ]]; then
    output_path="$OUTPUT_PATH"
  else
    prediction_file="$(basename "${prediction_path:-tce_results.json}")"
    output_file="${prediction_file/tce_results/tce_eval}"
    if [[ "$output_file" == "$prediction_file" ]]; then
      output_file="tce_eval.json"
    fi
    output_path="$eval_dir/$output_file"
  fi
  mkdir -p "$(dirname "$output_path")"

  if [[ ! -f "$benchmark_path" ]]; then
    echo "[ORACLE-TCE-EVAL] SKIP $user_dir: benchmark not found: $benchmark_path"
    continue
  fi

  if [[ ! -f "$prediction_path" ]]; then
    echo "[ORACLE-TCE-EVAL] SKIP $user_dir: prediction not found: $prediction_path"
    continue
  fi

  echo "[ORACLE-TCE-EVAL] user=$user_dir llm_judge=$ENABLE_LLM_JUDGE"

  cmd=(
    "$PYTHON_BIN" -m eval.eval_tce
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
