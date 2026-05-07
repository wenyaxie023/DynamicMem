#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
PROJECT_ROOT="xxx/mem_bench/dynamicmem" # Replace with the actual path to the DynamicMem project root
BENCHMARK_ROOT="${BENCHMARK_ROOT:-$PROJECT_ROOT/data_construction/generated_outputs/gemini_3_flash_preview}"
BASELINE="${BASELINE:-rag}"
USER_DIR="${USER_DIR:-001_user_001}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-}"
TOPK="${TOPK:-${RETRIEVAL_TOP_K:-5}}"

ENABLE_LLM_JUDGE="${ENABLE_LLM_JUDGE:-true}"
LLM_PROVIDER="${LLM_PROVIDER:-azure}"
if [[ "$LLM_PROVIDER" == "azure" ]]; then
  LLM_MODEL="${LLM_MODEL:-${AZURE_OPENAI_DEPLOYMENT:-gpt-5-mini}}"
else
  LLM_MODEL="${LLM_MODEL:-gpt-5-mini}"
fi
LLM_MAX_WORKERS="${LLM_MAX_WORKERS:-4}"

RESULT_ROOT="$PROJECT_ROOT/generation/$BASELINE/results/$USER_DIR"
if [[ -n "$EXPERIMENT_NAME" ]]; then
  PREDICTION_DIR="$RESULT_ROOT/prediction/$EXPERIMENT_NAME"
  EVAL_DIR="$RESULT_ROOT/eval/$EXPERIMENT_NAME"
else
  PREDICTION_DIR="$RESULT_ROOT/prediction"
  EVAL_DIR="$RESULT_ROOT/eval"
fi
BENCHMARK_DIR="$BENCHMARK_ROOT/$USER_DIR"

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

if [[ -n "${BENCHMARK_PATH:-}" ]]; then
  benchmark_path="$BENCHMARK_PATH"
else
  benchmark_path="$(resolve_latest_named_file "$BENCHMARK_DIR" 'tce_benchmark_vnext_*task_packs*.json' || true)"
  if [[ -z "$benchmark_path" ]]; then
    benchmark_path="$(resolve_latest_named_file "$BENCHMARK_DIR" 'tce_benchmark*task_packs*.json' || true)"
  fi
  if [[ -z "$benchmark_path" ]]; then
    benchmark_path="$(resolve_existing_file "$BENCHMARK_DIR/tce_benchmark.json" || true)"
  fi
fi

if [[ -n "${PREDICTION_PATH:-}" ]]; then
  prediction_path="$PREDICTION_PATH"
else
  prediction_path=""
  if [[ -n "$TOPK" ]]; then
    prediction_path="$(resolve_latest_named_file "$PREDICTION_DIR" "tce_results*topk${TOPK}*.json" || true)"
  fi
  if [[ -z "$prediction_path" ]]; then
    prediction_path="$(resolve_latest_named_file "$PREDICTION_DIR" 'tce_results_v14_taskabc*.json' || true)"
  fi
  if [[ -z "$prediction_path" ]]; then
    prediction_path="$(resolve_latest_named_file "$PREDICTION_DIR" 'tce_results*taskabc*.json' || true)"
  fi
  if [[ -z "$prediction_path" ]]; then
    prediction_path="$(resolve_latest_named_file "$PREDICTION_DIR" 'tce_results*.json' || true)"
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
  output_path="$EVAL_DIR/$output_file"
fi

if [[ ! -f "${benchmark_path:-}" ]]; then
  echo "[TCE-EVAL] benchmark not found under $BENCHMARK_DIR" >&2
  exit 1
fi

if [[ ! -f "${prediction_path:-}" ]]; then
  echo "[TCE-EVAL] prediction not found under $PREDICTION_DIR" >&2
  exit 1
fi

mkdir -p "$(dirname "$output_path")"
cd "$PROJECT_ROOT"

cmd=(
  "$PYTHON_BIN" -m eval.eval_tce
  --benchmark "$benchmark_path"
  --prediction "$prediction_path"
  --output "$output_path"
)

if [[ "$ENABLE_LLM_JUDGE" == "true" ]]; then
  cmd+=(
    --enable-llm-judge
    --llm-provider "$LLM_PROVIDER"
    --llm-model "$LLM_MODEL"
    --llm-max-workers "$LLM_MAX_WORKERS"
  )
fi

echo "[TCE-EVAL] baseline=$BASELINE user=$USER_DIR experiment=${EXPERIMENT_NAME:-default} topk=${TOPK:-default}"
"${cmd[@]}"
