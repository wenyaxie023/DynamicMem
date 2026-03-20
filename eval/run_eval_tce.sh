#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
PROJECT_ROOT="/users/4/xie00470/mem_bench/dynamicmem"
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

if [[ -n "${BENCHMARK_PATH:-}" ]]; then
  benchmark_path="$BENCHMARK_PATH"
else
  benchmark_path="$PREDICTION_DIR/tce_subset_checkpoint.json"
fi

prediction_file="tce_results.json"
output_file="tce_eval_llm_judge.json"
if [[ -n "$TOPK" ]]; then
  prediction_file="tce_results_topk${TOPK}.json"
  output_file="tce_eval_llm_judge_topk${TOPK}.json"
fi

prediction_path="$PREDICTION_DIR/$prediction_file"
output_path="$EVAL_DIR/$output_file"

mkdir -p "$EVAL_DIR"
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
