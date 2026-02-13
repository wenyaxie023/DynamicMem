#!/usr/bin/env bash
set -euo pipefail

# Batch runner for RAG dynamic state prediction generation.
# Produces: generation/rag/results/<user_id>/prediction/dynamic_state_prediction_results.json

PYTHON_BIN="python"
PROJECT_ROOT="/users/4/xie00470/mem_bench/behavior_and_conversation"
BENCHMARK_ROOT="$PROJECT_ROOT/data_construction/generated_outputs/gemini_3_flash_preview"
OUTPUT_ROOT="$PROJECT_ROOT/generation/rag/results"

# Accept either numeric IDs (1,2,3) or full user dirs (001_user_001)
USERS=("001_user_001")
# Optional truncation for debugging.
# Leave empty to use full history until each checkpoint.
MAX_VISIBLE_LOGS=""
RETRIEVAL_TOP_K="10"
LLM_PROVIDER="openai"
LLM_MODEL="gpt-5-mini"
LLM_MAX_WORKERS="1"
RETRIEVER_PROVIDER="openai"
RETRIEVER_MODEL="text-embedding-3-large"
RETRIEVER_BATCH_SIZE="64"
RESUME="false"
SAVE_PROMPT_AND_RAW="true"

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
  output_path="$OUTPUT_ROOT/$user_dir/prediction/dynamic_state_prediction_results_topk${RETRIEVAL_TOP_K}.json"

  if [[ ! -f "$benchmark_path" ]]; then
    echo "[RAG-DSP] SKIP $user_dir: benchmark not found"
    continue
  fi

  if [[ ! -f "$app_logs_path" ]]; then
    echo "[RAG-DSP] SKIP $user_dir: app_log_large not found: $app_logs_path"
    continue
  fi

  echo "[RAG-DSP] user=$user_dir model=$LLM_MODEL max_visible_logs=${MAX_VISIBLE_LOGS:-full} retrieval_top_k=$RETRIEVAL_TOP_K"

  cmd=(
    "$PYTHON_BIN" -m generation.rag.rag_dynamic_state_prediction
    --benchmark "$benchmark_path"
    --app-logs-path "$app_logs_path"
    --output "$output_path"
    --retrieval-top-k "$RETRIEVAL_TOP_K"
    --llm-provider "$LLM_PROVIDER"
    --llm-model "$LLM_MODEL"
    --llm-max-workers "$LLM_MAX_WORKERS"
    --retriever-provider "$RETRIEVER_PROVIDER"
    --retriever-model "$RETRIEVER_MODEL"
    --retriever-batch-size "$RETRIEVER_BATCH_SIZE"
  )

  if [[ -n "$MAX_VISIBLE_LOGS" ]]; then
    cmd+=(--max-visible-logs "$MAX_VISIBLE_LOGS")
  fi

  if [[ "$RESUME" == "true" ]]; then
    cmd+=(--resume)
  fi

  if [[ "$SAVE_PROMPT_AND_RAW" == "true" ]]; then
    cmd+=(--save-prompt-and-raw)
  fi

  "${cmd[@]}"
done
