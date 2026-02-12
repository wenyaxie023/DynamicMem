#!/usr/bin/env bash
set -euo pipefail

# Edit options below as needed.

PYTHON_BIN="python"
INPUT_ROOT_DIR="/users/4/xie00470/mem_bench/behavior_and_conversation/data_construction/generated_outputs/gemini_3_flash_preview"
OUTPUT_ROOT_DIR="/users/4/xie00470/mem_bench/behavior_and_conversation/generation/rag/results"
QA_DIR="/users/4/xie00470/mem_bench/behavior_and_conversation/generation/qa"
USERS=("2")
# USERS=("1")
TOPKS=("5" "10" "20")
# MODE="all" # all | retrieve | generate
# MODE="generate"
MODE="all"
SKIP_RETRIEVE="false"
WRITE_EACH="true"
RESUME="false"
# Extra args forwarded to rag.py (e.g. --retriever-type openai --llm-model gpt-5-mini)
# EXTRA_ARGS=(--llm-provider azure)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

WRITE_EACH_FLAG=()
if [[ "$WRITE_EACH" == "true" ]]; then
  WRITE_EACH_FLAG=(--write-each)
fi

RESUME_FLAG=()
if [[ "$RESUME" == "true" ]]; then
  RESUME_FLAG=(--resume)
fi

for user in "${USERS[@]}"; do
  if [[ "$MODE" == "all" || "$MODE" == "retrieve" ]]; then
    echo "[RAG] retrieve: user=${user}"
    "$PYTHON_BIN" "$SCRIPT_DIR/rag.py" \
      --user-idx "$user" \
      --retrieve-only \
      --input-root-dir "$INPUT_ROOT_DIR" \
      --output-root-dir "$OUTPUT_ROOT_DIR" \
      --qa-dir "$QA_DIR" \
      "${WRITE_EACH_FLAG[@]}" \
      "${RESUME_FLAG[@]}"
  fi

  if [[ "$MODE" == "all" || "$MODE" == "generate" ]]; then
    for topk in "${TOPKS[@]}"; do
      echo "[RAG] generate: user=${user} topk=${topk}"
      if [[ "$SKIP_RETRIEVE" == "true" || "$MODE" == "all" ]]; then
        "$PYTHON_BIN" "$SCRIPT_DIR/rag.py" \
          --user-idx "$user" \
          --gen-topk "$topk" \
          --skip-retrieve \
          --input-root-dir "$INPUT_ROOT_DIR" \
          --output-root-dir "$OUTPUT_ROOT_DIR" \
          --qa-dir "$QA_DIR" \
          "${WRITE_EACH_FLAG[@]}" \
          "${RESUME_FLAG[@]}"
      else
        "$PYTHON_BIN" "$SCRIPT_DIR/rag.py" \
          --user-idx "$user" \
          --gen-topk "$topk" \
          --input-root-dir "$INPUT_ROOT_DIR" \
          --output-root-dir "$OUTPUT_ROOT_DIR" \
          --qa-dir "$QA_DIR" \
          "${WRITE_EACH_FLAG[@]}" \
          "${RESUME_FLAG[@]}"
      fi
    done
  fi
done
