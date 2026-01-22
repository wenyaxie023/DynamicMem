#!/usr/bin/env bash
set -euo pipefail

# Edit options below as needed.

PYTHON_BIN="python"
ROOT_DIR="/users/4/xie00470/mem_bench/behavior_and_conversation/data_construction/generated_outputs/gemini_3_flash_preview"
USERS=("1")
SIZES=("medium" "large")
TOPKS=("5" "10" "20")
# MODE="all" # all | retrieve | generate
# MODE="retrieve"
MODE="all"
SKIP_RETRIEVE="false"

# Extra args forwarded to rag.py (e.g. --retriever-type openai --llm-model gpt-5-mini)
EXTRA_ARGS=()

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for user in "${USERS[@]}"; do
  for size in "${SIZES[@]}"; do
    if [[ "$MODE" == "all" || "$MODE" == "retrieve" ]]; then
      echo "[RAG] retrieve: user=${user} size=${size}"
      "$PYTHON_BIN" "$SCRIPT_DIR/rag.py" \
        --user-idx "$user" \
        --log-size "$size" \
        --retrieve-only \
        --root-dir "$ROOT_DIR" \
        "${EXTRA_ARGS[@]}"
    fi

    if [[ "$MODE" == "all" || "$MODE" == "generate" ]]; then
      for topk in "${TOPKS[@]}"; do
        echo "[RAG] generate: user=${user} size=${size} topk=${topk}"
        if [[ "$SKIP_RETRIEVE" == "true" || "$MODE" == "all" ]]; then
          "$PYTHON_BIN" "$SCRIPT_DIR/rag.py" \
            --user-idx "$user" \
            --log-size "$size" \
            --gen-topk "$topk" \
            --skip-retrieve \
            --root-dir "$ROOT_DIR" \
            "${EXTRA_ARGS[@]}"
        else
          "$PYTHON_BIN" "$SCRIPT_DIR/rag.py" \
            --user-idx "$user" \
            --log-size "$size" \
            --gen-topk "$topk" \
            --root-dir "$ROOT_DIR" \
            "${EXTRA_ARGS[@]}"
        fi
      done
    fi
  done
done
