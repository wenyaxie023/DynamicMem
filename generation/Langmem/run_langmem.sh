#!/usr/bin/env bash
set -euo pipefail

# Run Langmem Full Pipeline (Hybrid Mode)

PYTHON_BIN="python"

# Default paths
INPUT_ROOT_DIR="/home/zyli/MUSE/.data/data_construction/generated_outputs/gemini_3_flash_preview"
OUTPUT_ROOT_DIR="/home/zyli/MUSE/generation/Langmem/results"
QA_DIR="/home/zyli/MUSE/generation/qa"

# User to process
USERS=("001_user_001")

# Langmem config
TOP_K=5
EMBED_MODEL="openai:text-embedding-3-small"
LLM_PROVIDER="openai"
LLM_MODEL="gpt-4o-mini"
QUERY_MODEL="gpt-4o-mini"

# Memory mode: triple, profile, episodic, hybrid
MODE="hybrid"

# Resume or not
RESUME="false"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Add project root to PYTHONPATH
PYTHONPATH="${PYTHONPATH:-}"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

RESUME_FLAG=()
if [[ "$RESUME" == "true" ]]; then
  RESUME_FLAG=(--resume)
fi

for user in "${USERS[@]}"; do
  echo "[Langmem Full] Running: user=${user} mode=${MODE} topk=${TOP_K}"
  "$PYTHON_BIN" "$SCRIPT_DIR/langmem_full.py" \
    --user-idx "$user" \
    --mode "$MODE" \
    --input-root-dir "$INPUT_ROOT_DIR" \
    --output-root-dir "$OUTPUT_ROOT_DIR" \
    --qa-dir "$QA_DIR" \
    --top-k "$TOP_K" \
    --embed-model "$EMBED_MODEL" \
    --llm-provider "$LLM_PROVIDER" \
    --llm-model "$LLM_MODEL" \
    --query-model "$QUERY_MODEL" \
    "${RESUME_FLAG[@]}"
done
