#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="python"
PROJECT_ROOT="/users/4/xie00470/mem_bench/behavior_and_conversation"
INPUT_ROOT_DIR="$PROJECT_ROOT/data_construction/generated_outputs/gemini_3_flash_preview"
OUTPUT_ROOT_DIR="$PROJECT_ROOT/generation/letta/results"
QA_DIR="$PROJECT_ROOT/generation/qa"
USERS=("001_user_001")

RETRIEVAL_TOP_K="10"
LLM_PROVIDER="openai"
LLM_MODEL="gpt-5-mini"
LLM_MAX_WORKERS="4"

LETTA_MODE="sdk" # sdk | local
ALLOW_LOCAL_FALLBACK="true"
RESUME="true"
PERSONA=""
HUMAN=""

cd "$PROJECT_ROOT"

for user in "${USERS[@]}"; do
  cmd=(
    "$PYTHON_BIN" -m generation.letta.letta
    --user-idx "$user"
    --input-root-dir "$INPUT_ROOT_DIR"
    --output-root-dir "$OUTPUT_ROOT_DIR"
    --qa-dir "$QA_DIR"
    --retrieval-top-k "$RETRIEVAL_TOP_K"
    --llm-provider "$LLM_PROVIDER"
    --llm-model "$LLM_MODEL"
    --llm-max-workers "$LLM_MAX_WORKERS"
    --letta-mode "$LETTA_MODE"
  )

  if [[ "$ALLOW_LOCAL_FALLBACK" != "true" ]]; then
    cmd+=(--no-local-fallback)
  fi

  if [[ "$RESUME" == "true" ]]; then
    cmd+=(--resume)
  fi
  if [[ -n "$PERSONA" ]]; then
    cmd+=(--persona "$PERSONA")
  fi
  if [[ -n "$HUMAN" ]]; then
    cmd+=(--human "$HUMAN")
  fi

  "${cmd[@]}"
done
