#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="python"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
INPUT_ROOT_DIR="${INPUT_ROOT_DIR:-$PROJECT_ROOT/data}"
OUTPUT_ROOT_DIR="${OUTPUT_ROOT_DIR:-$PROJECT_ROOT/generation/letta/results}"
QA_DIR="${QA_DIR:-$PROJECT_ROOT/data}"
USERS=(${USERS:-user1})
APP_LOGS_FILENAME="${APP_LOGS_FILENAME:-app_log_large.json}"
QA_FILENAME="${QA_FILENAME:-qa.json}"

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
    --app-logs-filename "$APP_LOGS_FILENAME"
    --qa-filename "$QA_FILENAME"
  )

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
