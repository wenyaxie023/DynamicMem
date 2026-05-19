#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

USER_DIR="$PROJECT_ROOT/data/user1"
APP_LOGS_FILENAME="app_log_test.json"
QA_FILENAME="qa_test.json"

APP_LOG_PATH="$USER_DIR/$APP_LOGS_FILENAME"
QA_PATH="$USER_DIR/$QA_FILENAME"
if [[ ! -f "$APP_LOG_PATH" ]]; then
  echo "[LETTA-TEST] app logs not found: $APP_LOG_PATH"
  exit 1
fi
if [[ ! -f "$QA_PATH" ]]; then
  echo "[LETTA-TEST] QA file not found: $QA_PATH"
  exit 1
fi

cd "$PROJECT_ROOT"

"$PYTHON_BIN" -m baseline_prediction.letta.letta \
  --input-user-dir "$USER_DIR" \
  --qa-user-dir "$USER_DIR" \
  --prediction-dir "$USER_DIR" \
  --app-logs-filename "$APP_LOGS_FILENAME" \
  --qa-filename "$QA_FILENAME"
