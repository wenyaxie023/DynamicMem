#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

USER_ID="${USER_ID:-user1}"
ACTION="${ACTION:-all}"
RESUME="${RESUME:-true}"
HOT_RESUME_LATEST="${HOT_RESUME_LATEST:-true}"
KEEP_IMPORTED_AGENTS="${KEEP_IMPORTED_AGENTS:-false}"

DATA_ROOT="${DATA_ROOT:-$PROJECT_ROOT/data}"
RESULT_ROOT="${RESULT_ROOT:-$PROJECT_ROOT/baseline_prediction/letta/results}"
AGENTS_DIR="${AGENTS_DIR:-$PROJECT_ROOT/baseline_prediction/letta/agents}"

APP_LOGS_FILENAME="${APP_LOGS_FILENAME:-app_log_large.json}"
QA_FILENAME="${QA_FILENAME:-qa.json}"
BENCHMARK_FILENAME="${BENCHMARK_FILENAME:-dynamic_state_prediction_benchmark.json}"

cd "$PROJECT_ROOT"

cmd=(
  "$PYTHON_BIN" -m baseline_prediction.letta.pipeline
  --action "$ACTION"
  --user "$USER_ID"
  --data-root "$DATA_ROOT"
  --result-root "$RESULT_ROOT"
  --agents-dir "$AGENTS_DIR"
  --app-logs-filename "$APP_LOGS_FILENAME"
  --qa-filename "$QA_FILENAME"
  --benchmark-filename "$BENCHMARK_FILENAME"
)

if [[ "$RESUME" == "true" ]]; then
  cmd+=(--resume)
fi
if [[ "$HOT_RESUME_LATEST" == "true" ]]; then
  cmd+=(--hot-resume-latest)
fi
if [[ "$KEEP_IMPORTED_AGENTS" == "true" ]]; then
  cmd+=(--keep-imported-agents)
fi

echo "[LETTA-PIPELINE] Running command:"
printf ' %q' "${cmd[@]}"
echo

"${cmd[@]}"
