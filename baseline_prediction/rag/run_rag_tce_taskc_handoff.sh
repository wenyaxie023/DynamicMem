#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CONFIG_PATH="$REPO_ROOT/configs/experiments/tce/rag_user1_taskc_only_handoff_5ckpt_openai.yaml"
BENCHMARK_PATH="$REPO_ROOT/handoff_inputs/tce_benchmark_taskc_v2.json"
APP_LOGS_PATH="$REPO_ROOT/handoff_inputs/app_log_large.json"

if [[ ! -f "$BENCHMARK_PATH" ]]; then
  echo "[ERROR] Missing benchmark pack JSON: $BENCHMARK_PATH"
  echo "Place the downloaded TCE v2 Task C pack JSON at that path, then rerun."
  exit 1
fi

if [[ ! -f "$APP_LOGS_PATH" ]]; then
  echo "[ERROR] Missing app log JSON: $APP_LOGS_PATH"
  echo "RAG retrieval also needs app_log_large.json at that path, then rerun."
  exit 1
fi

cd "$REPO_ROOT"
python3 -m baseline_prediction.run_tce --config "$CONFIG_PATH" "$@"
