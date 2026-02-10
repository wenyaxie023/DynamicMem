#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="python3"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

USER_DIR="001_user_001"
BASELINE="demo"
MODE="checkpoint"
CHECKPOINT_IDS="cp_0001,cp_0010,cp_0050"

BENCHMARK="$PROJECT_ROOT/data_construction/generated_outputs/gemini_3_flash_preview/$USER_DIR/state_abstraction_benchmark.json"
APP_LOGS="$PROJECT_ROOT/data_construction/generated_outputs/gemini_3_flash_preview/$USER_DIR/app_log_large.json"
PRED_DIR="$PROJECT_ROOT/generation/$BASELINE/results/$USER_DIR/prediction"
EVAL_DIR="$PROJECT_ROOT/generation/$BASELINE/results/$USER_DIR/eval"
TASKS="$PRED_DIR/state_abstraction_tasks_${MODE}.json"
SUBSET_BENCHMARK="$PRED_DIR/state_abstraction_subset_${MODE}.json"
PREDICTION="$PRED_DIR/state_abstraction_results_${MODE}.json"
EVAL_OUT="$EVAL_DIR/state_abstraction_eval_api_${MODE}.json"

mkdir -p "$PRED_DIR" "$EVAL_DIR"

cd "$PROJECT_ROOT"

echo "[1/3] Export tasks"
"$PYTHON_BIN" -m state_abstraction_api.cli export \
  --benchmark "$BENCHMARK" \
  --app-logs "$APP_LOGS" \
  --mode "$MODE" \
  --checkpoint-ids "$CHECKPOINT_IDS" \
  --output "$TASKS" \
  --subset-benchmark-output "$SUBSET_BENCHMARK" \
  --include-targets

echo "[2/3] Run demo predictor"
"$PYTHON_BIN" state_abstraction_api/demo_predictor.py \
  --tasks "$TASKS" \
  --output "$PREDICTION" \
  --mode oracle

echo "[3/3] Evaluate"
"$PYTHON_BIN" -m state_abstraction_api.cli evaluate \
  --benchmark "$SUBSET_BENCHMARK" \
  --prediction "$PREDICTION" \
  --output "$EVAL_OUT"

echo "Done."
echo "Prediction: $PREDICTION"
echo "Eval: $EVAL_OUT"
