#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="python3"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

USER_DIR="001_user_001"
BENCHMARK="$PROJECT_ROOT/data_construction/generated_outputs/gemini_3_flash_preview/$USER_DIR/state_abstraction_benchmark.json"
APP_LOGS="$PROJECT_ROOT/data_construction/generated_outputs/gemini_3_flash_preview/$USER_DIR/app_log_large.json"

# mode: checkpoint | date
MODE="checkpoint"

# used when MODE=checkpoint
CHECKPOINT_IDS="cp_0001,cp_0010,cp_0050"

# used when MODE=date (YYYY-MM-DD comma-separated)
DATES="2023-12-31,2024-03-31,2024-06-30"

BASELINE="rag"
OUT_DIR="$PROJECT_ROOT/generation/$BASELINE/results/$USER_DIR/prediction"
TASKS_OUT="$OUT_DIR/state_abstraction_tasks_${MODE}.json"
SUBSET_BENCHMARK_OUT="$OUT_DIR/state_abstraction_subset_${MODE}.json"

mkdir -p "$OUT_DIR"

CMD=(
  "$PYTHON_BIN" -m state_abstraction_api.cli export
  --benchmark "$BENCHMARK"
  --app-logs "$APP_LOGS"
  --mode "$MODE"
  --output "$TASKS_OUT"
  --subset-benchmark-output "$SUBSET_BENCHMARK_OUT"
)

if [[ "$MODE" == "checkpoint" ]]; then
  CMD+=(--checkpoint-ids "$CHECKPOINT_IDS")
elif [[ "$MODE" == "date" ]]; then
  CMD+=(--dates "$DATES")
else
  echo "Unsupported MODE: $MODE (must be checkpoint or date)" >&2
  exit 1
fi

cd "$PROJECT_ROOT"
"${CMD[@]}"

echo "Done."
echo "Tasks: $TASKS_OUT"
echo "Subset benchmark: $SUBSET_BENCHMARK_OUT"
