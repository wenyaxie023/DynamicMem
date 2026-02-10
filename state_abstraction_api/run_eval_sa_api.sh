#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="python3"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

USER_DIR="001_user_001"
BASELINE="rag"

# Must match run_export_sa_tasks.sh MODE
MODE="checkpoint"

SUBSET_BENCHMARK="$PROJECT_ROOT/generation/$BASELINE/results/$USER_DIR/prediction/state_abstraction_subset_${MODE}.json"
PREDICTION="$PROJECT_ROOT/generation/$BASELINE/results/$USER_DIR/prediction/state_abstraction_results.json"
OUTPUT="$PROJECT_ROOT/generation/$BASELINE/results/$USER_DIR/eval/state_abstraction_eval_api_${MODE}.json"

mkdir -p "$(dirname "$OUTPUT")"

cd "$PROJECT_ROOT"
"$PYTHON_BIN" -m state_abstraction_api.cli evaluate \
  --benchmark "$SUBSET_BENCHMARK" \
  --prediction "$PREDICTION" \
  --output "$OUTPUT"

echo "Done."
echo "Eval output: $OUTPUT"
