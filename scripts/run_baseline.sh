#!/usr/bin/env bash
#
# Run a DynamicMem baseline end-to-end: prediction, then evaluation.
#
# Usage:
#   scripts/run_baseline.sh <baseline>
#
#   <baseline> is one of: amem rag oracle simplemem memoryos hipporag2
#
# It runs the matching example configs:
#   configs/experiments/tce/<baseline>_predict.yaml
#   configs/experiments/tce/<baseline>_eval.yaml
#
# Prerequisites:
#   - benchmark data downloaded under outputs/<user_id>/ (see README Quick Start)
#   - export OPENAI_API_KEY=...
#   - memoryos and hipporag2 must run in their own conda env (see README)
#
set -euo pipefail

BASELINE="${1:-}"
if [[ -z "$BASELINE" ]]; then
  echo "Usage: $0 <baseline>   (amem|rag|oracle|simplemem|memoryos|hipporag2)" >&2
  exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PREDICT_CONFIG="configs/experiments/tce/${BASELINE}_predict.yaml"
EVAL_CONFIG="configs/experiments/tce/${BASELINE}_eval.yaml"
for c in "$PREDICT_CONFIG" "$EVAL_CONFIG"; do
  if [[ ! -f "$c" ]]; then
    echo "Config not found: $c" >&2
    echo "Available baselines: amem rag oracle simplemem memoryos hipporag2" >&2
    exit 1
  fi
done

echo ">>> [1/2] Prediction  ($PREDICT_CONFIG)"
python -m baseline_prediction.run_tce --config "$PREDICT_CONFIG"

echo ">>> [2/2] Evaluation  ($EVAL_CONFIG)"
python -m evaluation.eval_tce --config "$EVAL_CONFIG"

echo ">>> Done. Scores are in the eval summary above; full payload under"
echo "    results/<baseline>/results/<user_id>/eval/"
