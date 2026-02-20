#!/usr/bin/env bash
set -euo pipefail

# Usage examples:
#   bash generation/run_qa.sh
#   bash generation/run_qa.sh --config configs/experiments/qa/oracle.yaml
#   bash generation/run_qa.sh --config configs/experiments/qa/oracle.yaml --user-id 001_user_001
#   bash generation/run_qa.sh --config configs/experiments/qa/oracle.yaml --dry-run

CONFIG="configs/experiments/qa/default.yaml"
DEFAULTS="configs/qa.default.yaml"

if [[ $# -gt 0 ]]; then
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --config)
        CONFIG="$2"
        shift 2
        ;;
      --defaults)
        DEFAULTS="$2"
        shift 2
        ;;
      *)
        break
        ;;
    esac
  done
fi

python3 -m generation.run_qa \
  --config "$CONFIG" \
  --defaults "$DEFAULTS" \
  "$@"
