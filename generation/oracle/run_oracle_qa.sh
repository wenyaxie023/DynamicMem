#!/usr/bin/env bash
set -euo pipefail

bash generation/run_qa_batch.sh \
  --config configs/experiments/qa/oracle.yaml \
  "$@"
