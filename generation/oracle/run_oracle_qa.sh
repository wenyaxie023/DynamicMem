#!/usr/bin/env bash
set -euo pipefail

bash generation/run_qa_batch.sh \
  --config configs/baselines/oracle_qa.yaml \
  "$@"
