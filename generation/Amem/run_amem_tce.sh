#!/usr/bin/env bash
set -euo pipefail

python3 -m generation.run_tce_batch \
  --config configs/baselines/amem_tce.yaml \
  "$@"
