#!/usr/bin/env bash
set -euo pipefail

python3 -m generation.run_tce_batch \
  --config configs/experiments/tce/letta_v14_user1.yaml \
  "$@"
