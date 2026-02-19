#!/usr/bin/env bash
set -euo pipefail

python3 -m generation.run_dsp_batch \
  --config configs/experiments/dsp/nemori.yaml \
  "$@"
