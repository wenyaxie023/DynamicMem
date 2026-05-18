#!/usr/bin/env bash
set -euo pipefail

# Ensure PyTorch loads CUDA libs from the active conda env first.
# This avoids picking incompatible system CUDA libs (e.g., nvJitLink mismatch).
# SITE_PACKAGES=$(python3 - <<'PY'
# import site
# paths = site.getsitepackages()
# print(paths[0] if paths else "")
# PY
# )

# if [ -n "${SITE_PACKAGES}" ]; then
#   NVIDIA_LIB_PATHS=(
#     "${SITE_PACKAGES}/nvidia/nvjitlink/lib"
#     "${SITE_PACKAGES}/nvidia/cusparse/lib"
#     "${SITE_PACKAGES}/nvidia/cublas/lib"
#     "${SITE_PACKAGES}/nvidia/cuda_runtime/lib"
#   )
#   for p in "${NVIDIA_LIB_PATHS[@]}"; do
#     if [ -d "${p}" ]; then
#       export LD_LIBRARY_PATH="${p}:${LD_LIBRARY_PATH:-}"
#     fi
#   done
# fi

python3 -m generation.run_tce_batch \
  --config configs/experiments/tce/hipporag2.yaml \
  "$@"
