#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="python3"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
OUTPUT_ROOT="$PROJECT_ROOT/outputs/gemini_3_flash_preview"
TASK_CONTRACT_VERSION="${TASK_CONTRACT_VERSION:-taskabc_v2}"
RESEARCH_FRAME_VERSION="${RESEARCH_FRAME_VERSION:-rq_20260413}"
# If unset, build_tce_benchmark.py falls back to its built-in default
# (tce_contracts.CANONICAL_RESEARCH_DOC_V2). Set this env var to override.
CANONICAL_RESEARCH_DOC="${CANONICAL_RESEARCH_DOC:-}"
RESEARCH_DOC_ARGS=()
if [[ -n "$CANONICAL_RESEARCH_DOC" ]]; then
  RESEARCH_DOC_ARGS=(--canonical-research-doc "$CANONICAL_RESEARCH_DOC")
fi

# supported user ids: "1" or "001_user_001"
USERS=("001_user_001")

normalize_user_dir() {
  local x="$1"
  if [[ "$x" =~ ^[0-9]+$ ]]; then
    printf "%03d_user_%03d" "$x" "$x"
  else
    printf "%s" "$x"
  fi
}

for user in "${USERS[@]}"; do
  user_dir="$(normalize_user_dir "$user")"
  app_logs_final="$OUTPUT_ROOT/$user_dir/app_logs_final.json"

  if [[ ! -f "$app_logs_final" ]]; then
    echo "[TCE-BENCH] SKIP $user_dir: missing $app_logs_final"
    continue
  fi

  echo "[TCE-BENCH] Building benchmark for $user_dir"
  (
    cd "$PROJECT_ROOT"
    "$PYTHON_BIN" benchmark_construction/build_tce_benchmark.py \
      --app-logs-final "$app_logs_final" \
      --task-contract-version "$TASK_CONTRACT_VERSION" \
      --research-frame-version "$RESEARCH_FRAME_VERSION" \
      "${RESEARCH_DOC_ARGS[@]}"
  )
done
