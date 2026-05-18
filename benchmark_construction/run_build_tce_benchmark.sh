#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="python3"
PROJECT_ROOT="/users/4/xie00470/mem_bench/dynamicmem"
OUTPUT_ROOT="$PROJECT_ROOT/data_construction/generated_outputs/gemini_3_flash_preview"
TASK_CONTRACT_VERSION="${TASK_CONTRACT_VERSION:-taskabc_v2}"
RESEARCH_FRAME_VERSION="${RESEARCH_FRAME_VERSION:-rq_20260413}"
CANONICAL_RESEARCH_DOC="${CANONICAL_RESEARCH_DOC:-analysis_tools/tce_research_questions/001_user_001/new_research_question.md}"

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
    "$PYTHON_BIN" data_construction/build_tce_benchmark.py \
      --app-logs-final "$app_logs_final" \
      --task-contract-version "$TASK_CONTRACT_VERSION" \
      --research-frame-version "$RESEARCH_FRAME_VERSION" \
      --canonical-research-doc "$CANONICAL_RESEARCH_DOC"
  )
done
