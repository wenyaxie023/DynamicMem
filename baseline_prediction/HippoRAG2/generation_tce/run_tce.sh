#!/bin/bash
set -e

# Default arguments
USER_ID="003_user_003"
LLM="gpt-5-mini"
EMBED="text-embedding-3-small"
HIPPORAG_DIR="outputs/${USER_ID}_large/${LLM}_${EMBED}"

# Check if dir exists
if [ ! -d "$HIPPORAG_DIR" ]; then
    echo "Error: HippoRAG directory not found at $HIPPORAG_DIR"
    exit 1
fi

resolve_benchmark_path() {
    local user_dir="user_data/${USER_ID}"
    local path
    path="$(find "$user_dir" -maxdepth 1 -type f -name 'tce_benchmark_vnext_*task_packs*.json' | LC_ALL=C sort | tail -n 1)"
    if [ -z "$path" ]; then
        path="$(find "$user_dir" -maxdepth 1 -type f -name 'tce_benchmark*task_packs*.json' | LC_ALL=C sort | tail -n 1)"
    fi
    if [ -z "$path" ] && [ -f "$user_dir/tce_benchmark.json" ]; then
        path="$user_dir/tce_benchmark.json"
    fi
    if [ -z "$path" ]; then
        echo "Error: benchmark not found under $user_dir" >&2
        exit 1
    fi
    printf '%s\n' "$path"
}

echo "Running HippoRAG TCE for $USER_ID using index at $HIPPORAG_DIR"

python3 -u generation_tce/tce.py \
    --benchmark "$(resolve_benchmark_path)" \
    --app-logs-path "user_data/${USER_ID}/app_log_large.json" \
    --output "results/HippoRAG2/results/${USER_ID}/prediction/tce_results_v14_taskabc.json" \
    --hipporag-dir "$HIPPORAG_DIR" \
    --llm-provider "openai" \
    --llm-model "$LLM" \
    --debug \
    "$@"
