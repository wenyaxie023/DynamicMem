#!/bin/bash
set -e

# Default arguments
USER_ID="003_user_003"
LLM="gpt-5-mini"
EMBED="text-embedding-3-large" 
# Note: Original script used 'text-embedding-3-small' in var but 'text-embedding-3-large' in arg? 
# The log says 'text-embedding-3-large'.

HIPPORAG_DIR="outputs/online_test_u003"

# Check if dir exists
if [ ! -d "$HIPPORAG_DIR" ]; then
    echo "Creating HippoRAG directory at $HIPPORAG_DIR"
    mkdir -p "$HIPPORAG_DIR"
fi

resolve_benchmark_path() {
    local user_dir="../../user_data/${USER_ID}"
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

echo "Running ONLINE HippoRAG TCE for $USER_ID using index at $HIPPORAG_DIR"

python3 -u generation_tce/online_tce.py \
    --benchmark "$(resolve_benchmark_path)" \
    --app-logs-path "../../user_data/${USER_ID}/app_log_large.json" \
    --output "$HIPPORAG_DIR/tce_results.json" \
    --save-dir "$HIPPORAG_DIR" \
    --llm-model "$LLM" \
    --embedding-model "$EMBED" \
    --batch-size 40 \
    --debug \
    --resume \
    "$@" | tee -a "$HIPPORAG_DIR/run.log"
