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

echo "Running ONLINE HippoRAG TCE for $USER_ID using index at $HIPPORAG_DIR"

python3 -u generation_tce/online_tce.py \
    --benchmark "../../user_data/${USER_ID}/tce_benchmark.json" \
    --app-logs-path "../../user_data/${USER_ID}/app_log_large.json" \
    --output "$HIPPORAG_DIR/predictions.json" \
    --save-dir "$HIPPORAG_DIR" \
    --llm-model "$LLM" \
    --embedding-model "$EMBED" \
    --batch-size 40 \
    --debug \
    --save-prompt-and-raw \
    --resume \
    "$@" | tee -a "$HIPPORAG_DIR/run.log"
