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

echo "Running HippoRAG TCE for $USER_ID using index at $HIPPORAG_DIR"

python3 -u generation_tce/tce.py \
    --benchmark "user_data/${USER_ID}/tce_benchmark.json" \
    --app-logs-path "user_data/${USER_ID}/app_log_large.json" \
    --output "generation/HippoRAG2/results/${USER_ID}/prediction/results.json" \
    --hipporag-dir "$HIPPORAG_DIR" \
    --llm-provider "openai" \
    --llm-model "$LLM" \
    --debug \
    --save-prompt-and-raw \
    "$@"
