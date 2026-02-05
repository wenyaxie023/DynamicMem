#!/bin/bash

# =============================================================================
# Evaluation Script for Memory Benchmark
# =============================================================================

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Define baselines to evaluate
BASELINES=(
    "rag"
    # "oracle"
    # "MemoryOS"
    # "mem0"
    # "Amem"
    # "HippoRAG2"
    # "Mamba"
    # "Kimi-Linear"
    # "MemAgent"
)

# Define users to evaluate
USERS=(
    "001_user_001"
    "002_user_002"
    "003_user_003"
    "004_user_004"
    "005_user_005"
    "006_user_006"
    "007_user_007"
    "008_user_008"
    "009_user_009"
    "010_user_010"
    # Add more users as needed
)

# Define metrics to compute
METRICS=(
    # "evidence_recall"
    # "exact_match"
    # "rouge"
    # "bert_score"
    "llm_judge"
)

# Only evaluate rag_results_topK.json for the given K values (optional)
RAG_TOPK=(
    "5"
    "10"
    "20"
)

# Output directory (optional, leave empty to save next to prediction files)
OUTPUT_DIR=""

# Enable partial re-eval (idx 151-180) by passing --enable to this script
ENABLE_PARTIAL="false"
for arg in "$@"; do
    if [[ "$arg" == "--enable" ]]; then
        ENABLE_PARTIAL="true"
    fi
done
ENABLE_FLAG=""
if [[ "$ENABLE_PARTIAL" == "true" ]]; then
    ENABLE_FLAG="--enable"
fi

# =============================================================================
# Run Evaluation
# =============================================================================

echo "=========================================="
echo "Starting Evaluation"
echo "=========================================="
echo "Baselines: ${BASELINES[*]}"
echo "Users: ${USERS[*]}"
echo "Metrics: ${METRICS[*]}"
echo "=========================================="

cd "$PROJECT_ROOT"

# Run batch evaluation
echo "Running batch evaluation..."
for user in "${USERS[@]}"; do
    for topk in "${RAG_TOPK[@]}"; do
        echo "Evaluating: $user"
        echo "----------------------------------------"
        python -m eval.eval \
            --mode batch \
            --baselines "${BASELINES[@]}" \
            --users "$user" \
            --metrics "${METRICS[@]}" \
            --llm-model gpt-5.1-chat \
            --llm-provider azure \
            --rerun-llm-failed \
            --rag-topk "$topk" \
            $ENABLE_FLAG \
            ${OUTPUT_DIR:+--output-dir "$OUTPUT_DIR"}
    done
done
 

echo "=========================================="
echo "Evaluation Complete"
echo "=========================================="
