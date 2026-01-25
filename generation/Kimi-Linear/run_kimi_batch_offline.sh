#!/bin/bash
set -e # Exit on error

# Configuration
# GPU Configuration: Use all 4 GPUs
export CUDA_VISIBLE_DEVICES=0,1,2,3 
OUTPUT_DIR="results"

# Data Paths
QUESTIONS_ROOT="../../questions" 
LOGS_ROOT="../../user_app_logs" 

mkdir -p $OUTPUT_DIR

# --- Run Offline QA Benchmark ---
echo "[Runner] Starting Kimi-Linear Offline QA (TP=4)..."
source .venv/bin/activate

process_user() {
    USER_ID=$1       # e.g., 003
    SHORT_ID=$2      # e.g., user3
    LOG_DIR_PATH=$3  # Path to user log dir
    
    # Iterate Question Files
    for q_file in "$QUESTIONS_ROOT/$SHORT_ID"/qa_*.json; do
        [ -e "$q_file" ] || continue
        
        Q_NAME=$(basename "$q_file") # qa_w0.json
        
        # Determine Context File based on Q_NAME
        CONTEXT_FILE=""
        CLEAN_ID=""
        
        # Match qa_w0 (Small)
        if [[ "$Q_NAME" == "qa_w0.json" ]] || [[ "$Q_NAME" == "qa_w0_with_app_logs.json" ]]; then
            CONTEXT_FILE="$LOG_DIR_PATH/app_log_small.json"
            CLEAN_ID="qa_w0"
        # Match qa_w0_w1 (Medium)
        elif [[ "$Q_NAME" == "qa_w0_w1.json" ]] || [[ "$Q_NAME" == "qa_w0_w1_with_app_logs.json" ]]; then
            CONTEXT_FILE="$LOG_DIR_PATH/app_log_medium.json"
            CLEAN_ID="qa_w0_w1"
        # Match qa_w0_w4 (Large)
        elif [[ "$Q_NAME" == "qa_w0_w4.json" ]] || [[ "$Q_NAME" == "qa_w0_w4_with_app_logs.json" ]]; then
            CONTEXT_FILE="$LOG_DIR_PATH/app_log_large.json"
            CLEAN_ID="qa_w0_w4"
        else
            echo "[Runner] Unknown question file format: $Q_NAME. Skipping."
            continue
        fi
        
        if [ ! -f "$CONTEXT_FILE" ]; then
             echo "[Runner] Context file not found: $CONTEXT_FILE for $Q_NAME"
             continue
        fi

	    RESULT_DIR="$OUTPUT_DIR/${USER_ID}_user_${USER_ID}/prediction"
	    mkdir -p "$RESULT_DIR"

        RUN_ID="${USER_ID}_${CLEAN_ID}" # e.g. 003_qa_w0
        RESULT_FILE="$RESULT_DIR/kimi_${RUN_ID}.json"
        
        if [ -f "$RESULT_FILE" ]; then
            echo "[Runner] Skipping $RUN_ID (Exists)."
            continue
        fi

        echo "------------------------------------------------"
        echo "Processing $SHORT_ID ($USER_ID)"
        echo "Questions: $Q_NAME"
        echo "Context:   $(basename "$CONTEXT_FILE")"
        
        python kimi_offline_inference.py \
            --log_file "$CONTEXT_FILE" \
            --question_file "$q_file" \
            --output_file "$RESULT_FILE"
    done
}

# User 003
LOG_DIR_003="$LOGS_ROOT/003_user_003"
if [ -d "$LOG_DIR_003" ]; then
    process_user "003" "003_user_003" "$LOG_DIR_003"
else
    echo "[!] Log dir not found for User 003"
fi

# User 004
LOG_DIR_004="$LOGS_ROOT/004_user_004"
if [ -d "$LOG_DIR_004" ]; then
    process_user "004" "004_user_004" "$LOG_DIR_004"
else
    echo "[!] Log dir not found for User 004"
fi

echo "[Runner] All benchmarks completed."
