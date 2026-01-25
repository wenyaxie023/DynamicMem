#!/bin/bash
set -m # Enable job control

# Configuration
VLLM_PORT=8000
# GPU Configuration: Use all 4 GPUs
export CUDA_VISIBLE_DEVICES=0,1,2,3 
MODEL_PATH="moonshotai/Kimi-Linear-48B-A3B-Instruct"     
LOG_DIR="logs_kimi_qa"
OUTPUT_DIR="results_kimi_qa"

# Data Paths
QUESTIONS_ROOT="../../questions" 
# Use the relocated user logs
LOGS_ROOT="../../user_app_logs" 

mkdir -p $LOG_DIR
mkdir -p $OUTPUT_DIR

# --- 1. Start vLLM Server (TP=4) ---
echo "[Runner] Checking Kimi-Linear vLLM Server (TP=4)..."
source .venv/bin/activate

# Check if port 8000 is open
if lsof -i :$VLLM_PORT >/dev/null; then
    echo "[Runner] Server is already running on port $VLLM_PORT. Skipping startup."
    SERVER_ALREADY_RUNNING=true
else
    echo "[Runner] Starting new server..."
    # Start vLLM with Prefix Caching Enabled for Speedup
    nohup python -m vllm.entrypoints.openai.api_server \
        --model $MODEL_PATH \
        --port $VLLM_PORT \
        --trust-remote-code \
        --tensor-parallel-size 4 \
        --enable-prefix-caching \
        --gpu-memory-utilization 0.90 \
        --max-model-len 1000000 \
        > "$LOG_DIR/server.log" 2>&1 &

    SERVER_PID=$!
    echo "[Runner] Server PID: $SERVER_PID. Waiting for startup..."
    SERVER_ALREADY_RUNNING=false
    
    # Wait for server to be ready
    MAX_RETRIES=120
    count=0
    while true; do
        if grep -q "Application startup complete" "$LOG_DIR/server.log"; then
            echo "[Runner] Server is ready!"
            break
        fi
        if ! kill -0 $SERVER_PID 2>/dev/null; then
            echo "[Runner] Server process died! Check logs."
            cat "$LOG_DIR/server.log"
            exit 1
        fi
        sleep 5
        count=$((count+1))
        if [ $count -ge $MAX_RETRIES ]; then
            echo "[Runner] Timeout waiting for server."
            kill $SERVER_PID
            exit 1
        fi
        echo -n "."
    done
    echo ""
fi

# --- 2. Run QA Benchmark ---
echo "[Runner] Starting Batch QA..."

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
        if [[ "$Q_NAME" == "qa_w0.json" ]]; then
            CONTEXT_FILE="$LOG_DIR_PATH/app_log_small.json"
        elif [[ "$Q_NAME" == "qa_w0_w1.json" ]]; then
            CONTEXT_FILE="$LOG_DIR_PATH/app_log_medium.json"
        elif [[ "$Q_NAME" == "qa_w0_w4.json" ]]; then
            CONTEXT_FILE="$LOG_DIR_PATH/app_log_large.json"
        else
            echo "[Runner] Unknown question file format: $Q_NAME. Skipping."
            continue
        fi
        
        if [ ! -f "$CONTEXT_FILE" ]; then
             echo "[Runner] Context file not found: $CONTEXT_FILE for $Q_NAME"
             continue
        fi

        RUN_ID="${USER_ID}_${Q_NAME%.*}" # 003_qa_w0
        RESULT_FILE="$OUTPUT_DIR/kimi_${RUN_ID}.json"
        
        if [ -f "$RESULT_FILE" ]; then
            echo "[Runner] Skipping $RUN_ID (Exists)."
            continue
        fi

        echo "------------------------------------------------"
        echo "Processing $SHORT_ID ($USER_ID)"
        echo "Questions: $Q_NAME"
        echo "Context:   $(basename "$CONTEXT_FILE")"
        
        python kimi_qa_benchmark.py \
            --log_file "$CONTEXT_FILE" \
            --question_file "$q_file" \
            --output_file "$RESULT_FILE"
    done
}

# User 003
LOG_DIR_003="$LOGS_ROOT/003_user_003"
if [ -d "$LOG_DIR_003" ]; then
    process_user "003" "user3" "$LOG_DIR_003"
else
    echo "[!] Log dir not found for User 003"
fi

# User 004
LOG_DIR_004="$LOGS_ROOT/004_user_004"
if [ -d "$LOG_DIR_004" ]; then
    process_user "004" "user4" "$LOG_DIR_004"
else
    echo "[!] Log dir not found for User 004"
fi

# --- 3. Cleanup ---
if [ "$SERVER_ALREADY_RUNNING" = false ]; then
    echo "[Runner] Batch processing complete. Stopping server..."
    kill $SERVER_PID
    wait $SERVER_PID 2>/dev/null
    echo "[Runner] Server stopped."
else
    echo "[Runner] Batch processing complete. Leaving external server running."
fi
