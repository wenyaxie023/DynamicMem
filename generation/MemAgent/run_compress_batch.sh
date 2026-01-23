#!/bin/bash

# Configuration
VLLM_PORT=8003
GPU_ID=2
MODEL_PATH="BytedTsinghua-SIA/RL-MemoryAgent-7B"
LOG_DIR="logs_compress_phase"
OUTPUT_DIR="memory_storage"
DATA_ROOT="${1:-../../user_app_logs}" # Default to user_app_logs in project root

mkdir -p $LOG_DIR
mkdir -p $OUTPUT_DIR

# --- 1. Start vLLM Server (Check if running first) ---
if ! pgrep -f "vllm.entrypoints.openai.api_server" > /dev/null; then
    echo "[Runner] Starting vLLM Server on GPU $GPU_ID..."
    source venv-vllm/bin/activate

    # Start server in background
    CUDA_VISIBLE_DEVICES=$GPU_ID python3 -m vllm.entrypoints.openai.api_server \
        --model $MODEL_PATH \
        --port $VLLM_PORT \
        --trust-remote-code \
        --gpu-memory-utilization 0.9 \
        --max-model-len 32768 \
        > "$LOG_DIR/server.log" 2>&1 &
    SERVER_PID=$!

    echo "[Runner] Server PID: $SERVER_PID. Waiting for startup..."

    # Wait for server ready
    MAX_RETRIES=60
    for i in $(seq 1 $MAX_RETRIES); do
        if curl -s http://localhost:$VLLM_PORT/health > /dev/null; then
            echo "[Runner] Server is READY!"
            break
        fi
        echo -n "."
        sleep 5
    done
else
    echo "[Runner] vLLM Server already running. Skipping startup."
fi

# Double check readiness
if ! curl -s http://localhost:$VLLM_PORT/health > /dev/null; then
    echo "[Runner] Server is NOT responding."
    exit 1
fi

# --- 2. Run Compression Phase (Dynamic Discovery) ---
echo "[Runner] Switching to MemAgent Client Environment..."
# Only deactivate if we activated vllm above (not strictly necessary to check, just deactivate)
deactivate 2>/dev/null 
source venv-memagent/bin/activate

echo "[Runner] Scanning for datasets in: $DATA_ROOT"

# Find all app_log_*.json files
find "$DATA_ROOT" -name "app_log_*.json" | sort | while read -r FILEPATH; do
    # Extract Size (small/medium/large)
    FILENAME=$(basename "$FILEPATH")
    SIZE_RAW="${FILENAME#app_log_}"
    SIZE="${SIZE_RAW%.json}"
    
    # Extract User ID from parent directory name
    # Assuming format: .../003_user_003/app_log_small.json
    PARENT_DIR=$(basename "$(dirname "$FILEPATH")")
    # Extract the leading numbers: 003_user_003 -> 003
    USER_ID=$(echo "$PARENT_DIR" | cut -d'_' -f1)
    
    OUTPUT_JSON="$OUTPUT_DIR/memory_${USER_ID}_${SIZE}.json"
    
    echo "--------------------------------------------------------"
    echo "Found: $FILEPATH"
    echo "Target User: $USER_ID | Size: $SIZE"
    
    if [ -f "$OUTPUT_JSON" ]; then
        echo "[Skip] Result already exists: $OUTPUT_JSON"
        continue
    fi
    
    echo "[Running] Compressing..."
    python3 mem_compress_phase.py \
        --data_file "$FILEPATH" \
        --output_file "$OUTPUT_JSON" \
        --user_id "$USER_ID" \
        --dataset_size "$SIZE"
        
    if [ $? -eq 0 ]; then
         echo "[Success] Finished $USER_ID - $SIZE"
    else
         echo "[Failed] Script returned error code."
    fi
done

echo "[Runner] Batch Complete. (Server left running for Phase 2)"
# Note: I removed the auto-kill to facilitate Phase 2 immediate run, or continuous usage.
# If user wants to stop, they can pkill.
