#!/bin/bash
set -mq # Enable job control

# Configuration
VLLM_PORT=8003
GPU_ID=2
# Memory Agent Model
MODEL_PATH="BytedTsinghua-SIA/RL-MemoryAgent-7B"     
LOG_DIR="logs_answer_phase"
MEMORY_DIR="memory_storage"
OUTPUT_DIR="results_answer_phase"
QUESTIONS_ROOT="../../questions" # Assuming questions are in MemBench/questions

mkdir -p $LOG_DIR
mkdir -p $OUTPUT_DIR

# --- 1. Start vLLM Server ---
echo "[Runner] Starting vLLM Server on GPU $GPU_ID..."
source venv-vllm/bin/activate

CUDA_VISIBLE_DEVICES=$GPU_ID python -m vllm.entrypoints.openai.api_server \
    --model $MODEL_PATH \
    --port $VLLM_PORT \
    --trust-remote-code \
    --gpu-memory-utilization 0.90 \
    --max-model-len 32768 \
    > "$LOG_DIR/server.log" 2>&1 &

SERVER_PID=$!
echo "[Runner] Server PID: $SERVER_PID. Waiting for startup..."

# Wait for server to be ready
MAX_RETRIES=60
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

deactivate

# --- 2. Run Answering Phase ---
echo "[Runner] Starting Batch Answering..."
source venv-memagent/bin/activate

# Mapping Size to Question File
declare -A SIZE_TO_QFILE
SIZE_TO_QFILE["small"]="qa_w0.json"
SIZE_TO_QFILE["medium"]="qa_w0_w1.json"
SIZE_TO_QFILE["large"]="qa_w0_w4.json"

# Iterate over generated memory files
# Format: memory_{USER_ID}_{SIZE}.json (e.g., memory_003_large.json)
for mem_file in "$MEMORY_DIR"/memory_*.json; do
    [ -e "$mem_file" ] || continue
    
    filename=$(basename "$mem_file")
    # Extract User ID and Size
    # Assumes format memory_003_large.json
    USER_ID=$(echo "$filename" | cut -d'_' -f2)
    SIZE_RAW=$(echo "$filename" | cut -d'_' -f3 | cut -d'.' -f1) # "large"
    
    # Map 003 -> user3
    # Use simple string replacement: remove leading zeros? 
    # Actually user directories are user3, user4. So 003 -> user3.
    # ${USER_ID#00} removes leading 00.
    SHORT_USER_ID="${USER_ID#0}"     # 003 -> 03
    SHORT_USER_ID="${SHORT_USER_ID#0}" # 03 -> 3
    USER_Q_DIR="$QUESTIONS_ROOT/user${SHORT_USER_ID}"
    
    # Check if User Question Directory exists
    if [ ! -d "$USER_Q_DIR" ]; then
        echo "[Runner] Check: Question directory $USER_Q_DIR not found for User $USER_ID. Skipping."
        continue
    fi
    
    # Key mapping logic
    Q_FILENAME=${SIZE_TO_QFILE[$SIZE_RAW]}
    if [ -z "$Q_FILENAME" ]; then
        echo "[Runner] Check: Unknown size '$SIZE_RAW' for file $filename. Skipping."
        continue
    fi
    
    Q_FILE_PATH="$USER_Q_DIR/$Q_FILENAME"
    if [ ! -f "$Q_FILE_PATH" ]; then
         echo "[Runner] Check: Question file $Q_FILE_PATH not found. Skipping."
         continue
    fi
    
    RESULT_FILE="$OUTPUT_DIR/answer_${USER_ID}_${SIZE_RAW}.json"
    
    if [ -f "$RESULT_FILE" ]; then
        echo "[Runner] Skipping $filename (Result exists)."
        continue
    fi
    
    echo "========================================================"
    echo "[Task] Processing User: $USER_ID | Size: $SIZE_RAW"
    echo "       Memory: $mem_file"
    echo "       Questions: $Q_FILE_PATH"
    echo "========================================================"
    
    python mem_answer_phase.py \
        --memory_file "$mem_file" \
        --question_file "$Q_FILE_PATH" \
        --output_file "$RESULT_FILE"
        
done

# --- 3. Cleanup ---
echo "[Runner] Batch processing complete. Stopping server..."
kill $SERVER_PID
wait $SERVER_PID 2>/dev/null
echo "[Runner] Server stopped."
