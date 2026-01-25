#!/bin/bash
set -mq # Enable job control

# Configuration
VLLM_PORT=8003
GPU_ID=3
# Memory Agent Model
MODEL_PATH="BytedTsinghua-SIA/RL-MemoryAgent-7B"     
LOG_DIR="logs_fixed"
MEMORY_DIR="memory_storage"
OUTPUT_DIR="results_fixed"
DATA_ROOT="../../user_data"
QUESTIONS_ROOT="../../questions"

mkdir -p $LOG_DIR
mkdir -p $MEMORY_DIR
mkdir -p $OUTPUT_DIR

# ==============================================================================
# 1. Start vLLM Server
# ==============================================================================
echo "[Runner] Checking vLLM Server..."

if ! pgrep -f "vllm.entrypoints.openai.api_server" > /dev/null; then
    echo "[Runner] Starting vLLM Server on GPU $GPU_ID..."
    source venv-vllm/bin/activate

    CUDA_VISIBLE_DEVICES=$GPU_ID python3 -m vllm.entrypoints.openai.api_server \
        --model $MODEL_PATH \
        --port $VLLM_PORT \
        --trust-remote-code \
        --gpu-memory-utilization 0.90 \
        --max-model-len 32768 \
        > "$LOG_DIR/server.log" 2>&1 &
    
    SERVER_PID=$!
    echo "[Runner] Server PID: $SERVER_PID. Waiting for startup..."
    
    # Wait for server ready
    MAX_RETRIES=60
    for i in $(seq 1 $MAX_RETRIES); do
        if grep -q "Application startup complete" "$LOG_DIR/server.log"; then
             echo "[Runner] Server is READY!"
             break
        fi
        if ! kill -0 $SERVER_PID 2>/dev/null; then
             echo "[Runner] Server DIED. Check logs."
             cat "$LOG_DIR/server.log"
             exit 1
        fi
        echo -n "."
        sleep 5
    done
    deactivate
else
    echo "[Runner] vLLM Server already running. Using existing instance."
fi

# Double check readiness
if ! curl -s http://localhost:$VLLM_PORT/health > /dev/null; then
    echo "[Runner] Server is NOT responding on port $VLLM_PORT. Exiting."
    exit 1
fi

# ==============================================================================
# 2. Run Compression Phase
# ==============================================================================
echo "========================================================"
echo "[Runner] Phase 1: Compression (MemGen)"
echo "========================================================"
source venv-memagent/bin/activate

find "$DATA_ROOT" -name "app_log_*.json" | sort | while read -r FILEPATH; do
    FILENAME=$(basename "$FILEPATH")
    SIZE_RAW="${FILENAME#app_log_}"
    SIZE="${SIZE_RAW%.json}"
    
    # Parent dir is like 003_user_003
    PARENT_DIR=$(basename "$(dirname "$FILEPATH")")
    USER_ID=$(echo "$PARENT_DIR" | cut -d'_' -f1) # 003

    OUTPUT_JSON="$MEMORY_DIR/memory_${USER_ID}_${SIZE}.json"
    
    if [ -f "$OUTPUT_JSON" ]; then
        echo "[Skip] Memory exists for User $USER_ID ($SIZE)"
        continue
    fi
    
    echo "[Processing] Compressing User $USER_ID ($SIZE)..."
    python3 mem_compress_phase.py \
        --data_file "$FILEPATH" \
        --output_file "$OUTPUT_JSON" \
        --user_id "$USER_ID" \
        --dataset_size "$SIZE"
done

# ==============================================================================
# 3. Run Answer Phase
# ==============================================================================
echo "========================================================"
echo "[Runner] Phase 2: Answering (QA)"
echo "========================================================"

# Mapping Size to Question Filename (FIXED MAPPING)
declare -A SIZE_TO_QFILE
SIZE_TO_QFILE["small"]="qa_w0_with_app_logs.json"
SIZE_TO_QFILE["medium"]="qa_w0_w1_with_app_logs.json"
SIZE_TO_QFILE["large"]="qa_w0_w4_with_app_logs.json"

for mem_file in "$MEMORY_DIR"/memory_*.json; do
    [ -e "$mem_file" ] || continue
    
    filename=$(basename "$mem_file")
    # memory_003_large.json
    USER_ID=$(echo "$filename" | cut -d'_' -f2) # 003
    SIZE_RAW=$(echo "$filename" | cut -d'_' -f3 | cut -d'.' -f1) # large
    
    # Construct Question Directory: 003_user_003
    # Assuming the pattern is matches the user_data directory logic
    USER_FULL_DIR="${USER_ID}_user_${USER_ID}"
    USER_Q_DIR="$QUESTIONS_ROOT/$USER_FULL_DIR"
    
    if [ ! -d "$USER_Q_DIR" ]; then
        echo "[Error] Question dir not found: $USER_Q_DIR"
        continue
    fi
    
    Q_FILENAME=${SIZE_TO_QFILE[$SIZE_RAW]}
    if [ -z "$Q_FILENAME" ]; then
        echo "[Error] Unknown size mapping for: $SIZE_RAW"
        continue
    fi
    
    Q_FILE_PATH="$USER_Q_DIR/$Q_FILENAME"
    if [ ! -f "$Q_FILE_PATH" ]; then
         echo "[Error] Question file not found: $Q_FILE_PATH"
         continue
    fi
    
    RESULT_FILE="$OUTPUT_DIR/answer_${USER_ID}_${SIZE_RAW}.json"
    
    if [ -f "$RESULT_FILE" ]; then
        echo "[Skip] Result exists for $USER_ID ($SIZE_RAW)"
        continue
    fi
    
    echo "[Processing] Answering for User $USER_ID ($SIZE_RAW)..."
    python mem_answer_phase.py \
        --memory_file "$mem_file" \
        --question_file "$Q_FILE_PATH" \
        --output_file "$RESULT_FILE"
done

echo "[Runner] All tasks complete."
# We do not kill the server automatically to allow manual inspection/repeated runs.
# kill $SERVER_PID
