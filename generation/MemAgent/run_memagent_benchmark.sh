#!/bin/bash
# run_memagent_benchmark.sh
# Dual-environment benchmark script for MemAgent

# Generate data using client environment
echo "Generating benchmark data..."
venv-memagent/bin/python3 gen_benchmark_data.py

mkdir -p results_logs

# Function to run one benchmark
run_one() {
    FACTOR=$1
    FILE="benchmark_data/app_log_${FACTOR}x.json"
    RESULT_FILE="${FILE}.result.json"
    LOGFILE="results_logs/vram_${FACTOR}x.log"
    
    echo "========================================"
    echo "Checking Benchmark for ${FACTOR}x Data..."
    
    if [ -f "$RESULT_FILE" ]; then
        echo "Result file $RESULT_FILE exists. Skipping..."
        echo "========================================"
        return
    fi
    
    echo "Running Benchmark for ${FACTOR}x Data..."
    echo "========================================"
    
    # Start Server using venv-vllm
    # Set CUDA_VISIBLE_DEVICES=3 for the server process to isolate it
    export CUDA_VISIBLE_DEVICES=3
    
    # Note: Using port 8003 as per original design
    nohup venv-vllm/bin/vllm serve BytedTsinghua-SIA/RL-MemoryAgent-7B \
      --port 8003 \
      --tensor-parallel-size 1 \
      --gpu-memory-utilization 0.9 \
      --trust-remote-code > results_logs/server_${FACTOR}x.log 2>&1 &
    PID=$!
    
    echo "Server PID: $PID. Waiting for readiness on port 8003..."
    
    # Wait for ready
    MAX_RETRIES=120
    count=0
    while [ $count -lt $MAX_RETRIES ]; do
        if curl -s http://localhost:8003/v1/models > /dev/null; then
            echo "Server is ready!"
            break
        fi
        sleep 5
        count=$((count + 1))
    done
    
    if [ $count -ge $MAX_RETRIES ]; then
        echo "Server failed to start for ${FACTOR}x"
        kill -9 $PID
        return
    fi
    
    # Monitor VRAM
    (
        while true; do
            date "+%H:%M:%S" >> $LOGFILE
            nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader >> $LOGFILE
            sleep 1
        done
    ) &
    MON_PID=$!
    
    # Run Benchmark using venv-memagent
    echo "Starting client benchmark..."
    venv-memagent/bin/python3 benchmark_single.py $FILE --port 8003
    
    # Kill Monitor and Server
    kill $MON_PID
    kill $PID
    sleep 10 # wait for cleanup
}

# Run 1x, 2x, 3x, 4x
run_one 1
run_one 2
run_one 3
run_one 4

echo "All Benchmarks Completed."
