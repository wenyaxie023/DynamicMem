#!/bin/bash
source .venv/bin/activate

# Generate data
python3 gen_benchmark_data.py

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
    
    # Start Server
    nohup vllm serve moonshotai/Kimi-Linear-48B-A3B-Instruct \
      --port 8000 \
      --tensor-parallel-size 4 \
      --max-model-len 1048576 \
      --trust-remote-code > results_logs/server_${FACTOR}x.log 2>&1 &
    PID=$!
    
    # Wait for ready
    MAX_RETRIES=60
    count=0
    while [ $count -lt $MAX_RETRIES ]; do
        if curl -s http://localhost:8000/v1/models > /dev/null; then
            break
        fi
        sleep 5
        count=$((count + 1))
    done
    
    if [ $count -ge $MAX_RETRIES ]; then
        echo "Server failed to start for ${FACTOR}x"
        kill $PID
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
    
    # Run Benchmark
    python3 benchmark_single.py $FILE
    
    # Kill Monitor and Server
    kill $MON_PID
    kill $PID
    sleep 5 # wait for cleanup
}

# Run 1x, 2x, 3x, 4x
run_one 1
run_one 2
run_one 3
run_one 4

echo "All Benchmarks Completed."
