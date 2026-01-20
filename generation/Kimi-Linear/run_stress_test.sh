#!/bin/bash
source .venv/bin/activate

# Start server in background
echo "Starting vLLM server..."
nohup vllm serve moonshotai/Kimi-Linear-48B-A3B-Instruct \
  --port 8000 \
  --tensor-parallel-size 4 \
  --max-model-len 1048576 \
  --trust-remote-code > vllm_stress.log 2>&1 &

SERVER_PID=$!
echo "Server PID: $SERVER_PID"

# Wait for server
echo "Waiting for server to be ready..."
MAX_RETRIES=120
count=0
while [ $count -lt $MAX_RETRIES ]; do
    if curl -s http://localhost:8000/v1/models > /dev/null; then
        echo "Server is ready!"
        break
    fi
    echo "Waiting for server... ($count/$MAX_RETRIES)"
    sleep 10
    count=$((count + 1))
done

if [ $count -ge $MAX_RETRIES ]; then
    echo "Server failed to start."
    tail -n 20 vllm_stress.log
    kill $SERVER_PID
    exit 1
fi

# Start VRAM monitoring
echo "Starting VRAM monitoring..."
rm -f vram_usage.log
(
    while true; do
        timestamp=$(date "+%H:%M:%S")
        smi_output=$(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits)
        echo "$timestamp, $smi_output" >> vram_usage.log
        sleep 1
    done
) &
MONITOR_PID=$!

# Run stress test
echo "Running stress test..."
python3 stress_test_kimi.py 

# Stop monitor
kill $MONITOR_PID

# Cleanup server
echo "Stress test finished. Killing server..."
kill $SERVER_PID

echo "Done. Check vram_usage.log for details."
