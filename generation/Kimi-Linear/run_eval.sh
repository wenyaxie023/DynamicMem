#!/bin/bash
source .venv/bin/activate

# Start server in background
echo "Starting vLLM server..."
# Check if port 8000 is free, if not kill whatever is on it? 
# For now, assume it's free or we can just fail.
# Actually, I'll use the command from kimi.sh but modify it to use the current environment's `vllm`
# and redirect output to a log file so we can check it.

nohup vllm serve moonshotai/Kimi-Linear-48B-A3B-Instruct \
  --port 8000 \
  --tensor-parallel-size 4 \
  --max-model-len 1048576 \
  --trust-remote-code > vllm_server.log 2>&1 &

SERVER_PID=$!
echo "Server PID: $SERVER_PID"

# Wait for server to be ready
echo "Waiting for server to be ready..."
MAX_RETRIES=60
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
    echo "Server failed to start. Check vllm_server.log"
    tail -n 20 vllm_server.log
    kill $SERVER_PID
    exit 1
fi

# Run evaluation
echo "Running evaluation..."
python3 eval_kimi.py --limit 5 --output_path kimi_results_sample.json

# Cleanup
echo "Evaluation finished. Killing server..."
kill $SERVER_PID
