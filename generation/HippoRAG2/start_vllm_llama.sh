export CUDA_VISIBLE_DEVICES=1
vllm serve "meta-llama/Meta-Llama-3.1-8B-Instruct" \
    --quantization fp8 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.3 \
    --port 8001
