export CUDA_VISIBLE_DEVICES=0
vllm serve "Qwen/Qwen2.5-14B-Instruct" \
    --quantization fp8 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.9 \
    --port 8000
