# Dynamic State Prediction (DSP) using HippoRAG

This directory contains the implementation for MemBench Dynamic State Prediction using HippoRAG.

> [!IMPORTANT]
> **Version Note**: This is the **canonical location** for MemBench DSP code.
> Always run DSP scripts from this directory (or via the wrapper `run_dynamic_state_prediction.sh`) to ensure compatibility with HippoRAG v2 indices.

## Key Files
- `dynamic_state_prediction.py`: Main driver script.
- `run_dynamic_state_prediction.sh`: Runner script with default arguments.
- `analyze_generation_tokens.py`: Token usage analysis for DSP generation.
- `deep_log_investigation.py`: Deep dive log analysis.
- `final_verify.py`: Final verification script.

## How to Run
```bash
# Run from parent directory or using relative paths
bash run_dynamic_state_prediction.sh --max-checkpoints 30
```

## Tips for Complexity
- **Checkpoints 1-5**: Simple (1-3 keys), fast retrieval.
- **Checkpoints 6+**: Complex (8-15+ keys). Retrieval may take >10s (up to 5 mins for massive queries).
- **LLM Timeout**: The logic includes a 60s timeout + retry to handle complex Schema generation.

## Online/Global Prefetch Mode
To run the optimized online mode with global prefetching:
```bash
# Recommended command for high throughput (adjust batch-size as needed)
nohup sh generation_dsp/run_online.sh --batch-size <LARGE_BATCH_SIZE> > outputs/online_test_u003/nohup_run.log 2>&1 &
```
- **Global Prefetch**: The script will now pre-compute OpenIE for ALL logs before starting the pipeline.
- **Large Batch Size**: Ensures efficient utilization of TPM and reduces total runtime.
