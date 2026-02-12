# Dynamic State Prediction API

API layer for exporting/evaluating dynamic state prediction checkpoints.

## One-click demo

```bash
bash dynamic_state_prediction_api/run_one_click_demo.sh
```

## CLI

List checkpoints:

```bash
python3 -m dynamic_state_prediction_api.cli list \
  --benchmark data_construction/generated_outputs/gemini_3_flash_preview/001_user_001/dynamic_state_prediction_benchmark.json
```

Export tasks:

```bash
python3 -m dynamic_state_prediction_api.cli export \
  --benchmark data_construction/generated_outputs/gemini_3_flash_preview/001_user_001/dynamic_state_prediction_benchmark.json \
  --app-logs data_construction/generated_outputs/gemini_3_flash_preview/001_user_001/app_log_large.json \
  --mode checkpoint \
  --checkpoint-ids cp_0001,cp_0010 \
  --output generation/rag/results/001_user_001/prediction/dynamic_state_prediction_tasks_checkpoint.json \
  --subset-benchmark-output generation/rag/results/001_user_001/prediction/dynamic_state_prediction_subset_checkpoint.json
```

Evaluate predictions:

```bash
python3 -m dynamic_state_prediction_api.cli evaluate \
  --benchmark generation/rag/results/001_user_001/prediction/dynamic_state_prediction_subset_checkpoint.json \
  --prediction generation/rag/results/001_user_001/prediction/dynamic_state_prediction_results.json \
  --output generation/rag/results/001_user_001/eval/dynamic_state_prediction_eval_api.json
```

## Python

```python
from dynamic_state_prediction_api import DynamicStatePredictionAPI
```
