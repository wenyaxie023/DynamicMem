# State Abstraction API

Community-facing API layer for MemBench state abstraction.

It provides a unified workflow to:

- evaluate by **event-chain checkpoint**
- evaluate by **date** (mapped to latest checkpoint up to that day)
- export baseline tasks in a stable JSON contract
- evaluate baseline predictions in one command

## One-Click Demo

Run a full demo pipeline in one command:

```bash
cd <repo_root>
bash state_abstraction_api/run_one_click_demo.sh
```

This script executes:

1. export tasks (checkpoint mode)
2. run a demo predictor
3. evaluate and print summary

Demo outputs:

- prediction: `generation/demo/results/001_user_001/prediction/state_abstraction_results_checkpoint.json`
- eval: `generation/demo/results/001_user_001/eval/state_abstraction_eval_api_checkpoint.json`

Note: the demo predictor uses `--include-targets` + `oracle` mode to prove the pipeline is wired correctly. Replace step 2 with your own baseline for real benchmarking.

## 1) List Checkpoints

```bash
cd <repo_root>
python3 -m state_abstraction_api.cli list \
  --benchmark data_construction/generated_outputs/gemini_3_flash_preview/001_user_001/state_abstraction_benchmark.json
```

## 2) Export Tasks

### 2.1 By Checkpoint IDs

```bash
python3 -m state_abstraction_api.cli export \
  --benchmark data_construction/generated_outputs/gemini_3_flash_preview/001_user_001/state_abstraction_benchmark.json \
  --app-logs data_construction/generated_outputs/gemini_3_flash_preview/001_user_001/app_log_large.json \
  --mode checkpoint \
  --checkpoint-ids cp_0001,cp_0010,cp_0050 \
  --output generation/rag/results/001_user_001/prediction/state_abstraction_tasks_checkpoint.json \
  --subset-benchmark-output generation/rag/results/001_user_001/prediction/state_abstraction_subset_checkpoint.json
```

### 2.2 By Date(s)

```bash
python3 -m state_abstraction_api.cli export \
  --benchmark data_construction/generated_outputs/gemini_3_flash_preview/001_user_001/state_abstraction_benchmark.json \
  --app-logs data_construction/generated_outputs/gemini_3_flash_preview/001_user_001/app_log_large.json \
  --mode date \
  --dates 2023-12-31,2024-03-31,2024-06-30 \
  --output generation/rag/results/001_user_001/prediction/state_abstraction_tasks_date.json \
  --subset-benchmark-output generation/rag/results/001_user_001/prediction/state_abstraction_subset_date.json
```

Notes:

- Date mode selects one checkpoint per date: **latest checkpoint <= date 23:59:59**.
- Exported task `input.app_logs` always includes full logs visible up to checkpoint timestamp.

## 3) Baseline Prediction Contract

Your baseline should output:

```json
{
  "predictions": [
    {
      "checkpoint_id": "cp_0001",
      "snapshot_state": {
        "habits_state": {
          "weekend_neighborhood_walk": {"timing": {"start_time": "06:30", "end_time": "07:30"}}
        },
        "user_attributes_state": {},
        "preferences_state": {}
      },
      "delta_prediction": {
        "added": {},
        "updated": {},
        "removed": {}
      },
      "uncertainty": {
        "habits_state:weekend_neighborhood_walk": 0.15
      }
    }
  ]
}
```

## 4) Evaluate Predictions

```bash
python3 -m state_abstraction_api.cli evaluate \
  --benchmark generation/rag/results/001_user_001/prediction/state_abstraction_subset_checkpoint.json \
  --prediction generation/rag/results/001_user_001/prediction/state_abstraction_results.json \
  --output generation/rag/results/001_user_001/eval/state_abstraction_eval_api.json
```

Scoring policy:

- averages are computed only over checkpoints that have predictions (`checkpoint_id` matched)
- output includes `evaluated_checkpoints` and `skipped_checkpoints`

## 5) Python API

```python
from pathlib import Path
from state_abstraction_api import StateAbstractionAPI

api = StateAbstractionAPI(Path(".../state_abstraction_benchmark.json"))
rows = api.list_checkpoints()
subset = api.select_checkpoints_by_id(["cp_0001", "cp_0002"])
api.export_tasks(subset, Path(".../app_log_large.json"), Path("tasks.json"))
api.export_subset_benchmark(subset, Path("subset_benchmark.json"))
api.evaluate_predictions(Path("predictions.json"), Path("eval.json"))
```
