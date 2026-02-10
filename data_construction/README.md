# Data Construction Pipeline

This directory owns benchmark data construction and state-abstraction benchmark building.

## Scope

- User/profile generation pipeline (Stage 1 -> Stage 2 -> Stage 3)
- App-log dataset preparation (`app_logs_final.json`, `app_log_large.json`, etc.)
- State-abstraction benchmark checkpoint construction

For stage-level implementation details, see `stages/README.md`.

## Quick Start

From repository root:

```bash
cd data_construction
python batch_generation_runner.py --debug
```

Common arguments (`batch_generation_runner.py`):

- `--provider {google,openai,aimlapi,custom}`
- `--model <model_name>`
- `--api-key <key>` / `--base-url <url>`
- `--user-count <N>`
- `--user-index <1-based>`
- `--skip-stage1` / `--skip-stage2` / `--skip-stage3`
- `--cutoff-date <YYYY-MM-DD or MM.DD>`
- `--dry-run-world-bg`
- `--output-dir <path>`

## Typical Outputs

Under `generated_outputs/<model_name>/<user_id>/`:

- `user_basic_profile.json`
- `dynamic_profiles_final.json`
- `all_events_chains.json`
- `app_logs_final.json`
- `app_log_large.json` (after `prepare_test_data.py`)
- `golden_evidence_index.json`

## State Abstraction Benchmark

To evaluate memory behavior beyond QA, build checkpoint-based state-abstraction ground truth.

### 1) Build Checkpoints

```bash
cd data_construction
python3 build_state_abstraction_benchmark.py \
  --app-logs-final generated_outputs/gemini_3_flash_preview/001_user_001/app_logs_final.json
```

Or use the wrapper:

```bash
cd <repo_root>
bash data_construction/run_build_state_abstraction_benchmark.sh
```

Output:

- `data_construction/generated_outputs/gemini_3_flash_preview/001_user_001/state_abstraction_benchmark.json`

Checkpoint rule (current implementation):

- A checkpoint is created at each chain completion (the last app log for a `chain_id`).

### 2) Produce Baseline Predictions

Prediction contract (`generation/<baseline>/results/<user_id>/prediction/state_abstraction_results.json`):

```json
{
  "predictions": [
    {
      "checkpoint_id": "cp_0001",
      "snapshot_state": {
        "habits_state:weekend_neighborhood_walk": {"timing": {"start_time": "06:30"}}
      },
      "delta_prediction": {
        "added": {},
        "updated": {},
        "removed": []
      },
      "uncertainty": {
        "habits_state:weekend_neighborhood_walk": 0.12
      }
    }
  ]
}
```

Example baseline runner:

```bash
cd <repo_root>
python3 generation/rag/rag_state_abstraction.py \
  --benchmark data_construction/generated_outputs/gemini_3_flash_preview/001_user_001/state_abstraction_benchmark.json \
  --app-logs-path data_construction/generated_outputs/gemini_3_flash_preview/001_user_001/app_log_large.json \
  --output generation/rag/results/001_user_001/prediction/state_abstraction_results.json \
  --llm-provider openai \
  --llm-model gpt-5-mini \
  --resume
```

### 3) Evaluate Predictions

```bash
cd <repo_root>
python3 -m eval.eval_state_abstraction \
  --benchmark data_construction/generated_outputs/gemini_3_flash_preview/001_user_001/state_abstraction_benchmark.json \
  --prediction generation/<baseline>/results/001_user_001/prediction/state_abstraction_results.json \
  --output generation/<baseline>/results/001_user_001/eval/state_abstraction_eval.json
```

Main metrics:

- snapshot: `snapshot_key_f1`, `snapshot_value_accuracy_on_expected`, `snapshot_exact_match`
- delta: `delta_op_f1`, `delta_exact_match`
- uncertainty: `uncertainty_brier_error`, `uncertainty_coverage`

For checkpoint/date export APIs and a one-click demo, see `state_abstraction_api/README.md`.
