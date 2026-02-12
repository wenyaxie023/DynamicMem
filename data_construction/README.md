# Data Construction Pipeline

This directory owns benchmark data construction and dynamic-state-prediction benchmark building.

## Scope

- User/profile generation pipeline (Stage 1 -> Stage 2 -> Stage 3)
- App-log dataset preparation (`app_logs_final.json`, `app_log_large.json`, etc.)
- Dynamic state prediction benchmark checkpoint construction

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

## Dynamic State Prediction Benchmark

To evaluate memory behavior beyond QA, build checkpoint-based dynamic state prediction ground truth.

### 1) Build Checkpoints

```bash
cd data_construction
python3 build_dynamic_state_prediction_benchmark.py \
  --app-logs-final generated_outputs/gemini_3_flash_preview/001_user_001/app_logs_final.json
```

Or use the wrapper:

```bash
cd <repo_root>
bash data_construction/run_build_dynamic_state_prediction_benchmark.sh
```

Output:

- `data_construction/generated_outputs/gemini_3_flash_preview/001_user_001/dynamic_state_prediction_benchmark.json`

Checkpoint rule (current implementation):

- A checkpoint is created at each chain completion (the last app log for a `chain_id`).
- Valid-state filtering is applied before checkpoint export:
  - state validity is checked from `all_events_chains.json` at chain level.
  - required observable fields must be supported by the chain events.
  - if a resolved state explicitly contains `schedule_dates`, it must match the union of `events[*].time_specification.schedule_dates` for that state.
  - invalid states are removed from checkpoint targets; checkpoints with no valid states are skipped.
- New-information filtering is applied after valid-state filtering:
  - if a checkpoint has no new valid observable snapshot compared to the previous exported checkpoint, it is skipped.
  - this avoids evaluating chain completions that do not change evaluable state targets.

### 2) Produce Baseline Predictions

Prediction contract (`generation/<baseline>/results/<user_id>/prediction/dynamic_state_prediction_results.json`):

```json
{
  "predictions": [
    {
      "checkpoint_id": "cp_0001",
      "snapshot_state": {
        "habits_state:weekend_neighborhood_walk": {"timing": {"start_time": "06:30"}}
      },
      "evidence": {
        "habits_state:weekend_neighborhood_walk": ["log_00001", "log_00008"]
      }
    }
  ]
}
```

Implementation note:
- generation now uses structured response (dynamic schema) for openai/azure providers, with automatic fallback to JSON prompting for other providers.

Example baseline runner:

```bash
cd <repo_root>
python3 generation/rag/rag_dynamic_state_prediction.py \
  --benchmark data_construction/generated_outputs/gemini_3_flash_preview/001_user_001/dynamic_state_prediction_benchmark.json \
  --app-logs-path data_construction/generated_outputs/gemini_3_flash_preview/001_user_001/app_log_large.json \
  --output generation/rag/results/001_user_001/prediction/dynamic_state_prediction_results.json \
  --llm-provider openai \
  --llm-model gpt-5-mini \
  --resume
```

### 3) Evaluate Predictions

```bash
cd <repo_root>
python3 -m eval.eval_dynamic_state_prediction \
  --benchmark data_construction/generated_outputs/gemini_3_flash_preview/001_user_001/dynamic_state_prediction_benchmark.json \
  --prediction generation/<baseline>/results/001_user_001/prediction/dynamic_state_prediction_results.json \
  --output generation/<baseline>/results/001_user_001/eval/dynamic_state_prediction_eval.json
```

Main metrics:

- snapshot/value: `snapshot_value_f1_mean_on_expected`, `snapshot_value_accuracy_on_expected`, `snapshot_exact_match`
- snapshot/evidence: `snapshot_evidence_recall_mean_on_expected`, `snapshot_evidence_precision_mean_on_expected`, `snapshot_evidence_f1_mean_on_expected`
- llm-as-judge (optional): per-key correctness decisions are requested from LLM, and score is computed by program as `correct_pairs / total_pairs` (range `[0,1]`).

Enable LLM judge:

```bash
python3 -m eval.eval_dynamic_state_prediction \
  --benchmark data_construction/generated_outputs/gemini_3_flash_preview/001_user_001/dynamic_state_prediction_benchmark.json \
  --prediction generation/<baseline>/results/001_user_001/prediction/dynamic_state_prediction_results.json \
  --output generation/<baseline>/results/001_user_001/eval/dynamic_state_prediction_eval.json \
  --enable-llm-judge \
  --llm-provider openai \
  --llm-model gpt-5-mini
```

For checkpoint/date export APIs and a one-click demo, see `dynamic_state_prediction_api/README.md`.
