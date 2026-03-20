# Evaluation Usage

Canonical QA contract:
`docs/protocols/qa_generation_and_eval_contract.md`

This directory provides evaluation scripts for single-file or batch evaluation across baselines/users.

## Quick Start (Recommended)

1. Edit `eval/run_eval.sh` and set `BASELINES`, `USERS`, and `METRICS`.
2. Run:

```
bash eval/run_eval.sh
```

The script runs batch evaluation and prints summaries.

## CLI Usage

### Single File

```
python -m eval.eval \
  --mode single \
  --input generation/<baseline>/results/<user_id>/prediction/<file>.json \
  --metrics exact_match rouge bert_score llm_judge evidence_recall
```

### Batch

```
python -m eval.eval \
  --mode batch \
  --baselines MemAgent rag \
  --users 003_user_003 004_user_004 \
  --metrics exact_match rouge bert_score llm_judge evidence_recall \
  --output-dir <optional_output_dir>
```

## Output Locations

- Default: results are written to `eval/` next to each prediction file, e.g.
  `generation/<baseline>/results/<user_id>/eval/`
- File names:
  - `*_eval.json`: detailed results
  - `*_eval.csv`: per-sample scores
- In batch mode, if `--output-dir` is provided, a `batch_summary.csv` is also created.

## Notes

- If `--metrics` is omitted, defaults are:
  `exact_match`, `rouge`, `bert_score`, `llm_judge`, `evidence_recall`.
- Batch mode scans `generation/<baseline>/results/<user_id>/prediction/*.json`.


## TCE Evaluation

You can evaluate behavior-style TCE (not QA) with:

```bash
python -m eval.eval_tce \
  --benchmark data_construction/generated_outputs/gemini_3_flash_preview/<user_id>/tce_benchmark.json \
  --prediction generation/<baseline>/results/<user_id>/prediction/tce_results.json \
  --output generation/<baseline>/results/<user_id>/eval/tce_eval.json \
  --save-eyeball
```


Prediction format:

- `checkpoint_id`: checkpoint id from benchmark
- `snapshot_state`: predicted full state map

Evaluation focus for this task:
- Keys are treated as fixed by benchmark.
- Main automatic metric is average value F1 across keys (`snapshot_value_f1_mean_on_expected`).

TCE eval implementation details:
- prediction-to-benchmark alignment is done by `metadata.checkpoint_timestamp` first; unmatched timestamps are excluded to avoid silent checkpoint-id drift.
- when `--enable-llm-judge` is enabled, results are written incrementally (checkpoint-by-checkpoint) to the output json.
- openai/azure judge calls use structured response with dynamic pydantic schemas; other providers fall back to JSON parsing.
- TCE semantic scoring now uses slot-level LLM judge only.
- legacy whole-state / whole-change `1-5` rubric judge is removed from the active protocol.
- evidence alignment is NOT judged by llm; evidence quality still comes from app_log_id matching metrics.
- main eval JSON stores split canonical slot-eval payloads only; judge prompts and raw judge outputs belong in the separate audit artifact, not the main eval file.
- RQ3 deterministic option metrics:
  - `rq3_apply_option_extractable_item_count`
  - `rq3_apply_option_prediction_coverage_on_extractable`
  - `rq3_apply_option_accuracy_on_extractable`
  - these are computed only for apply items where the reference answer contains a stably extractable option label such as `A/B/C`
- Slot-level LLM judge request granularity:
  - Task A: one request per `state_key`, containing all slots for that key
  - Task B: one request per changed `state_key`, containing all `before/after/change_reason` slots for that key
  - Task C: one request per `(state_key, qa_id)` item, containing all answer atomic-fact slots for that item
- `--save-eyeball` stores `groundtruth_snapshot`, `prediction_snapshot`, `groundtruth_evidence`, and `prediction_evidence` for manual inspection.

### TCE Slot-Level LLM Judge I/O Schema

Generic slot judge input:
```json
{
  "state_key": "state:key",
  "slots": [
    {
      "point_id": "scp_1",
      "point_type": "field",
      "polarity": "positive",
      "reference_value": "06:30",
      "predicted_value": "06:30",
      "target_path": "timing.start_time"
    },
    {
      "point_id": "scp_2",
      "point_type": "micro",
      "polarity": "positive",
      "point_text": "The answer avoids a live conference recommendation.",
      "predicted_value": "Use webinars instead of a live conference."
    }
  ]
}
```

Generic slot judge output:
```json
{
  "judgments": [
    {
      "point_id": "scp_1",
      "analysis": "The predicted value matches the required start time.",
      "correct": true
    }
  ]
}
```

Canonical main-eval row payloads:
- Task A:
  - `snapshot_slot_eval_by_key[state_key] = { score_0_1, slot_count, slot_context, judgments }`
- Task B:
  - `change_slot_eval_by_key[state_key].before|after|state_predict|change_reason = { score_0_1, slot_count, slot_context, judgments }`
- Task C:
  - `rq3_apply_slot_eval_by_item[item_id] = { state_key, qa_id, score_0_1, slot_count, slot_context, judgments }`

### Item-Level Trend Analysis

For changed-vs-unchanged and per-key temporal trends, run:

```bash
python -m eval.analyze_tce_item_trends \
  --benchmark data_construction/generated_outputs/gemini_3_flash_preview/<user_id>/tce_benchmark.json \
  --prediction generation/rag/results/<user_id>/prediction/tce_results_topk5_perkey_impl.json \
  --output-dir generation/rag/results/<user_id>/analysis/perkey_trends \
  --group-change-mode first_seen \
  --metrics exact,f1 \
  --rolling-days 7 \
  --top-n-keys 12
```

Outputs:
- `item_level_metrics.csv`
- `changed_vs_unchanged_daily.csv`
- `key_daily_metrics.csv`
- `changed_vs_unchanged_trend.png`
- `top_keys_trend.png`

### Milestone1 TCE Analysis Pack

For checkpoint-level RQ1/RQ2/RQ3 analysis on frozen TCE artifacts, run:

```bash
python -m eval.build_tce_analysis_pack \
  --benchmark data_construction/generated_outputs/gemini_3_flash_preview/<user_id>/tce_benchmark_vnext_*.json \
  --prediction generation/rag/results/<user_id>/prediction/tce_results_*.json \
  --eval generation/rag/results/<user_id>/eval/tce_eval_*.json \
  --output-dir generation/rag/results/<user_id>/analysis/milestone1_<tag>
```

This analysis pack:
- uses `benchmark + prediction + eval` directly, not viewer payloads
- writes reusable CSV summaries, per-unit CSVs, PNG plots, and a Markdown note
- treats checkpoint as the primary x-axis
- annotates checkpoints with `actual_tokens_at_cutoff`
- computes:
  - `RQ1`: Task A / Task B state / Task C score vs checkpoint
  - `RQ2`: Task B `change_reason` as the primary attribution metric
  - `RQ3`: overlap-state `apply - know` gap

Expected outputs:
- `checkpoint_summary.csv`
- `rq1_task_scores_by_checkpoint.csv`
- `rq2_taskb_attribution_by_checkpoint.csv`
- `rq3_know_apply_gap_by_checkpoint.csv`
- `correlation_summary.csv`
- `task_a_units.csv`
- `task_b_units.csv`
- `task_c_units.csv`
- heatmap PNGs for Task A / Task B state / Task B reason / Task C / RQ3 gap
- plot PNGs for RQ1/RQ2/RQ3, evidence-vs-score scatter plots, and debug state-line overlays
- ranking CSVs for top-variable states and top positive/negative RQ3 gap states
- `milestone1_analysis_summary.md`

Historical artifact note: existing files under `results*/` and `generated_outputs/` may still use legacy names and are kept unchanged intentionally.
