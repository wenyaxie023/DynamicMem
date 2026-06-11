# Evaluation Usage

This directory evaluates baseline predictions against TCE benchmark task packs.

Generation / adapter contract:
- [`docs/protocols/tce_generation_and_adapter_contract.md`](../docs/protocols/tce_generation_and_adapter_contract.md)

If this README conflicts with the contract document, follow the contract.

## Quick Start

Batch wrapper (recommended):
```bash
bash evaluation/run_eval_tce.sh
```
Override the defaults via env: `PROJECT_ROOT`, `BENCHMARK_ROOT`, `BASELINE`,
`USER_DIR`, `EXPERIMENT_NAME`, `LLM_PROVIDER`, `LLM_MODEL`, `LLM_MAX_WORKERS`.

Direct CLI (single file):
```bash
python -m evaluation.eval_tce \
  --benchmark outputs/<user_id>/<benchmark_task_packs>.json \
  --prediction baseline_prediction/<baseline>/results/<user_id>/prediction/<run_name>/tce_results.json \
  --output baseline_prediction/<baseline>/results/<user_id>/evaluation/<run_name>/tce_eval.json
```

## Output Locations

Evaluation outputs are written next to each prediction:
- `baseline_prediction/<baseline>/results/<user_id>/evaluation/<run_name>/tce_eval.json`
  — main eval payload
- optional audit artifacts in the same directory

## TCE Evaluation Details

Prediction format:

- top-level payload:
  - `task_contract_version`
  - `research_frame_version`
  - optional `canonical_research_doc`
  - `predictions`
- each prediction should provide:
  - `checkpoint_id`
  - `snapshot_state`
  - `evidence`
  - `change_analysis` when Task B is enabled
  - `rq3_apply_answers` when Task C is enabled

Evaluation focus for this task:
- Keys are treated as fixed by benchmark.
- Main automatic metric is average value F1 across keys (`snapshot_value_f1_mean_on_expected`).

TCE eval implementation details:
- prediction-to-benchmark alignment is done by `metadata.checkpoint_timestamp` first; unmatched timestamps are excluded to avoid silent checkpoint-id drift.
- when `--enable-llm-judge` is enabled, results are written incrementally (checkpoint-by-checkpoint) to the output json.
- `--resume` reuses cached slot-judge units only when their stored payload is complete and does not contain evaluator-generated `judge_error:` judgments; failed judge units are retried automatically.
- openai/azure judge calls use structured response with dynamic pydantic schemas; other providers fall back to JSON parsing.
- TCE semantic scoring now uses slot-level LLM judge only.
- legacy whole-state / whole-change `1-5` rubric judge is removed from the active protocol.
- evidence alignment is NOT judged by llm; evidence quality still comes from app_log_id matching metrics.
- main eval JSON stores split canonical slot-eval payloads only; judge prompts and raw judge outputs belong in the separate audit artifact, not the main eval file.
- Task C currently uses slot-level LLM judge over per-item `answer_scoring_points[]`.
- if a current Task C pack item is missing `answer_scoring_points[]`, evaluator treats it as invalid protocol input and fails instead of falling back to option-style scoring.
- legacy option metrics may appear only when inspecting historical artifacts; they are not part of the current write/eval contract.
- active terminology is point-specific evaluation: `field`, `list_item`, and `micro` are all scoring-point types; `atomic fact` is only legacy shorthand for a `micro` point.
- Slot-level LLM judge request granularity:
  - Task A: one request per `state_key`, containing all slots for that key
  - Task B: one request per changed `state_key`, containing all `before/after/change_reason` slots for that key
    - legacy/v1 only; active `taskabc_v2` no longer requires standalone Task B
  - Task C: one request per `(state_key, qa_id)` item, containing all answer scoring-point slots for that item
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
  - legacy/v1 only
- Task C:
  - `rq3_apply_slot_eval_by_item[item_id] = { state_key, qa_id, score_0_1, slot_count, slot_context, judgments }`
