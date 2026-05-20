# tce_core

Shared baseline_prediction/evaluation utilities for TCE:
- value prediction (`snapshot_state`)
- evidence prediction (`evidence` with `app_log_id` + `evidence_content` per state key)

Terminology (canonical):
- `sampling_strategy`: full checkpoint selection rule
- `sampling_mode`: strategy type (`exposure_token` or `calendar_time`)
- `checkpoint`: final evaluation unit at one selected cutoff

Notes:
- "anchor"/"anchor_specs" are kept as internal builder terms for backward
  compatibility and should not be treated as external API concepts.

Prediction metadata contract:
- `sampling_mode`: `exposure_token` or `calendar_time`
- `sampling_params`:
  - exposure: `{"exposure_percent": int, "tokenizer_model": str}`
  - calendar: `{"calendar_anchor_freq": str, "anchor_index": int, "anchor_timestamp": str, "actual_tokens_at_cutoff": int, "total_tokens": int, "tokenizer_model": str}`

Change task contract:
- `change_analysis`:
  - `<key>` -> `{"before": ..., "after": ..., "change_reason": str, "evidence": [{"app_log_id": ..., "evidence_content": ...}]}`

Task-contract metadata:
- benchmark / prediction / eval artifacts should carry top-level:
  - `task_contract_version`
  - `research_frame_version`
  - `canonical_research_doc` when available
- current active contract family is `taskabc_v2`
- legacy frozen artifacts without explicit metadata should be interpreted as `taskabc_v1`
- `taskabc_v2` default task set is `Task A + Task C`
- `RQ2` is analyzed via `Task A` changed-vs-unchanged slices rather than via standalone Task B

Task C personalized-service contract:
- under `taskabc_v2`, Task C supports `RQ3: Personalization Utility`
- `Task C - Task A` remains a useful diagnostic gap, but it is not the canonical task definition
- checkpoint may include `state_questionability` and `rq3_apply_service_qa` with per-key apply QA items
- prediction may include `rq3_apply_answers` with per-item answer/evidence
- full protocol supplement: `docs/protocols/tce_rq3_know_apply_contract.md`
- under active `taskabc_v2`, Task C no longer uses the legacy apply-QA-style `{service_category, question, reference_answer}` write path
- current `taskabc_v2` Task C write path generates one family-specific item per key:
  - `user_communication`: `scenario`, `task_instruction`, `reference_answer`
  - structured families: `scenario`, `task_instruction`, `output_template`, `reference_output`
- active scoring terminology is point-specific evaluation: `micro` and `field` are point types; `atomic fact` is only legacy shorthand for a `micro` point
- `user_communication` materializes deterministic micro `answer_scoring_points[]` from the validated state after item validation passes
- structured families materialize deterministic field-based `answer_scoring_points[]` from `reference_output` after item validation passes

RQ3 apply prediction snippet:
```json
{
  "rq3_apply_answers": {
    "state:key": {
      "items": [
        {
          "qa_id": "q1",
          "answer": "...",
          "evidence": [
            {
              "app_log_id": "log_0002",
              "evidence_content": "..."
            }
          ]
        }
      ]
    }
  }
}
```

BREAKING CHANGES:
- Removed legacy `anchor_*` compatibility fields from prediction metadata.
- Removed legacy parallel API entrypoint.
- Removed `tce_core/checkpoint_sampler.py`.
- Change payload uses `change_reason` (replacing `reason`).
- Removed LLM-judge `evidence_alignment`; evidence quality is measured by app_log_id matching metrics only.
- under `taskabc_v2`, Task A pack build excludes `priority`, `schedule_date`, and `schedule_dates` from answer templates and scoring points

LLM Judge I/O (eval layer):

Snapshot judge input (`value_pairs`):
```json
{
  "habits_state:morning_walk": {
    "expected_value": {"timing": {"start_time": "06:30"}},
    "predicted_value": {"timing": {"start_time": "07:00"}}
  }
}
```

Snapshot judge output (`judgments`):
```json
{
  "judgments": [
    {
      "key": "habits_state:morning_walk",
      "reason": "short text",
      "correctness": 3,
      "completeness": 4,
      "specificity": 4
    }
  ]
}
```

Change judge input (`change_pairs`):
```json
{
  "habits_state:morning_walk": {
    "expected_before": {"timing": {"start_time": "06:30"}},
    "expected_after": {"timing": {"start_time": "07:00"}},
    "predicted_before": {"timing": {"start_time": "06:30"}},
    "predicted_after": {"timing": {"start_time": "07:00"}},
    "predicted_change_reason": "routine shifted later",
    "expected_evidence_ids": ["log_0002"],
    "predicted_evidence_ids": ["log_0002"]
  }
}
```

Change judge output (`judgments`):
```json
{
  "judgments": [
    {
      "key": "habits_state:morning_walk",
      "reason": "short text",
      "before_after_correctness": 5,
      "change_reason": 4
    }
  ]
}
```

Eval I/O:

Input benchmark (minimal):
```json
{
  "checkpoints": [
    {
      "checkpoint_id": "cp_0001",
      "as_of": {"timestamp": "2025-01-01 08:00:00"},
      "expected_snapshot_state": {},
      "state_observability": {}
    }
  ]
}
```

Input prediction (minimal):
```json
{
  "predictions": [
    {
      "checkpoint_id": "cp_0001",
      "metadata": {"checkpoint_timestamp": "2025-01-01 08:00:00"},
      "snapshot_state": {},
      "evidence": {},
      "change_analysis": {}
    }
  ]
}
```

Output result (top-level):
```json
{
  "summary": {},
  "checkpoints": [],
  "prediction_alignment": {}
}
```

```python
from tce_core import evaluate_checkpoints, normalize_predictions
```
