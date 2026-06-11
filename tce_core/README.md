# tce_core

Shared baseline_prediction/evaluation utilities for **TCE** (Temporal Checkpoint
Evaluation). The benchmark has two tasks:

- **State Completion** — reconstruct the user's current state at the checkpoint.
  In code this is the `snapshot` / `Task A` path: value prediction (`snapshot_state`)
  plus evidence prediction (`evidence` with `app_log_id` + `evidence_content` per
  state key).
- **Personalized Service** — act on the remembered state. In code this is the
  `rq3_apply` / `Task C` path.

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

Task-contract metadata:
- benchmark / prediction / eval artifacts should carry top-level:
  - `task_contract_version`
  - `research_frame_version`
  - `canonical_research_doc` when available
- current active contract family is `taskabc_v2`
- legacy frozen artifacts without explicit metadata should be interpreted as `taskabc_v1`
- the active `taskabc_v2` task set is the two tasks: State Completion (`Task A`)
  and Personalized Service (`Task C`)

Personalized Service contract:
- checkpoint may include `state_questionability` and `rq3_apply_service_qa` with per-key apply items
- prediction may include `rq3_apply_answers` with per-item answer/evidence
- the `taskabc_v2` Task C write path generates one family-specific item per key:
  - `user_communication`: `scenario`, `task_instruction`, `reference_answer`
  - structured families: `scenario`, `task_instruction`, `output_template`, `reference_output`
- scoring terminology is point-specific evaluation: `micro` and `field` are point types; `atomic fact` is only legacy shorthand for a `micro` point
- `user_communication` materializes deterministic micro `answer_scoring_points[]` from the validated state after item validation passes
- structured families materialize deterministic field-based `answer_scoring_points[]` from `reference_output` after item validation passes

Personalized Service prediction snippet:
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

LLM Judge I/O (eval layer):

State Completion judge input (`value_pairs`):
```json
{
  "habits_state:morning_walk": {
    "expected_value": {"timing": {"start_time": "06:30"}},
    "predicted_value": {"timing": {"start_time": "07:00"}}
  }
}
```

State Completion judge output (`judgments`):
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
      "evidence": {}
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
