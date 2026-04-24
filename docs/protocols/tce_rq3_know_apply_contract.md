# TCE Task C Personalized-Service Contract

Status: active supplement

Canonical protocol:
- `docs/protocols/temporal_checkpoint_evaluation_developer_manual.md`

This file only records Task C / apply-specific additions beyond the shared TCE manual.

Legacy note:
- the filename keeps `rq3_know_apply` for backward compatibility
- under the current `taskabc_v2` contract, Task C belongs to `RQ3: Personalization Utility`
- `Task C - Task A` is still a useful diagnostic gap, but it is not the task's canonical name
- `taskabc_v1` keeps the legacy apply-QA-style Task C contract
- `taskabc_v2` uses the structured proactive service-completion contract documented here

Under `taskabc_v2`, Task C measures personalized-service utility with:
- `Task C`: mixed-family proactive personalized-service score
- optional diagnostic gap: `Task C - Task A`

Current Task C v2 design:
- move from apply-QA-style outputs to structured proactive personalized-service completion
- keep `user_communication` in natural-language assistant-response form
- use three service-interface families:
  - `Habit-Conditioned User Communication`
  - `Preference-Conditioned Filtering Parameter Completion`
  - `Attribute-Conditioned Action Configuration`
- use the strict canonical construction mapping:
  - `habits -> Habit-Conditioned User Communication`
  - `preferences -> Preference-Conditioned Filtering Parameter Completion`
  - `attributes -> Attribute-Conditioned Action Configuration`
- treat `full-field dependency` and `point_pairability` as build-time validity rules
- use family-appropriate point-based scoring as the primary direction
- require a family-appropriate service object rather than a raw copy of the source state
- preserve one rubric-backed scoring point per required output leaf so slot scoring stays deterministic

## Benchmark Extension

Each checkpoint may include:

```json
{
  "state_questionability": {
    "state:key": {
      "is_questionable": true,
      "reason_codes": [],
      "askable_fields": ["start_time"],
      "validator_version": "qv2_l1_l2"
    }
  },
  "rq3_apply_service_qa": {
    "version": "v9",
    "generator": {
      "provider": "openai",
      "model": "gpt-5",
      "generated_at_utc": "2026-03-04T00:00:00+00:00"
    },
    "validator": {
      "provider": "azure",
      "model": "gpt-5-mini",
      "policy": {
        "max_rewrites": 2,
        "rule_and_llm_validation": true
      }
    },
    "pair_count_per_key": 1,
    "keys": {
      "state:key": {
        "items": [
          {
            "qa_id": "q1",
            "service_family": "user_communication",
            "scenario": "...",
            "task_instruction": "...",
            "reference_answer": "Send a reminder that the user's regular morning run window starts at 06:30.",
            "answer_scoring_points": [
              {
                "point_id": "ap_q1_p1",
                "point_type": "micro",
                "polarity": "positive",
                "point_text": "The answer mentions that the reminder is for a recurring morning run.",
                "reference_value": "06:30"
              }
            ],
            "gold_memory_evidence_app_log_ids": ["log_0001"],
            "retrieval_query": "...",
            "item_validation": {
              "is_valid": true,
              "semantic_criteria": [],
              "failed_rules": [],
              "rewrite_attempts": 1
            },
            "scoring_validation": {
              "is_valid": true,
              "failed_rules": [],
              "rewrite_attempts": 0
            }
          }
        ]
      }
    }
  }
}
```

## Prediction Extension

```json
{
  "rq3_apply_answers": {
    "state:key": {
      "items": [
        {
          "qa_id": "q1",
          "answer": "Send a reminder that the user's regular morning run window starts at 06:30.",
          "evidence": [
            {
              "app_log_id": "log_0001",
              "evidence_content": "..."
            }
          ]
        }
      ]
    }
  }
}
```

Prediction contract notes:
- `taskabc_v1` prediction continues to use `{answer, evidence}`.
- `taskabc_v2` prediction uses:
  - `{answer, evidence}` for `user_communication`
  - `{output, evidence}` for `information_request_construction` / `action_configuration`
- `gold_memory_evidence_app_log_ids` are benchmark-side memory evidence anchors, not generated text.
- build-time generator prompt in `taskabc_v2` emits one structured service-completion item:
  - `scenario`
  - `task_instruction`
  - `reference_answer` for `user_communication`
  - `output_template` / `reference_output` for structured families
- `taskabc_v2` structured-family `output_template` and `reference_output` must form a family-appropriate structured scoring contract
- they must not be a raw copy of the current contracted `state_value`
- after semantic acceptance, the builder must materialize structured-family scoring points programmatically so they align one-to-one with the scalar leaves in `reference_output`
- for `preferences -> Preference-Conditioned Filtering Parameter Completion`, the contracted Task C v2 source input is `{"statement": ...}` only
- auxiliary preference `signals` may remain benchmark-side support metadata, but they are not shown to the answering assistant and are not part of paired slot scoring
- `taskabc_v2` `answer_scoring_points[]` are:
  - micro answer points materialized from retained source-state fields for `user_communication`
- active `taskabc_v2` `habits -> user_communication` also prepends one micro point with `point_role = identity_gate`; this point checks whether the answer is actually about the targeted habit/routine itself
- active `taskabc_v2` `user_communication` leaf point text is generated deterministically from the code-generated field id and validated state value, not from LLM-generated scoring descriptions
  - positive-only field points materialized from `reference_output` for structured families
- active terminology: both families use point-specific evaluation; `atomic fact` is only legacy shorthand for a `micro` point, not a separate evaluation contract
- `reference_answer` for `user_communication` is a canonical gold realization for context; current slot-level eval judges the lowered `answer_scoring_points[]`, not holistic answer similarity
- when a current `taskabc_v2` `habits -> user_communication` item includes an `identity_gate` point and that gate is judged incorrect, evaluator must assign the whole item score `0` instead of averaging any remaining leaf-level points
- missing / empty `answer_scoring_points[]` is a Task C protocol violation for current pack-first eval; evaluator must fail
- `taskabc_v2` programmatic structural checks must validate:
  - canonical family mapping
  - family-appropriate structured service object
  - non-empty scenario / task instruction
  - `output_template` / `reference_output` shape match
  - rubric coverage over required `reference_output` leaves

Final answering prompt contract:
- shared Task C prompting comes from `tce_core/prompts.py`
- Task C v2 synthesis prompt must dispatch by `service_family`
- each Task C v2 family uses a separate generation prompt with family-specific example and wording
- each family-specific generation prompt should state one fixed canonical source `state_type`, not a generic `attribute|habit|preference` menu
- builder-side normalization should inject the final `service_family` field; synthesis output may omit it
- the stable runtime family id `information_request_construction` is retained for compatibility, but its paper-facing meaning is now `Preference-Conditioned Filtering Parameter Completion`
- `taskabc_v1` visible prompt uses the pack-authored item `question`
- `taskabc_v2` visible prompt reuses pack-authored item `retrieval_query` as the canonical task body
- for structured families, that `retrieval_query` must already include `output_template`
- retrieval query for apply answering comes from `items[*].retrieval_query`
- `taskabc_v2` answering prompt must require:
  - one short assistant response for `user_communication`
  - a structured `output` object for structured families
- `service_family` may be included as lightweight context

## Eval Metrics

- `rq3_apply_item_count`
- `rq3_apply_key_coverage`
- `rq3_apply_answer_point_score_mean`
- `rq3_apply_evidence_precision`
- `rq3_apply_evidence_recall`
- `rq3_apply_evidence_f1`
- `rq3_apply_evidence_app_log_id_nonempty_rate`
- `rq3_apply_evidence_content_nonempty_rate`
- `rq3_apply_evidence_content_with_id_rate`

Legacy note:
- old option-based apply metrics may still appear only when reading legacy artifacts
- old “know vs apply gap” wording may still appear in historical analysis outputs
- new write paths and current eval must not require option labels

## Execution

Build apply QA pack:

```bash
python -m data_construction.build_tce_rq3_apply_pack \
  --benchmark <benchmark.json> \
  --output <benchmark_with_rq3_apply.json> \
  --provider openai \
  --model gpt-5 \
  --item-count-per-key 1 \
  --reuse-scope key_value_signature \
  --validator-provider azure \
  --validator-model gpt-5-mini \
  --max-rewrites 2
```

Formal reuse policy:
- `--reuse-scope key_value_signature` is the canonical protocol default
- if `validated_state_value_signature` is unchanged, the builder must reuse the prior checkpoint's authored Task C item set instead of regenerating it
- when a reused Task C item set is copied into a later checkpoint, checkpoint-local gold evidence anchors must be refreshed from the current checkpoint's `state_observability`
- unchanged apply states must not be re-generated under the formal protocol

Run TCE generation:

```bash
python -m generation.run_tce --config <config.yaml>
```

Key runtime config:
- `runtime.enable_rq3_apply_service_qa: true`
- `runtime.rq3_apply_save_prompt_and_raw: true`
- optional `retrieval.rq3_apply_top_k`

Runtime semantics:
- generation runtime consumes one apply item per key from `rq3_apply_service_qa`
- under `taskabc_v2`, the builder canonicalizes `pair_count_per_key` to `1`
