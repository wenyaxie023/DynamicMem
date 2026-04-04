# TCE RQ3 Apply-Service Contract

Status: active supplement

Canonical protocol:
- `docs/protocols/temporal_checkpoint_evaluation_developer_manual.md`

This file only records Task C / apply-specific additions beyond the shared TCE manual.

RQ3 now measures **Know vs Apply gap** with:
- `know`: snapshot cloze/fill-the-blank score (existing TCE snapshot task)
- `apply`: apply-service QA score (new apply-only QA pack)

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
    "version": "v1",
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
            "service_category": "...",
            "question": "...",
            "reference_answer": "...",
            "answer_scoring_points": [
              {
                "point_id": "ap_q1_p1",
                "point_type": "micro",
                "polarity": "positive",
                "point_text": "..."
              }
            ],
            "gold_memory_evidence_app_log_ids": ["log_0001"],
            "retrieval_query": "...",
            "validation": {
              "is_valid": true,
              "semantic_criteria": [],
              "failed_rules": [],
              "rewrite_attempts": 1
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
          "answer": "...",
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
- vNext Task C prediction does **not** include `analysis`.
- `gold_memory_evidence_app_log_ids` are benchmark-side memory evidence anchors, not generated text.
- `validation` 不再单独暴露 `contract_checks`；programmatic structural failures、semantic failures、以及 rubric validation failures 统一进入 `failed_rules`
- build-time generator prompt now emits `service_category`, `question`, `reference_answer`, and atomic `rubric[]`; pack build converts `rubric[]` into `answer_scoring_points[]`
- `answer_scoring_points[]` are scored as binary `0/1` atomic-fact hits and averaged per item
- Task C programmatic structural check is intentionally minimal:
  - it only validates generated `rubric[]` shape
  - it does not invalidate items for question length, wording style, punctuation, or missing gold evidence anchors
  - question quality remains the responsibility of semantic validation
  - scoring quality remains the responsibility of rubric validation

Final answering prompt contract:
- four-block structure only:
  - `[Task]`
  - `[User memory]`
  - `[Output format]`
  - `[Rules]`
- `[Task]` must include:
  - `Instruction:`
  - `Query:`
- retrieval query for apply answering comes from `items[*].retrieval_query`
- apply answering prompt must include the item `question`
- `service_category` may be included as lightweight context, but no option-style scenario block is required
- `definitions` such as `evidence_content` granularity are expressed inside `[Rules]`, not a separate block

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
- old option-based apply metrics may still appear when reading legacy artifacts
- new write paths must not require option labels

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
- if `validated_state_value_signature` and `evidence_signature` are unchanged, the builder must reuse the prior checkpoint's full Task C pack
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
