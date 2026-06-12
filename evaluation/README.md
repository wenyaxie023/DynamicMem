# Evaluation

Scores baseline predictions against the benchmark task packs. The full prediction
format and scoring schema live in the
[adapter contract](../docs/protocols/tce_generation_and_adapter_contract.md).

## Quick Start

```bash
python -m evaluation.eval_tce --config configs/experiments/tce/<baseline>_eval.yaml
```

Or point at the files directly:

```bash
python -m evaluation.eval_tce \
  --benchmark outputs/<user_id>/task_packs.json \
  --prediction baseline_prediction/<baseline>/results/<user_id>/prediction/<run_name>/tce_results.json \
  --output baseline_prediction/<baseline>/results/<user_id>/evaluation/<run_name>/tce_eval.json
```

The eval prints a summary and writes `tce_eval.json` next to the prediction.

## Prediction format

Each prediction provides:
- `checkpoint_id`
- `snapshot_state` — State Completion
- `evidence`
- `rq3_apply_answers` — Personalized Service

## Scoring

Semantic scoring uses a slot-level LLM judge. The headline scores are:
- **State Completion** → `snapshot_point_score`
- **Personalized Service** → `rq3_apply_answer_point_score`
