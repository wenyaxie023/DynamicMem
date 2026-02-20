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


## Dynamic State Prediction Evaluation

You can evaluate behavior-style dynamic state prediction (not QA) with:

```bash
python -m eval.eval_dynamic_state_prediction \
  --benchmark data_construction/generated_outputs/gemini_3_flash_preview/<user_id>/dynamic_state_prediction_benchmark.json \
  --prediction generation/<baseline>/results/<user_id>/prediction/dynamic_state_prediction_results.json \
  --output generation/<baseline>/results/<user_id>/eval/dynamic_state_prediction_eval.json \
  --save-eyeball
```


Prediction format:

- `checkpoint_id`: checkpoint id from benchmark
- `snapshot_state`: predicted full state map

Evaluation focus for this task:
- Keys are treated as fixed by benchmark.
- Main automatic metric is average value F1 across keys (`snapshot_value_f1_mean_on_expected`).

Dynamic state prediction eval implementation details:
- prediction-to-benchmark alignment is done by `metadata.checkpoint_timestamp` first; unmatched timestamps are excluded to avoid silent checkpoint-id drift.
- when `--enable-llm-judge` is enabled, results are written incrementally (checkpoint-by-checkpoint) to the output json.
- `llm_judge_raw_output`, `llm_judge_prompt`, and (if available) `llm_judge_prompt_from_response` are saved for debugging.
- openai/azure judge calls use structured response with dynamic pydantic schemas; other providers fall back to JSON parsing.
- llm-as-judge now uses per-key `score` in `[0,10]` (not only boolean correctness):
  - `llm_judge_avg_score_0_10`: per-checkpoint average key score
  - `llm_judge_score`: normalized `[0,1]` value (= avg score / 10)
- `--save-eyeball` stores `groundtruth_snapshot`, `prediction_snapshot`, `groundtruth_evidence`, and `prediction_evidence` for manual inspection.
