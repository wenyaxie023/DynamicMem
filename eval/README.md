# Evaluation Usage

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

