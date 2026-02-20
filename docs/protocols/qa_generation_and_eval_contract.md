# QA Protocol Contract (Single Source of Truth)

Status: active  
Scope: QA generation artifacts and QA evaluation input/output contracts.

This document is the canonical contract for:
- baseline contributors who only need to produce/evaluate QA artifacts
- maintainers who evolve the QA pipeline

If any README conflicts with this file, this file wins.

## 1. Contributor Boundary

Contributors do **not** need to know internal QA data-construction steps (`states/events/raw/task_t*`).

Contributor-facing contract is only:
- where prediction files are discovered
- JSON schema required by evaluator
- metric behavior on required fields
- unified entrypoint usage

Internal pipeline implementation under `QA/` is maintainers-only.

## 2. Directory Contract

Required layout for QA evaluation inputs:

```text
generation/<baseline>/results/<user_id>/prediction/*.json
```

- `<baseline>`: baseline name used in evaluator `--baselines`
- `<user_id>`: e.g. `003_user_003`
- each `*.json`: one QA prediction artifact

Evaluator outputs are written to:

```text
generation/<baseline>/results/<user_id>/eval/
```

## 3. QA Prediction JSON Contract

Top-level type:
- JSON array of objects

Each item:
- `query` (string, required)
- `reference` (string, required)
- `prediction` (string, optional, default empty string)
- `id` (string, optional)
- `metadata` (object, optional)
- `reference_app_logs` (array<object>, optional)

Evidence prediction contract:
- preferred location: `metadata.evidence_prediction`
- type: `array<object>`
- each evidence object should include `app_log_id` when available

Notes:
- evaluator currently reads evidence prediction from `metadata.evidence_prediction`
- evaluator currently reads golden evidence from `reference_app_logs`

## 4. Evaluation Contract

Entrypoint:

```bash
python -m eval.eval --mode single --input <prediction.json>
python -m eval.eval --mode batch --baselines <...> --users <...>
```

Default metrics when omitted:
- `exact_match`
- `rouge`
- `bert_score`
- `llm_judge`
- `evidence_recall`

Expected outputs:
- `<file_stem>_eval.json`
- `<file_stem>_eval.csv`
- batch mode with `--output-dir`: `batch_summary.csv`

## 5. Unified QA Runner Contract

Recommended entrypoint:

```bash
python -m generation.run_qa --config <yaml>
```

Supports:
- `--dry-run` to print resolved settings
- run-settings persistence (`*_run_settings.yaml`) for reproducibility

Runner is the stable interface for contributors and automation.

YAML runtime contract:
- `runtime.baseline` selects QA baseline adapter.
- currently wired baselines: `qa_pipeline`, `oracle`, `icl`, `rag`.
- `baseline_params` is passed to adapter as string map (`QaAdapterArgs.extras`).

Adding new QA baseline:
1. add `generation/qa_adapters/<baseline>.py` with `run(args: QaAdapterArgs) -> dict`
2. register baseline in `generation/qa_adapters/registry.py`
3. add config template `configs/experiments/qa/<baseline>.yaml`
4. validate with `python -m generation.run_qa --config ... --dry-run`

## 6. Internal Pipeline Contract (Maintainers)

Internal files under `data/user{user_id}` are implementation details:
- `schema.json`, `app_log_large.json`, `app_logs_final.json`
- `states.json`, `events.json`, `logs.json`, `raw.json`
- `task_t1..t6.json`, `task_final_t1..t6.json`, `tasks.json`
- `tasks_sampled.json`, `qa.json`, `qa_context.json`, `qa_refine.json`, `qa_doublecheck.json`

These are not part of contributor-facing protocol and may change with versioned migration notes.

## 7. Backward Compatibility Policy

When schema or field semantics change:
1. update this file first
2. add or update acceptance tests under `tests/acceptance_qa_*.py`
3. update downstream READMEs to reference this file
4. document migration impact in PR notes

## 8. Source of Enforced Truth

Contract is enforced by:
- `tests/acceptance_qa_contracts.py`
- `tests/acceptance_qa_runner_dry_run.py`
- `tests/acceptance_qa_runner_and_registry.py`
