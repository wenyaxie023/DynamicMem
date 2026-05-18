# Data Construction

`data_construction` owns the user-history build pipeline that turns a persona-like
description into structured benchmark artifacts.

This README is the module-level onboarding page. It explains what this directory
produces, how to run the main entrypoint, and where to find deeper references. If
this README conflicts with a protocol document or execution runbook, follow the
protocol document or runbook.

## What This Module Owns

This module is responsible for:

- user and profile generation across multiple domains
- event-chain generation across time windows
- app-log generation from those events
- preparation of the main artifacts consumed by QA and TCE workflows

The recommended external entrypoint is `batch_generation_runner.py`.

## Three-Stage Pipeline

The pipeline is organized as three stages:

1. `Stage 1`: generate dynamic profiles and resolve conflicts
2. `Stage 2`: convert resolved profiles into event chains
3. `Stage 3`: convert event chains into chronological app logs

For stage-level implementation details, see [stages/README.md](stages/README.md).

## Quick Start

For a newcomer, the shortest reliable path is:

1. Resolve the sampled user set and download the matching world backgrounds.
2. Run the full Stage 1 -> Stage 2 -> Stage 3 pipeline.

Example:

```bash
cd data_construction
python download_world_backgrounds.py --model gemini_3_flash_preview --user-count 10
python batch_generation_runner.py --debug --user-count 10
```

By default, Stage 1 runs with `--world-bg-mode require_existing`, so the pipeline
expects world background artifacts to exist before dynamic profile generation
continues.

Precomputed world backgrounds are currently hosted at:

- `https://huggingface.co/datasets/xiewenya/user-world-backgrounds`

The helper script mirrors the runner's current persona-sampling selection using the
same sample file, seed, and user-count logic. If your sampled user set changes,
rerun the helper with matching arguments before running the main pipeline.

Stage 1 accepts either of these local input formats:

- `generated_outputs/<model_name>/<user_id>/world_backgrounds.json`
- `generated_outputs/<model_name>/<user_id>/<domain_slug>_world_background.json`

If only the aggregated file is present, Stage 1 will load it and sync the per-domain
cache files automatically. If neither format is present, Stage 1 fails fast with a
missing-prerequisite error.

Useful variants:

- Download only the second sampled user:

```bash
cd data_construction
python download_world_backgrounds.py \
  --model gemini_3_flash_preview \
  --user-count 10 \
  --user-index 2
```

- Download explicit known users without mirroring sampling:

```bash
cd data_construction
python download_world_backgrounds.py \
  --model gemini_3_flash_preview \
  --users 001_user_001 002_user_002
```

- for sampled runs, run `download_world_backgrounds.py` with the same sampling inputs
  you will use for `batch_generation_runner.py`
- for ad hoc runs driven by `--user-description`, use `--world-bg-mode generate_if_missing`
  unless you have already prepared those files manually

If you intentionally want to fall back to automatic generation for missing domains:

```bash
cd data_construction
python batch_generation_runner.py --debug --world-bg-mode generate_if_missing
```

`--dry-run-world-bg` remains available for prompt-only preparation, but it is not a
full pipeline run.

## Key Outputs

Under `generated_outputs/<model_name>/<user_id>/`, a newcomer should recognize at
least these artifacts:

- `user_basic_profile.json`: normalized user profile derived from the input description
- `dynamic_profiles_final.json`: resolved multi-domain dynamic state
- `all_events_chains.json`: aggregated event chains produced from the resolved state
- `app_logs_final.json`: final app-log sequence used by downstream pipelines
- `app_log_large.json`: large app-log view used by QA and TCE consumers
- `golden_evidence_index.json`: evidence mapping artifact for downstream inspection

These files form the handoff boundary for most downstream generation and evaluation
work in the repo.

## Handoff to TCE

`data_construction` is also the source of the artifacts used to build TCE benchmarks.
In practice, TCE build workflows start from outputs such as:

- `all_events_chains.json`
- `app_logs_final.json`
- `app_log_large.json`

This README intentionally does not duplicate the full TCE benchmark build procedure.
Use the following documents instead:

- [docs/protocols/temporal_checkpoint_evaluation_developer_manual.md](../docs/protocols/temporal_checkpoint_evaluation_developer_manual.md):
  canonical TCE task and artifact contract
- [docs/runbooks/tce_execution_runbook.md](../docs/runbooks/tce_execution_runbook.md):
  executable workflow for benchmark build, generation, and evaluation

## Where To Go Next

- [stages/README.md](stages/README.md): internal Stage 1/2/3 responsibilities and outputs
- [generation/README.md](../generation/README.md): baseline generation entrypoints
- [eval/README.md](../eval/README.md): evaluator entrypoints
- [docs/protocols/qa_generation_and_eval_contract.md](../docs/protocols/qa_generation_and_eval_contract.md):
  QA generation/evaluation contract

Documentation boundary:

- onboarding READMEs explain usage and navigation
- protocol docs define contracts and semantics
- runbooks define executable operational steps
- plans are active or historical work records, not onboarding docs
