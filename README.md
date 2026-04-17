# DynamicMem

DynamicMem is a benchmark and experimentation repo for long-horizon memory systems.
It builds synthetic user histories, turns them into observable app logs, and evaluates
memory-aware baselines on both behavior-style and end-to-end tasks.

This README is the top-level onboarding entrypoint. It explains where to start and
where the canonical protocol documents live. If this README conflicts with a protocol
document or execution runbook, follow the protocol or runbook.

## Project Overview

At a high level, the repo supports four recurring activities:

1. construct benchmark data for synthetic users
2. build or inspect benchmark artifacts
3. run baseline generation pipelines
4. evaluate predictions against benchmark ground truth

DynamicMem currently uses two main evaluation settings:

- `Temporal Checkpoint Evaluation (TCE)`: checkpointed memory behavior over time
- `End-to-End QA`: question answering over the full user history

## Repository Map

- `data_construction/`: builds user histories, event chains, app logs, and TCE inputs
- `generation/`: contributor-facing generation entrypoints for DSP, QA, and TCE
- `eval/`: evaluation entrypoints and analysis utilities
- `QA/`: internal QA pipeline for task construction and QA generation
- `tce_core/`: shared TCE data structures and evaluation helpers
- `docs/`: protocol specs, runbooks, plans, and governance notes

## Start Here

For a new reader, the recommended path is:

1. Read [data_construction/README.md](data_construction/README.md) to understand how
   user data is built.
2. Inspect the generated artifact types you will see under
   `data_construction/generated_outputs/<model>/<user_id>/`.
3. Move to [generation/README.md](generation/README.md) to run a baseline.
4. Move to [eval/README.md](eval/README.md) to evaluate predictions.

If you need internal stage-level details for data construction, continue to
[data_construction/stages/README.md](data_construction/stages/README.md).

## Docs Map

Onboarding and module guides:

- This file: repo-level onboarding and navigation
- [data_construction/README.md](data_construction/README.md): module overview for data construction
- [data_construction/stages/README.md](data_construction/stages/README.md): Stage 1/2/3 implementation guide
- [generation/README.md](generation/README.md): generation entrypoints and adapter conventions
- [eval/README.md](eval/README.md): evaluation entrypoints
- [QA/README.md](QA/README.md): QA pipeline overview

Normative protocol and execution references:

- [docs/protocols/temporal_checkpoint_evaluation_developer_manual.md](docs/protocols/temporal_checkpoint_evaluation_developer_manual.md):
  canonical TCE protocol
- [docs/runbooks/tce_execution_runbook.md](docs/runbooks/tce_execution_runbook.md):
  executable TCE workflow and operational checks
- [docs/protocols/tce_generation_and_adapter_contract.md](docs/protocols/tce_generation_and_adapter_contract.md):
  contributor-facing TCE generation and adapter contract
- [docs/protocols/qa_generation_and_eval_contract.md](docs/protocols/qa_generation_and_eval_contract.md):
  canonical QA generation/evaluation contract

Documentation boundary:

- onboarding READMEs explain usage and navigation
- protocol docs define contracts and semantics
- runbooks define executable workflows
- plans record ongoing or historical work, not onboarding guidance

## Example Flow

Prepare and build data for one or more users:

```bash
cd data_construction
python download_world_backgrounds.py --model gemini_3_flash_preview --user-count 10
python batch_generation_runner.py --debug --user-count 10
```

See [data_construction/README.md](data_construction/README.md) for how the sampled
user set is resolved and how precomputed world backgrounds are downloaded from
`xiewenya/user-world-backgrounds`.

Inspect a TCE generation config without launching a full run:

```bash
python -m generation.run_tce --config configs/experiments/tce/<baseline>.yaml --dry-run
```
