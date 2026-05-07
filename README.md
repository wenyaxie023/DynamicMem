# DynamicMem

DynamicMem is a benchmark and experimentation repo for long-horizon memory systems.
It builds synthetic user histories, turns them into observable app logs, and evaluates
memory-aware baselines on temporal-checkpoint tasks.

## Project Overview

At a high level, the repo supports four recurring activities:

1. construct benchmark data for synthetic users
2. build or inspect benchmark artifacts
3. run baseline generation pipelines
4. evaluate predictions against benchmark ground truth

The main evaluation setting is `Temporal Checkpoint Evaluation (TCE)`: checkpointed
memory behavior over time.

## Repository Map

- `data_construction/`: builds user histories, event chains, app logs, and TCE inputs
- `generation/`: baseline generation entrypoints and adapter conventions
- `eval/`: evaluation entrypoints and analysis utilities
- `tce_core/`: shared TCE data structures and evaluation helpers

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

## Example Flow

Prepare and build data for one or more users:

```bash
cd data_construction
python download_world_backgrounds.py --model gemini_3_flash_preview --user-count 10
python batch_generation_runner.py --debug --user-count 10
```

See [data_construction/README.md](data_construction/README.md) for how the sampled
user set is resolved and how precomputed world backgrounds are downloaded.

Inspect a TCE generation config without launching a full run:

```bash
python -m generation.run_tce --config configs/<config>.yaml --dry-run
```
