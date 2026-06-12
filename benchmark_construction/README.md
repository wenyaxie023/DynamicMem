# Benchmark Construction (Part 2)

This directory contains the entrypoints that turn Part 1's synthetic user
trajectories into the Temporal Checkpoint Evaluation (TCE) benchmark task packs
consumed by Part 3 (baseline prediction) and Part 4 (evaluation).

Most of the construction logic lives in the shared
[`tce_core/`](../tce_core/) package (`task_packs`, `task_spec`,
`exposure_checkpoint_builder`, `questionability`, `scoring_points`,
`state_validation`, `pipeline`). Scripts here are the user-facing build
entrypoints that wire those pieces together for one user at a time.

For the generation/adapter contract and field semantics, see
[`docs/protocols/tce_generation_and_adapter_contract.md`](../docs/protocols/tce_generation_and_adapter_contract.md).

## Inputs (produced by Part 1)

For each user under `outputs/<user_id>/`:

- `app_logs_final.json`: final app-log sequence
- `app_log_large.json`: large app-log view
- `all_events_chains.json`: aggregated event chains

## Outputs

Written back under the same `outputs/<user_id>/` directory, in
several stages:

- `tce_benchmark_vnext_raw.json` (raw benchmark)
- `tce_benchmark_state_validated.json` (after state-validation pass)
- `tce_benchmark_task_packs.json` (final task packs consumed by baselines / eval)

## Entrypoints

- `build_tce_benchmark.py` — raw benchmark from Part 1 artifacts
- `build_tce_state_validation.py` — state-validation pass
- `build_tce_task_packs.py` — task-pack assembly (State Completion + Personalized Service)
- `build_tce_rq3_apply_pack.py` — Task C (personalized service) apply pack
- `verify_tce_groundtruth.py` — sanity / contract checks on a built benchmark
- `run_build_tce_benchmark.sh` — convenience wrapper that drives the steps above

## Quick Start

End-to-end benchmark build for one user (with sensible defaults):

```bash
bash benchmark_construction/run_build_tce_benchmark.sh
```

Override defaults via env vars: `PROJECT_ROOT`, `PYTHON_BIN`,
`TASK_CONTRACT_VERSION`, `RESEARCH_FRAME_VERSION`.
Set `USERS=(...)` inside the script (or fork it) to target a different list.

Direct module invocation for one stage:

```bash
python -m benchmark_construction.build_tce_benchmark \
  --app-logs-final outputs/<user_id>/app_logs_final.json \
  --task-contract-version taskabc_v2 \
  --research-frame-version rq_v2
```
