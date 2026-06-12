# HippoRAG2 TCE Runtime

This directory contains the protocol-aligned TCE runtime for `hipporag2`.

## Canonical Entry

- Runtime: `baseline_prediction/HippoRAG2/generation_tce/online_tce.py`
- Adapter: `baseline_prediction/adapters/hipporag2.py`
- Runner: `python -m baseline_prediction.run_tce --config configs/experiments/tce/hipporag2_predict.yaml`

## Current Design

- `hipporag2` runs through shared `tce_core.run_pipeline(...)`.
- A build phase first ingests raw app logs in chronological order and persists checkpoint snapshots plus builder progress under `baseline_params.save_dir`.
- Test-time `prepare_checkpoint_state(...)` is load-only and opens the persisted HippoRAG snapshot for the requested checkpoint.
- Retrieval is read-only and consumes shared `QuerySpec.retrieval_query_text` directly.
- Both tasks (State Completion and Personalized Service) require a pack-first benchmark artifact.

## Config Notes

- Shared embedding backend settings come from `retriever.provider`, `retriever.model`, `retriever.batch_size`.
- Shared retrieval budget comes from `retrieval.top_k`, `retrieval.rq3_apply_top_k`, `retrieval.final_qa_top_k`.
- Backend-specific storage root uses `baseline_params.save_dir`.
- Periodic builder persistence cadence uses `baseline_params.builder_save_every_logs` (default `5`).
