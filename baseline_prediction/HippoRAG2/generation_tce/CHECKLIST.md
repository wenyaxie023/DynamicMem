# HippoRAG2 TCE Checklist

Last updated: 2026-04-05
Status: protocol-aligned

## Canonical Files

| File | Role |
|---|---|
| `configs/experiments/tce/hipporag2_predict.yaml` | Formal pack-first config |
| `generation/adapters/hipporag2.py` | Shared adapter wiring |
| `generation/HippoRAG2/generation_tce/online_tce.py` | Canonical runtime |
| `generation/HippoRAG2/generation_tce/tce.py` | Compatibility wrapper |

## Contract Checks

- [x] Uses shared `tce_core.run_pipeline(...)`
- [x] Implements `prepare_checkpoint_state(...)`
- [x] Implements `retrieve_context_for_query(...)`
- [x] Consumes shared `QuerySpec.retrieval_query_text`
- [x] Uses pack-first benchmark artifacts for Task A / B / C
- [x] Uses shared `retriever.*` config for embedding backend settings
- [x] Uses shared `retrieval.*` config for top-k settings
- [x] Emits runtime metadata under baseline name `hipporag2`
- [x] Materializes persisted checkpoint snapshots before shared test-time pipeline execution
- [x] Uses load-only `prepare_checkpoint_state(...)` during test phase
- [x] Writes authoritative builder progress for builder-level resume
- [x] Keeps query-time retrieval read-only

## Runtime Notes

- Two-phase builder:
  - `preprocess/` stores full-corpus per-log preprocessing artifacts and deduped chunk/entity/fact embeddings
  - `builder/workspace/` replays only the confirmed prefix into the active graph
- Builder workspace lives under `baseline_params.save_dir/builder/workspace`.
- Periodic snapshots live under `baseline_params.save_dir/builder/periodic`.
- Formal checkpoint snapshots live under `baseline_params.save_dir/checkpoints/<checkpoint_id>`.
- Checkpoint manifest lives at `baseline_params.save_dir/checkpoints/manifest.json`.
- Preprocess progress lives at `baseline_params.save_dir/preprocess/progress.json`.
- Preprocess manifest lives at `baseline_params.save_dir/preprocess/manifest.json`.
- Per-log preprocess artifacts live under `baseline_params.save_dir/preprocess/logs/`.
- `baseline_params.save_dir` is the canonical backend-specific storage knob.
- `baseline_params.builder_save_every_logs` controls periodic builder snapshot cadence.
- `baseline_params.interleave_build_and_test=true` enables checkpoint-by-checkpoint interleaved materialize-and-test; default formal mode remains full build then shared test.
- `baseline_params.stop_after_build=true` stops after materializing up to the requested checkpoint scope; useful for viewer/debug inspection before test.
- `baseline_params.hipporag_dir` is supported only as a backward-compatible alias in the adapter/CLI layer.

## Verification

- `python3 -m unittest discover -s tests -p 'test_hipporag2_online_no_future_leakage.py'`
- `python3 -m unittest discover -s tests -p 'acceptance_adapters_and_cli.py'`
