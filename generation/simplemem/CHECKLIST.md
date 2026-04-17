# SimpleMem TCE Baseline Checklist

Status: current  
Last Updated: 2026-04-06

## 1. Architecture

Implementation entry:
- `generation/simplemem/generation_tce/tce.py`

Vendor-style upstream components:
- `generation/simplemem/upstream_vendor/models/`
- `generation/simplemem/upstream_vendor/core/`
- `generation/simplemem/upstream_vendor/database/`
- `generation/simplemem/upstream_vendor/main.py`

Adapter entry:
- `generation/adapters/simplemem.py`

Runtime shape:
- build phase sequentially ingests raw app logs
- build phase writes:
  - `builder_progress.json`
  - `simplemem_live_db/`
  - `simplemem_snapshots/manifest.json`
  - per-checkpoint snapshot bundles under `simplemem_snapshots/snapshots/`
- test phase uses shared `run_pipeline(...)`

## 2. Protocol Contract

`simplemem` now follows the shared TCE contract with one approved stateful-baseline allowance:
- builder ingest still consumes raw app log objects
- dialogue-native transport uses:
  - `User: {raw_json}`
  - `Assistant: [ingested]`
- checkpoint retrieval state may use derived memory entries
- each derived memory entry must have auditable raw-log lineage
- lineage is currently stored in companion `entry_lineage.json`, keyed by `entry_id`
- retrieval is still driven by shared `QuerySpec.retrieval_query_text`
- shared visible prompting now uses inline-memory rendering of raw app logs recovered from sidecar lineage

Still required:
- retrieval query must come directly from shared `QuerySpec.retrieval_query_text`
- Task A/B/C visible prompt text stays in shared `tce_core` prompt builders
- no local retrieval-query rewriting or fallback
- no query-time mutation of shared memory state

## 3. Memory Representation

Stored SimpleMem entries contain:
- `lossless_restatement`
- `keywords`
- `timestamp`
- `location`
- `persons`
- `entities`
- `topic`

Companion lineage artifact:
- `simplemem_live_db/entry_lineage.json`
- each snapshot DB copy carries its own `entry_lineage.json`

Retrieval flow:
1. retrieve memory entries from the checkpoint snapshot
2. keep auditable `retrieved_memory_entries` and `retrieved_app_log_ids` in retrieval metadata
3. expand retrieved entry lineage back to raw app logs for `inline_memory_blocks`
4. let shared `run_pipeline(...)` answer with the shared inline-memory prompt path

## 4. Config Contract

Shared sections remain:
- `runtime`
- `data`
- `output`
- `llm`
- `retriever`
- `retrieval`
- `final_qa`

SimpleMem-specific knobs live in `baseline_params`:
- `builder_llm_provider`
- `builder_llm_model`
- `builder_llm_max_workers`
- `builder_llm_temperature`
- `window_size`
- `overlap_size`
- `save_every_logs`
- `semantic_top_k`
- `keyword_top_k`
- `structured_top_k`
- `enable_planning`
- `enable_reflection`
- `max_reflection_rounds`
- `enable_parallel_processing`
- `max_parallel_workers`
- `enable_parallel_retrieval`
- `max_retrieval_workers`

Current formal config:
- `configs/experiments/tce/simplemem.yaml`

## 5. Resume + Usage

Authoritative builder resume source:
- `builder_progress.json`

Minimum fields to inspect:
- `confirmed_last_log_idx`
- `confirmed_app_log_id`
- `pending_dialogue_buffer`
- `previous_entries`
- `processed_count`
- `completed_snapshot_ids`
- `status`

Usage sidecars:
- `usage_cost_live.json`
- `usage_cost.json`

Tracked buckets:
- `build_memory`
- `retrieval`
- `answer_llm`

## 6. Viewer

Viewer exporter:
- `generation/simplemem/build_viewer.py`

Default outputs:
- `analysis_tools/simplemem_build_viewer/simplemem_build_viewer_data.json`
- `analysis_tools/simplemem_build_viewer/simplemem_build_viewer_report.html`

The viewer now reads:
- `builder_progress.json`
- `simplemem_live_db/`
- checkpoint timeline
- run-log tail
- derived memory entries plus their sidecar lineage
- vendored SimpleMem entry fields and retrieval metadata

## 7. Quick Verification

Dry-run config resolution:

```bash
python3.11 -m generation.run_tce --config configs/experiments/tce/simplemem.yaml --dry-run
python3.11 -m generation.run_tce_batch --config configs/experiments/tce/simplemem.yaml --dry-run
```

Targeted tests:

```bash
python3.11 -m unittest discover -s tests -p 'test_simplemem*.py'
PYTHONPATH=tests:${PYTHONPATH} python3.11 -m unittest \
  acceptance_adapters_and_cli.AdapterCliAcceptance.test_run_tce_dry_run_with_yaml
```
