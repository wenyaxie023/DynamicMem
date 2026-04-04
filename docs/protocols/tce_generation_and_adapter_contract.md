# TCE Generation and Adapter Contract

Status: active  
Scope: contributor-facing TCE generation, adapter wiring, prediction discovery, and baseline onboarding.

This document is the canonical contract for:
- baseline contributors who only need to adapt a memory/retrieval system to TCE
- maintainers who review or extend the shared TCE runner + adapter layer

If any README, run script, or baseline note conflicts with this file, this file wins.

Normative TCE protocol references:
- `docs/protocols/temporal_checkpoint_evaluation_developer_manual.md`
- `docs/runbooks/tce_execution_runbook.md`

## 1. Contributor Boundary

Baseline contributors do **not** need to modify:
- benchmark build internals
- state validation internals
- task-pack authoring internals
- evaluator metric definitions

Contributor-facing responsibility is only:
- adapt one baseline to consume a pack-first TCE benchmark
- ensure checkpoint-time memory state is built from sequential app-log ingestion
- ensure Task A/B/C querying does not pollute shared memory state
- emit prediction JSON in the evaluator-compatible schema
- wire the baseline through the unified TCE runner and tests

Shared generation config contract:
- cross-baseline knobs must use shared top-level config sections, not `baseline_params`
- canonical shared sections are:
  - `runtime`
    - includes shared per-run identity such as `user_id`
  - `llm`
  - `retriever`
  - `retrieval`
  - `final_qa`
- `baseline_params` is reserved for backend-specific knobs only

## 2. Canonical Entrypoints

Recommended single-run entrypoint:

```bash
python -m generation.run_tce --config configs/experiments/tce/<baseline>.yaml
```

Recommended batch entrypoint:

```bash
python -m generation.run_tce_batch --config configs/experiments/tce/<baseline>.yaml
```

Batch config templating supports:
- `{user_id}` for per-user path expansion
- `{run_id}` for an explicit per-run instance label
- `{run_name}` for a derived namespace:
  - `run_name = experiment_name` when `run_id == "main"`
  - otherwise `run_name = experiment_name + "__" + run_id`

Recommended naming split:
- `runtime.experiment_name`: stable experiment family / config version
- `runtime.run_id`: concrete run instance; default `main`

Recommended usage:
- use `run_id: main` for the canonical resumable run of an experiment
- use a manual `run_id` label when you explicitly want a separate run namespace
- prefer `{run_name}` in output / artifact paths so one stable `experiment_name` can support both resume and isolated runs

Both entrypoints support:
- `--dry-run` for resolved config inspection
- `*_run_settings.yaml` persistence next to prediction output

Shared user identity contract:
- `user_id` is a shared runtime / adapter field, not a baseline-specific extra
- `generation.run_tce_batch` must inject the selected batch user into shared `runtime.user_id`
- direct single-run configs may set `runtime.user_id` explicitly
- when a config used with `generation.run_tce` has exactly one top-level `users` entry, the runner may resolve shared `runtime.user_id` from that single user
- `baseline_params.user_id` is not part of the current contract

## 3. Directory Contract

Required prediction discovery layout:

```text
generation/<baseline>/results/<user_id>/prediction/*.json
```

Evaluator outputs are written to:

```text
generation/<baseline>/results/<user_id>/eval/
```

Rules:
- `<baseline>` must match `runtime.baseline` and the adapter registry key
- `<user_id>` should use canonical IDs such as `001_user_001`
- prediction JSON files in `prediction/` are the evaluator inputs

## 4. Baseline Classes

### 4.1 Unified TCE Baseline Route

All baselines must use one shared `tce_core` orchestrator.

The orchestrator drives three explicit phases:
1. `prepare_checkpoint_state`
2. `retrieve_context_for_query`
3. `answer_query`

Contributor responsibility is to implement backend hooks for these phases.
Legacy `retrieve_context(...)` callback is no longer a valid baseline contract.

Current baselines:
- `rag`
- `oracle`
- `icl`
- `hipporag2`
- `amem`
- `memoryos`
- `mem0`
- `letta`
- `memgpt`

`letta` / `memgpt` are not protocol exceptions. They receive the same task/query contract as other baselines and differ only in backend state preparation and answering behavior.

## 5. Pack-First Input Contract

Generation must default to a benchmark that already contains:
- `state_completion_pack`
- `change_tracking_pack`
- `rq3_apply_service_qa`

Generation may read legacy raw benchmarks only as backward-compat fallback. New contributor configs and new formal runs must point to a pack-first benchmark artifact.

## 6. Stateful Baseline Contract

All stateful TCE baselines must satisfy these invariants:

1. Sequential ingest
- user app logs must be ingested in chronological order
- checkpoint memory state must reflect exactly the logs visible up to that checkpoint
- canonical app-log payload is the raw app log object itself
- this applies to builder ingest, retrieval / indexing corpus construction, and inline-memory rendering
- backends may apply lossless serialization or transport wrappers, but they must not add, drop, rename, summarize, or otherwise semantically rewrite app-log fields before those stages
- agent-loop baselines may expose a single formal `run_generation(...)` entry, but they must still keep build and test logically separated through `prepare_checkpoint_state`, `retrieve_context_for_query`, and `answer_query`
- when build is part of the formal entry, `prepare_checkpoint_state(...)` is the only place allowed to advance baseline-local memory state to the current checkpoint
- checkpoint-snapshot baselines must keep build and test as two formal phases:
  - build phase sequentially ingests logs and writes builder progress plus checkpoint snapshots / `manifest.json`
  - test phase consumes those persisted checkpoint artifacts through shared `run_pipeline(...)`
- for checkpoint-snapshot baselines, `prepare_checkpoint_state(...)` must only resolve the current checkpoint's prepared snapshot / collection; it must not advance builder state during test
- build and test must consume the same shared `data.benchmark` and `data.app_logs_path` inputs rather than silently switching to a baseline-local checkpoint source

2. Checkpoint isolation
- retrieval at checkpoint `cp_i` may read only the snapshot / collection / manifest entry corresponding to `cp_i`
- later checkpoint state must never leak into earlier checkpoint queries

3. Query non-pollution
- Task A/B/C queries must not mutate shared memory state
- if the backend requires mutable query-time state, the adapter must use an isolated per-query or per-checkpoint copy

4. Resume semantics for builder baselines
- local builder progress is the authoritative source for confirmed ingest depth
- `resume` must continue ingest from the confirmed local prefix recorded in builder progress files
- message-history-derived ingest reconstruction is not part of the canonical resume contract
- for checkpoint-snapshot baselines, checkpoint snapshots / `manifest.json` are required persisted artifacts for checkpoint retrieval / testing, but they are not the authoritative source for builder/ingest resume

5. Shared orchestrator behavior
- Task A must remain per-key
- Task B and Task C must continue to use pack-first scope from the benchmark
- contributors must not fork local prediction JSON schemas
- contributors must not reintroduce baseline-local task-query semantics outside shared `tce_core`

## 7. Concurrency Contract

Prediction metadata must always record:
- `concurrency_policy`
- `requested_checkpoint_workers`
- `requested_within_checkpoint_workers`
- `effective_checkpoint_workers`
- `effective_within_checkpoint_workers`

Policy classes:
- `rag`, `oracle`, `icl`, `hipporag2`, `memoryos`, `mem0`
  - `checkpoint_parallelism = allowed`
  - `within_checkpoint_parallelism = allowed`
- `amem`
  - `checkpoint_parallelism = forbidden`
  - `within_checkpoint_parallelism = allowed`
- `letta`, `memgpt`
  - `checkpoint_parallelism = forbidden`
  - `within_checkpoint_parallelism = forbidden`

Safety rule:
- if a baseline cannot prove that checkpoint retrieval and per-question answering are read-only with respect to shared memory state, it must force both worker counts to `1`

## 8. Adapter Contract

Adapter file:

```text
generation/adapters/<baseline>.py
```

Required interface:
- `run(args: TceAdapterArgs) -> dict`

`TceAdapterArgs` fields contributors should handle:
- routing:
  - `baseline`
- inputs:
  - `user_id`
  - `benchmark`
  - `app_logs_path`
- output:
  - `output`
- runtime:
  - `max_visible_logs`
  - `resume`
  - `max_checkpoints`
  - `debug`
  - `debug_dir`
  - `save_prompt_and_raw`
  - `enable_rq3_apply_service_qa`
  - `rq3_apply_save_prompt_and_raw`
- llm:
  - `llm_provider`
  - `llm_model`
  - `llm_max_workers`
- retrieval / execution:
  - `checkpoint_workers`
  - `within_checkpoint_workers`
  - `save_every_generation_keys`
  - `retrieval_top_k`
  - `rq3_apply_retrieval_top_k`
  - `enable_final_qa`
  - `final_qa_path`
  - `final_qa_output_path`
  - `final_qa_retrieval_top_k`
  - `final_qa_save_prompt_and_raw`
- explicit retriever:
  - `retriever_provider`
  - `retriever_model`
  - `retriever_batch_size`
- backend extras:
  - string map from `baseline_params`

Expected adapter behavior:
- resolve only backend-specific config from `baseline_params`
- run one baseline on one user / one benchmark artifact
- write predictions to `args.output`
- return a payload containing at least `predictions`

## 9. Shared Config Sections

Shared runtime config:
- `runtime.user_id`
- `runtime.enable_change_reasoning`
- `runtime.enable_rq3_apply_service_qa`
- `runtime.rq3_apply_save_prompt_and_raw`
- `runtime.checkpoint_workers`
- `runtime.within_checkpoint_workers`
- `runtime.save_every_generation_keys`

Shared explicit-retriever config:
- `retriever.provider`
- `retriever.model`
- `retriever.batch_size`
- `retriever.model` is the shared retrieval / indexing embedding-model knob for explicit-retrieval baselines, including implementations that previously exposed baseline-local names such as `embedding_model` or `embedding_model_name`

Shared retrieval execution config:
- `retrieval.top_k`
- `retrieval.rq3_apply_top_k`
- `retrieval.final_qa_top_k`

Shared final-QA config:
- `final_qa.enabled`
- `final_qa.path`
- `final_qa.output_path`
- `final_qa.save_prompt_and_raw`

Rules:
- `enable_change_reasoning=true` is the standard switch for Task B generation
- `enable_rq3_apply_service_qa` remains the standard runtime switch for Task C generation
- shared retrieval top-k settings apply only to baselines that implement explicit query-time retrieval
- explicit-retrieval baselines must consume shared `QuerySpec.retrieval_query_text` directly; they must not regenerate or fallback to baseline-local retrieval query text
- agent-memory baselines such as `letta` / `memgpt` may ignore explicit-retriever config when they do not perform query-time retrieval

Letta / MemGPT self-hosted runtime note:
- canonical self-hosted local Letta uses `letta_mode: sdk`
- `letta_mode: local` means the repository's local compatibility fallback, not a self-hosted Letta server
- to target a local/self-hosted Letta server, the runtime environment must set `LETTA_BASE_URL=http://127.0.0.1:<port>` (or another explicit self-hosted base URL)
- if `LETTA_BASE_URL` is missing, SDK mode does not imply self-hosted routing
- when `.env` carries a hosted `LETTA_API_KEY`, local/self-hosted runs should explicitly `unset LETTA_API_KEY` unless the local server itself is configured with auth
- for this project, self-hosted Letta service should run on a login node rather than a transient compute-node job
- reason: TCE resume, builder-agent reuse, and viewer/debug URLs assume a stable long-lived `LETTA_BASE_URL`; compute-node teardown will invalidate that endpoint
- Task C runtime consumes one pack item per key; item-count is not a shared runtime config
- generation runtime assumes pack-first Task C data is already valid; missing/invalid `rq3_apply_service_qa` should be treated as task-pack build issues rather than a baseline runtime policy knob

## 10. Baseline Params Contract

`baseline_params` must contain backend-specific knobs only.

Examples may include:
- `snapshot_dir`
- `checkpoint_dir`
- backend storage roots / collection names / dependency config
- backend-specific size labels when needed

Contributors must document any backend-specific extras in the baseline config template.

## 11. Prediction Contract

Minimum prediction fields:
- `predictions[].checkpoint_id`
- `predictions[].snapshot_state`
- `predictions[].evidence`
- `predictions[].change_analysis` when Task B is enabled
- `predictions[].rq3_apply_answers` when Task C is enabled

Prediction metadata must remain compatible with `eval/eval_tce.py`.
Contributors must not introduce baseline-local top-level fields that the evaluator depends on.

Evidence note:
- output evidence schema remains `{"app_log_id", "evidence_content"}`
- `app_log_id` may be an empty string when a baseline cannot stably map returned memory content back to raw app logs
- current `app_log_id`-based evidence metrics remain auxiliary and should not be treated as a hard baseline capability gate

## 12. Optional Final-Checkpoint QA Hook

Baselines may optionally add a `post-TCE final-memory QA` stage after the unified orchestrator finishes Task A/B/C prediction.

Contract:
- run standard TCE Task A/B/C prediction first
- after the last executed checkpoint completes, reuse that checkpoint's memory state to answer a QA question set
- final QA should reuse the same shared hook family:
  - `prepare_checkpoint_state`
  - `retrieve_context_for_query`
  - `answer_query`
- final QA retrieval must remain checkpoint-isolated and query non-polluting
- final QA output must be written as a separate QA artifact, not embedded into the main TCE prediction JSON
- the QA artifact must follow `docs/protocols/qa_generation_and_eval_contract.md`
- the legacy `generation.run_qa` runner and QA adapter registry remain unchanged

Recommended shared config for baselines that implement this hook:
- `final_qa.enabled`
- `final_qa.path`
- `final_qa.output_path`
- `final_qa.save_prompt_and_raw`
- `retrieval.final_qa_top_k`

Agent-memory baseline note:
- `letta` / `memgpt` may implement final QA, but they do not use `retrieval.final_qa_top_k` because final QA answers come from checkpoint-scoped agent memory rather than an explicit retriever.

## 13. Evaluation Contract

Canonical evaluator:

```bash
python -m eval.eval_tce
```

Contributor validation should confirm:
- prediction artifact is discoverable under `generation/<baseline>/results/<user_id>/prediction/`
- evaluator can load the artifact without schema drift
- output lands in the corresponding `eval/` directory

## 14. Onboarding Checklist For A New TCE Baseline

1. Add `generation/adapters/<baseline>.py`
2. Register the baseline in `generation/adapters/registry.py`
3. Add `configs/experiments/tce/<baseline>.yaml`
4. Ensure the baseline consumes a pack-first benchmark by default
5. Add at least one minimal acceptance test covering:
- config dry-run
- pack-first benchmark consumption
- evaluator-compatible prediction shape
- concurrency metadata presence
6. Add or update stateful snapshot semantics coverage if the baseline maintains persistent memory state
7. Validate with:
- `python -m generation.run_tce --config ... --dry-run`
- `python -m generation.run_tce_batch --config ... --dry-run`
- a minimal executable prediction run
- `python -m eval.eval_tce ...` on the produced artifact

## 15. Source of Enforced Truth

This contract is enforced by:
- `tests/acceptance_tce_contracts.py`
- `tests/acceptance_adapters_and_cli.py`
- `tests/acceptance_tce_batch.py`
- baseline-specific minimal acceptance tests
- stateful snapshot acceptance tests
