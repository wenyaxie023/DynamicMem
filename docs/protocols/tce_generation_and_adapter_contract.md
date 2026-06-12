# TCE Generation and Adapter Contract

The contract for adapting a memory/retrieval system to TCE: how a baseline
consumes the benchmark, builds checkpoint memory, emits predictions, and is
scored. If a README or run script conflicts with this file, this file wins.

The two tasks are **State Completion** (`snapshot_state`) and **Personalized
Service** (`rq3_apply_answers`).

## 1. Contributor boundary

A baseline contributor does **not** touch benchmark build, state validation,
task-pack authoring, or evaluator metrics. The contributor only:

- adapts one baseline to consume a pack-first benchmark
- builds checkpoint-time memory from sequential app-log ingestion
- ensures queries do not pollute shared memory state
- emits prediction JSON in the evaluator schema
- wires the baseline through the shared runner

Config convention: cross-baseline knobs live in the shared top-level sections
(`runtime`, `llm`, `retriever`, `retrieval`, `final_qa`); `baseline_params` holds
backend-specific knobs only.

## 2. Entrypoints

```bash
python -m baseline_prediction.run_tce        --config configs/experiments/tce/<baseline>_predict.yaml   # single user
python -m baseline_prediction.run_tce_batch  --config configs/experiments/tce/<baseline>_predict.yaml   # multi-user
```

- `runtime.experiment_name`: stable experiment label; `runtime.run_id`: run
  instance (default `main`). `run_name = experiment_name` when `run_id == main`,
  else `experiment_name__run_id`. Prefer `{run_name}` in output paths.
- `user_id` is a shared `runtime` field; `run_tce_batch` injects the selected
  user into `runtime.user_id`. A single-user config may set it directly, or the
  runner resolves it from a one-entry `users` list.
- Both entrypoints support `--dry-run` and persist a `*_run_settings.yaml` next to
  the prediction output.

## 3. Directory contract

```text
baseline_prediction/<baseline>/results/<user_id>/prediction/*.json   # evaluator inputs
baseline_prediction/<baseline>/results/<user_id>/evaluation/         # evaluator outputs
```

- `<baseline>` matches `runtime.baseline` and the adapter registry key.
- `<user_id>` uses canonical IDs such as `001_user_001`.

## 4. Orchestrator phases

All baselines run through the shared `tce_core` orchestrator, which drives three
phases the contributor implements:

1. `prepare_checkpoint_state`
2. `retrieve_context_for_query`
3. `answer_query`

Current baselines: `rag`, `oracle`, `hipporag2`, `amem`, `memoryos`, `simplemem`.

## 5. Pack-first input

The benchmark passed to a baseline already contains the authored task packs
(`state_completion_pack`, `rq3_apply_service_qa`). Configs must point at a
pack-first benchmark artifact.

## 6. Stateful baseline invariants

For baselines that maintain persistent memory:

- **Sequential ingest**: app logs are ingested in chronological order; checkpoint
  memory reflects exactly the logs visible up to that checkpoint. The canonical
  ingest payload is the raw app-log object; adapters may apply only a lossless
  transport wrapper (no adding, dropping, renaming, summarizing, or rewriting
  fields). Derived memory units are allowed only if each keeps auditable lineage
  back to the raw logs visible at that checkpoint.
- **Build / test split**: the build phase sequentially ingests logs and persists
  checkpoint snapshots + builder progress (`manifest.json`); the test phase
  consumes those persisted artifacts through `run_pipeline(...)`.
  `prepare_checkpoint_state(...)` is load-only and must not advance builder state.
- **Checkpoint isolation**: retrieval at checkpoint `cp_i` reads only `cp_i`'s
  state; later state must never leak into earlier queries.
- **Query non-pollution**: queries must not mutate shared memory; use an isolated
  per-query/per-checkpoint copy if the backend needs mutable query state.
- **Resume**: builder progress files are the authoritative confirmed-ingest depth;
  `resume` continues from that prefix. Destructive rebuilds (clearing existing
  state / databases / snapshots) require `runtime.allow_destructive_rebuild=true`;
  otherwise a builder that cannot safely reuse artifacts must fail closed.

## 7. Concurrency

Prediction metadata records `concurrency_policy`, the requested and effective
`checkpoint_workers`, and the requested and effective `within_checkpoint_workers`.

Per-baseline policy:

| Baseline | checkpoint parallelism | within-checkpoint parallelism |
|----------|------------------------|-------------------------------|
| `rag`, `oracle`, `hipporag2`, `memoryos` | allowed | allowed |
| `amem` | forbidden | allowed |
| `simplemem` | forbidden | forbidden |

If a baseline cannot guarantee that checkpoint retrieval and answering are
read-only w.r.t. shared memory, it must force both worker counts to `1`.

## 8. Adapter interface

Adapter file: `baseline_prediction/adapters/<baseline>.py`, exposing
`run(args: TceAdapterArgs) -> dict`.

`TceAdapterArgs` carries: routing (`baseline`); inputs (`user_id`, `benchmark`,
`app_logs_path`); `output`; runtime knobs (`resume`, `allow_destructive_rebuild`,
`max_checkpoints`, `save_prompt_and_raw`, `enable_rq3_apply_service_qa`, …); LLM
(`llm_provider`, `llm_model`, `llm_max_workers`); retriever (`retriever_provider`,
`retriever_model`, `retriever_batch_size`); retrieval/execution
(`checkpoint_workers`, `within_checkpoint_workers`, `retrieval_top_k`,
`rq3_apply_retrieval_top_k`, optional `final_qa_*`); and backend extras (the
`baseline_params` string map).

The adapter resolves backend config from `baseline_params`, runs one baseline on
one user/benchmark, writes predictions to `args.output`, and returns a payload
containing at least `predictions`.

## 9. Prediction contract

Each prediction artifact carries top-level `task_contract_version` and
`research_frame_version` (current: `taskabc_v2`, `rq_v2`), and a
`predictions` list where each entry provides:

- `checkpoint_id`
- `snapshot_state` — State Completion
- `evidence` — `{"app_log_id", "evidence_content"}` (`app_log_id` may be empty if
  the baseline cannot map memory content back to a raw log)
- `rq3_apply_answers` — Personalized Service (when `enable_rq3_apply_service_qa`)

Prediction metadata must stay compatible with `evaluation/eval_tce.py`;
contributors must not add baseline-local top-level fields the evaluator depends on.

## 10. Optional final-checkpoint QA hook

A baseline may optionally answer a separate QA set against the last checkpoint's
memory after the main run, reusing the same three phases. It must stay
checkpoint-isolated and non-polluting, and write a **separate** QA artifact (not
the main prediction JSON). Configured via the `final_qa.*` and
`retrieval.final_qa_top_k` sections; disabled by default.

## 11. Evaluation

```bash
python -m evaluation.eval_tce --config configs/experiments/tce/<baseline>_eval.yaml
```

Confirm the prediction is discoverable under
`baseline_prediction/<baseline>/results/<user_id>/prediction/`, loads without
schema drift, and writes output to the sibling `evaluation/` directory.

## 12. Onboarding a new baseline

1. Add `baseline_prediction/adapters/<baseline>.py` and register it in
   `baseline_prediction/adapters/registry.py`.
2. Add `configs/experiments/tce/<baseline>_predict.yaml` (and `_eval.yaml`),
   pointing at a pack-first benchmark.
3. Implement the three orchestrator phases; satisfy the stateful invariants (§6)
   if the baseline keeps persistent memory.
4. Validate:
   - `python -m baseline_prediction.run_tce --config ... --dry-run`
   - a minimal real prediction run
   - `python -m evaluation.eval_tce --config ...` on the produced artifact
