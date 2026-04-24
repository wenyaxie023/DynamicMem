# TCE Developer Code Map

Status: active
Audience: developers reviewing or changing TCE code
Last Updated: 2026-04-22

This file is not a protocol spec.
It is a short developer-facing map for finding the right code entrypoints quickly.

Active scope:
- `taskabc_v2`
- Task A
- Task C

Archive-only for new development:
- standalone Task B / `change_tracking_pack`
- `taskabc_v1`
- legacy Task C QA-style path

## 1. Main Pipeline
If you only want the big picture, remember these four layers:

1. Raw benchmark build
- file: `data_construction/build_tce_benchmark.py`
- role:
  - turn app logs / event chains into sampled checkpoints
  - write `expected_snapshot_state`
  - write `state_observability`
- not responsible for authoring tasks

2. State validation
- CLI entry: `data_construction/build_tce_state_validation.py`
- core logic: `tce_core/state_validation.py`
- role:
  - decide whether a state/key is eligible to enter later task-pack build
  - write:
    - `state_questionability`
    - `validated_snapshot_state`
    - `state_validation_summary`

3. Task-pack build
- CLI entry: `data_construction/build_tce_task_packs.py`
- core logic: `tce_core/task_packs.py`
- role:
  - turn `validated_snapshot_state` into formal prebuilt packs
  - author:
    - `state_completion_pack`
    - `rq3_apply_service_qa`

4. Runtime consumption
- file: `tce_core/pipeline.py`
- role:
  - read prebuilt packs
  - build `QuerySpec`
  - run retrieval + answer
- this is not where tasks are authored

## 2. Most Important Files
If you are reviewing TCE task authoring, focus on these files first.

### `tce_core/state_validation.py`
This decides whether a state is allowed to enter Stage 2.

Read this first when:
- a state/key is unexpectedly filtered out
- `validated_snapshot_state` looks wrong
- evidence support seems too strict or too loose

Core question answered here:
- can this state be asked about?

### `tce_core/task_packs.py`
This is the main task authoring file.

Read this first when:
- you want to change which keys get packed
- you want to change Task A item structure
- you want to change Task C item structure
- you want to inspect `filtered_keys`, `reason_codes`, or pack schema

Core question answered here:
- what task item gets written into the benchmark artifact?

### `tce_core/prompts.py`
This contains:
- task generation prompts
- validation prompts
- rewrite prompts
- runtime answer prompts

Read this first when:
- pack quality is poor because prompt wording is poor
- Task C family-specific generation needs adjustment
- validation / rewrite behavior needs to change
- you want to inspect exactly what the LLM sees

Core question answered here:
- what prompt text is shown to the model?

### `tce_core/scoring_points.py`
This builds and validates `scoring_points`.

Read this first when:
- the issue is scoring, not question authoring
- point-specific scoring behavior looks wrong
- safe fallback vs rewrite behavior matters

Core question answered here:
- how will this task be scored?

### `tce_contracts.py`
This is the shared contract / compatibility helper layer.

It is not a task authoring entrypoint.
It defines the shared rules that other layers use.

Read this first when:
- you need to know whether code is running under `taskabc_v1` or `taskabc_v2`
- you want to understand active vs legacy branching
- you want to inspect contract-level field exclusion rules
- you want to know how Task A / Task C values are normalized before pack build
- you want to know how artifact metadata is stamped

Core questions answered here:
- which contract version is this artifact using?
- what contract-specific normalization should happen before authoring or evaluation?
- what metadata must be written on benchmark / prediction / eval artifacts?

### `tce_core/pipeline.py`
This is the pack consumer.

Read this first when:
- runtime is ignoring pack-authored fields
- retrieval query vs answer prompt looks mismatched
- you want to confirm how Task A / Task C are actually executed

Core question answered here:
- how does runtime consume the prebuilt pack?

## 3. Practical Entry Points
### If you want to change which keys can enter the task pool
Look at:
- `tce_core/state_validation.py`

### If you want to change Task A item authoring
Look at:
- `tce_core/task_packs.py`

Specifically:
- `build_state_completion_pack_inplace(...)`

### If you want to change Task C v2 authoring
Look at:
- `tce_core/task_packs.py`
- `tce_core/prompts.py`

Specifically:
- Task C pack build in `task_packs.py`
- family-specific Task C v2 prompt builders in `prompts.py`

### If you want to change Task C validation or rewrite rules
Look at:
- `tce_core/task_packs.py`
- `tce_core/prompts.py`

### If you want to change scoring logic instead of prompt wording
Look at:
- `tce_core/scoring_points.py`

### If you want to change contract-level behavior instead of task logic
Look at:
- `tce_contracts.py`

Typical examples:
- changing the active/default contract version
- changing v2 field exclusion rules
- changing Task A current-state normalization
- changing Task C source-value normalization
- changing how Stage 2 task aliases like `all` are expanded
- changing how top-level contract metadata is inferred or applied

### If you want to confirm runtime pack consumption
Look at:
- `tce_core/pipeline.py`

## 4. What To Ignore At First
If your current task is only about task authoring, you can usually ignore these files on the first pass:
- `tce_core/evaluation.py`
- `tce_core/state_timeline_viewer.py`
- `tce_core/manual_review.py`
- `tce_core/final_checkpoint_qa.py`
- `tce_core/orchestrator_protocol.py`

They matter later, but they are not the main entrypoints for authoring Task A / Task C packs.

## 5. One-Line Summary
For active TCE task authoring:
- `state_validation.py` decides whether a key can enter the pool
- `task_packs.py` decides what task item gets written
- `prompts.py` decides what the model sees
- `scoring_points.py` decides how the answer is scored
- `tce_contracts.py` decides which contract rules the other layers should follow
- `pipeline.py` decides how runtime consumes the pack
