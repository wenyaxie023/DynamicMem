# Temporal Checkpoint Evaluation Developer Manual

Status: active
Owner: DynamicMem team
Last Updated: 2026-04-24

Legacy archive:
- `docs/protocols/archive/temporal_checkpoint_evaluation_developer_manual_legacy.md`

This file is the only active developer manual for TCE.
Anything marked legacy in code, old artifacts, or older docs is not part of the active contract unless this file explicitly says otherwise.

## 1. Purpose & Scope
This manual defines the active `Temporal Checkpoint Evaluation` (TCE) development contract for DynamicMem.

Goals:
- keep the active protocol short, explicit, and implementation-aligned
- separate active requirements from archived compatibility behavior
- reduce doc drift across benchmark build, task-pack build, generation, and eval

Out of scope:
- legacy protocol details
- archived `taskabc_v1` task definitions
- standalone End-to-End QA protocol details
- baseline-specific operational commands

Related active docs:
- execution runbook: `docs/runbooks/tce_execution_runbook.md`
- generation/adapter contract: `docs/protocols/tce_generation_and_adapter_contract.md`
- prompt/retrieval checklist: `docs/protocols/tce_prompt_retrieval_checklist.md`
- QA contract: `docs/protocols/qa_generation_and_eval_contract.md`
- developer code map: `docs/tce_developer_code_map.md`

## 2. Active vs Archived
Active protocol:
- contract family: `taskabc_v2`
- active benchmark objects:
  - `Task A = State Reconstruction`
  - `Task C = Personalization Utility`
- active research mapping:
  - `RQ1` is measured by `Task A`
  - `RQ2` is analyzed through `Task A` changed-vs-unchanged slices
  - `RQ3` is measured by `Task C`

Archived for new development:
- `taskabc_v1`
- standalone `Task B / change_tracking_pack`
- legacy Task C apply-QA-style contract
- raw-benchmark fallback as a normative path
- legacy prompt/retrieval regeneration behavior

Rule:
- archived items may still exist in old artifacts or compatibility code paths
- they are invalid for new protocol design, new benchmark authoring, and new acceptance criteria

## 3. Artifact Metadata & Compatibility
All new benchmark / prediction / eval artifacts must carry:
- `task_contract_version`
- `research_frame_version`
- `canonical_research_doc` when available

Active value:
- `task_contract_version = taskabc_v2`

Compatibility rule:
- artifacts created before `2026-04-13` that lack explicit `task_contract_version` may still be interpreted as `taskabc_v1`
- that compatibility behavior exists only for archive reading, not for active development

Pack names kept for compatibility:
- `state_completion_pack`
- `rq3_apply_service_qa`

Archive-only pack name:
- `change_tracking_pack`

## 4. Active Execution Stages
Stage 0. Raw benchmark build
- entrypoint: `data_construction/build_tce_benchmark.py`
- outputs:
  - sampled checkpoints
  - `expected_snapshot_state`
  - `state_observability`
- does not do state validation
- does not author task packs

Stage 1. Independent state validation
- entrypoint: `data_construction/build_tce_state_validation.py`
- outputs:
  - `state_questionability`
  - `validated_snapshot_state`
  - `state_validation_summary`

Stage 2. Prebuilt task packs
- entrypoint: `data_construction/build_tce_task_packs.py`
- outputs:
  - `state_completion_pack`
  - `rq3_apply_service_qa`

Stage 3. Generation / evaluation
- runtime must consume prebuilt packs
- Task A consumes `state_completion_pack`
- Task C consumes `rq3_apply_service_qa`
- for stateful baselines with persisted checkpoint memory artifacts:
  - build/reuse validity is determined by source-timeline inputs needed to materialize checkpoint memory state
  - this source scope includes the app-log stream and checkpoint cut definitions, not the authored Task A / Task C evaluation packs
  - if source-timeline inputs are unchanged, later benchmark revisions may reuse previously built memory artifacts and rerun generation/evaluation only
  - changing only the authored evaluation packs must not silently discard, clear, or rebuild existing persisted memory artifacts
  - if a backend cannot safely resume from the existing persisted memory state, generation must fail closed and require an explicit destructive-rebuild opt-in rather than deleting prior memory by default

Optional post-hook:
- final-memory QA is explicit opt-in only
- it is not part of the Task A / Task C main contract

## 5. Checkpoint Semantics
A checkpoint is the canonical time-slice evaluation unit.

Shared rules:
- each checkpoint has one target state view and one observability view
- all tasks are bounded by memory visible up to that checkpoint
- benchmark build is the canonical place where checkpoint sampling is materialized
- benchmark sampling metadata must be stored in the benchmark artifact, not regenerated during generation

## 6. Task-Level Ground-Truth Resolution
`validated_snapshot_state` is task-agnostic.
Before authoring active task packs, the builder must apply deterministic code-only task-level resolution.

Transition-style state rule:
- if a validated object has only top-level keys from `{"from", "to"}`, treat it as transition-style

Task A current-state projection:
- non-transition state:
  - use the validated value directly
- transition state:
  - use `.to`
- if current state cannot be resolved, filter the key from `state_completion_pack`

Task C current-state projection:
- must reuse the Task A current-state projection
- if current state cannot be resolved, filter the key from `rq3_apply_service_qa`

Active v2 pre-authoring filtering rule:
- `priority`, `schedule_date`, and `schedule_dates` now belong to the Stage 1 pre-validation exclusion set, together with preference support fields such as `signal` / `signals`
- Stage 2 may still defensively normalize these fields away when reading older artifacts, but this is compatibility behavior rather than a separate active filtering step

Implementation rules:
- this layer must be deterministic and code-only
- filtered keys must record `reason_codes`
- `validated_snapshot_state` itself must not be rewritten by Stage 2 filtering

## 7. Stage 1 State Validation Contract
Stage 1 determines whether a state is sufficiently evidence-supported to enter active task-pack build.

Inputs:
- `expected_snapshot_state`
- `state_observability`
- checkpoint-bounded evidence logs

Required outputs per checkpoint:
- `state_questionability`
- `validated_snapshot_state`
- `state_validation_summary`

Required behavior:
- validation is field-aware, not just key-aware
- `validated_snapshot_state` must agree with the accepted field-level result
- evidence must be recomputed from checkpoint-local `state_observability`
- resume must skip already valid Stage 1 results when signatures still match
- active v2 must apply a deterministic pre-validation exclusion set before L1/L2 questionability:
  - exclude preference support fields `signal` / `signals`
  - exclude authoring-only habit metadata fields `priority`, `schedule_date`, and `schedule_dates`
  - these excluded fields must not appear in `askable_fields`, must not affect `is_questionable`, and must not survive into `validated_state_value`

Active acceptance:
- only Stage 1 outputs above are required for active v2
- legacy `change_reason_validation` may still appear in compatibility flows, but it is not part of active acceptance

## 8. Task A Active Contract
Objective:
- reconstruct the user’s current state for each active key

Stage 2 authoring requirements:
- author only from `validated_snapshot_state`
- build one item per retained `state_key`
- each item must contain:
  - `item_id`
  - `state_key`
  - `question_text`
  - `answer_template`
  - `retrieval_query`
  - `scoring_points`
- filtered-out keys should appear under `filtered_keys`

Generation contract:
- Task A must run per key
- each `state_key` gets its own:
  - retrieval query
  - retrieval context
  - answer prompt
  - raw-output audit record

Prompt contract:
- visible query text must come directly from pack-authored `question_text`
- shared prompt builders must not prepend extra checkpoint/task wrappers
- explicit-retrieval baselines must consume pack-authored `retrieval_query` directly

Output contract:
- `snapshot_state[key] = predicted_value`
- `evidence[key] = [{"app_log_id": str, "evidence_content": str}, ...]`

Primary scoring:
- `scoring_points[]` is the canonical semantic scoring source
- field-like leaves use field points
- complex text uses `micro` points
- `atomic fact` is legacy shorthand for one `micro` point; the active evaluation unit is always the scoring point
- generated `micro` points must pass validation; otherwise rewrite or safe fallback is required

## 9. Task C Active Contract
Objective:
- test whether the model can use user state to complete a personalized service task

Active v2 family mapping:
- `habits -> user_communication`
- `preferences -> information_request_construction`
- `attributes -> action_configuration`

Stage 2 authoring requirements:
- author only from `validated_snapshot_state`
- keep at least one accepted item for each retained key
- every accepted item must contain:
  - `qa_id`
  - `service_family`
  - `retrieval_query`
  - `answer_scoring_points`
  - `gold_memory_evidence_app_log_ids`
  - validation metadata

Family-specific accepted item fields:
- `user_communication`
  - `scenario`
  - `task_instruction`
  - `reference_answer`
- structured families
  - `scenario`
  - `task_instruction`
  - `output_template`
  - `reference_output`

Generation contract:
- runtime consumes pack-authored Task C items only
- retrieval uses pack-authored `retrieval_query`
- active `taskabc_v2` structured-family semantic generation prompts author only:
  - `scenario`
  - `task_instruction`
  - `output_template`
  - `reference_output`
- active `taskabc_v2` semantic prompts must not ask the model to author scoring rubrics, scoring criteria, or scoring points
- for structured families, scoring points are materialized programmatically only after semantic acceptance
- answering uses:
  - pack-authored item `retrieval_query` as the canonical visible task body
  - for structured families, that canonical task body must already include `output_template`

Prediction output contract:
- `rq3_apply_answers[state_key].items[*].answer` for `user_communication`
- `rq3_apply_answers[state_key].items[*].output` for structured families
- all items include `evidence`

Validation contract:
- Task C validation is split into two stages:
  - item-semantic validation
  - code-only scoring-point materialization
- item-semantic validation must run first and must judge whether the authored item itself is answerable and state-dependent
- item-semantic validation must not use rubric pairability or scoring-point coverage as semantic pass/fail criteria
- item-semantic validation prompt input must not include `answer_scoring_points`, `scoring_rubric`, scoring criteria, or scoring-point coverage hints
- active `taskabc_v2` semantic validation uses one shared five-criterion schema across all Task C families:
  - `answerability`
  - `service_completion_quality`
  - `full_field_dependency`
  - `low_leakage`
  - `output_groundedness`
- criterion definitions should direct the validator toward the relevant input fields:
  - `full_field_dependency` should check which `state_value` field paths are actually required
  - `low_leakage` should compare `scenario` / `task_instruction` against the field paths in `state_value`
  - `output_groundedness` should check which parts of `reference_answer` or `reference_output` are grounded by which parts of `state_value`
- scoring-point materialization runs only after item-semantic validation passes
- scoring-point materialization is deterministic code-only behavior
- `user_communication` semantic rewrite may update `reference_answer`
- `user_communication` semantic rewrite must not inspect, generate, or rewrite scoring points / scoring criteria
- structured-family semantic rewrite must not inspect, generate, or rewrite scoring points / scoring criteria
- for structured families, the builder materializes scoring points programmatically from the semantically accepted `reference_output`
- Task C v2 semantic rewrite should return only a delta over mutable item fields; unchanged fields may be omitted
- builder-side rewrite application must merge that delta back onto the original item rather than treating the rewrite payload as a full replacement object
- semantic acceptance and scoring acceptance remain separate decisions

Primary scoring:
- `answer_scoring_points[]` is the canonical scoring source
- Task A and Task C both use point-specific evaluation rather than holistic answer similarity
- `user_communication` first evaluates whether the predicted message is about the targeted state / routine itself; if this gate fails, the whole item score is `0`
- after the gate passes, `user_communication` uses one deterministic micro answer point per retained state field
- `user_communication` leaf point text must be generated by code from the field id and validated state value, not by the authoring LLM
- active `taskabc_v2` `habits -> user_communication` must additionally prepend one micro `identity_gate` point that checks whether the message is actually about the targeted habit/routine itself rather than a different routine that happens to share some leaf details
- active evaluator semantics for that `identity_gate` are hard-gated:
  - if any `identity_gate` point for the item is judged incorrect, the whole Task C item score is `0`
  - only when the `identity_gate` passes may the evaluator average the remaining non-gate points
- active `user_communication` main-path materialization must preserve validated state-field coverage; it must not truncate, regroup, or regenerate criteria through legacy apply atomic-fact prompts
- structured families use deterministic field points over the service object, with one point per required output leaf
- active `taskabc_v2` Task C points do not require `polarity`; compatibility readers may still interpret missing `polarity` as positive when reading older shared code paths
- `reference_answer` / `reference_output` are gold-answer context artifacts; slot-level evaluation still judges `answer_scoring_points[]`

## 10. Shared Prompt / Retrieval / Runtime Rules
Pack-first rule:
- generation must consume pack-authored task text and retrieval text
- baselines must not regenerate missing retrieval queries or question text locally

Shared `QuerySpec` rule:
- `retrieval_query_text` comes from task-pack `retrieval_query`
- `answer_query_text` comes from task-pack visible task text
- active `taskabc_v2` Task C is the exception:
  - retrieval still uses item `retrieval_query`
  - visible prompting reuses that same pack-authored `retrieval_query` as the canonical task body
  - for structured families, `retrieval_query` must already include `output_template`
  - no separate Task C question-only answer surface should be introduced through `answer_query_text`

Shared orchestrator rule:
- all baselines execute through the shared `tce_core` route
- orchestrator phases are:
  1. `prepare_checkpoint_state`
  2. `retrieve_context_for_query`
  3. `answer_query`

Concurrency rule:
- runtime must record both requested and effective worker counts
- if a baseline forbids a parallelism dimension, runtime must force that worker count to `1`

Memory safety rule:
- retrieval may read only the prepared checkpoint state
- answering must not mutate shared memory state for Task A / Task C

Evaluation resume rule:
- `eval/eval_tce.py --resume` reuses only slot-judge units whose cached payload is structurally complete and free of recorded judge-call errors
- any cached Task A key / Task B key / Task C item whose stored judgments contain evaluator-generated `judge_error:` placeholders must be treated as incomplete and retried on resume
- resume may skip successfully judged units even when sibling units in the same checkpoint are retried
- partial outputs with transport / rate-limit / provider failures must therefore converge by rerunning against the same eval artifact, rather than requiring manual deletion of failed units

## 11. Metrics & Acceptance
Primary public metrics:
- Task A:
  - `snapshot_point_score_mean_on_expected`
- Task C:
  - point-based answer score family derived from `answer_scoring_points`

Auxiliary metrics:
- value continuity diagnostics
- evidence id/content quality diagnostics

Stage acceptance summary:
- Stage 1 input must already have raw benchmark state + observability
- Stage 1 output must contain valid `state_questionability`, `validated_snapshot_state`, `state_validation_summary`
- Stage 2 input must already contain Stage 1 outputs
- Stage 2 output must contain:
  - `state_completion_pack`
  - `rq3_apply_service_qa`
- Generation must consume prebuilt packs directly

## 12. Change Governance
If a change affects any of the following, update this active manual first:
- task definitions
- schemas / contracts
- metrics / evaluation protocol

Then update:
1. implementation
2. tests
3. supporting docs / runbooks

## 13. Archive Policy
The following belong in the archive document, not in this active manual:
- `taskabc_v1`
- standalone Task B / change attribution contract
- legacy Task C QA-style contract
- legacy fallback semantics that are kept only for reading old artifacts
- historical metric definitions no longer used for active acceptance

Archive reference:
- `docs/protocols/archive/temporal_checkpoint_evaluation_developer_manual_legacy.md`
