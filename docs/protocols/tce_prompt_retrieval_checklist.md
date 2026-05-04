# TCE Prompt/Retrieval Checklist

This checklist tracks where TCE (especially RAG baseline) builds retrieval query, LLM input prompt, and where to inspect artifacts.

## 1) Retrieval Query: where generated

- Source field: task-pack `retrieval_query`
- QuerySpec injection: `_build_task_query_spec(...)` in `tce_core/pipeline.py`
- Runtime contract:
  - `retrieval_query_text` is loaded from the pack and carried through `QuerySpec`
  - explicit-retrieval baselines must consume `QuerySpec.retrieval_query_text` directly
  - agent-memory baselines that use backend-native retrieval/answer APIs must still use `QuerySpec.retrieval_query_text` as the backend retrieval question, even if final answering happens later inside `answer_query(...)`
  - baselines must not regenerate or fallback to local retrieval-query builders

Pack-first runtime contract:
- `retrieval_query_text` comes from task-pack `retrieval_query`
- `answer_query_text` comes from task-pack `question_text` / `question`
- Task C v2 does not use a separate question-only answer surface:
  - retrieval uses pack-authored `retrieval_query`
  - answering reuses that same `retrieval_query` as the canonical visible task body
  - for structured families, `retrieval_query` must already include `output_template`
- `checkpoint_timestamp` may still be carried in shared runtime metadata such as `QuerySpec`, but active Task C v2 pack-authoring prompts do not expose it as a prompt input field
- shared Task A / B / C answer prompts now use the pack-authored direct question text without baseline-local checkpoint wrappers
- explicit-retrieval baselines must consume `QuerySpec.retrieval_query_text` directly; they must not regenerate or fallback to baseline-local retrieval query text

## 2) LLM Input Template: where defined

Snapshot generation prompt template:
- Functions:
  - `build_state_completion_prompt_with_inline_memory(...)`
  - `build_state_completion_prompt_with_agent_memory(...)`
- File: `tce_core/prompts.py`
- Includes sections:
  - `[Instructions]`
  - `[Output format]`
  - direct question block from pack `question_text` only
  - `[Memory]`
  - inline-memory mode pastes retrieved memory blocks into `[Memory]`
  - agent-memory mode uses `[Memory]` to instruct the model to rely on the system's stored checkpoint-bounded memory about the user's trajectory
  - model-facing state output key is `user_state`; runtime normalizes it into
    persisted prediction `snapshot_state`
  - habit completions include schedule vocabulary and encoding guidance; non-habit
    completions omit that schedule supplement
  - evidence schema uses `{"app_log_id", "evidence_content"}`

Change reasoning prompt template:
- Functions:
  - `build_change_reasoning_prompt_with_inline_memory(...)`
  - `build_change_reasoning_prompt_with_agent_memory(...)`
- File: `tce_core/prompts.py`
- Legacy compatibility prompt currently keeps the direct question and memory before its instructions/output schema
- Output schema key:
  - `change_analysis[key] = {before, after, change_reason, evidence}`

State questionability (L2 validate) prompt template:
- Function: `build_state_questionability_validation_prompt(...)`
- File: `tce_core/prompts.py`
- Output schema key:
  - prompt output: `field_verdicts[field_path] = {reason_analysis, is_valid}`
  - stored artifact: `field_verdicts[*] = {field_name, reason_analysis, is_valid}`
  - no prompt-authored top-level `is_questionable`
  - no prompt-authored `reason_codes`
- Runtime behavior:
  - prompt renders a fixed `field_verdicts` object with one key per `candidate_field_paths` entry
  - code normalizes `field_verdicts` by expected candidate field path and ignores extra/renamed model keys
  - code filters individual fields strictly from valid field verdicts
  - no active state-level questionability decision is produced by L2
  - L2 evidence logs are built from `state_observability.evidence_app_log_ids`

Change reason metadata validation prompt template:
- Function: `build_change_reason_validation_prompt(...)`
- File: `tce_core/prompts.py`
- Purpose:
  - validate whether `state_observability.last_change_reason` is evidence-supported enough to be used as Task B `reference_change_reason`
- Output schema key:
  - prompt output: `change_reason_verdict = {reason_analysis, is_valid}`
  - stored artifact: `change_reason_validation = {reason_analysis, is_valid}`

Apply pack generation prompt template:
- Function: `build_rq3_apply_question_pack_prompt(...)`
- File: `tce_core/prompts.py`
- Required section order (prompt-writing checklist aligned):
  - `[Task Instruction]`
  - `[Definitions]`
  - `[Constraints]`
  - `[Example]`
  - `[Input/Output Format]`
- Current design intent:
  - this prompt now generates QA only:
    - `service_category`
    - `question`
    - `reference_answer`
  - it no longer generates `rubric`
  - prefer concrete service-choice / policy / routing / bounded service-operation questions
  - for `habit` states, schedule-grounded service operations are valid when the item asks what the assistant should do around the routine
  - avoid naked timestamp/date recall or one-step reminder-offset questions that lack a real assistant-action layer

Apply atomic-fact generation / validation prompt templates:
- Functions:
  - `build_apply_answer_scoring_points_prompt(...)`
  - `build_apply_rubric_validation_prompt(...)`
  - `build_apply_rubric_rewrite_prompt(...)`
- File: `tce_core/prompts.py`
- Current design intent:
  - atomic facts are generated only after QA passes question-level validation
  - Task C atomic-fact generation is grounded by:
    - `state_value`
    - `question`
    - `reference_answer`
    - not question-only surface wording
  - active `taskabc_v2` `user_communication` should materialize deterministic micro points from validated state fields after semantic acceptance
  - active `taskabc_v2` `user_communication` main-path materialization should preserve state-field coverage directly; prompt-generated atomic facts are legacy / fallback behavior only
  - atomic facts should prefer stable semantic requirements over brittle exact surface forms
  - repeated slots that all hinge on the same narrow identifier / model / spec should be rejected
  - atomic-fact rewrite is atomic-fact-only; it must not rewrite QA
  - if atomic facts still fail after rewrites, Task C uses safe fallback from `reference_answer`

Apply validation / rewrite prompt templates:
- Functions:
  - `build_task_c_v2_validation_prompt(...)`
  - `build_task_c_v2_rewrite_prompt(...)`
- File: `tce_core/prompts.py`
- Structured-family generation prompt intent:
  - preference / attribute prompts should generate one or two filled output leaves
  - preference / attribute structured outputs should use string-valued leaves, usually in a compact object with one or two keys
  - `core` is a leaf/field-level role, not a state-level role; at least one filled output leaf should be a field-local core leaf
  - detail leaves are allowed when they add grounded, service-useful precision for the same service object
  - `reference_output` remains the intended structured gold answer and must stay grounded in `state_value`
  - require one `reference_anchors` entry per filled leaf so each fill records its state / reference basis and role (`core` or `detail`)
- Validation prompt contract:
  - LLM returns only `criteria[*] = {criterion, pass, analysis}`
  - this is item-semantic validation, not scoring-contract validation
  - active `taskabc_v2` uses one shared five-criterion schema across families:
    - `answerability`
    - `service_realism`
    - `full_field_dependency`
    - `low_leakage`
    - `output_groundedness`
  - `service_realism` means the item should describe a realistic assistant-mediated service action that a user could naturally be doing now, not a backend placeholder, arbitrary workflow, contrived form, or task invented only to expose the state
  - criterion definitions should point the validator to the relevant fields or payload regions to inspect, rather than relying on vague free-form analysis
  - semantic validation must not use rubric pairability or scoring-point coverage as semantic pass/fail criteria
  - semantic validation prompt inputs must not include `answer_scoring_points`, `scoring_rubric`, scoring criteria, or scoring-point coverage hints
  - final artifact field `item_validation.is_valid` is computed programmatically in `tce_core/task_packs.py`, not delegated to the LLM
  - answer scoring points are materialized later by deterministic code in `tce_core/scoring_points.py`
- Rewrite prompt contract:
  - rewrite prompt must include the failed item itself
- rewrite prompt must include validator feedback:
  - `failed_rules`
  - `criteria`
- `user_communication` validation should not mark the fixed generic task instruction as leakage merely because it mentions using routine details
- `user_communication` validation may use the `state_key` suffix as grounding for the routine label when `state_value` stores only schedule/timing/location fields
- `user_communication` rewrite may update `reference_answer`
- rewrite prompts must not include, generate, or rewrite `answer_scoring_points`, `scoring_rubric`, scoring criteria, scoring descriptions, or scoring-point coverage hints
- structured-family scoring points are materialized programmatically from the accepted `reference_output`
- Task C v2 rewrite output should be a delta patch over mutable fields only; unchanged fields may be omitted
- builder-side rewrite application should merge that delta onto the invalid item

Task C runtime answer prompt template:
- Function:
  - `_build_task_c_runtime_prompt(...)`
- File: `tce_core/prompts.py`
- Current design intent:
  - text mode is used by `user_communication`
  - structured mode is used by `information_request_construction` and `action_configuration`
  - use a concise runtime-answering prompt rather than the synthesis-data prompt checklist structure
  - the prompt must place runtime answering instructions and output format before actual answer materials
  - actual answer materials should appear after the rules as `[Assistant Task]` followed by `[Memory]`
  - for `user_communication`, the pack-authored assistant task's `[Task Instruction]` defines what assistant message to write
  - the runtime prompt's answering rules define completeness, evidence, and JSON-only output requirements
  - runtime instructions should define `[Memory]...[/Memory]` as the system memory about the user's trajectory
  - structured-family prompts should ask the model to return the completed object under `answer`
  - for `user_communication`, the `answer` field should contain a concise but complete assistant message: usually one or two natural sentences, while retaining all task-relevant routine details supported by memory
  - for `user_communication`, the model-facing output schema should describe `answer` as a concise but complete assistant message, not merely a short response

Slot-level evaluation judge prompt template:
- Functions:
  - `build_snapshot_slot_judge_prompt(...)`
  - `build_change_slot_judge_prompt(...)`
  - `build_apply_slot_judge_prompt(...)`
- File: `eval/prompts_tce.py`
- Required section order:
  - `[Task Instruction]`
  - `[Definitions]`
  - `[Constraints]`
  - `[Example]`
  - `[Input/Output Format]`
- Current design intent:
  - judge one boolean `correct` value for each checklist `point_id`
  - Task A slot-level judge requests are opt-in and disabled by default; leave them off when using the Task A holistic judge as the active LLM-judge metric
  - Task C slot-level judge requests are opt-in and disabled by default; leave them off when using the Task C holistic judge as the active LLM-judge metric
  - use short prompt-local `point_id` values, then map them back to canonical scoring-point ids in evaluator code
  - mark a point correct when the prediction satisfies the checklist item's core idea / practical value
  - do not require exact wording, formatting, field name, or identical detail organization unless the checklist item explicitly requires it
  - still mark a point incorrect when the core idea is omitted, contradicted, unrelated, too vague to establish, or conflicts with other prediction content

Task A holistic evaluation judge prompt template:
- Function:
  - `build_snapshot_holistic_judge_prompt(...)`
- File: `eval/prompts_tce.py`
- Required section order:
  - `[Task Instruction]`
  - `[Definitions]`
  - `[Constraints]`
  - `[Example]`
  - `[Input/Output Format]`
- Current design intent:
  - frame the method as Core + Detail field evaluation
  - judge deterministic evaluator-derived fields under the predicted state value against the golden user-state value
  - do not show or consume task-pack `scoring_points`
  - choose habit / preference / attribute instructions and examples from the state-key family
  - habit timing guidance uses explicit minute-offset boundaries: exact detail, within 5 minutes may preserve core, within 10 minutes is partial detail, and larger offsets are wrong detail unless the field is an approximate window
  - habit location guidance treats compatible added specificity as preserving core, such as `sofa in living room` for reference `sofa`
  - return one `field_judgments[]` item for every requested field
  - in each field judgment, write `analysis` before boolean `core_correct` and integer `detail_quality`
  - compute each field's 0-1 score in evaluator code as `0.8 * core_correct + 0.2 * (detail_quality / 2)`

Task C holistic evaluation judge prompt template:
- Function:
  - `build_apply_holistic_judge_prompt(...)`
- File: `eval/prompts_tce.py`
- Required section order:
  - `[Task Instruction]`
  - `[Definitions]`
  - `[Constraints]`
  - `[Example]`
  - `[Input/Output Format]`
- Current design intent:
  - frame the model-facing method as Core + Detail field evaluation for personalized service responses
  - avoid benchmark-internal labels or source names such as `Task C`, `service_family`, `service_type`, raw `state_key`, and memory terminology
  - do not show or consume task-pack `answer_scoring_points`
  - for user-communication items, judge the already-materialized checklist fields: the identity gate when present plus one field for each retained state-field requirement, each against the full reference and predicted assistant message
  - for structured items, judge deterministic evaluator-derived fields under the reference service output
  - core is the service output field's central practical value for the service moment, not raw state reconstruction
  - details are service-useful precision such as time, date, place, cadence, qualifiers, exclusions, constraints, encoding, tier/version, branch/address, examples, or scope
  - return one `field_judgments[]` item for every requested field
  - in each field judgment, write `analysis` before boolean `core_correct` and integer `detail_quality`
  - compute each field's 0-1 score in evaluator code as `0.8 * core_correct + 0.2 * (detail_quality / 2)`, then aggregate item scores as `rq3_apply_holistic_score_mean`

## 3) Prompt Assembly + LLM Call: where happens

Main orchestration:
- File: `tce_core/pipeline.py`
- Shared phases:
  1. `prepare_checkpoint_state(...)`
  2. `build_task_query_spec(...)`
  3. `retrieve_context_for_query(...)`
  4. `answer_query(...)`
  5. normalize + persist + finalize
- legacy `retrieve_context(...)` compatibility path is not part of the canonical protocol
- for agent-memory baselines with backend-native retrieval/answer APIs:
  - `retrieve_context_for_query(...)` may materialize backend retrieval state and audit metadata without returning inline memory blocks
  - `answer_query(...)` may then call the backend-native `ask()` / answer generator using the shared agent-memory prompt plus that prepared retrieval state

Snapshot / retrieval path:
- Build task text from prebuilt pack `retrieval_query`
- Retrieval hook returns inline-memory payload for the answer prompt
- Build final prompt:
  - `build_state_completion_prompt_with_inline_memory(...)` for inline-memory baselines
  - `build_state_completion_prompt_with_agent_memory(...)` for agent-memory baselines
- Send to model:
  - `ask_structured(prompt, text_format)` when structured response is available
  - fallback `ask_json(prompt)`

Per-key mode:
- still in `pipeline.py`, loop over each key and build one prompt per key.

Change path:
- Build task text from prebuilt `change_tracking_pack.keys[*].retrieval_query`
- Build prompt:
  - `build_change_reasoning_prompt_with_inline_memory(...)` or
  - `build_change_reasoning_prompt_with_agent_memory(...)`
- Send to model: `ask_json(change_prompt)`
- legacy note:
  - this path is only part of the frozen `taskabc_v1` contract
  - active `taskabc_v2` uses `Task A` changed-vs-unchanged analysis for `RQ2` and does not require standalone Task B by default

State questionability validate path:
- File: `data_construction/build_tce_state_validation.py`
- Build prompt + call model:
  - `validate_state_questionability_with_llm(...)`
  - uses `build_state_questionability_validation_prompt(...)`
  - `validate_change_reason_with_llm(...)`
  - uses `build_change_reason_validation_prompt(...)`
- Evidence source for one state:
  - benchmark field `checkpoint.state_observability[state_key].evidence_app_log_ids`
  - helper path:
    - `collect_evidence_ids(...)`
    - `compute_evidence_signature(...)`
    - `build_evidence_logs(...)`
  - all in `tce_core/state_validation.py`
- Important runtime behavior:
  - evidence is recomputed per checkpoint from that checkpoint's `state_observability`
  - if evidence changes, `evidence_signature` changes and L2 validation is recomputed
  - current benchmark semantics are cumulative: later checkpoints usually include earlier evidence plus newly-added evidence for the same state

Pack-first generation path:
- File: `tce_core/pipeline.py`
- Task A:
  - requires `state_completion_pack`
  - helper: `build_state_completion_targets_from_pack(...)`
  - no fallback to `build_prediction_task_from_checkpoint(...)` or baseline-local retrieval-query regeneration when pack is missing or `retrieval_query` is blank
  - current protocol requires per-key retrieval / prompt / raw-output records
  - each `state_key` should appear in `metadata.per_key_retrieval[*]`
- Task B:
  - requires `change_tracking_pack` when `enable_change_reasoning=true` on legacy/v1 artifacts
  - helper: `build_change_targets_from_pack(...)`
  - no fallback to legacy/generated change task text or baseline-local retrieval-query regeneration when pack is missing or `retrieval_query` is blank
  - on active `taskabc_v2`, `enable_change_reasoning=true` should be treated as a no-op when the benchmark omits `change_tracking_pack`
- Task C:
  - requires pack-authored `rq3_apply_service_qa.keys[*].items[*].retrieval_query`
  - Task C v2 synthesis prompt should be selected by `service_family`, with family-specific example wording
  - Task C v2 `user_communication` synthesis prompt should implement `Habit-Conditioned User Communication` in natural-language assistant-response form with a `reference_answer`
  - Task C v2 `user_communication` synthesis prompt should align few-shot state schema with the real nested `schedule.* / timing.* / location` input shape used by Stage 2
- Task C v2 preference-family prompt should implement `Preference-Conditioned Search-Filter Completion`
- Task C v2 internal family id `information_request_construction` should contract preference inputs to statement-only when the raw preference state also includes auxiliary `signals`
- Task C v2 `action_configuration` synthesis prompt should implement `Attribute-Conditioned Action Configuration`
- Task C v2 structured-family synthesis prompts should ask for a family-appropriate service object, not a raw copy of the source state
- Task C v2 structured-family scenarios should read like natural user product moments:
  - preference filtering: the user is browsing, searching, comparing, or planning options, and the assistant fills search/filter fields before showing matches
  - action configuration: the user is setting up, connecting, completing, or submitting something, and the assistant fills setup/form/configuration fields
  - structured tasks should not make the correct output depend on an extra user choice not determined by `state_value`
  - avoid backend-ish wording such as payload dispatch, downstream modules, coordinator workflows, or unexplained "shortlists being prepared"
- Task C v2 structured-family semantic synthesis prompts should author only `output_template` plus `reference_output`; scoring points are materialized later in code
- Task C v2 `user_communication` scoring points should align one-to-one with retained source-state field paths
- `taskabc_v1` final answer prompt must include the item `question`
- `taskabc_v2` final answer prompt must reuse pack-authored item `retrieval_query` as the visible task body
- `taskabc_v2` structured-family `retrieval_query` must already embed `output_template`
- Task C v2 runtime prompting should not reintroduce a separate question-only `answer_query_text` surface
- internal family ids such as `service_family` should not be exposed to runtime answer prompts unless a prompt genuinely depends on them
- no fallback to generated service-application retrieval query or baseline-local retrieval-query regeneration when item `retrieval_query` is blank

## 4) Runtime Output: where to inspect

Prediction output file:
- `generation/<baseline>/results/<user_id>/prediction/*.json`

For Task A review, inspect:
- `predictions[i].metadata.per_key_retrieval[*].retrieval_query`
- `predictions[i].metadata.prompt` (list of per-key prompts)
- `predictions[i].metadata.raw_model_output` (raw outputs)

Prompt/raw fields are only present when enabled:
- `runtime.save_prompt_and_raw: true` in run config

Your current example:
- `generation/rag/results/001_user_001/prediction/tce_results_vnext_20260319_formal_topk20_gpt5mini_v14_taskabc.json`

## 5) Quick Debug Checklist (copy/paste)

1. Confirm retrieval query shape:
   - check `metadata.per_key_retrieval[*].retrieval_query`.
   - Task A retrieval query should be item-specific, one `state_key` per record.
   - for Task C v1, retrieval query should contain scenario + question, not question-only.
   - for Task C v2, retrieval query should contain scenario + task instruction.
   - Task C v2 retrieval query should not prepend wrapper labels such as `Service family:`, `Retrieval objective:`, or `Required output fields:`.
2. Confirm final prompt text:
   - check `metadata.prompt[*]`.
   - Task A/B prompts should start from the pack-authored question text.
   - Task C v1 should start from the pack-authored question text.
   - Task C v2 `user_communication` should render the pack-authored scenario/task block before `[Memory]`.
   - Task C v2 structured families should render the pack-authored scenario/task/output object before `[Memory]`.
3. Confirm retrieved inline memory payload:
   - check `metadata.per_key_retrieval[*].retrieval_metadata`.
   - inline memory should reflect the backend retrieval content itself via lossless serialization, not an adapter-local rematerialized raw-log view.
   - inspect backend-native retrieval counts / ids only when that baseline intentionally exposes them.
4. Confirm model raw output before normalization:
   - check `metadata.raw_model_output`.
5. If output is empty/null-heavy:
   - check retrieval top-k, target template shape, and whether the prompt contains the expected key.
