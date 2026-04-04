# TCE Prompt/Retrieval Checklist

This checklist tracks where TCE (especially RAG baseline) builds retrieval query, LLM input prompt, and where to inspect artifacts.

## 1) Retrieval Query: where generated

- Source field: task-pack `retrieval_query`
- QuerySpec injection: `_build_task_query_spec(...)` in `tce_core/pipeline.py`
- Runtime contract:
  - `retrieval_query_text` is loaded from the pack and carried through `QuerySpec`
  - explicit-retrieval baselines must consume `QuerySpec.retrieval_query_text` directly
  - baselines must not regenerate or fallback to local retrieval-query builders

Pack-first runtime contract:
- `retrieval_query_text` comes from task-pack `retrieval_query`
- `answer_query_text` comes from task-pack `question_text` / `apply_question`
- `checkpoint_timestamp` is still injected by shared `tce_core` into `QuerySpec`, not by baseline-local wrappers
- shared Task A / B / C answer prompts now use the pack-authored direct question text without baseline-local checkpoint wrappers
- explicit-retrieval baselines must consume `QuerySpec.retrieval_query_text` directly; they must not regenerate or fallback to baseline-local retrieval query text

## 2) LLM Input Template: where defined

Snapshot generation prompt template:
- Functions:
  - `build_state_completion_prompt_with_inline_memory(...)`
  - `build_state_completion_prompt_with_agent_memory(...)`
- File: `tce_core/prompts.py`
- Includes sections:
  - direct question block from pack `question_text` only
  - `[Memory]`
  - inline-memory mode pastes retrieved memory blocks into `[Memory]`
  - agent-memory mode uses `[Memory]` to instruct the model to rely on agent-stored memory
  - `[Instructions]`
  - `[Output format]`
  - evidence schema uses `{"app_log_id", "evidence_content"}`

Change reasoning prompt template:
- Functions:
  - `build_change_reasoning_prompt_with_inline_memory(...)`
  - `build_change_reasoning_prompt_with_agent_memory(...)`
- File: `tce_core/prompts.py`
- Same direct-question + memory/output/rules structure as Task A
- Output schema key:
  - `change_analysis[key] = {before, after, change_reason, evidence}`

Service application answer prompt template:
- Functions:
  - `build_service_application_prompt_with_inline_memory(...)`
  - `build_service_application_prompt_with_agent_memory(...)`
- File: `tce_core/prompts.py`
- Same direct-question + memory/output/rules structure as Task A/B
- Output schema key:
  - `{answer, evidence}`

State questionability (L2 validate) prompt template:
- Function: `build_state_questionability_validation_prompt(...)`
- File: `tce_core/prompts.py`
- Required section order (prompt-writing checklist aligned):
  - `[Task Instruction]`
  - `[Definitions]`
  - `[Constraints]`
  - `[Example]`
  - `[Input/Output Format]`
- Output schema key:
  - `field_verdicts[*] = {field_name, reason_analysis, is_valid}`
  - top-level `{is_questionable, reason_codes, field_verdicts}`

Change reason metadata validation prompt template:
- Function: `build_change_reason_validation_prompt(...)`
- File: `tce_core/prompts.py`
- Purpose:
  - validate whether `state_observability.last_change_reason` is evidence-supported enough to be used as Task B `reference_change_reason`
- Output schema key:
  - `{exists, reason_analysis, is_valid, reason_codes}`

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
  - atomic facts should prefer stable semantic requirements over brittle exact surface forms
  - repeated slots that all hinge on the same narrow identifier / model / spec should be rejected
  - atomic-fact rewrite is atomic-fact-only; it must not rewrite QA
  - if atomic facts still fail after rewrites, Task C uses safe fallback from `reference_answer`

Apply validation / rewrite prompt templates:
- Functions:
  - `build_rq3_apply_validation_prompt(...)`
  - `build_rq3_apply_rewrite_prompt(...)`
- File: `tce_core/prompts.py`
- Validation prompt contract:
  - LLM returns only `criteria[*] = {criterion, pass, analysis}`
  - required criteria:
    - `personalization_necessity`
    - `service_decision_quality`
    - `answer_groundedness`
  - `service_decision_quality` should judge whether the item is a real state-dependent assistant action; for `habit` states, schedule-grounded service execution / conflict-handling items can still pass
  - `answer_groundedness` should judge whether the reference answer stays within information explicit in `state_value` or the question
  - final artifact field `qa_validation.is_valid` is computed programmatically in `tce_core/task_packs.py`, not delegated to the LLM
- Rewrite prompt contract:
  - rewrite prompt must include the failed QA item itself
  - rewrite prompt must include validator feedback:
    - `failed_rules`
    - `criteria`
  - rewrite prompt rewrites QA only; it must not accept or emit `rubric`

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
  - requires `change_tracking_pack` when `enable_change_reasoning=true`
  - helper: `build_change_targets_from_pack(...)`
  - no fallback to legacy/generated change task text or baseline-local retrieval-query regeneration when pack is missing or `retrieval_query` is blank
- Task C:
  - requires pack-authored `rq3_apply_service_qa.keys[*].items[*].retrieval_query`
  - final answer prompt must include the item `question`
  - `service_category` may be included as lightweight context
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
   - for Task C, retrieval query should contain scenario + question, not question-only.
2. Confirm final prompt text:
   - check `metadata.prompt[*]`.
   - Task A/B/C prompts should now start directly from the pack-authored question text, then include `[Memory] / [Output format] / [Rules]`.
3. Confirm retrieved inline memory payload:
   - check `metadata.per_key_retrieval[*].retrieval_metadata`.
   - for log-grounded baselines, `retrieved_app_log_ids` should still be inspectable when available.
4. Confirm model raw output before normalization:
   - check `metadata.raw_model_output`.
5. If output is empty/null-heavy:
   - check retrieval top-k, target template shape, and whether the prompt contains the expected key.
