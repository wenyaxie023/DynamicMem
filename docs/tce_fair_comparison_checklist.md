# TCE Fair Comparison Checklist (Draft)

This checklist is tailored to the current codebase for TCE (TCE).

## 1. Baseline Taxonomy (for paper reporting)

Use explicit groups instead of mixing all methods in one pool.

1. `Upper-bound oracle`
- `oracle` (`generation/oracle/tce.py`)
- Retrieval uses benchmark observability evidence (`state_observability.evidence_app_log_ids`), so this is not a deployable baseline and should be reported as an upper bound.

2. `Prompted direct baselines (same core TCE prompt template)`
- `icl` (`generation/icl/tce.py`)
- `rag` (`generation/rag/rag_tce.py`)
- `amem` (`generation/Amem/tce.py`)
- `hipporag2` (`generation/HippoRAG2/generation_tce/tce.py`)
- `memoryos` (`generation/MemoryOS/tce_adapter.py`)
- `mem0` (`generation/mem0/tce.py`)
- These run through shared pipeline/prompt assembly (`tce_core/pipeline.py`, `tce_core/prompts.py`), but retrieval context differs.

3. `Agent-loop / non-equivalent prompting`
- `letta` / `memgpt` (`generation/letta/tce.py`, adapter alias in `generation/adapters/registry.py`)
- This path intentionally ignores pipeline prompt and uses its own agent prompt; report separately from “same-prompt” baselines.

4. `Not implemented in TCE adapter layer (do not include as comparable TCE results yet)`
- `nemori`, `zep` remain stubs in `generation/adapters/registry.py`.
- YAML presence under `configs/experiments/tce/*.yaml` is not sufficient; compare only baselines with runnable adapter support.

## 2. Core Comparability Rule

Only compare in one table if all methods satisfy:

- Same benchmark file and checkpoint set.
- Same evaluation script and judge settings.
- Same generation LLM family/deployment constraints (or explicitly stratified by model).
- No method has privileged ground-truth evidence (except oracle upper-bound table).
- Prompting protocol is equivalent (same prompt construction path), otherwise place in separate group/table.

## 3. Mandatory Alignment Controls (Generation)

For each experiment table, lock these fields across baselines unless explicitly studied:

1. Data and split
- `data.benchmark`
- `data.app_logs_path`
- user set and checkpoint coverage (same `users`, same `max_checkpoints` policy)

2. LLM generation setup
- `llm.provider`, `llm.model`, `llm.max_workers`
- timeout/retry policy (see `generation/common/llm_client.py`)
- structured-output mode policy (openai/azure support vs others)

3. Runtime policy
- `resume` handling policy (report whether continued runs were allowed)
- `debug`, `save_prompt_and_raw`
- `max_visible_logs` (window size) must match

4. Retrieval budget policy (critical)
- Define one common retrieval budget variable (e.g., `k=5`).
- Report and enforce per-checkpoint input count via metadata:
  - `num_input_context_logs`
  - `input_context_app_log_ids`
- If a baseline internally uses a different cap, either patch it or report as non-comparable.

## 4. Mandatory Alignment Controls (Evaluation)

1. Use one evaluator for all TCE baselines
- `eval/eval_tce.py`

2. LLM-as-judge parity
- Same `--llm-provider`, `--llm-model`, `--llm-max-workers`
- Same `enable_llm_judge` on/off policy for all baselines in a table
- Report both automatic metrics and judge metrics together

3. Prediction-to-benchmark alignment
- Keep timestamp-based alignment enabled (default in evaluator) to avoid ID drift issues.

## 5. Current Codebase Risks to Address Before Final Paper Runs

1. Provider mismatch across configs
- Example: `oracle.yaml` currently uses `azure`, others often `openai`.
- If unintentional, unify provider/model or stratify results by provider.

2. Unequal user sets in configs
- Example: `configs/experiments/tce/rag_user1_v14_top20_c4.yaml` is explicitly a single-user config.
- Must align user set / benchmark scope for fair comparison.

3. `letta` prompt protocol differs
- `ask_json` ignores pipeline prompt and uses agent-specific prompt (`generation/letta/tce.py`).
- Keep in a separate “agent-loop” section/table.

4. `hipporag2` retrieval cap currently hardcoded to top-5
- In `generation/HippoRAG2/generation_tce/tce.py`, `selected_logs = retrieved_logs_candidates[:5]`.
- This can silently violate declared retrieval budget if other methods use different `k`.

5. `oracle` must be labeled upper bound
- Uses ground-truth evidence IDs; should not be presented as a standard baseline.

## 6. Recommended Reporting Template (per table)

For each baseline, report:

- Baseline group (`upper-bound`, `same-prompt retrieval`, `agent-loop`, etc.)
- Generation LLM provider/model
- Retrieval backend/model
- Retrieval budget (`k`) and observed `num_input_context_logs` stats (mean/p50/p95)
- Checkpoint coverage (#evaluated checkpoints)
- Key metrics:
  - `snapshot_value_f1_mean_on_expected_mean`
  - `snapshot_evidence_f1_mean_on_expected_mean`
  - `snapshot_evidence_exact_match_mean_on_expected_mean`
  - `llm_judge_score_mean` (if enabled)

## 7. Pre-Run Audit Checklist (copy/paste)

- [ ] Baselines selected are implemented and runnable in `generation/adapters/registry.py`
- [ ] Baselines are grouped by comparability class (oracle upper-bound separated)
- [ ] Same users/checkpoints across compared baselines
- [ ] Same benchmark and app log source paths
- [ ] Same generation provider/model (or explicitly stratified)
- [ ] Same retrieval budget policy (`k`)
- [ ] `num_input_context_logs` present in output metadata
- [ ] Same evaluation command and same judge settings
- [ ] No stale/partial outputs included unintentionally (resume policy documented)
- [ ] Final table includes both quality and context-budget statistics
