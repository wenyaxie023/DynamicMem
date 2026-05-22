# TCE Execution Runbook

Status: active
Owner: DynamicMem team
Last Updated: 2026-04-16

Protocol spec:
- `docs/protocols/temporal_checkpoint_evaluation_developer_manual.md`
- `docs/protocols/tce_generation_and_adapter_contract.md`

This runbook only contains executable workflow and operational checks.
Contributor-facing generation and adapter obligations are maintained in:
- `docs/protocols/tce_generation_and_adapter_contract.md`

Current execution note:
- This runbook is the broad execution reference for TCE.
- For the current TCE v2 pre-batch build validation pass across baselines, use `docs/plans/tce_v2_build_progress_master_sheet.md` as the active command sheet and status tracker.
- Current workflow policy: `final_qa` is disabled unless explicitly re-enabled for a specific purpose.

## 0. Execution Prerequisites

Shell bootstrap for repo commands:

```bash
cd /users/4/xie00470/mem_bench/dynamicmem
source ~/miniconda/etc/profile.d/conda.sh
conda activate mem0311
```

Environment / credentials:
- Commands in this runbook assume the repo root `.env` is present when Azure-backed configs are used.
- `baseline_prediction.run_tce_batch` adapters such as `memoryos` load the repo-root `.env` automatically.
- Required Azure credential vars for Azure-backed runs:
  - `AZURE_OPENAI_API_KEY`
  - `AZURE_OPENAI_BASE_URL`
- Optional MemoryOS-specific overrides, only when intentionally overriding the shared provider resolution:
  - `LLM_CONTROLLER_API_KEY`
  - `LLM_CONTROLLER_API_BASE_URL`
  - `EMBEDDING_API_KEY`
  - `EMBEDDING_API_BASE_URL`

Recommended preflight before a new build batch:

```bash
python3 debug_utils/test_azure_key.py
```

Node selection:
- Do not launch TCE builds, generation jobs, or evaluations from login nodes.
- On this cluster, hosts such as `ahl02` are login nodes.
- Use compute nodes such as `aga02` for actual build and generation jobs.
- Viewer serving and other lightweight browser-facing helpers may still run on login nodes.

## 0.1 Recommended Entry Points

Use the narrowest entrypoint that matches the job:
- Current TCE v2 cross-baseline pre-batch validation:
  - use `docs/plans/tce_v2_build_progress_master_sheet.md`
- Single-user generation smoke or bounded rerun:
  - use `python3 -m baseline_prediction.run_tce --config ...`
- Multi-user batch generation:
  - use `python3 -m baseline_prediction.run_tce_batch --config ...`
- Prediction evaluation:
  - use `python3 -m evaluation.eval_tce ...`

Operational rule:
- Keep this runbook as the general TCE reference.
- Keep baseline-specific validation commands and current batch status in the master sheet.

## 1. Part I - Pack Build Execution (with 10-state eyeball)

Raw benchmark path for build-only workflows:
- For memory-building baselines that need checkpoint boundaries before task definitions are finalized, first build a raw benchmark and use that for `build_only` runs.
- Recommended raw benchmark filename:
  - `outputs/gemini_3_flash_preview/<user_id>/tce_benchmark_vnext_raw.json`

### 1.0 Build benchmark (pre-sampled checkpoints)
```bash
python3 -m benchmark_construction.build_tce_benchmark \
  --app-logs-final outputs/gemini_3_flash_preview/<user_id>/app_logs_final.json \
  --all-events-chains outputs/gemini_3_flash_preview/<user_id>/all_events_chains.json \
  --output outputs/gemini_3_flash_preview/<user_id>/tce_benchmark_vnext_raw.json \
  --sampling-mode calendar \
  --calendar-anchor-freq quarterly \
  --app-logs-large outputs/gemini_3_flash_preview/<user_id>/app_log_large.json \
  --sampling-tokenizer-model gpt-4o-mini \
  --task-contract-version taskabc_v2 \
  --research-frame-version rq_20260413 \
  --canonical-research-doc analysis_tools/tce_research_questions/001_user_001/new_research_question.md
```

Build-only note:
- A raw benchmark is sufficient for memory-building baselines such as `memoryos` when the immediate goal is only to materialize checkpointed memory state.
- Task-pack benchmarks are still needed later for full generation and evaluation.

Example raw-benchmark build-only run for `memoryos`:
```bash
python3 -m baseline_prediction.run_tce_batch \
  --config configs/experiments/tce/memoryos_build_only_raw.yaml \
  --users <user_id>
```

### 1.1 Run standalone state validation
```bash
python3 -m benchmark_construction.build_tce_state_validation \
  --benchmark outputs/gemini_3_flash_preview/<user_id>/<benchmark_base>.json \
  --output outputs/gemini_3_flash_preview/<user_id>/<benchmark_state_validated>.json \
  --validator-provider gemini \
  --validator-model gemini-3-flash-preview \
  --l2-evidence-top-k 0 \
  --save-every-states 5 \
  --resume
```

### 1.2 Build precomputed task packs
```bash
python3 -m benchmark_construction.build_tce_task_packs \
  --benchmark outputs/gemini_3_flash_preview/<user_id>/<benchmark_state_validated>.json \
  --output outputs/gemini_3_flash_preview/<user_id>/<benchmark_task_packs>.json \
  --tasks all \
  --provider gemini \
  --model gemini-3-flash-preview \
  --validator-provider gemini \
  --validator-model gemini-3-flash-preview \
  --item-count-per-key 1 \
  --reuse-scope key_value_signature \
  --max-rewrites 2 \
  --save-every-apply-keys 5 \
  --save-raw \
  --task-contract-version taskabc_v2 \
  --research-frame-version rq_20260413 \
  --canonical-research-doc analysis_tools/tce_research_questions/001_user_001/new_research_question.md
```

Versioning note:
- benchmark / task-pack / prediction / eval artifacts for the current research frame should all carry:
  - `task_contract_version = taskabc_v2`
  - `research_frame_version = rq_20260413`
- if an older artifact lacks those fields, treat it as frozen legacy `taskabc_v1`
- under active `taskabc_v2`, `--tasks all` means `Task A + Task C` only
- `enable_change_reasoning` should be treated as legacy-only; do not expect a standalone Task B pack in the default v2 workflow

Task C apply-only rerun:
- If an `all` task-pack build completed Task A but stopped during Task C, do not use any review-only shortcut.
- Re-run the existing task-pack benchmark in `apply-only` mode so completed Task A artifacts are preserved and `rq3_apply_service_qa` is rebuilt formally:
```bash
python3 -m benchmark_construction.build_tce_task_packs \
  --benchmark outputs/gemini_3_flash_preview/<user_id>/tce_benchmark_task_packs_partial.json \
  --output outputs/gemini_3_flash_preview/<user_id>/tce_benchmark_task_packs_final.json \
  --tasks apply \
  --provider gemini \
  --model gemini-3-flash-preview \
  --validator-provider gemini \
  --validator-model gemini-3-flash-preview \
  --item-count-per-key 1 \
  --reuse-scope key_value_signature \
  --max-rewrites 2 \
  --save-every-apply-keys 5
```
- This rerun is still a protocol-faithful Task C rebuild, not a temporary inspection artifact.

Apply-only compatibility wrapper:
```bash
python3 -m benchmark_construction.build_tce_rq3_apply_pack \
  --benchmark outputs/gemini_3_flash_preview/<user_id>/tce_benchmark_state_validated.json \
  --output outputs/gemini_3_flash_preview/<user_id>/tce_benchmark_task_packs_apply_only.json \
  --provider gemini \
  --model gemini-3-flash-preview \
  --validator-provider gemini \
  --validator-model gemini-3-flash-preview \
  --item-count-per-key 1 \
  --reuse-scope key_value_signature \
  --max-rewrites 2 \
  --save-every-apply-keys 5 \
  --save-raw
```

### 1.3 Generate 10-state eyeball set
```bash
python3 - <<'PY'
import json
from pathlib import Path

p = Path("outputs/gemini_3_flash_preview/<user_id>/<benchmark_task_packs>.json")
out = Path("baseline_prediction/rag/results/<user_id>/analysis/rq3_manual_review/rq3_question_preview10.csv")
out.parent.mkdir(parents=True, exist_ok=True)

payload = json.loads(p.read_text(encoding="utf-8"))
rows = []
for cp in payload.get("checkpoints", []):
    qa = (cp.get("rq3_apply_service_qa") or {}).get("keys") or {}
    for sk, node in qa.items():
        for item in (node.get("items") or []):
            rows.append(
                (
                    cp.get("checkpoint_id"),
                    sk,
                    item.get("qa_id"),
                    item.get("service_category"),
                    item.get("question"),
                    item.get("reference_answer"),
                )
            )

rows = rows[:10]
with out.open("w", encoding="utf-8") as f:
    f.write("checkpoint_id,state_key,qa_id,service_category,question,reference_answer\n")
    for r in rows:
        f.write(",".join('"{}"'.format(str(x).replace('"', '""')) for x in r) + "\n")
print("saved", out)
PY
```

### 1.4 Generate stratified manual-review pack
For deeper manual review of Stage 1 and Stage 2, prefer a purpose-specific review pack over a single mixed sample table.

```bash
python3 debug_utils/build_tce_manual_review_samples.py \
  --validated outputs/gemini_3_flash_preview/<user_id>/tce_benchmark_state_validated.json \
  --task-packs outputs/gemini_3_flash_preview/<user_id>/tce_benchmark_task_packs.json \
  --output-dir baseline_prediction/rag/results/<user_id>/analysis/tce_manual_review
```

Expected outputs:
- Overview:
  - `tce_manual_review_plan_20260308.md`
  - `tce_manual_review_purpose_manifest_20260308.json`
  - `tce_manual_review_sample_summary_20260308.json`
- Purpose-specific sample sets:
  - `stage1_l1_fail_sample_set_20260308.json`
  - `stage1_l2_fail_sample_set_20260308.json`
  - `stage1_pass_sample_set_20260308.json`
  - `stage2_state_completion_sample_set_20260308.json`
  - `stage2_change_tracking_sample_set_20260308.json`
  - `stage2_apply_accepted_sample_set_20260308.json`
  - `stage2_apply_discarded_sample_set_20260308.json`
- Each sample set should also include:
  - `*_sample_set_20260308.csv`
  - `*_review_sheet_20260308.csv`
- Suggested starting points:
  - `stage1_l1_fail` and `stage1_l2_fail` for gate strictness
  - `stage2_state_completion` for template and reuse quality
  - `stage2_apply_accepted` and `stage2_apply_discarded` for Task C item quality

### 1.5 Part I gate checks
- Check `state_questionability` exists in checkpoint payload.
- Check `validated_snapshot_state` exists in checkpoint payload.
- Check `state_completion_pack` exists when using new benchmark.
- Check `rq3_apply_service_qa` exists when running active `taskabc_v2`.
- Check `change_tracking_pack` only when auditing legacy `taskabc_v1` / standalone Task B artifacts.
- Check state validate evidence semantics:
  - per checkpoint, L2 evidence is rebuilt from that checkpoint's `state_observability.evidence_app_log_ids`
  - if a later checkpoint has additional evidence for the same state, it must trigger re-validation rather than silent reuse
  - evidence is cumulative up to checkpoint time; later checkpoints typically include earlier evidence plus newly-added evidence
- Check Stage 1 persistence semantics:
  - Stage 1 should incrementally refresh the output artifact every 5 processed states by default
  - `--resume` should continue from the same output artifact and skip keys already present in `state_questionability`
  - resumed keys should still seed later cross-checkpoint reuse
- Check each accepted item has `validation` payload.
- During Stage 2 apply build, confirm the output artifact is incrementally refreshed every configured `--save-every-apply-keys` interval.
- Check `discard_rate <= 30%`.
- Check each `questionable state` has at least one valid item.
- Perform manual eyeball on 10 states.
- For deeper review, prefer the stratified manual-review pack under `baseline_prediction/rag/results/<user_id>/analysis/tce_manual_review/` over one mixed JSON.
- Note:
  - Stage 2 requires a Stage 1 artifact on disk.
  - If Stage 1 stops mid-run, resume from the persisted partial output with `--resume`.

## 2. Part II - Generation Execution

Generation mode selection:
- Use `baseline_prediction.run_tce` for single-user smoke runs, bounded reruns, and `--max-checkpoints` cases.
- Use `baseline_prediction.run_tce_batch` for multi-user execution from templated YAML configs.
- For the current TCE v2 pre-batch baseline validation pass, prefer the master sheet over the generic examples below.

### 2.1 Retrieval query audit (manual)
- Verify retrieval query source and final text before run.
- Verify know/apply query separation rules match spec.

### 2.2 Answer prompt audit (manual)
- Verify prompt template path and assembly path.
- Verify task instruction is consistent across baselines.

### 2.2a Stage 2 pack acceptance (manual)
- Task A:
  - verify `state_completion_pack.keys[*].answer_template` matches `validated_snapshot_state`
  - verify repeated unchanged validated states are reused, not regenerated
- Task B:
  - legacy/v1 only
  - verify only validated-state intersection changes become `change_tracking_pack.keys`
- Task C:
  - verify `rq3_apply_service_qa` only uses validated states
  - verify each accepted item contains `validation`

### 2.3 RAG single-checkpoint smoke
```bash
python3 -m baseline_prediction.run_tce \
  --config configs/experiments/tce/rag_predict_users.yaml \
  --max-checkpoints 1
```
- RAG/TCE supports:
  - key-level incremental save via `runtime.save_every_generation_keys` (recommended smoke value: `1`)
  - checkpoint concurrency via `runtime.checkpoint_workers`
  - within-checkpoint concurrency via `runtime.within_checkpoint_workers`
  - resume skips only checkpoints marked `metadata._checkpoint_complete=true`; incomplete partial checkpoints are rerun
- Under the current protocol, Task A runs per key:
  - one retrieval per `state_key`
  - one answer prompt per `state_key`
  - prediction metadata should include `per_key_retrieval[*]`
- Do not use checkpoint-level combined Task A retrieval as a formal protocol path.
- For long reruns, avoid depending on unstable partial artifacts. If `save_prompt_and_raw=true`, very small `save_every_generation_keys` values can cause excessive full-file rewrites and unstable intermediate outputs.
- Baseline concurrency policy must follow the protocol:
  - `rag / oracle / hipporag2 / memoryos`: both worker dimensions may be enabled
  - `amem`: `checkpoint_workers` is effectively forced to `1`; `within_checkpoint_workers` may still be parallel
- After the run, inspect prediction metadata:
  - `concurrency_policy`
  - `requested_checkpoint_workers`
  - `requested_within_checkpoint_workers`
  - `effective_checkpoint_workers`
  - `effective_within_checkpoint_workers`
- If evaluation later runs with LLM judge enabled, verify request granularity:
  - Task A: one request per `state_key`
  - Task B: one request per changed `state_key`
  - Task C: one request per `(state_key, qa_id)`
- The primary Task C answer metric is `rq3_apply_answer_point_score_mean`, derived from slot-level LLM judging over each item's `answer_scoring_points[]`. Missing `answer_scoring_points[]` should be treated as a task-pack or protocol error, not as a signal to fall back to option-style metrics.

### 2.4 Letta/MemGPT expansion run
```bash
python3 -m baseline_prediction.run_tce \
  --config configs/experiments/tce/memgpt_v14.yaml \
  --max-checkpoints 1
```

Expected output:
- `baseline_prediction/<baseline>/results/<user_id>/prediction/tce_results*.json`
- sibling `*_run_settings.yaml`

## 3. Part III - Evaluation Execution

### 3.1 Evaluate prediction
```bash
python3 -m evaluation.eval_tce \
  --benchmark outputs/gemini_3_flash_preview/<user_id>/<benchmark_task_packs>.json \
  --prediction baseline_prediction/<baseline>/results/<user_id>/prediction/<run_name>/tce_results.json \
  --output baseline_prediction/<baseline>/results/<user_id>/evaluation/<run_name>/tce_eval.json \
  --enable-llm-judge \
  --llm-provider azure \
  --llm-model gpt-5-mini \
  --save-eyeball
```

### 3.2 Evaluation checks
- Check `summary` + `checkpoints` + `prediction_alignment` exist.
- Check snapshot/change/apply/gap metrics exist.
- Check evidence metrics are id-matching based.
- If resuming eval with `--resume`, confirm any prior slot-judge units that contain evaluator-generated `judge_error:` judgments are retried rather than skipped.
- Check LLM judge request granularity:
  - snapshot judge should issue one request per `state_key`
  - change judge should issue one request per changed `state_key`
  - apply judge should issue one request per `(state_key, qa_id)`

### 3.3 Generate eval manual-review pack
When auditing LLM judge strictness or rubric consistency across Task A/B/C, prefer generating the eval manual-review pack instead of reading the full eval JSON directly.

```bash
python3 debug_utils/build_tce_eval_manual_review_pack.py \
  --benchmark outputs/gemini_3_flash_preview/<user_id>/<benchmark_task_packs>.json \
  --prediction baseline_prediction/<baseline>/results/<user_id>/prediction/<run_name>/tce_results.json \
  --eval baseline_prediction/<baseline>/results/<user_id>/evaluation/<run_name>/tce_eval.json \
  --output-dir baseline_prediction/<baseline>/results/<user_id>/analysis/tce_eval_manual_review \
  --top-n 20
```

Expected outputs:
- `tce_eval_manual_review_summary.json`
- `snapshot_judge_suspicious_cases.json`
- `change_judge_suspicious_cases.json`
- `apply_judge_suspicious_cases.json`
- corresponding `*.csv`
- `README.md`

Suggested starting point:
- review `apply_judge_suspicious_cases.json` first, then `change`, then `snapshot`

### 3.4 Build state timeline viewer data
When auditing a user's full timeline from the perspective of a single `state`, prefer generating state timeline viewer data instead of manually cross-referencing benchmark, prediction, eval, and app logs.

```bash
python3.11 debug_utils/build_tce_state_timeline_viewer.py \
  --benchmark outputs/gemini_3_flash_preview/<user_id>/<benchmark_task_packs>.json \
  --prediction baseline_prediction/<baseline>/results/<user_id>/prediction/<run_name>/tce_results.json \
  --eval baseline_prediction/<baseline>/results/<user_id>/evaluation/<run_name>/tce_eval.json \
  --app-logs outputs/gemini_3_flash_preview/<user_id>/app_log_large.json \
  --output baseline_prediction/<baseline>/results/<user_id>/analysis/state_timeline_viewer/state_timeline_viewer_data.json
```

For visualization-focused review, prefer `--compact` to avoid very large JSON artifacts:

```bash
python3.11 debug_utils/build_tce_state_timeline_viewer.py \
  --benchmark outputs/gemini_3_flash_preview/<user_id>/<benchmark_task_packs>.json \
  --prediction baseline_prediction/<baseline>/results/<user_id>/prediction/<run_name>/tce_results.json \
  --eval baseline_prediction/<baseline>/results/<user_id>/evaluation/<run_name>/tce_eval.json \
  --app-logs outputs/gemini_3_flash_preview/<user_id>/app_log_large.json \
  --output baseline_prediction/<baseline>/results/<user_id>/analysis/state_timeline_viewer/state_timeline_viewer_data_compact.json \
  --compact
```

Viewer entry points:
- `analysis_tools/tce_state_timeline_viewer/index.html`
- `analysis_tools/tce_state_timeline_viewer/app.js`
- `analysis_tools/tce_state_timeline_viewer/styles.css`

Serve locally:
```bash
cd analysis_tools/tce_state_timeline_viewer
python3.11 -m http.server 8000
```

Then open in the browser:
```text
http://127.0.0.1:8000/index.html?data=../../baseline_prediction/<baseline>/results/<user_id>/analysis/state_timeline_viewer/state_timeline_viewer_data.json
```

Viewer display rules:
- The main view is keyed off `validated_snapshot_state`.
- Blank cells mean `not present / not evaluated`, not `0 score`.
- Task A and Task B prefer the newer 4-rubric judge view; legacy eval files are shown through a compatibility path.
- Task C prefers `rq3_apply_slot_eval_by_item` slot-level item eval; legacy and deterministic compatibility views are only fallbacks for older eval artifacts.
- `--compact` removes large fields such as `request`, `response`, `prompt`, and `raw_model_output`, and is the preferred default review artifact.

## 4. Failure Routing & Manual Intervention

### 4.1 Pack build blocked
Trigger any of:
- `discard_rate > 30%`
- some `questionable state` has 0 valid items
- missing validation payload

Action:
1. Inspect `discarded_items[].validation.failed_rules` distribution.
2. Re-run with same config and `--save-raw` for trace.
3. If still blocked, mark `manual_review_required=true` and open manual review issue.

### 4.2 Generation blocked
Trigger examples:
- malformed prediction schema
- missing task pack and fail mode enabled

Action:
1. Validate config and run settings.
2. Validate retrieval query and prompt assembly.
3. Re-run single checkpoint before batch resume.

### 4.3 Evaluation blocked
Trigger examples:
- schema mismatch
- LLM judge parse failure

Action:
1. Validate prediction schema against spec.
2. Run eval without LLM judge to isolate structural issues.
3. Re-enable LLM judge after structural pass.

## 5. Artifact Paths & Command Checklist

### 5.1 Canonical paths
- Benchmark: `outputs/.../tce_benchmark*.json`
- Prediction: `baseline_prediction/<baseline>/results/<user_id>/prediction/tce_results*.json`
- Eval: `baseline_prediction/<baseline>/results/<user_id>/evaluation/tce_eval*.json`
- Manual review: `baseline_prediction/rag/results/<user_id>/analysis/rq3_manual_review/`
- Stratified manual review: `baseline_prediction/rag/results/<user_id>/analysis/tce_manual_review/`

Storage note:
- To avoid filling the home directory, these large artifact trees may live in project storage and be symlinked back to the repo paths:
  - `outputs/...`
  - `baseline_prediction/<baseline>/results/...`
- Recommended project storage root:
  - `/projects/standard/zrliu/shared/wenya/xie00470/dynamicmem/`
- If the repo path is already a symlink, keep using the repo-local path in scripts and commands.

### 5.2 Quick checks
```bash
rg -n "DRAFT_MARKER" docs/protocols/temporal_checkpoint_evaluation_developer_manual.md

python3 -m baseline_prediction.run_tce --help
python3 -m baseline_prediction.run_tce_batch --help
python3 -m evaluation.eval_tce --help
python3 benchmark_construction/build_tce_benchmark.py --help
```

Historical artifact note:
- `results*/` and `generated_outputs/` may contain legacy-name files and are intentionally not renamed.
