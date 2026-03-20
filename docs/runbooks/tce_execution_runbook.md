# TCE Execution Runbook

Status: active
Owner: DynamicMem team
Last Updated: 2026-03-17

Protocol spec:
- `docs/protocols/temporal_checkpoint_evaluation_developer_manual.md`

This runbook only contains executable workflow and operational checks.

## 1. Part I - Pack Build Execution (with 10-state eyeball)

### 1.0 Build benchmark (pre-sampled checkpoints)
```bash
python3 -m data_construction.build_tce_benchmark \
  --app-logs-final data_construction/generated_outputs/gemini_3_flash_preview/<user_id>/app_logs_final.json \
  --all-events-chains data_construction/generated_outputs/gemini_3_flash_preview/<user_id>/all_events_chains.json \
  --output data_construction/generated_outputs/gemini_3_flash_preview/<user_id>/tce_benchmark.json \
  --sampling-mode calendar \
  --calendar-anchor-freq quarterly \
  --app-logs-large data_construction/generated_outputs/gemini_3_flash_preview/<user_id>/app_log_large.json \
  --sampling-tokenizer-model gpt-4o-mini
```

### 1.1 Run standalone state validation
```bash
python3 -m data_construction.build_tce_state_validation \
  --benchmark data_construction/generated_outputs/gemini_3_flash_preview/<user_id>/tce_benchmark.json \
  --output data_construction/generated_outputs/gemini_3_flash_preview/<user_id>/tce_benchmark_state_validated.json \
  --validator-provider gemini \
  --validator-model gemini-3-flash-preview \
  --l2-evidence-top-k 0 \
  --save-every-states 5 \
  --resume
```

### 1.2 Build precomputed task packs
```bash
python3 -m data_construction.build_tce_task_packs \
  --benchmark data_construction/generated_outputs/gemini_3_flash_preview/<user_id>/tce_benchmark_state_validated.json \
  --output data_construction/generated_outputs/gemini_3_flash_preview/<user_id>/tce_benchmark_task_packs.json \
  --tasks all \
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

Task C apply-only rerun:
- 若 `all` task-pack build 已经完成 Task A / Task B，但 Task C 中途中断，不要再使用任何 review-only shortcut。
- 直接对现有 task-pack benchmark 做 `apply-only` 正式重跑，保留已完成的 Task A / Task B，并完整重建 `rq3_apply_service_qa`：
```bash
python3 -m data_construction.build_tce_task_packs \
  --benchmark data_construction/generated_outputs/gemini_3_flash_preview/<user_id>/tce_benchmark_task_packs_partial.json \
  --output data_construction/generated_outputs/gemini_3_flash_preview/<user_id>/tce_benchmark_task_packs_final.json \
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
- 该 rerun 仍然是 protocol-faithful 的正式 Task C 重建，不是临时浏览产物。

Apply-only compatibility wrapper:
```bash
python3 -m data_construction.build_tce_rq3_apply_pack \
  --benchmark data_construction/generated_outputs/gemini_3_flash_preview/<user_id>/tce_benchmark_state_validated.json \
  --output data_construction/generated_outputs/gemini_3_flash_preview/<user_id>/tce_benchmark_task_packs_apply_only.json \
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

p = Path("data_construction/generated_outputs/gemini_3_flash_preview/<user_id>/tce_benchmark_task_packs.json")
out = Path("generation/rag/results/<user_id>/analysis/rq3_manual_review/rq3_apply_question_preview10.csv")
out.parent.mkdir(parents=True, exist_ok=True)

payload = json.loads(p.read_text(encoding="utf-8"))
rows = []
for cp in payload.get("checkpoints", []):
    qa = (cp.get("rq3_apply_service_qa") or {}).get("keys") or {}
    for sk, node in qa.items():
        for item in (node.get("items") or []):
            rows.append((cp.get("checkpoint_id"), sk, item.get("qa_id"), item.get("apply_scenario"), item.get("apply_question"), item.get("apply_reference_answer")))

rows = rows[:10]
with out.open("w", encoding="utf-8") as f:
    f.write("checkpoint_id,state_key,qa_id,apply_scenario,apply_question,apply_reference_answer\n")
    for r in rows:
        f.write(",".join('"{}"'.format(str(x).replace('"', '""')) for x in r) + "\n")
print("saved", out)
PY
```

### 1.4 Generate stratified manual-review pack
当需要对 Stage 1 + Stage 2 做更细的人工审核时，优先生成“分目的 review pack”，不要只看一个混合样本表。

```bash
python3 debug_utils/build_tce_manual_review_samples.py \
  --validated data_construction/generated_outputs/gemini_3_flash_preview/<user_id>/tce_benchmark_state_validated.json \
  --task-packs data_construction/generated_outputs/gemini_3_flash_preview/<user_id>/tce_benchmark_task_packs.json \
  --output-dir generation/rag/results/<user_id>/analysis/tce_manual_review
```

产物约定：
- 总览：
  - `tce_manual_review_plan_20260308.md`
  - `tce_manual_review_purpose_manifest_20260308.json`
  - `tce_manual_review_sample_summary_20260308.json`
- 分目的 sample sets（建议直接按这些文件审）：
  - `stage1_l1_fail_sample_set_20260308.json`
  - `stage1_l2_fail_sample_set_20260308.json`
  - `stage1_pass_sample_set_20260308.json`
  - `stage2_state_completion_sample_set_20260308.json`
  - `stage2_change_tracking_sample_set_20260308.json`
  - `stage2_apply_accepted_sample_set_20260308.json`
  - `stage2_apply_discarded_sample_set_20260308.json`
- 每个 sample set 都应配套：
  - `*_sample_set_20260308.csv`
  - `*_review_sheet_20260308.csv`

建议阅读顺序：
1. `stage1_l1_fail`
2. `stage1_l2_fail`
3. `stage1_pass`
4. `stage2_state_completion`
5. `stage2_change_tracking`
6. `stage2_apply_accepted`
7. `stage2_apply_discarded`

目的说明：
- `stage1_l1_fail`: 检查 deterministic gate 是否误伤
- `stage1_l2_fail`: 检查 evidence inferability 是否过严
- `stage1_pass`: 检查 validated state 是否过度/不足保留
- `stage2_state_completion`: 检查 template 与 reuse
- `stage2_change_tracking`: 检查 changed-key 选择与 before/after
- `stage2_apply_accepted`: 检查题目质量与 validator 是否偏松
- `stage2_apply_discarded`: 检查 discard 是否合理、是否可救回

### 1.5 Part I gate checks
- Check `state_questionability` exists in checkpoint payload.
- Check `validated_snapshot_state` exists in checkpoint payload.
- Check `state_completion_pack` and `change_tracking_pack` exist when using new benchmark.
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
- For deeper review, prefer the stratified manual-review pack under `generation/rag/results/<user_id>/analysis/tce_manual_review/` over one mixed JSON.
- Note:
  - Stage 2 requires a Stage 1 artifact on disk.
  - 若 Stage 1 中途停止，可基于已落盘的 partial output 使用 `--resume` 继续。

## 2. Part II - Generation Execution

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
  - verify only validated-state intersection changes become `change_tracking_pack.keys`
- Task C:
  - verify `rq3_apply_service_qa` only uses validated states
  - verify each accepted item contains `validation`

### 2.3 RAG single-checkpoint smoke
```bash
python3 -m generation.run_tce \
  --config configs/experiments/tce/rag_per_key_time_quarterly.yaml \
  --max-checkpoints 1
```

Quarterly-time checkpoints:
- 使用 `build_tce_benchmark.py` 的默认采样（`--sampling-mode calendar --calendar-anchor-freq quarterly`）在 benchmark 构建阶段生成每 3 个月一个 checkpoint。
- 预测 metadata 将记录 `sampling_params.actual_tokens_at_cutoff`，可在时间轴图上标注 token 规模。
- generation 阶段应直接消费 benchmark 中已采样 checkpoint；当 benchmark 含 `sampling_strategy.stage=benchmark_build` 时，运行时采样参数会被忽略。

Note:
- 若 config 路径仍含 `{user_id}` 占位符，请改用 `generation.run_tce_batch` 或在 `generation.run_tce` 中传入显式 `--benchmark/--app-logs-path/--output`。
- RAG/TCE generation 现在支持：
  - key-level incremental save（通过 `baseline_params.save_every_generation_keys` 控制，推荐 `1`）
  - checkpoint-level concurrency（通过 `baseline_params.checkpoint_workers` 控制）
  - within-checkpoint concurrency（通过 `baseline_params.within_checkpoint_workers` 控制）
  - resume 只会跳过 `metadata._checkpoint_complete=true` 的完整 checkpoint；不完整 partial checkpoint 会重跑，避免吃到半成品
- 当前协议下，Task A state completion 必须 per-key 运行：
  - 每个 `state_key` 独立 retrieval
  - 每个 `state_key` 独立 answering prompt
  - prediction metadata 中应有 `per_key_retrieval[*]`
- 不要再用 checkpoint-level combined Task A retrieval 作为正式协议路径
- 长时间 formal rerun 的操作注意事项：
  - 如果 `Task A` 已切到 per-key，且 `save_prompt_and_raw=true`，`save_every_generation_keys=1` 会导致频繁整文件重写 prediction artifact。
  - 这会显著拖慢 formal run，并且在人工中断 / resume / 并发 checkpoint 执行时，使磁盘上的中间 prediction 文件变得不稳定。
  - 正式 rerun 若主要目标是拿最终 review artifact，优先使用较大的 `save_every_generation_keys`，避免依赖中途 partial artifact。
  - 若需要一个稳定的最终 prediction 文件，优先在 generation 函数返回后，把返回的 `result` 再单独序列化成一个 stable copy，而不是直接把中途 output 路径当作最终 review artifact。
- baseline concurrency policy 必须遵守协议中的 allowed/forbidden 约束：
  - `rag / oracle / icl / hipporag / amem_baseline`: 可同时开 `checkpoint_workers` 和 `within_checkpoint_workers`
  - `letta`: 两者都会被运行时强制降为 `1`
- 运行后应抽查 prediction metadata：
  - `concurrency_policy`
  - `requested_checkpoint_workers`
  - `requested_within_checkpoint_workers`
  - `effective_checkpoint_workers`
  - `effective_within_checkpoint_workers`
- 若随后运行 evaluation 且启用了 LLM judge，应确认 judge 请求粒度为：
  - Task A: 每个 `state_key` 一次
  - Task B: 每个 changed `state_key` 一次
  - Task C: 每个 `(state_key, qa_id)` 一次
- 对 Task C 还应额外检查 deterministic option metrics：
  - `rq3_apply_option_extractable_item_count`
  - `rq3_apply_option_prediction_coverage_on_extractable`
  - `rq3_apply_option_accuracy_on_extractable`
  - 这是 Task C 的主指标；Task C 已不再运行 LLM-as-a-judge

### 2.4 Letta/MemGPT expansion run
```bash
python3 -m generation.run_tce \
  --config configs/experiments/tce/memgpt.yaml \
  --max-checkpoints 1
```

Expected output:
- `generation/<baseline>/results/<user_id>/prediction/tce_results*.json`
- sibling `*_run_settings.yaml`

## 3. Part III - Evaluation Execution

### 3.1 Evaluate prediction
```bash
python3 -m eval.eval_tce \
  --benchmark data_construction/generated_outputs/gemini_3_flash_preview/<user_id>/tce_benchmark_task_packs.json \
  --prediction generation/<baseline>/results/<user_id>/prediction/tce_results.json \
  --output generation/<baseline>/results/<user_id>/eval/tce_eval.json \
  --enable-llm-judge \
  --llm-provider azure \
  --llm-model gpt-5-mini \
  --save-eyeball
```

### 3.2 Evaluation checks
- Check `summary` + `checkpoints` + `prediction_alignment` exist.
- Check snapshot/change/apply/gap metrics exist.
- Check evidence metrics are id-matching based.
- Check LLM judge request granularity:
  - snapshot judge should issue one request per `state_key`
  - change judge should issue one request per changed `state_key`
  - apply judge should issue one request per `(state_key, qa_id)`

### 3.3 Generate eval manual-review pack
当需要人工检查“LLM judge 是否过松/过严”或“Task A/B/C 的 rubric 是否不一致”时，优先生成 eval manual-review pack，不要直接手翻完整 eval JSON。

```bash
python3 debug_utils/build_tce_eval_manual_review_pack.py \
  --benchmark data_construction/generated_outputs/gemini_3_flash_preview/<user_id>/tce_benchmark_task_packs.json \
  --prediction generation/<baseline>/results/<user_id>/prediction/tce_results.json \
  --eval generation/<baseline>/results/<user_id>/eval/tce_eval.json \
  --output-dir generation/<baseline>/results/<user_id>/analysis/tce_eval_manual_review \
  --top-n 20
```

产物约定：
- `tce_eval_manual_review_summary.json`
- `snapshot_judge_suspicious_cases.json`
- `change_judge_suspicious_cases.json`
- `apply_judge_suspicious_cases.json`
- 对应 `*.csv`
- `README.md`

建议阅读顺序：
1. `apply_judge_suspicious_cases.json`
2. `change_judge_suspicious_cases.json`
3. `snapshot_judge_suspicious_cases.json`

### 3.4 Build state timeline viewer data
当需要以 `state` 为中心检查用户整条时间线上的 `expected / validated / predicted / evidence / evaluation` 时，优先生成 state timeline viewer 数据，而不是手工来回对照 benchmark、prediction、eval 和 app logs。

```bash
python3.11 debug_utils/build_tce_state_timeline_viewer.py \
  --benchmark data_construction/generated_outputs/gemini_3_flash_preview/<user_id>/tce_benchmark_task_packs.json \
  --prediction generation/<baseline>/results/<user_id>/prediction/tce_results.json \
  --eval generation/<baseline>/results/<user_id>/eval/tce_eval.json \
  --app-logs data_construction/generated_outputs/gemini_3_flash_preview/<user_id>/app_log_large.json \
  --output generation/<baseline>/results/<user_id>/analysis/state_timeline_viewer/state_timeline_viewer_data.json
```

如果只做可视化 review，优先使用 compact 模式，避免生成超大 JSON：

```bash
python3.11 debug_utils/build_tce_state_timeline_viewer.py \
  --benchmark data_construction/generated_outputs/gemini_3_flash_preview/<user_id>/tce_benchmark_task_packs.json \
  --prediction generation/<baseline>/results/<user_id>/prediction/tce_results.json \
  --eval generation/<baseline>/results/<user_id>/eval/tce_eval.json \
  --app-logs data_construction/generated_outputs/gemini_3_flash_preview/<user_id>/app_log_large.json \
  --output generation/<baseline>/results/<user_id>/analysis/state_timeline_viewer/state_timeline_viewer_data_compact.json \
  --compact
```

Viewer 入口：
- `analysis_tools/tce_state_timeline_viewer/index.html`
- `analysis_tools/tce_state_timeline_viewer/app.js`
- `analysis_tools/tce_state_timeline_viewer/styles.css`

本地打开方式：
```bash
cd analysis_tools/tce_state_timeline_viewer
python3.11 -m http.server 8000
```

然后在浏览器中打开：
```text
http://127.0.0.1:8000/index.html?data=../../generation/<baseline>/results/<user_id>/analysis/state_timeline_viewer/state_timeline_viewer_data.json
```

Viewer 显示规则：
- 主视图以 `validated_snapshot_state` 为准
- 空白格表示 `not present / not evaluated`，不是 `0 score`
- Task A / Task B 优先显示新 4-rubric judge；若 eval 仍是旧协议，则在 viewer 中兼容展示 legacy judge
- Task C 主视图只显示 deterministic metrics；若旧 eval 文件带 `rq3_llm_*`，会放入 legacy metrics 折叠块
- `--compact` 模式会去掉 `request/response/prompt/raw_model_output` 等大字段，适合作为默认 review 产物

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
- Benchmark: `data_construction/generated_outputs/.../tce_benchmark*.json`
- Prediction: `generation/<baseline>/results/<user_id>/prediction/tce_results*.json`
- Eval: `generation/<baseline>/results/<user_id>/eval/tce_eval*.json`
- Manual review: `generation/rag/results/<user_id>/analysis/rq3_manual_review/`
- Stratified manual review: `generation/rag/results/<user_id>/analysis/tce_manual_review/`

Storage note:
- 为避免 home 目录爆满，以下大产物目录允许长期落在 project storage，再通过 symlink 回填到 repo 原路径：
  - `data_construction/generated_outputs/...`
  - `generation/<baseline>/results/...`
- 当前推荐 project storage 根目录：
  - `/projects/standard/zrliu/shared/wenya/xie00470/dynamicmem/`
- 如果原路径已经是 symlink，运行脚本与 runbook 命令无需修改；继续使用 repo 内的原路径即可。

### 5.2 Quick checks
```bash
rg -n "DRAFT_MARKER" docs/protocols/temporal_checkpoint_evaluation_developer_manual.md

python3 -m generation.run_tce --help
python3 -m generation.run_tce_batch --help
python3 -m eval.eval_tce --help
python3 data_construction/build_tce_benchmark.py --help
```

Historical artifact note:
- `results*/` and `generated_outputs/` may contain legacy-name files and are intentionally not renamed.
