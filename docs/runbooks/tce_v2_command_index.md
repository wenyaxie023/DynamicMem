# TCE v2 Command Index

Status: active
Owner: DynamicMem team
Last Updated: 2026-04-23

Purpose:
- keep one copy-pasteable place for TCE v2 pack build, baseline build-only, generation, evaluation, and viewer commands
- prefer commands and configs that exist in the current repo checkout
- mark viewer/export gaps explicitly when the current worktree does not contain a tracked exporter entrypoint

Related docs:
- protocol: `docs/protocols/temporal_checkpoint_evaluation_developer_manual.md`
- main runbook: `docs/runbooks/tce_execution_runbook.md`
- current RAG Task C handoff notes: `docs/runbooks/rag_tce_v2_taskc_handoff.md`

## 0. Shell Bootstrap

Use a compute node for build / generation / evaluation. Viewer serving can run on a login node.

```bash
cd /users/4/xie00470/mem_bench/dynamicmem
source ~/miniconda/etc/profile.d/conda.sh
conda activate mem0311
```

Optional preflight:

```bash
python3 debug_utils/test_azure_key.py
```

## 1. TCE v2 Pack Build

### 1.1 Raw benchmark

```bash
python3 -m data_construction.build_tce_benchmark \
  --app-logs-final data_construction/generated_outputs/gemini_3_flash_preview/<USER_ID>/app_logs_final.json \
  --all-events-chains data_construction/generated_outputs/gemini_3_flash_preview/<USER_ID>/all_events_chains.json \
  --output data_construction/generated_outputs/gemini_3_flash_preview/<USER_ID>/tce_benchmark_vnext_raw.json \
  --sampling-mode calendar \
  --calendar-anchor-freq quarterly \
  --app-logs-large data_construction/generated_outputs/gemini_3_flash_preview/<USER_ID>/app_log_large.json \
  --sampling-tokenizer-model gpt-4o-mini \
  --task-contract-version taskabc_v2 \
  --research-frame-version rq_20260413 \
  --canonical-research-doc analysis_tools/tce_research_questions/001_user_001/new_research_question.md
```

### 1.2 State validation

```bash
python3 -m data_construction.build_tce_state_validation \
  --benchmark data_construction/generated_outputs/gemini_3_flash_preview/<USER_ID>/tce_benchmark_vnext_raw.json \
  --output data_construction/generated_outputs/gemini_3_flash_preview/<USER_ID>/tce_benchmark_vnext_state_validated.json \
  --validator-provider gemini \
  --validator-model gemini-3-flash-preview \
  --l2-evidence-top-k 0 \
  --save-every-states 5 \
  --resume
```

### 1.3 Task-pack build

```bash
python3 -m data_construction.build_tce_task_packs \
  --benchmark data_construction/generated_outputs/gemini_3_flash_preview/<USER_ID>/tce_benchmark_vnext_state_validated.json \
  --output data_construction/generated_outputs/gemini_3_flash_preview/<USER_ID>/tce_benchmark_vnext_task_packs.json \
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

## 2. TCE Pack Viewer

Build compact review payload from a task-pack benchmark:

```bash
python3.11 debug_utils/build_tce_task_pack_review.py \
  --benchmark data_construction/generated_outputs/gemini_3_flash_preview/001_user_001/tce_benchmark_vnext_20260423_taskc_v2_rubric_rewrite_from_state_validated.json \
  --output analysis_tools/tce_task_pack_review/task_pack_review_data.json \
  --focus task_c
```

Serve the static page:

```bash
python3 -m http.server 8310 --directory .
```

Open:

```text
http://127.0.0.1:8310/analysis_tools/tce_task_pack_review/index.html
```

## 3. Baseline Build-Only / Memory Materialization

These are the tracked build-only configs in the current repo.

### 3.1 RAG embedding-cache build-only

```bash
python3 -m generation.run_tce \
  --config configs/experiments/tce/rag_user1_taskc_only_20260420_build_only.yaml
```

Notes:
- this prebuilds the app-log embedding cache only
- no standalone memory-build viewer is tracked for RAG

### 3.2 AMEM build-only

```bash
python3 -m generation.run_tce \
  --config configs/experiments/tce/amem_raw_build_only.yaml
```

### 3.3 MemoryOS build-only

```bash
python3 -m generation.run_tce \
  --config configs/experiments/tce/memoryos_build_only_user001.yaml
```

Raw-benchmark build-only variant:

```bash
python3 -m generation.run_tce \
  --config configs/experiments/tce/memoryos_build_only_raw.yaml
```

### 3.4 SimpleMem build-only

```bash
python3 -m generation.run_tce \
  --config configs/experiments/tce/simplemem_build_only.yaml
```

Raw-benchmark build-only variant:

```bash
python3 -m generation.run_tce \
  --config configs/experiments/tce/simplemem_build_only_raw.yaml
```

### 3.5 HippoRAG2 build-only

```bash
python3 -m generation.run_tce_batch \
  --config configs/experiments/tce/hipporag2_build_only_raw.yaml --users 004_user_004
```

Raw-benchmark build-only variant:

```bash
python3 -m generation.run_tce \
  --config configs/experiments/tce/hipporag2_build_only_raw.yaml
```

### 3.6 Build-only status for other tracked baselines

- `memoryos`, `amem`, `simplemem`, `hipporag2`, and `rag` have tracked build-only configs
- no tracked build-only config currently exists under `configs/experiments/tce/` for `oracle`, `icl`, `zep`, `letta`, `memgpt`, `mem0`, or `nemori`

## 4. Baseline Generation

### 4.1 Current Task C-only runs

RAG user001 Task C only 5 checkpoints:

```bash
python3 -m generation.run_tce \
  --config configs/experiments/tce/rag_user1_taskc_only_20260423_5ckpt.yaml
```

SimpleMem user001 Task C only 5 checkpoints:

```bash
python3 -m generation.run_tce \
  --config configs/experiments/tce/simplemem_user1_taskc_only_20260423_5ckpt.yaml
```

RAG handoff config:

```bash
bash generation/rag/run_rag_tce_taskc_handoff.sh
```

### 4.2 Full / general TCE configs currently tracked

RAG:

```bash
python3 -m generation.run_tce \
  --config configs/experiments/tce/rag_user1_v14_top20_c4.yaml
```

MemoryOS:

```bash
python3 -m generation.run_tce \
  --config configs/experiments/tce/memoryos.yaml
```

HippoRAG2:

```bash
python3 -m generation.run_tce \
  --config configs/experiments/tce/hipporag2.yaml
```

Letta:

```bash
python3 -m generation.run_tce \
  --config configs/experiments/tce/letta_v14_user1.yaml
```

MemGPT:

```bash
python3 -m generation.run_tce \
  --config configs/experiments/tce/memgpt_v14.yaml
```

mem0:

```bash
python3 -m generation.run_tce \
  --config configs/experiments/tce/mem0.yaml
```

Oracle batch config:

```bash
python3 -m generation.run_tce_batch \
  --config configs/experiments/tce/oracle.yaml
```

ICL batch config:

```bash
python3 -m generation.run_tce_batch \
  --config configs/experiments/tce/icl.yaml
```

Zep:

```bash
python3 -m generation.run_tce \
  --config configs/experiments/tce/zep.yaml
```

Nemori batch config:

```bash
python3 -m generation.run_tce_batch \
  --config configs/experiments/tce/nemori.yaml
```

Notes:
- the current tracked `amem` config is build-only, not a full prediction config
- some configs are user001-specific because their benchmark/output paths are hard-coded or single-user by design

## 5. Evaluation

### 5.1 Generic TCE eval

```bash
python3 -m eval.eval_tce \
  --benchmark <BENCHMARK_TASK_PACKS_JSON> \
  --prediction <PREDICTION_JSON> \
  --output <EVAL_JSON> \
  --enable-llm-judge \
  --llm-provider azure \
  --llm-model gpt-5-mini \
  --llm-max-workers 4 \
  --save-eyeball
```

### 5.2 Current RAG Task C-only eval

```bash
python3 -m eval.eval_tce \
  --benchmark /projects/standard/zrliu/shared/wenya/xie00470/dynamicmem/data_construction/generated_outputs/gemini_3_flash_preview/001_user_001/tce_benchmark_vnext_20260423_taskc_v2_rubric_rewrite_from_state_validated.json \
  --prediction /users/4/xie00470/mem_bench/dynamicmem/generation/rag/results/001_user_001/prediction/rag_user1_taskc_only_20260423_5ckpt/tce_results_taskc_only_5ckpt.json \
  --output /users/4/xie00470/mem_bench/dynamicmem/generation/rag/results/001_user_001/eval/rag_user1_taskc_only_20260423_5ckpt/tce_eval_taskc_only_5ckpt.json \
  --enable-llm-judge \
  --llm-provider azure \
  --llm-model gpt-5-mini \
  --llm-max-workers 4 \
  --save-eyeball
```

### 5.3 Existing eval wrapper scripts

- generic wrapper: `eval/run_eval_tce.sh`
- RAG: `generation/rag/run_eval_rag_tce.sh`
- Oracle: `generation/oracle/run_eval_oracle_tce.sh`
- ICL: `generation/icl/run_eval_icl_tce.sh`
- Letta: `generation/letta/run_eval_letta_tce.sh`

Resume note:
- add `--resume` to continue from an existing eval JSON at `--output`
- resumed eval now retries cached slot-judge units that only contain evaluator-generated `judge_error:` judgments; successful units are still skipped

## 6. TCE State Timeline Viewer

Build compact viewer payload:

```bash
python3.11 debug_utils/build_tce_state_timeline_viewer.py \
  --benchmark /projects/standard/zrliu/shared/wenya/xie00470/dynamicmem/data_construction/generated_outputs/gemini_3_flash_preview/001_user_001/tce_benchmark_vnext_20260423_taskc_v2_rubric_rewrite_from_state_validated.json \
  --prediction /users/4/xie00470/mem_bench/dynamicmem/generation/rag/results/001_user_001/prediction/rag_user1_taskc_only_20260423_5ckpt/tce_results_taskc_only_5ckpt.json \
  --eval /users/4/xie00470/mem_bench/dynamicmem/generation/rag/results/001_user_001/eval/rag_user1_taskc_only_20260423_5ckpt/tce_eval_taskc_only_5ckpt.json \
  --app-logs /projects/standard/zrliu/shared/wenya/xie00470/dynamicmem/data_construction/generated_outputs/gemini_3_flash_preview/001_user_001/app_log_large.json \
  --output generation/rag/results/001_user_001/analysis/rag_user1_taskc_only_20260423_5ckpt/state_timeline_viewer_data_compact.json \
  --compact
```

Serve:

```bash
python3 -m http.server 8000 --directory .
```

Open:

```text
http://127.0.0.1:8000/analysis_tools/tce_state_timeline_viewer/index.html?data=/generation/rag/results/001_user_001/analysis/rag_user1_taskc_only_20260423_5ckpt/state_timeline_viewer_data_compact.json
```

## 7. Baseline Memory Viewers

### 7.1 AMEM memory viewer

Export viewer payload:

```bash
python3.11 -m generation.Amem.export_memory_viewer \
  --state-pkl results/Amem/snapshots/amem_build_only__main/membench_amem_001_user_001_large.pkl \
  --checkpoint-json results/Amem/snapshots/amem_build_only__main/membench_amem_001_user_001_large.json \
  --output analysis_tools/amem_memory_viewer/amem_memory_viewer_data.json
```

Serve:

```bash
python3 -m http.server 8301 --directory .
```

Open:

```text
http://127.0.0.1:8301/analysis_tools/amem_memory_viewer/index.html
```

### 7.2 MemoryOS memory viewer

Refresh viewer payload:

```bash
python3.11 debug_utils/build_memoryos_memory_viewer.py \
  --snapshot-root results/MemoryOS/snapshots/memoryos_v14_user1_azure_build_only/001_user_001_large \
  --output results/MemoryOS/results/001_user_001/prediction/memoryos_v14_user1_azure_build_only/memory_viewer_data.json
```

Serve:

```bash
python3 -m http.server 8302 --directory .
```

Open:

```text
http://127.0.0.1:8302/analysis_tools/memoryos_memory_viewer/index.html?data=/results/MemoryOS/results/001_user_001/prediction/memoryos_v14_user1_azure_build_only/memory_viewer_data.json
```

### 7.3 SimpleMem memory viewer

Tracked static viewer assets exist:
- `analysis_tools/simplemem_build_viewer/simplemem_build_viewer_report.html`
- `analysis_tools/simplemem_build_viewer/simplemem_build_viewer_data.json`

Serve:

```bash
python3 -m http.server 8304 --directory .
```

Open:

```text
http://127.0.0.1:8304/analysis_tools/simplemem_build_viewer/simplemem_build_viewer_report.html
```

Note:
- the current worktree contains the static viewer assets
- a tracked `generation.simplemem.build_viewer` exporter entrypoint is not present in this checkout, so there is no verified refresh command to record here

### 7.4 HippoRAG2 memory viewer

Tracked static viewer assets exist:
- `analysis_tools/hipporag2_memory_viewer/index.html`
- `analysis_tools/hipporag2_memory_viewer/hipporag2_memory_viewer_data.json`

Serve:

```bash
python3 -m http.server 8303 --directory .
```

Open:

```text
http://127.0.0.1:8303/analysis_tools/hipporag2_memory_viewer/index.html
```

Note:
- the current worktree contains the static viewer assets
- a tracked `generation.HippoRAG2.export_memory_viewer` exporter entrypoint is not present in this checkout, so there is no verified refresh command to record here

### 7.5 Letta, mem0, Zep, and other viewer surfaces

Tracked static viewer assets exist for some baselines:
- `analysis_tools/letta_agent_memory_viewer/`
- `analysis_tools/mem0_build_viewer/`
- `analysis_tools/zep_memory_viewer/`

Current repo status:
- no verified refresh/export command was found in the tracked worktree for these viewer payloads
- serve them only when you already have a matching `*_viewer_data.json` artifact to inspect

## 8. Wrapper Script Index

Generation wrappers currently present:
- `generation/rag/run_rag_tce.sh`
- `generation/oracle/run_oracle_tce.sh`
- `generation/icl/run_icl_tce.sh`
- `generation/Amem/run_amem_tce.sh`
- `generation/run_hipporag2_tce.sh`
- `generation/run_memgpt_tce.sh`
- `generation/run_nemori_tce.sh`
- `generation/run_zep_tce.sh`
- `generation/letta/run_letta_tce.sh`
- `generation/mem0/run_mem0_tce.sh`

Eval wrappers currently present:
- `eval/run_eval_tce.sh`
- `generation/rag/run_eval_rag_tce.sh`
- `generation/oracle/run_eval_oracle_tce.sh`
- `generation/icl/run_eval_icl_tce.sh`
- `generation/letta/run_eval_letta_tce.sh`
