# DynamicMem Open-Source Restructure Plan

Working branch: `opensource-prep` (cut from `dev`; `dev` untouched).

Public name: **DynamicMem**. The paper anonymizes the system as "MUSE" for
NeurIPS double-blind review; the open-source repo and READMEs use **DynamicMem**.
Paper section → repo part mapping is kept so readers can align code with the paper.

## 1. Target top-level structure

```
dynamicmem/
├── trajectory_synthesis/      # PART 1  (paper §"MUSE: Trajectory Synthesis Pipeline")
├── benchmark_construction/    # PART 2  (paper §"MUSE Benchmark")
├── baseline_prediction/       # PART 3  (paper §"Experiment → Baselines")
├── evaluation/                # PART 4  (paper §"Evaluation Protocol & Metrics / Results")
├── tce_core/                  # SHARED  TCE protocol / data contracts (kept as-is)
├── bench_core/                # SHARED  evaluator contracts (kept as-is)
├── configs/                   # kept; YAML module paths rewritten
└── README.md                  # top-level navigation, one section per part
```

Rationale for keeping `tce_core/` + `bench_core/` as a shared layer instead of
splitting them across parts: `tce_core` is imported by part 2 (build scripts),
part 3 (10+ baselines), and part 4 (eval). Splitting file-by-file would shred
imports; a shared protocol layer is both cleaner and faithful to what it is.

## 2. Per-part migration map (tracked files only)

### PART 1 → `trajectory_synthesis/` (from `data_construction/`)
- `stages/` (stage1_dynamic_profile, stage1_utils, stage2_events_chain, stage3_app_logs, README)
- `dynamic_profile_generator.py`, `events_chain_generator.py`
- `elite_persona_sampler.py`, `download_world_backgrounds.py`
- `batch_generation_runner.py`, `generation_pipeline.py`
- `add_reference_app_logs.py`
- `app_catalog.py`, `app_models.py`, `app_system.py`, `domains.py` (app world model)
- `prompt_templates.py`, `llm_client.py`, `logging_utils.py`
- `README.md`
- TRIAGE: `prepare_test_data.py` (keep only if used by a public flow)

### PART 2 → `benchmark_construction/` (from `data_construction/build_tce_*`)
- `build_tce_benchmark.py`, `build_tce_task_packs.py`
- `build_tce_rq3_apply_pack.py`, `build_tce_rq3_know_apply_pack.py`
- `build_tce_state_validation.py`, `verify_tce_groundtruth.py`
- `run_build_tce_benchmark.sh`
- (logic lives in shared `tce_core/`; this dir is the construction entrypoints)

### PART 3 → `baseline_prediction/` (from `generation/`, minus QA)
- KEEP: `run_tce.py`, `run_tce_batch.py`, `tce_config.py`, `tce_safety.py`,
  `adapters/`, `common/`, all baseline dirs (Amem, HippoRAG2, Kimi-Linear,
  Mamba, MemAgent, MemoryOS, icl, letta, mem0, oracle, rag, simplemem, zep),
  `run_*_tce.sh`, `README.md`
- CUT (QA / End-to-End, per decision): `qa_adapters/`, `run_qa.py`, `run_qa.sh`,
  `run_qa_batch.sh`, `qa_config.py`, `load_dataset.py` (CONFIRMED cut)

### PART 4 → `evaluation/` (from `eval/` + `bench_core/`)
- KEEP: `eval_tce.py`, `metrics.py`, `prompts_tce.py`, `client.py`, `config.py`,
  `generation.py`, `logger.py`, `std_eval_data.py`, `run_eval_tce.sh`,
  `build_tce_analysis_pack.py`, `analyze_tce_*.py`, `README.md`
- CUT (CONFIRMED, QA end-to-end): `eval.py`, `run_eval.sh`, `prompts.py`
- `bench_core/` stays at top level as shared evaluator contract

### SHARED → `tce_core/`, `bench_core/` (kept, not moved)
- All current `tce_core/*` and `bench_core/*` stay. `tce_core/final_checkpoint_qa.py`
  is TCE final-checkpoint (not End-to-End QA) — KEEP unless verified otherwise.

## 3. Cut list (open-source hygiene)

- `QA/` (entire dir) — End-to-End QA pipeline, out of scope
- `generation/qa_adapters/`, `generation/run_qa*`, `generation/qa_config.py`
- `eval/eval.py`, `eval/run_eval.sh`, `eval/prompts.py` (pending confirm = QA)
- `docs/` QA contract docs; prune docs to TCE-only
- Root one-off scripts: `analyze_topk_refusal.py`, `calc_final_cost.py`,
  `generate_stress_data.py`, `run_missing_users.sh`, `tce_contracts.py`
- `data_construction/prepare_test_data.py` (CONFIRMED cut, QA-only)
- `eval/eval.py`, `eval/run_eval.sh`, `eval/prompts.py`, `generation/load_dataset.py`,
  `generation/qa_config.py` (CONFIRMED cut)
- `tests/` + `acceptance_*`: drop QA-only ones, keep TCE ones (dedicated pass)
- Root duplicate `app_catalog.py`/`app_system.py` (differ from data_construction
  copies — keep the data_construction versions, delete root)
- `analysis_tools/` viewers — delete, or relocate to `tools/` if any are useful
- Already gitignored (no action, just confirm absent from tarball): `.env`,
  `*.json` dumps, `analysis*.{ipynb,md}`, `.tmp_*_upstream/`, `analyze_code_v*/`,
  `debug_outputs/`, `formal*_task_pack_statistics.md`

## Execution status (updated)

DONE (step 1–2, branch `opensource-prep`, uncommitted — review then commit):
- Removed 3 git submodules (HippoRAG/MemAgent/graphiti) + `.gitmodules`;
  they become documented optional external deps (decided).
- `generation/` → `baseline_prediction/`; QA files cut.
- `eval/` → `evaluation/`; QA files cut (`eval.py`, `run_eval.sh`, `prompts.py`).
- `data_construction/` split → `trajectory_synthesis/` (21 files) +
  `benchmark_construction/` (7 files); `prepare_test_data.py` cut; dir removed.
- Cut: `QA/`, `analysis_tools/`, root one-off scripts, root `app_catalog.py`/
  `app_system.py` duplicates.
- Top level now: trajectory_synthesis, benchmark_construction,
  baseline_prediction, evaluation, tce_core, bench_core, configs + leftovers.

CORRECTION (load_dataset.py was misclassified):
- `load_dataset.py` is NOT QA-only. It is a shared benchmark dataset loader
  (`load_membench_dataset`, `build_membench_memory_from_event`, `MemBenchSample`)
  used by 3 kept TCE baselines (Amem, MemoryOS, mem0). Restored in place as
  `baseline_prediction/load_dataset.py`. (`eval.py`, `eval/prompts.py`,
  `prepare_test_data.py` re-verified as correctly cut — no kept importers.)
- `letta` internal `run_qa` is a local function, not the cut `run_qa.py` — no
  dependency; letta's internal QA codepath is a later deep-prune item.
- `tce_contracts.py` was ALSO misclassified (was in "root one-off scripts" cut
  list). It is a core shared contracts module imported by 15+ kept files across
  ALL parts (tce_core x5, bench_core, benchmark_construction x3, evaluation x2,
  baseline_prediction/oracle). Restored in place at repo root `tce_contracts.py`.

VALIDATED (env mem0311, python 3.11):
- All tracked .py rewritten: 0 unresolved old-module imports.
- Import smoke: 11 key modules across 4 parts + shared core import 0-fail.
- `python -m baseline_prediction.run_tce --config <single-user> --dry-run` OK.

STEP DONE — imports + configs + outputs root:
- Python imports rewritten (74 files / 128 lines): data_construction split →
  trajectory_synthesis / benchmark_construction; eval → evaluation;
  generation → baseline_prediction.
- 12 TCE configs: `generation/` → `baseline_prediction/`.
- Artifact root decided **top-level `outputs/`**: 43 TCE configs +
  `.gitignore` + canonical writers (batch_generation_runner.py,
  download_world_backgrounds.py, add_reference_app_logs.py) rewritten.

STEP DONE — shell-script parameterization:
- All 35 tracked .sh: `-m generation.*` → `-m baseline_prediction.*`;
  `$PROJECT_ROOT/generation/` → `/baseline_prediction/`;
  `data_construction/generated_outputs/` → `outputs/`;
  build script python target → `benchmark_construction/`.
- 6 hardcoded `PROJECT_ROOT=/abs/path` → env-overridable script-relative
  derivation; verified resolve to repo root for all depths.
- `run_letta_apptainer_server.sh` storage default → `$HOME/.letta`.
- CUT `baseline_prediction/rag/run_rag.sh` (QA RAG runner pointing at a
  foreign `behavior_and_conversation` repo + `--qa-dir`; TCE rag uses
  `run_rag_tce.sh`).
- `run_build_tce_benchmark.sh` no longer hardcodes the (untracked,
  analysis_tools/) research doc; defers to Python built-in default unless
  `CANONICAL_RESEARCH_DOC` env is set.
- Validated: 35/35 `bash -n` pass; 0 residual `data_construction/` or
  absolute machine paths in scripts.

INCIDENT (found + fixed): `git mv --sparse` (used to work around the
uninitialized-submodule block) set SKIP_WORKTREE on 188 files. `git add -u`
silently ignored working-tree edits to skip-worktree files, so commits
4edfe8c / 62ca5ad were incomplete (48 files: 29 .py, 18 .sh, 1 .md).
Fixed: cleared skip-worktree repo-wide, corrective commit, re-validated
on the committed tree (0 old imports, 12/12 import smoke, 35/35 bash -n,
dry-run OK). RULE: after any `git mv --sparse`, immediately run
`git ls-files -v | grep ^S` and clear skip-worktree before further edits.

OPEN CONTENT GAP (needs user decision — surfaced):
- `tce_contracts.CANONICAL_RESEARCH_DOC_V2` →
  `analysis_tools/tce_research_questions/001_user_001/new_research_question.md`,
  which is UNTRACKED and lives under the cut `analysis_tools/`. Part 2
  benchmark construction needs this doc. analysis_tools/ cut was correct (only
  6 web-viewer assets were tracked; this md was never tracked), but the doc
  itself must be given a tracked home or documented as required input.

REMAINING path/string surface (later dedicated passes, NOT this step):
- `trajectory_synthesis/generation_pipeline.py` debug_v* / _v2 / _elite_sample
  scratch entrypoints (15+) — prune or repoint during scratch-cleanup pass.
- Docs (`docs/runbooks/*`, top READMEs, evaluation/README.md) + tests +
  `evaluation/statistics.ipynb` + `user_data/rebuild_checkpoint.py`: handled in
  docs-prune / tests-QA / notebook passes.
- Error-message strings still say "generation.run_tce" (cosmetic) —
  string-cleanup pass.

NOT YET DONE (repo will not run until imports/configs rewritten):
- Rewrite imports (`data_construction.*` / `eval.*` / `generation.*` → new pkgs)
- Rewrite `configs/experiments/tce/*.yaml` module paths; cut `configs/qa.*`,
  `configs/experiments/qa/`
- `tests/` + `acceptance_*` QA-vs-TCE pass (e.g. `tests/test_load_dataset_payload.py`)
- `docs/` prune (e.g. `docs/protocols/qa_generation_and_eval_contract.md`)
- Triage `debug_utils/` (3), `user_data/` (1)
- Rewrite 5 READMEs; packaging (pyproject + 3 env specs); add MIT LICENSE

## 4. Execution order

1. `git mv` files into the 4 part dirs (preserves history). Keep `tce_core/`,
   `bench_core/`, `configs/` in place.
2. Delete cut list.
3. Rewrite imports: `data_construction.*` → `trajectory_synthesis.* /
   benchmark_construction.*`; `eval.*` → `evaluation.*`; `generation.*` →
   `baseline_prediction.*`. (`tce_core` / `bench_core` import paths unchanged.)
4. Rewrite `configs/experiments/tce/*.yaml` module references.
5. Rewrite 4 part READMEs + top-level README, aligned to paper terminology
   (DynamicMem / Trajectory Synthesis / Benchmark / Baselines / Evaluation).
6. Validate: `python -m baseline_prediction.run_tce --config <...> --dry-run`
   and import smoke test across all 4 parts.

## 5. Resolved decisions (from review)

- CUT: `load_dataset.py`, `eval/eval.py`, `eval/run_eval.sh`, `eval/prompts.py`,
  `data_construction/prepare_test_data.py` (all QA-only)
- `tests/` + `acceptance_*`: drop QA-only, keep TCE — dedicated pass during step 2
- Public dir names: **plain** (`trajectory_synthesis/`, no number prefix) so they
  are importable Python packages
- Packaging + environments: in scope — see section 6

## 6. Packaging & environments

Current state: **no top-level dependency manifest**. Deps are scattered across
per-baseline dirs (`generation/Amem/requirements.txt`, `Kimi-Linear`, `Mamba`)
and three git submodules (`HippoRAG2/HippoRAG`, `MemAgent/MemAgent_src`,
`zep/graphiti`). This must be fixed for open source.

Multi-environment layout (conda):

| Env name   | Scope                                                            |
|------------|------------------------------------------------------------------|
| `mem0311`  | **default** — parts 1, 2, 4 + all baselines except the two below |
| `memoryos` | MemoryOS baseline only (part 3)                                  |
| `hipporag2`| HippoRAG2 baseline only (part 3)                                 |

Plan:
- Top-level `pyproject.toml` for the core package (parts 1/2/4 + shared
  `tce_core`/`bench_core`), pinned to the `mem0311` env contents → export via
  `conda env export --no-builds` then curate.
- `environment/` dir with three reproducible specs: `environment.default.yml`
  (mem0311), `environment.memoryos.yml`, `environment.hipporag2.yml`. README
  states which baseline needs which env.
- Per-baseline `requirements.txt` kept inside each `baseline_prediction/<X>/`
  and referenced from the baseline's local README.
- Submodules: document `git submodule update --init --recursive`; pin commits;
  verify their licenses are redistribution-compatible (HippoRAG, MemAgent,
  graphiti) before release.
- LICENSE: **MIT** (decided). Add top-level `LICENSE` file. Add NOTICE for
  vendored submodules; verify each submodule's license is MIT-compatible
  (HippoRAG, MemAgent, graphiti) before release.

## 7. Open items still to decide

- Whether submodule baselines ship as submodules or as documented optional
  add-ons (license review pending)
