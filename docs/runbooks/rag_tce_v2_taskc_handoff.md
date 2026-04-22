# RAG TCE v2 Task C Handoff

This note is for running the repo's `rag` baseline on the TCE v2 `Task C` benchmark only.

## Files to provide

Two input files are required:

1. Task-pack benchmark JSON  
   Google Drive link: `https://drive.google.com/file/d/1YyszIUXba4H0PvGPVHE_JJBWfn-Bdu4r/view?usp=drive_link`
2. Raw app log JSON used by RAG retrieval  
   `app_log_large.json`

Important:
- The benchmark pack JSON alone is not enough for the RAG baseline.
- RAG retrieval also needs `app_log_large.json`.

## Expected placement

Place the downloaded files under the repo root:

```text
handoff_inputs/tce_benchmark_taskc_v2.json
handoff_inputs/app_log_large.json
```

If you prefer different file locations, edit the two paths in:

`configs/experiments/tce/rag_user1_taskc_only_handoff_5ckpt_openai.yaml`

## One-time setup

From the `dynamicmem/` repo root:

```bash
cd dynamicmem
```

Then activate whichever Python environment you normally use for this repo.

Example with conda:

```bash
conda activate <your_env_name>
```

Credential setup:

- The current `rag` runner calls `load_dotenv()`, so if you run from the `dynamicmem/` repo root it will read credentials from the repo-root `.env`.
- Default handoff config uses `OpenAI`, so the relevant `.env` entries are:

```bash
OPENAI_API_KEY=...
OPENAI_BASE_URL=...
```

- If you want `Azure OpenAI` instead, update the config `llm.provider` and `retriever.provider` from `openai` to `azure`. Then the relevant `.env` entries are:

```bash
AZURE_OPENAI_API_KEY=...
AZURE_OPENAI_BASE_URL=...
```

- Exporting env vars manually in the shell is also fine, but it is not required if `.env` is already set up.

## Run

Use the generation YAML for the actual handoff run.
If you only want to prebuild the RAG embedding cache, use a separate `build_only` YAML instead.

### Generation

The default one-command entrypoint is:

```bash
bash generation/rag/run_rag_tce_taskc_handoff.sh
```

This wrapper uses:

`configs/experiments/tce/rag_user1_taskc_only_handoff_5ckpt_openai.yaml`

### Optional build-only path

`generation` also prepares the embedding cache when needed, but then continues into Task C retrieval and answer generation.
`build_only` stops after cache preparation, so it does not produce the Task C prediction JSON used for evaluation.

Repo example:

`configs/experiments/tce/rag_user1_taskc_only_20260420_build_only.yaml`

Before using that example for this handoff, update at least:

- `data.benchmark`
- `data.app_logs_path`
- `output.prediction_path`
- provider settings if you want `openai` instead of `azure`

Then run:

```bash
python3 -m generation.run_tce --config <your_build_only_yaml>
```

## What this config does

The default handoff generation YAML is:

`configs/experiments/tce/rag_user1_taskc_only_handoff_5ckpt_openai.yaml`

It is configured to:

- run baseline `rag`
- run `task_selection: task_c_only`
- run the first `5` checkpoints
- use `gpt-5-mini` for answering
- use `text-embedding-3-large` for retrieval embeddings
- write outputs under `generation/rag/results/001_user_001/prediction/`

Most important knobs:

- `runtime.task_selection: task_c_only`
  - only runs TCE v2 Task C
- `runtime.max_checkpoints: 5`
  - limits the run to the first 5 checkpoints
- `retrieval.top_k: 20`
  - shared default retrieval top-k
- `retrieval.rq3_apply_top_k: 20`
  - main Task C retrieval top-k setting for this handoff run
- `llm.provider` / `retriever.provider`
  - choose `openai` or `azure`
- `llm.model`
  - answer model
- `retriever.model`
  - embedding model

## Output location

Default prediction output:

`generation/rag/results/001_user_001/prediction/rag_user1_taskc_only_handoff_5ckpt_openai/tce_results_taskc_only_5ckpt.json`

If you change `runtime.experiment_name`, the output directory changes with it.
