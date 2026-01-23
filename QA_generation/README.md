# MemBench QA Generation Pipeline

This folder contains a lightweight QA generation pipeline that builds atomic facts, samples atom subsets by time window, and generates QA pairs with an LLM.

## Pipeline Overview

The pipeline runs in four stages:

1. **Expand macros**: Expand `qa_generation/atom_macros.yaml` into `qa_generation/atom_ir.yaml`.
2. **Build atoms**: Convert schema + IR into `data/atoms.json`.
3. **Sample atoms**: Sample per time-window group into `data/atoms_<tag>.json` and return a registry list.
4. **Generate QA**: Use the registry to generate QA JSON files per window group.

Key files:

- `qa_generation/run_pipeline.py`: main entry point with CLI flags.
- `qa_generation/sampling.py`: sampling logic; builds the registry list.
- `qa_generation/generation.py`: QA generation using registry entries.
- `qa_generation/prompts.py`: prompt templates.
- `qa_generation/config.py`: default config values and paths.

## Run the Pipeline

From the repo root:

```bash
python qa_generation/run_pipeline.py
```

### Common Options

- Skip macro expansion:

```bash
python qa_generation/run_pipeline.py --skip-expand
```

- Skip atom building:

```bash
python qa_generation/run_pipeline.py --skip-atoms
```

- Skip sampling (uses existing sampled atom files):

```bash
python qa_generation/run_pipeline.py --skip-sample
```

- Skip QA generation:

```bash
python qa_generation/run_pipeline.py --skip-qa
```

- Control flush size explicitly:

```bash
python qa_generation/run_pipeline.py --flush-size 50
```

- Add a suffix to QA output files:

```bash
python qa_generation/run_pipeline.py --output-suffix _v2
```

## Output Files

- `data/atoms.json`: full atom list.
- `data/atoms_<tag>.json`: sampled atoms per window group.
- `data/qa_<tag>.json`: QA output per window group.

If `--output-suffix` is provided, QA outputs become `data/qa_<tag><suffix>.json`.

## Notes

- Default flush size uses `flush_rate` from `qa_generation/config.py` (default 5% of the sampled count, minimum 1).
- The registry entries include `prompt` and optional `suffix`, allowing per-window customization.
