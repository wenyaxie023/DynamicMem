# QA

`QA` is the refactored MemBench pipeline with two clear stages:

1. `qa_context`: build normalized task data from schema + logs.
2. `qa_generation`: sample tasks and generate final `qa.json`.

All cross-stage contracts are in `shared/`.

## Structure

```text
QA/
  qa_context/
    extract_main_store.py   # schema/app_log_large -> states/events/logs
    context_builder.py      # states/events/logs -> raw.json
    task_builder.py         # raw.json -> task_t1..t6 / task_final_t1..t6
    task_store.py           # task_final_t1..t6 -> tasks.json
    cli.py

  qa_generation/
    core/sampler.py         # tasks.json -> tasks_sampled.json
    core/generator.py       # tasks_sampled.json + raw.json -> qa.json
    fix_qa_links.py         # post-fix event/log mapping in qa.json
    llm/client.py
    llm/parser.py
    prompts/
    cli.py

  shared/
    config.py               # ContextConfig / GenerationConfig
    task.py                 # Task dataclass contract

  cli.py                    # unified CLI
  __main__.py               # python -m QA
```

## Data Flow

Per user (`data/user{user_id}`):

1. `schema.json` + `app_log_large.json`
2. `states.json`, `events.json`, `logs.json`
3. `raw.json`
4. `task_t1..t6.json` + `task_final_t1..t6.json`
5. `tasks.json`
6. `tasks_sampled.json`
7. `qa.json`

## Unified CLI

Run from repo root (`MemBench`) so `python -m QA` can resolve:

```bash
python -m QA -h
```

### Context only

```bash
python -m QA context pipeline --user-id 10
python -m QA context extract --user-id 10
python -m QA context raw --user-id 10
python -m QA context tasks --user-id 10
python -m QA context store --user-id 10
```

### Generation only

```bash
python -m QA generation sample --user-id 10
python -m QA generation generate --user-id 10
python -m QA generation pipeline --user-id 10
```

Common generation args:

- `--provider`
- `--model`
- `--max-workers`
- `--retry-times`
- `--flush-every`
- `--sample-per-group`
- `--sample-seed`
- `--qtypes` (e.g. `1,2,3`)
- `--categories` (comma-separated)

### End-to-end

```bash
python -m QA pipeline --user-id 10
```

This runs:

1. `qa_context pipeline`
2. `qa_generation pipeline`

### Post-fix QA links

```bash
python -m QA fix-links --user-id 10
python -m QA fix-links --user-id 10 --inplace
```

`fix-links` re-maps evidence by qtype:

- `1-4`: `event_ids -> reference_evidence(app_log_ids)`
- `5-6`: `reference_evidence(app_log_ids) -> event_ids`

## Resume / Reliability

- `qa_generation` supports parallel requests (`max_workers`).
- Periodic hot flush via `flush_every`.
- Resume by existing `uid` in `qa.json`.
- Failures are counted and skipped; run can continue.

## Compatibility Script

For old habits:

```bash
python QA/scripts/run_pipeline.py --user-id 10
```

Equivalent to `python -m QA pipeline --user-id 10`.

