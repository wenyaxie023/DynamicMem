# Data Construction Pipeline (Stages)

This directory implements a 3-stage benchmark data construction pipeline:

1. `Stage 1`: Dynamic profile generation and conflict resolution
2. `Stage 2`: Event-chain generation (by time window)
3. `Stage 3`: App log generation (with resume support)

The main orchestrator is `../batch_generation_runner.py`.

For full end-to-end usage (including state-abstraction benchmark construction and evaluation workflow), see `../README.md`.

## Directory Overview

- `stage1_dynamic_profile.py`: Generates `user_basic_profile`, `dynamic_profiles`, and `world_backgrounds`, then applies multi-round conflict fixes.
- `stage1_utils.py`: Utility functions for Stage 1 rule checks, key alignment, conflict detection, and conflict application.
- `stage2_events_chain.py`: Converts dynamic states into observable event chains in window order (`w0 -> w4`).
- `stage3_app_logs.py`: Merges all events and generates structured app logs with checkpoint management.
- `__init__.py`: Exports `DynamicProfileStage / EventsChainStage / AppLogsStage`.

## End-to-End Flow

### Stage 1: Dynamic Profile

Key responsibilities:

- Generate `user_basic_profile` from raw user description
- Generate initial dynamic profiles per domain
- Apply in-domain rule1~rule5 fixes
- Perform cross-domain key alignment
- Resolve cross-domain attribute and temporal conflicts

Typical outputs (under each user output directory):

- `user_basic_profile.json`
- `*_world_background.json` (per domain)
- `world_backgrounds.json` (aggregated)
- `dynamic_profiles_raw.json`
- `dynamic_profiles_domain_level_fixes_applied.json`
- `dynamic_profiles_key_aligned.json`
- `dynamic_profiles_conflict_resolved.json`
- `dynamic_profiles_final.json`

Supports a world-background `dry run` mode: only saves world background prompts without generating background content.

### Stage 2: Events Chain

Key responsibilities:

- Read `dynamic_profiles_final.json` from Stage 1
- Generate event chains by domain and by window
- Maintain inter-window context continuity and state-to-event conversion markers

Typical outputs:

- `*_events_chain.json` (per domain)
- `all_events_chains.json` (aggregated)

### Stage 3: App Logs

Key responsibilities:

- Merge events across all domains
- Sort events chronologically
- Convert events into app API call logs via LLM
- Auto-save checkpoints for resumable execution

Typical outputs:

- `app_logs_final.json`
- `golden_evidence_index.json`
- `checkpoints/checkpoint_*.json`
- `stage3_checkpoint.json` (legacy compatibility)
- `app_logs_intermediate.json` (intermediate state)

## Recommended Run

Assuming your repo root is `mem_bench/behavior_and_conversation`, run:

```bash
cd data_construction
python batch_generation_runner.py --debug
```

Common arguments (`batch_generation_runner.py`):

- `--provider {google,openai,aimlapi,custom}`
- `--model <model_name>`
- `--api-key <key>` / `--base-url <url>`
- `--user-count <N>`
- `--user-index <1-based>`
- `--skip-stage1` / `--skip-stage2` / `--skip-stage3`
- `--cutoff-date <YYYY-MM-DD or MM.DD>`
- `--dry-run-world-bg`
- `--output-dir <path>`

## LLM Configuration

If `--api-key` is not provided, environment variables are used (see `../llm_client.py`):

- `google`: `GOOGLE_API_KEY`
- `openai`: `OPENAI_API_KEY` (optional `OPENAI_BASE_URL`)
- `aimlapi`: `AIMLAPI_API_KEY` (optional `AIMLAPI_BASE_URL`)
- `custom`: `CUSTOM_API_KEY` + `CUSTOM_BASE_URL`

## Output Structure (Example)

```text
<output_root>/
  stage0_user_descriptions.json
  001_user_001/
    input_user_description.txt
    user_basic_profile.json
    dynamic_profiles_final.json
    world_backgrounds.json
    all_events_chains.json
    app_logs_final.json
    golden_evidence_index.json
    pipeline_summary.json
    stage1_summary.json
    stage2_summary.json
    stage3_summary.json
    checkpoints/
      checkpoint_000123.json
    debug/
      stage1_dynamic_profile/
      stage2_events_chain/
      stage3_app_logs/
```

## Resume and Re-run Tips

- By default, Stage 3 runs with `resume_from_existing=True` in the runner and prefers resuming from checkpoints.
- If you only want to regenerate later stages:
  1. Keep Stage 1 results: `--skip-stage1`
  2. Regenerate only Stage 3: `--skip-stage1 --skip-stage2`
- To force Stage 3 to restart from scratch, clean the target user output directory:
  - `checkpoints/`
  - `stage3_checkpoint.json`
  - `app_logs_intermediate.json`
  - `app_logs_final.json` (optional, to avoid mixing old/new outputs)
