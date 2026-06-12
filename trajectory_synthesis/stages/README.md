# Trajectory Synthesis Stages

The Stage 1/2/3 implementation behind
`trajectory_synthesis/batch_generation_runner.py`. For the module-level overview
and entrypoint, see [../README.md](../README.md).

## Stage Directory Purpose

The stages directory exists to separate the three major transformations in the user
data pipeline:

1. resolve dynamic user state
2. convert state into observable event chains
3. convert events into structured app logs

Core files in this directory:

- `stage1_dynamic_profile.py`: Stage 1 orchestration for dynamic profile generation
- `stage1_utils.py`: Stage 1 conflict checks, alignment helpers, and fix application
- `stage2_events_chain.py`: Stage 2 event-chain generation across windows
- `stage3_app_logs.py`: Stage 3 app-log generation and checkpoint-aware resume logic
- `__init__.py`: exports `DynamicProfileStage`, `EventsChainStage`, and `AppLogsStage`

## Stage 1

`Stage 1` turns the input user description into resolved multi-domain dynamic state.

Main responsibilities:

- generate `user_basic_profile.json` from the raw description
- generate domain-level dynamic profiles
- apply in-domain rule-based fixes
- align keys across domains
- resolve cross-domain attribute and temporal conflicts

Typical inputs:

- raw user description
- domain definitions and prompting context

Typical outputs under each user directory:

- `user_basic_profile.json`
- `*_world_background.json`
- `world_backgrounds.json`
- `dynamic_profiles_raw.json`
- `dynamic_profiles_domain_level_fixes_applied.json`
- `dynamic_profiles_key_aligned.json`
- `dynamic_profiles_conflict_resolved.json`
- `dynamic_profiles_final.json`

Implementation note:

- by default, Stage 1 expects precomputed world backgrounds to exist and loads them
  from `world_backgrounds.json` and/or per-domain `*_world_background.json` files
- world-background generation also supports a dry-run path that saves prompts without
  generating the background content itself

## Stage 2

`Stage 2` converts the resolved dynamic profile into observable event chains.

Main responsibilities:

- read `dynamic_profiles_final.json` from Stage 1
- generate per-domain event chains
- preserve time-window ordering and continuity
- attach state-to-event conversion markers used downstream

Typical inputs:

- `dynamic_profiles_final.json`

Typical outputs:

- `*_events_chain.json`
- `all_events_chains.json`

## Stage 3

`Stage 3` converts aggregated event chains into chronological app logs.

Main responsibilities:

- merge events across domains
- sort events in temporal order
- generate app-style API interaction logs
- save intermediate progress for resumable execution

Typical inputs:

- `all_events_chains.json`

Typical outputs:

- `app_logs_final.json`
- `golden_evidence_index.json`
- `checkpoints/checkpoint_*.json`
- `stage3_checkpoint.json`
- `app_logs_intermediate.json`

Implementation note:

- `checkpoints/checkpoint_*.json` is the main resume substrate

## Output Layout

Example per-user output layout:

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

## Resume and Re-run Notes

- The main runner prefers resuming Stage 3 from existing checkpoint artifacts.
- If Stage 1 artifacts are still valid, rerun later stages with `--skip-stage1`.
- If only Stage 3 needs regeneration, use `--skip-stage1 --skip-stage2`.
- To force Stage 3 to restart cleanly for one user, remove:
  - `checkpoints/`
  - `stage3_checkpoint.json`
  - `app_logs_intermediate.json`
  - optionally `app_logs_final.json` to avoid mixing old and new outputs
