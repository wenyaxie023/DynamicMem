# Trajectory Synthesis (Part 1)

Turns persona-like descriptions into synthetic user trajectories — the artifacts
the benchmark is built from: multi-domain profiles → event chains over time →
chronological app logs. The external entrypoint is `batch_generation_runner.py`.

## Three-stage pipeline

1. **Stage 1** — generate dynamic profiles and resolve conflicts
2. **Stage 2** — convert resolved profiles into event chains
3. **Stage 3** — convert event chains into chronological app logs

Stage-level details: [stages/README.md](stages/README.md).

## Quick Start

```bash
# 1. download the world backgrounds for the sampled users
python -m trajectory_synthesis.download_world_backgrounds --model gemini_3_flash_preview --user-count 10
# 2. run the full Stage 1 -> 2 -> 3 pipeline
python -m trajectory_synthesis.batch_generation_runner --user-count 10
```

By default Stage 1 uses `--world-bg-mode require_existing`, so world backgrounds
must exist before generation. Precomputed backgrounds are hosted at
<https://huggingface.co/datasets/xiewenya/user-world-backgrounds>.

Run `download_world_backgrounds.py` with the same sampling inputs (sample file,
seed, user-count) you will pass to `batch_generation_runner.py`. For ad-hoc runs
driven by `--user-description`, use `--world-bg-mode generate_if_missing`.

## Outputs

Under `outputs/<user_id>/`:

- `user_basic_profile.json` — normalized user profile
- `dynamic_profiles_final.json` — resolved multi-domain dynamic state
- `all_events_chains.json` — aggregated event chains
- `app_logs_final.json` — final app-log sequence
- `app_log_large.json` — large app-log view consumed by the benchmark
- `golden_evidence_index.json` — evidence mapping

`all_events_chains.json`, `app_logs_final.json`, and `app_log_large.json` are the
handoff into benchmark construction (Part 2,
[benchmark_construction/README.md](../benchmark_construction/README.md)).
