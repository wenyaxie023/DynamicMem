# Letta Baseline

This directory contains Letta-based baselines for:

- Unified Python pipeline: `generation.letta.pipeline`
- QA generation: `generation.letta.letta`
- Dynamic state prediction: `generation.letta.dynamic_state_prediction`
- Checkpoint agent snapshot builder: `generation.letta.checkpoint_agent_builder`

Current implementation uses **agent-loop ingestion**:
- each app log is sent to Letta agent as one message in chronological order.
- ingestion-turn outputs are ignored.
- prediction-turn outputs are parsed as JSON for evaluation.

## Output Layout

- QA:
  - `generation/letta/results/<user_id>/prediction/letta_results.json`
- Dynamic state prediction:
  - `generation/letta/results/<user_id>/prediction/dynamic_state_prediction_results.json`
- Agent artifacts:
  - checkpoint state: `generation/letta/agents/<user_id>_checkpoint_state.json`
  - lease registry: `generation/letta/agents/<user_id>_leased_agent_ids.json`
  - checkpoint snapshots: `generation/letta/agents/<checkpoint_id>.af`
  - final snapshot: `generation/letta/agents/final_<total_logs>.af`

## Run

- Unified pipeline (recommended):
  - `python3 -m generation.letta.pipeline --action all --user user1 --resume`
  - optional pre-run strong cleanup: add `--gc-leased-agents`
  - optional hot resume from latest on-disk snapshot: add `--hot-resume-latest`
  - checkpoint only: `python3 -m generation.letta.pipeline --action checkpoint --user user1 --resume`
  - QA only: `python3 -m generation.letta.pipeline --action qa --user user1 --resume`
  - DSP only: `python3 -m generation.letta.pipeline --action dsp --user user1 --resume`
- Smoke test script (kept):
  - `bash generation/letta/run_letta_test.sh`
- End-to-end test pipeline script:
  - `bash generation/letta/run_letta_pipeline_test.sh`
- Build checkpoint snapshots directly:
  - `python3 -m generation.letta.checkpoint_agent_builder --logs-path <app_log_large.json> --benchmark-path <dynamic_state_prediction_benchmark.json> --resume`

## Docker Server (OpenAI only)

One-click script:

```bash
export OPENAI_API_KEY="your_openai_api_key"
bash generation/letta/run_letta_docker_server.sh
```

Default server URL:
- `http://localhost:8283`

Optional env vars:
- `LETTA_PORT` (default `8283`)
- `LETTA_PERSIST_DIR` (default `~/.letta/.persist/pgdata`)
- `LETTA_CONTAINER_NAME` (default `letta-server`)
- `LETTA_SECURE=true` and `LETTA_SERVER_PASSWORD=...` to enable password auth

## Letta SDK

- The implementation now requires `letta_client` SDK.
- Local fallback mode was removed.
- Make sure `LETTA_BASE_URL` (and `LETTA_API_KEY` if secure mode) is configured.

Checkpoint-based QA/DSP:
- QA supports `--checkpoint-state-path` or `--agentfile-path`.
- DSP supports `--checkpoint-state-path` to load per-checkpoint `.af` snapshots.
- Temporary imported agent ids can be tracked via `--lease-registry-path`.

Runtime behavior:
- QA/DSP snapshot mode uses temporary imported agents and deletes them immediately after each query (`import -> ask -> delete`).
- `--gc-leased-agents` only controls startup cleanup of leaked temporary agents from lease registry.
- Checkpoint `.af` files are saved when each checkpoint is reached.
- `final_<total_logs>.af` is saved once at the end of a successful builder run (not continuously updated).

QA now uses explicit filenames (no fallback probing):
- `--app-logs-filename` (default: `app_log_large.json`)
- `--qa-filename` (default: `qa.json`)
