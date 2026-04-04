# Letta Baseline

This directory contains Letta-based baselines for:

- Unified Python pipeline: `generation.letta.pipeline`
- QA generation: `generation.letta.letta`
- Dynamic state prediction: `generation.letta.dynamic_state_prediction`
- Checkpoint agent snapshot builder: `generation.letta.checkpoint_agent_builder`
- TCE: `generation.letta.tce`

Current implementation uses **agent-loop ingestion**:
- each app log is sent to Letta agent as one message in chronological order.
- ingestion-turn outputs are ignored.
- prediction-turn outputs are parsed as JSON for evaluation.

## Output Layout

- QA:
  - `generation/letta/results/<user_id>/prediction/letta_results.json`
- Dynamic state prediction:
  - `generation/letta/results/<user_id>/prediction/dynamic_state_prediction_results.json`
- TCE:
  - `generation/letta/results/<user_id>/prediction/tce_results.json`
  - v14 minimal Task A smoke:
    - `generation/letta/results/<user_id>/prediction/tce_results_v14_taska.json`
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
- QA:
  - `bash generation/letta/run_letta.sh`
- TCE:
  - `bash generation/letta/run_letta_tce.sh`
  - current default entrypoint uses:
    - `configs/experiments/tce/letta_v14_user1.yaml`
  - current default experiment support is:
    - Task A / Task B / Task C enabled on `001_user_001` v14 benchmark
    - optional final QA enabled after the last executed checkpoint
    - effective TCE workers forced to 1 / 1
    - formal path uses checkpoint snapshot isolation:
      - logs are ingested once into a builder agent
      - each key is answered by importing the same checkpoint `.af` into a temp agent
      - temp agent is deleted after each key
      - builder advancement is separate from query-time answering; formal TCE no longer uses a mutating `retrieve_context` callback to drive ingest
    - checkpoint snapshots and resume state are stored under:
      - `generation/letta/agents/tce/<user_id>/`
    - builder ingestion progress is also stored under:
      - `generation/letta/agents/tce/<user_id>/builder_progress.json`
      - confirmed ingest progress is monotonic
      - ambiguous ingest failures are recorded as `ingest_uncertain` instead of rewinding confirmed progress
    - a builder lock file is created under the same directory; concurrent builder reuse is rejected
    - formal resume priority is:
      - local `builder_progress.json` + recorded `builder_agent_id`
      - latest checkpoint `.af` only if local builder progress is unavailable
      - fresh builder agent from scratch
    - formal config is SDK-only by default:
      - `allow_local_fallback: false`
      - use `shared_agent` + fallback only for debugging
- TCE eval:
  - `bash generation/letta/run_eval_letta_tce.sh`

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

## Apptainer Server (MSI / no Docker)

One-click script:

```bash
export AZURE_API_KEY="..."
export AZURE_BASE_URL="https://...openai.azure.com"
export AZURE_API_VERSION="2024-10-21"
bash generation/letta/run_letta_apptainer_server.sh
```

Notes:
- This path is intended for MSI-style environments where `apptainer` is available but Docker is not.
- The script pulls `docker://letta/letta:latest` into a local SIF on first run.
- The Letta server is expected to be reachable from the same shell/session at `http://127.0.0.1:8283`.
- After the server is up, point the SDK client at it with:
  - `export LETTA_BASE_URL="http://127.0.0.1:8283"`
  - `unset LETTA_API_KEY` if the local server has no auth
  - or `export LETTA_API_KEY="$LETTA_SERVER_PASSWORD"` if local auth is enabled

Recommended MSI local self-hosted run:

```bash
source /users/4/xie00470/miniconda3/etc/profile.d/conda.sh
conda activate mem0311

export AZURE_API_KEY="..."
export AZURE_BASE_URL="https://...openai.azure.com"
export AZURE_API_VERSION="2024-10-21"

export LETTA_STORAGE_ROOT="/projects/standard/zrliu/shared/wenya/letta"
export LETTA_PORT=8384
export LETTA_BACKGROUND=true

bash generation/letta/run_letta_apptainer_server.sh

export LETTA_BASE_URL="http://127.0.0.1:8384"
unset LETTA_API_KEY
```

Optional quick connectivity check:

```bash
curl http://127.0.0.1:8384/v1/models/
```

## Letta SDK

- The implementation now requires `letta_client` SDK.
- TCE minimal runner supports `SDK` first and may fall back to local compatibility mode when `allow_local_fallback=true`.
- Hosted Letta:
  - configure `LETTA_API_KEY`
- Local/self-hosted Letta:
  - configure `LETTA_BASE_URL`
  - `LETTA_API_KEY` is only needed if the local server has auth enabled

Checkpoint-based QA/DSP:
- QA supports `--checkpoint-state-path` or `--agentfile-path`.
- DSP supports `--checkpoint-state-path` to load per-checkpoint `.af` snapshots.
- Temporary imported agent ids can be tracked via `--lease-registry-path`.

Runtime behavior:
- QA/DSP snapshot mode uses temporary imported agents and deletes them immediately after each query (`import -> ask -> delete`).
- `--gc-leased-agents` only controls startup cleanup of leaked temporary agents from lease registry.
- Checkpoint `.af` files are saved when each checkpoint is reached.
- `final_<total_logs>.af` is saved once at the end of a successful builder run (not continuously updated).
- TCE builder progress is saved at log granularity in `builder_progress.json`; interrupted ingest resumes from the locally recorded confirmed prefix, even before the first checkpoint `.af` is written.

QA now uses explicit filenames (no fallback probing):
- `--app-logs-filename` (default: `app_log_large.json`)
- `--qa-filename` (default: `qa.json`)
