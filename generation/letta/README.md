# Letta Baseline

This directory contains Letta-based baselines for:

- QA generation: `generation.letta.letta`
- Dynamic state prediction: `generation.letta.dynamic_state_prediction`

Current implementation uses **agent-loop ingestion**:
- each app log is sent to Letta agent as one message in chronological order.
- ingestion-turn outputs are ignored.
- prediction-turn outputs are parsed as JSON for evaluation.

## Output Layout

- QA:
  - `generation/letta/results/<user_id>/prediction/letta_results.json`
- Dynamic state prediction:
  - `generation/letta/results/<user_id>/prediction/dynamic_state_prediction_results.json`

## Run

- QA:
  - `bash generation/letta/run_letta.sh`
- Dynamic state prediction:
  - `bash generation/letta/run_letta_dynamic_state_prediction.sh`
- Dynamic state prediction eval:
  - `bash generation/letta/run_eval_letta_dynamic_state_prediction.sh`

## Letta Modes

- `--letta-mode sdk`:
  - tries Letta Python SDK agent loop first.
- `--letta-mode local`:
  - local fallback simulation (non-SDK).

By default `sdk` mode allows fallback (`--no-local-fallback` disables fallback and fails hard).

Optional core-memory context:
- `--persona "<text>"`
- `--human "<text>"`
