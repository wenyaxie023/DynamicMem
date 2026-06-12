# Baseline Prediction (Part 3)

Run memory-system baselines on the benchmark. For the full adapter and onboarding
contract, see the
[adapter contract](../docs/protocols/tce_generation_and_adapter_contract.md).

## Quick Start

```bash
# single user
python -m baseline_prediction.run_tce       --config configs/experiments/tce/<baseline>_predict.yaml
# multi-user
python -m baseline_prediction.run_tce_batch --config configs/experiments/tce/<baseline>_predict_users.yaml
# inspect the resolved config without running
python -m baseline_prediction.run_tce       --config configs/experiments/tce/<baseline>_predict.yaml --dry-run
```

Each run also writes `*_run_settings.yaml` next to its prediction output.

## Output layout

```text
baseline_prediction/<baseline>/results/<user_id>/
  prediction/*.json   # evaluator inputs
  memory/             # optional, baseline-built memory
  evaluation/         # written by the evaluator
```

`<baseline>` matches `runtime.baseline` and the adapter registry key.

## Structure

- `run_tce.py` / `run_tce_batch.py` — single / batch runner
- `tce_config.py`, `configs/tce.default.yaml`, `configs/experiments/tce/<baseline>_predict.yaml` — config layer
- `adapters/{base,registry,<baseline>}.py` — adapter layer
- `<BaselineName>/` — per-baseline implementation

## Add a baseline

Add `adapters/<baseline>.py` implementing `run(args: TceAdapterArgs) -> dict`,
register it in `adapters/registry.py`, and add
`configs/experiments/tce/<baseline>_predict.yaml`. The interface, required
behavior, and validation steps are specified in the
[adapter contract](../docs/protocols/tce_generation_and_adapter_contract.md)
(§8 and §12).
