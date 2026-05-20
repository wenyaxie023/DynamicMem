# DynamicMem

DynamicMem is a benchmark for long-horizon memory systems. It synthesizes
realistic user trajectories, builds **Temporal Checkpoint Evaluation (TCE)**
task packs over those trajectories, runs memory-system baselines against them,
and evaluates predictions with per-checkpoint and per-slot semantic metrics.

This repository is organized as four parts, each of which can be read or run
independently:

| Part | Directory | What it does |
|------|-----------|--------------|
| 1 | [`trajectory_synthesis/`](trajectory_synthesis/) | Generates synthetic user trajectories (profile → events → app logs) |
| 2 | [`benchmark_construction/`](benchmark_construction/) | Turns trajectories into TCE benchmark task packs (Tasks A / B / C) |
| 3 | [`baseline_prediction/`](baseline_prediction/) | Runs memory-system baselines against the benchmark |
| 4 | [`evaluation/`](evaluation/) | Scores predictions against benchmark ground truth |

Shared infrastructure lives in [`tce_core/`](tce_core/) (TCE data contracts /
pipeline) and [`bench_core/`](bench_core/) (evaluator contracts). The root-level
`tce_contracts.py` carries the canonical task-contract metadata.

## Repository Map

```
dynamicmem/
├── trajectory_synthesis/      Part 1: user trajectory synthesis
├── benchmark_construction/    Part 2: TCE benchmark task packs
├── baseline_prediction/       Part 3: memory-system baselines
├── evaluation/                Part 4: scoring
├── tce_core/                  shared TCE protocol layer
├── bench_core/                shared evaluator contracts
├── configs/                   YAML configs (TCE experiments)
├── environment/               conda env specs (default / memoryos / hipporag2)
├── docs/                      protocol manuals + execution runbooks
├── tce_contracts.py           canonical task-contract constants
├── pyproject.toml             package metadata
└── LICENSE                    MIT
```

Generated artifacts (gitignored) land under repo-root `outputs/<model>/<user_id>/`.

## Quick Start

### 1. Set up a conda environment

The default environment supports Parts 1 / 2 / 4 and most of Part 3:

```bash
conda env create -f environment/default.yml   # creates `mem0311`
conda activate mem0311
pip install -e .                              # editable install of the core packages
```

Two baselines need their own environments (conflicting deps):

```bash
conda env create -f environment/memoryos.yml   # MemoryOS baseline
conda env create -f environment/hipporag2.yml  # HippoRAG2 baseline
```

### 2. Inspect a TCE config without running

```bash
python -m baseline_prediction.run_tce \
  --config configs/experiments/tce/hipporag2_predict_user001.yaml \
  --dry-run
```

### 3. Read the per-part guides

- [`trajectory_synthesis/README.md`](trajectory_synthesis/README.md) — Part 1
- [`benchmark_construction/README.md`](benchmark_construction/README.md) — Part 2
- [`baseline_prediction/README.md`](baseline_prediction/README.md) — Part 3
- [`evaluation/README.md`](evaluation/README.md) — Part 4

## Protocol & Runbooks

Onboarding READMEs explain usage and navigation; protocol docs define the
canonical contracts; runbooks define executable workflows. When they conflict,
follow the protocol or runbook.

- [`docs/protocols/temporal_checkpoint_evaluation_developer_manual.md`](docs/protocols/temporal_checkpoint_evaluation_developer_manual.md) — canonical TCE protocol
- [`docs/protocols/tce_generation_and_adapter_contract.md`](docs/protocols/tce_generation_and_adapter_contract.md) — TCE generation/adapter contract
- [`docs/runbooks/tce_execution_runbook.md`](docs/runbooks/tce_execution_runbook.md) — executable TCE workflow

## License

MIT — see [`LICENSE`](LICENSE).
