# DynamicMem

**DynamicMem** is a benchmark for **long-horizon memory systems**. It evaluates
how well a memory system tracks a single user's evolving state across a long,
realistic activity stream — using **Temporal Checkpoint Evaluation (TCE)**:
at many checkpoints along each user's timeline, the system must reconstruct the
user's current state, track what changed, and act on it.

If you build a memory system (RAG-over-memory, agentic memory, memory OS, …),
DynamicMem lets you **benchmark it in a few commands** against a shared dataset
and a set of reference baselines.

---

## What the benchmark measures

Each user has a synthesized multi-month trajectory (profile → life events → app
logs). TCE cuts that trajectory at ordered **checkpoints**; at each checkpoint a
memory system sees everything up to that point and is scored on three tasks:

| Task | Name | The system must… |
|------|------|-------------------|
| **A** | State Reconstruction | reconstruct the user's current state (per-slot semantic match) |
| **B** | Change Tracking | identify what changed since the previous checkpoint |
| **C** | Personalization Utility | use the remembered state to complete a proactive, personalized service |

Scoring is per-checkpoint and per-slot, with semantic (LLM-judge) metrics. See
[`docs/protocols/temporal_checkpoint_evaluation_developer_manual.md`](docs/protocols/temporal_checkpoint_evaluation_developer_manual.md)
for the full protocol.

---

## Quick Start

### 1. Install

```bash
conda env create -f environment/default.yml   # creates the `mem0311` env
conda activate mem0311
pip install -e .                               # editable install of the core packages
```

Two baselines need their own environments (conflicting deps); skip unless you run them:

```bash
conda env create -f environment/memoryos.yml    # MemoryOS baseline
conda env create -f environment/hipporag2.yml   # HippoRAG2 baseline
```

### 2. Set your LLM credentials

Predictions and the LLM judge call an OpenAI-compatible endpoint. Set the key for
your provider (provider/model are configurable in each config's `llm:` block):

```bash
export OPENAI_API_KEY=sk-...
# Azure users: also set AZURE_OPENAI_API_KEY and the endpoint/base-url your config expects
```

### 3. Download the benchmark data

The benchmark task packs and app logs are published on the Hugging Face Hub.
Download them into the repo-root `outputs/` directory (the layout the configs expect):

```bash
huggingface-cli download xiewenya/dynamicmem-tce \
  --repo-type dataset --local-dir outputs/
```

This populates `outputs/<model>/<user_id>/` with each user's task packs
(`*_task_packs.json`) and `app_log_large.json`.

### 4. Run a reference baseline

```bash
# Example: A-Mem (runs in the default env)
python -m baseline_prediction.run_tce \
  --config configs/experiments/tce/amem_predict_user001.yaml
```

Predictions are written to
`baseline_prediction/<baseline>/results/<user_id>/prediction/<run_name>/tce_results.json`.

> Configs under [`configs/experiments/tce/`](configs/experiments/tce/) are
> templates — set `runtime.user_id` / `data` to match the users you downloaded.
> Use `--dry-run` first to print the resolved config without running.

### 5. Evaluate the predictions

```bash
python -m evaluation.eval_tce \
  --config configs/experiments/tce/amem_eval_predict.yaml
```

Eval payloads land next to the predictions under `.../evaluation/<run_name>/tce_eval.json`.
See [`evaluation/README.md`](evaluation/README.md) for the metric definitions and
the direct (`--benchmark/--prediction/--output`) CLI form.

---

## Benchmark your own memory system

This is the main use case. A memory system plugs in as a **TCE adapter**: it
receives the app-log stream up to each checkpoint and returns predictions in the
TCE format, which `evaluation.eval_tce` then scores.

- Adapter contract & prediction format:
  [`docs/protocols/tce_generation_and_adapter_contract.md`](docs/protocols/tce_generation_and_adapter_contract.md)
- Register your adapter: [`baseline_prediction/adapters/registry.py`](baseline_prediction/adapters/registry.py)
- Use any reference baseline in [`baseline_prediction/`](baseline_prediction/) as a worked example.

Once your adapter produces `tce_results.json`, evaluate it exactly like a baseline (step 5).

---

## Reference baselines included

| Baseline | Env | Notes |
|----------|-----|-------|
| A-Mem | `mem0311` (default) | agentic memory |
| RAG | `mem0311` (default) | retrieval-over-memory |
| SimpleMem | `mem0311` (default) | lightweight memory |
| Oracle | `mem0311` (default) | ground-truth-state ceiling |
| MemoryOS | `memoryos` | needs its own env |
| HippoRAG2 | `hipporag2` | needs its own env |

---

## Repository structure (for contributors)

DynamicMem is organized as four parts; consumers only need Parts 3–4 above, but
the full data-generation pipeline is included and reproducible.

```
dynamicmem/
├── trajectory_synthesis/      Part 1: synthesize user trajectories (profile → events → app logs)
├── benchmark_construction/    Part 2: build TCE benchmark task packs (Tasks A / B / C)
├── baseline_prediction/       Part 3: run memory-system baselines  ← you are here for "run baselines"
├── evaluation/                Part 4: score predictions            ← you are here for "evaluate"
├── tce_core/                  shared TCE protocol / data contracts
├── bench_core/                shared evaluator contracts
├── configs/                   YAML experiment configs
├── environment/               conda env specs (default / memoryos / hipporag2)
├── docs/                      protocol manuals + execution runbooks
├── tce_contracts.py           canonical task-contract constants
├── pyproject.toml             package metadata
└── LICENSE                    MIT
```

To regenerate the dataset from scratch (needs LLM API budget), follow Part 1 then
Part 2: [`trajectory_synthesis/README.md`](trajectory_synthesis/README.md) →
[`benchmark_construction/README.md`](benchmark_construction/README.md).
Generated artifacts (gitignored) land under repo-root `outputs/<model>/<user_id>/`.

### Protocol & runbooks

READMEs explain usage; protocol docs define the canonical contracts; runbooks
define executable workflows. When they conflict, follow the protocol or runbook.

- [`docs/protocols/temporal_checkpoint_evaluation_developer_manual.md`](docs/protocols/temporal_checkpoint_evaluation_developer_manual.md) — canonical TCE protocol
- [`docs/protocols/tce_generation_and_adapter_contract.md`](docs/protocols/tce_generation_and_adapter_contract.md) — generation / adapter contract
- [`docs/runbooks/tce_execution_runbook.md`](docs/runbooks/tce_execution_runbook.md) — executable TCE workflow

---

## Citation

A paper describing DynamicMem is under review. Citation information will be added
here upon publication.

## License

MIT — see [`LICENSE`](LICENSE). Vendored baselines retain their upstream licenses
(e.g. [`baseline_prediction/MemoryOS/LICENSE`](baseline_prediction/MemoryOS/LICENSE)).
