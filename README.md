# DynamicMem

**DynamicMem** is a benchmark for **long-horizon memory in personal-assistant
agents**. Such an agent must infer and maintain a model of the user's **profile** —
who they are (*attributes*), what they routinely do (*habits*), and what they
prefer (*preferences*) — and keep it current as that profile evolves with life
events and shifting routines. This evidence is seldom stated explicitly; it is
scattered across many small actions in different applications and must be inferred
from those distributed behavioral traces.

DynamicMem evaluates this ability on synthesized, user-consistent trajectories
that span many months of multi-application activity. From each trajectory it
constructs an ordered sequence of **checkpoints**; at every checkpoint a memory
system observes the full history up to that point and is assessed under **Temporal
Checkpoint Evaluation (TCE)** on two tasks — reconstructing the user's current
profile, and acting on it — which also exposes how performance scales as the
history grows.

This repository provides the dataset, the evaluation protocol, and a suite of
reference baselines, so that an arbitrary memory system can be evaluated under an
identical protocol with a few commands.

## Tasks

At each checkpoint a system is evaluated on two task families, both derived from
the same validated ground-truth state so as to separate *recovering* state from
*using* it:

- **State Completion** — reconstruct the user's current state: the attributes,
  habits, and preferences that hold at the checkpoint.
- **Personalized Service** — use the remembered state to complete a proactive,
  personalized service request grounded in that state.

## Evaluation

Predictions are scored by a **field-level Core+Detail** LLM-as-judge. Each
reference unit is decomposed into fields; every field is scored on a binary
**Core** axis (does the prediction recover the field's central meaning?) and a
three-level **Detail** axis (how completely are the supporting specifics
preserved?), combined as `s = 0.8·Core + 0.2·(Detail/2)` and averaged over fields
and units. (In the code and configs this protocol is abbreviated **TCE**.)

## Installation

```bash
conda env create -f environment/default.yml   # creates the `dynamicmem` env
conda activate dynamicmem
pip install -e .                               # editable install of the core packages
```

The default environment is lightweight and runs the `rag`, `oracle`, and
`simplemem` baselines plus trajectory synthesis, benchmark construction, and
evaluation. The other baselines need extra setup, only if you run them:

```bash
pip install -r baseline_prediction/Amem/requirements.txt   # amem (adds torch / sentence-transformers / chromadb)
conda env create -f environment/memoryos.yml               # memoryos (isolated env)
conda env create -f environment/hipporag2.yml              # hipporag2 (isolated env)
```

Predictions and the LLM judge call the OpenAI API; the example configs use
`provider: openai`:

```bash
export OPENAI_API_KEY=sk-...
```

Provider and model are configurable per config (the `llm:` / `retriever:`
blocks); Azure is also supported (`provider: azure` + `AZURE_OPENAI_API_KEY`).

## Dataset

The benchmark task packs and app-log streams are released on the Hugging Face
Hub. Download them into the repo-root `outputs/` directory expected by the configs:

```bash
hf download xiewenya/dynamicmem --repo-type dataset --local-dir outputs/
```

This populates `outputs/<user_id>/` with each user's `task_packs.json`
(the per-checkpoint evaluation targets) and `app_log_large.json` (the activity
stream a memory system ingests).

## Visualization

An interactive [DynamicMem Visualizer](https://zirui-ray-liu.github.io/dynamicmem-viz/viz.html)
walks through one user's trajectory end to end — the evolving profile, the event
chain it induces, the resulting multi-application `app_log_large.json` stream, and
the TCE checkpoints built from it — so you can see what a memory system actually
ingests and is scored on before running anything.

## Running a baseline

A run produces predictions for one baseline on one user, then scores them:

```bash
# 1. prediction
python -m baseline_prediction.run_tce  --config configs/experiments/tce/rag_predict.yaml
# 2. evaluation
python -m evaluation.eval_tce          --config configs/experiments/tce/rag_eval.yaml
# (or both at once)
bash scripts/run_baseline.sh rag       # rag | oracle | simplemem (default env); amem | memoryos | hipporag2 (extra setup)
```

Predictions are written under
`baseline_prediction/<baseline>/results/<user_id>/prediction/`, and the evaluator
prints a summary and writes `tce_eval.json` next to them. The headline scores are:

- **State Completion** → `snapshot_point_score`
- **Personalized Service** → `rq3_apply_answer_point_score`

A baseline always produces State Completion predictions; Personalized Service is
enabled by `runtime.enable_rq3_apply_service_qa: true` (set in the example
configs). Configs under [`configs/experiments/tce/`](configs/experiments/tce/) are
templates — adjust `runtime.user_id` / `data` to the users you downloaded.

### Memory build modes

For the memory-building baselines (**A-Mem, SimpleMem, MemoryOS, HippoRAG2**),
`baseline_params.memory_action` controls how building the memory relates to
producing predictions:

- `build_then_predict` (the default in the example configs) — at each checkpoint,
  ingest history *then* predict, in a single run. Build and prediction are
  interleaved.
- `build_only` — only construct the memory snapshots; write no predictions.
- `predict_from_prebuilt` — skip building, load prebuilt snapshots and predict
  (requires a prior `build_only` run; errors if the snapshots are missing).

So you can either run everything in one pass (`build_then_predict`), or **build
the full memory first** (`build_only`) and **predict separately afterwards**
(`predict_from_prebuilt`) — useful for inspecting the constructed memory before
spending prediction tokens, or for reusing one build across several prediction
settings. `RAG` and `Oracle` have no build phase and ignore this setting.

## Evaluating your own memory system

A memory system integrates as a **DynamicMem adapter**: it receives the app-log
stream up to each checkpoint and returns predictions in the evaluator's format,
which `evaluation.eval_tce` then scores under the same protocol as the reference
baselines.

- Adapter contract and prediction format:
  [`docs/protocols/tce_generation_and_adapter_contract.md`](docs/protocols/tce_generation_and_adapter_contract.md)
- Registry: [`baseline_prediction/adapters/registry.py`](baseline_prediction/adapters/registry.py)
- Any baseline under [`baseline_prediction/`](baseline_prediction/) serves as a
  worked example.

## Reference baselines

DynamicMem ships six baselines that bracket the task with an oracle ceiling and a
retrieval-only floor, and fill the middle with four memory systems of increasing
structure. Oracle, RAG, and SimpleMem are original to this repository; A-Mem,
MemoryOS, and HippoRAG2 are vendored or installed from their upstream projects
(see [`NOTICE`](NOTICE) for sources and licenses). The four memory-building
baselines support the [memory build modes](#memory-build-modes) described above;
Oracle and RAG have no build phase.

| Baseline | Env | How it represents memory |
|----------|-----|--------------------------|
| **Oracle** | `dynamicmem` | *No memory.* Answers each checkpoint from the validated ground-truth state — an upper-bound ceiling that isolates prediction/judging headroom from memory quality. |
| **RAG** | `dynamicmem` | *No memory.* Embeds the raw app-log history and retrieves the top-_k_ most relevant entries per query — a retrieval-only floor over unprocessed history. |
| **SimpleMem** | `dynamicmem` | Lightweight structured memory: incrementally distills the history into a compact structured user profile that is updated and queried at each checkpoint. |
| **A-Mem** | `dynamicmem` | Agentic memory ([agiresearch/A-mem](https://github.com/agiresearch/A-mem)): turns observations into interlinked memory notes that evolve over time, retrieving linked neighbors at query time. |
| **MemoryOS** | `memoryos` | Memory operating system ([BAI-LAB/MemoryOS](https://github.com/BAI-LAB/MemoryOS)): tiered short-/mid-/long-term stores governed by OS-style update and eviction policies. |
| **HippoRAG2** | `hipporag2` | Graph-structured memory ([OSU-NLP-Group/HippoRAG](https://github.com/OSU-NLP-Group/HippoRAG)): builds a knowledge graph over the history and retrieves by personalized-PageRank traversal. |

## Repository structure

DynamicMem is organized as four parts. Using the published dataset requires only
Parts 3–4; Parts 1–2 are the full, reproducible data-generation pipeline.

```
dynamicmem/
├── trajectory_synthesis/      Part 1 — synthesize user trajectories (profile → events → app logs)
├── benchmark_construction/    Part 2 — build benchmark task packs from trajectories
├── baseline_prediction/       Part 3 — run memory-system baselines
├── evaluation/                Part 4 — score predictions (Core+Detail judge)
├── tce_core/                  shared protocol: checkpointing, pack build, scoring, runtime
├── configs/                   YAML experiment configs
├── environment/               conda environment specs
├── docs/                      adapter contract
├── tce_contracts.py           canonical task-contract constants
└── pyproject.toml             package metadata
```

To regenerate the dataset from scratch (requires LLM API budget), follow Part 1
then Part 2: [`trajectory_synthesis/README.md`](trajectory_synthesis/README.md) →
[`benchmark_construction/README.md`](benchmark_construction/README.md).

## Citation

A paper describing DynamicMem is under review; citation information will be added
upon publication.

## License

MIT — see [`LICENSE`](LICENSE). Vendored and depended-on third-party baselines
retain their own licenses; see [`NOTICE`](NOTICE) for attribution.
