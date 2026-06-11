---
license: mit
task_categories:
  - other
tags:
  - memory
  - long-horizon
  - benchmark
  - agents
pretty_name: DynamicMem
---

# DynamicMem

Benchmark data for **DynamicMem**, a benchmark for long-horizon memory systems.
Each user has a synthesized multi-month trajectory; at ordered checkpoints a
memory system must reconstruct the user's current state (**State Completion**)
and act on it (**Personalized Service**).

- Code, baselines, and evaluation: https://github.com/wenyaxie023/DynamicMem
- Quick start, task definitions, and the adapter contract are in the repo README.

## Contents

```
<user_id>/
├── task_packs.json      # benchmark task packs (carry ground-truth state)
└── app_log_large.json   # the activity stream the memory system consumes
```

## Usage

```bash
hf download xiewenya/dynamicmem \
  --repo-type dataset --local-dir outputs/
```

This reproduces the `outputs/<user_id>/...` layout the repo's configs
expect. Then follow the repo Quick Start to run a baseline and evaluate.

## License

MIT. If you use DynamicMem, please cite the paper (see the repository for the
current citation).
