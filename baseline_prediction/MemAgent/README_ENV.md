# MemAgent Environment & Project Guide

> **CRITICAL FOR AI ASSISTANTS**: Read this first! This project uses a highly specific **dual-environment** setup to resolve dependency conflicts. Do NOT try to merge them or upgrade packages blindly.

## 🏗️ Architecture Overview

This project is separated into a **Server Environment** (vLLM) and a **Client Environment** (MemAgent logic) to handle conflicting requirements for `numpy`, `torch`, and `transformers`.

### 1. The Dual Environments
*   **`venv-vllm`** (Python 3.10)
    *   **Role**: Dedicated HTTP API Server.
    *   **Key Deps**: `vllm==0.8.2` (Strict), `torch==2.6.0`, `numpy<2`.
    *   **Quirks**: Contains a manual patch for `pyairports` required by `outlines`. Do not wipe this venv without backing up that patch logic.
*   **`venv-memagent`** (Python 3.12)
    *   **Role**: Client execution, data generation, and benchmark logic.
    *   **Key Deps**: `transformers>=4.40`, `openai`, `datasets`.
    *   **Quirks**: Used to run all Python scripts that *talk* to the server.

### 2. Directory Structure Clarity
*   **Outer Directory** (`.../baseline_prediction/MemAgent/`) **[WE WORK HERE]**
    *   Contains our custom scripts: `run_memagent_benchmark.sh`, `benchmark_single.py`.
    *   Contains the environments: `venv-vllm/`, `venv-memagent/`.
    *   Contains the frozen requirements: `requirements-*-stable.txt`.
*   **Inner Directory** (`.../baseline_prediction/MemAgent/MemAgent/`) **[REFERENCE ONLY]**
    *   Contains the original author's source code (`recurrent/*`, `verl/*`).
    *   **Do not edit these files** unless strictly necessary. We implement our own logic in the outer directory using the inner code as a logic reference (not a dependency).

## ⚠️ Known Pitfalls (踩坑指南)

1.  **Dependency Hell**:
    *   **Never** install `vllm` in `venv-memagent`.
    *   **Never** install project-specific `transformers` logic in `venv-vllm`.
    *   The `vllm` server needs `trust-remote-code=True` enabled to load the custom `BytedTsinghua/RL-MemoryAgent-7B` model.

2.  **Pyairports Issue**:
    *   vLLM depends on `outlines`, which depends on a broken package `pyairports`.
    *   **Solution**: We have a mock/patch in `venv-vllm`. If you rebuild the environment, you must verify `import outlines` works. vLLM 0.8.2 upgrade did NOT remove this dependency.

3.  **Context Length Limits**:
    *   The original `quickstart.py` had a hardcoded `120,000` token limit.
    *   **Status**: We have REMOVED this limit in our custom `benchmark_single.py`. Do not re-introduce truncation logic; the RL-7B model is designed to handle M-level context via the recurrent mechanism.

4.  **Zombie Processes**:
    *   If the script crashes, vLLM or Python processes (monitor threads) might stay alive on the GPU.
    *   **Cleanup Command**: `pkill -9 -f vllm; pkill -9 -f benchmark_single.py`

## 🚀 Usage Instructions

### 1. Launching the Baseline Benchmark
Use the consolidated orchestrator script. It handles server startup/shutdown automatically.

```bash
# In .../baseline_prediction/MemAgent/
./run_memagent_benchmark.sh
```

### 2. Manual Development Workflow
If you want to run custom scripts (e.g., the upcoming multi-user test):

**Step A: Start Server**
```bash
# Use the vllm venv explicitly
venv-vllm/bin/vllm serve BytedTsinghua-SIA/RL-MemoryAgent-7B \
  --port 8003 \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.9 \
  --trust-remote-code &
# Wait for "Application startup complete" on port 8003
```

**Step B: Run Client**
```bash
# Use the memagent venv explicitly
venv-memagent/bin/python3 your_custom_script.py --port 8003
```
