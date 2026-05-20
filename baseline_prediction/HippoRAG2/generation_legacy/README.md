# MemBench Legacy Generation Scripts

These are the original generation scripts moved from the `baseline_prediction/` root. They use the legacy logic (HippoRAG directly + `generation_prompt.py`).

## Key Files

| File | Description |
| :--- | :--- |
| `run_hipporag_generation_legacy.py` | Single-user generation script (formerly `run_hipporag_generation.py`). |
| `run_batch_generation_legacy.py` | Batch generation script for multiple users (formerly `run_batch_generation.py`). |
| `generation_prompt_legacy.py` | The prompt file used by the above scripts. |

## Usage
These scripts are maintained for backward compatibility and baseline comparisons. They generally expect the index to exist in the standard output structure.
