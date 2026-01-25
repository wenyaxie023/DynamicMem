# Kimi-Linear Optimization: Offline Inference

## Problem: The "Slow Tokenizer" Bottleneck
During the initial benchmark of `Kimi-Linear-48B`, we observed a significant performance bottleneck:
- **Small Context (250KB)**: ~23 seconds per query.
- **Medium Context (1.1MB)**: ~90 seconds per query.

Profiling revealed that **GPU Inference** only accounted for ~10 seconds of this time. The remaining ~80 seconds were spent on **CPU-bound Tokenization**.

### Root Cause
The model uses `tokenization_kimi.py`, which is a pure Python implementation. To workaround `tiktoken`'s 400k character limit, it implements a manual Python loop (`_split_whitespaces_or_nonwhitespaces`) to split large texts. This Python loop is extremely slow for inputs > 1 million characters, causing the CPU to be the bottleneck.

## Solution: Offline Inference with Pre-Tokenization
To solve this, we switched from `vLLM Client-Server` mode to **Offline Inference** (`vllm.LLM` class).

### Strategy
1.  **Tokenize Once**: We manually call the tokenizer *once* on the large Context Log to generate a list of `token_ids`.
2.  **Reusable Cache**: These `context_token_ids` are kept in memory.
3.  **Fast Inference**: For each question, we construct the prompt as `[context_token_ids] + [question_token_ids]` and pass this directly to `llm.generate()`.
4.  **Bypass Overhead**: This completely bypasses the Python tokenizer loop for the context part during the QA phase.

### Expected Performance
- **Tokenization Overhead**: ~0 seconds (amortized).
- **Inference Time**: Purely GPU-bound (estimated 10-15s per query).
