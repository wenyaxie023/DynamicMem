# HippoRAG v2 Indexing System (Optimized for Azure)

This directory contains the production-ready implementation for large-scale HippoRAG indexing using Azure OpenAI services. The system is optimized for high-concurrency stress testing across multiple users.

## 1. Core Configuration

- **LLM Engine**: `gpt-5-mini` (via Azure)
- **Embedding Engine**: `text-embedding-3-small` (via Azure)
- **Concurrency Strategy**: 
  - 10 Users running in parallel.
  - `batch_size=6` per user.
  - `OPENIE_MAX_WORKERS=6` for parallel triple extraction.
  - Total TPM target: ~180k-200k.

## 2. Key Files

| File | Description |
| :--- | :--- |
| `run_index_only.py` | The main entry point for indexing a single user. Supports checkpoints. |
| `launch_10_users.sh` | Orchestration script that launches 10 concurrent indexing sessions (User 001-010). |
| `check_progress.py` | Quick status script to see the processing progress of all active users. |
| `quick_check.py` | Deep verification script to inspect graph quality and edge semantics. |
| `analyze_tokens.py` | Analyzes token usage from logs. |
| `estimate_tokens.py` | Estimates token usage. |
| `inspect_graph.py` | Inspects graph structure. |

## 3. Operation Guide

### How to Launch Indexing (Multi-User)
To start indexing for all 10 users concurrently in background `tmux` sessions:
```bash
bash launch_10_users.sh
```

### How to Monitor Progress
To see the current completion percentage for all users:
```bash
python3 check_progress.py
```

### How to Verify Graph Quality
To check if the extracted triples have meaningful semantic relationships:
```bash
python3 quick_check.py ../outputs/003_user_003_large/gpt-5-mini_text-embedding-3-small/graph.pickle \
  --openie_json ../outputs/003_user_003_large/openie_results_ner_gpt-5-mini.json \
  --recent --limit 5
```

### How to Manage Processes
Each user runs in a dedicated tmux session named `hippo_001`, `hippo_002`, etc.
- **View a specific session**: `tmux attach -t hippo_003`
- **Kill all tasks**: `pkill -f run_index_only.py` or `tmux kill-server`

## 4. Output Directories
- **Logs**: `../logs_stable_4x/` (Individual log per user).
- **Graph Data**: `../outputs/{user_id}_large/{llm}_{embed}/graph.pickle`
- **OpenIE Cache**: `../outputs/{user_id}_large/openie_results_ner_{llm}.json`
- **Checkpoints**: `../outputs/{user_id}_large/indexing_checkpoint.json`

## 5. Stability Notes
- **API Errors**: The current configuration is tuned to stay within Azure's 200k TPM limit. 
- **Resumption**: If a process stops, simply re-running the script will automatically resume from the last checkpoint.
