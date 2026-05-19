# RAG TCE

Primary runner:

```bash
bash generation/rag/run_rag_tce.sh
```

Primary module:
- `generation/rag/rag_tce.py`

Output:
- `generation/rag/results/<user_id>/prediction/<run_name>/tce_results_v14_taskabc.json`

Embedding cache:
- `generation/rag/results/<user_id>/memory/rag_tce_applog_embeddings_<fingerprint>.npz`
- the cache key is tied to the app-log file path/stat plus retriever provider/model

Build-only:
- set `baseline_params.build_only: true` in batch config to prebuild the app-log embedding cache without running checkpoint generation
