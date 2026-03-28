# SimpleMem TCE Baseline Checklist

## 1. Configuration Verification

### 1.1 Model Configuration
Before running, verify the following configuration values are correctly set in your YAML config:

| Parameter | Config Path | Example Value | Verification Command |
|-----------|-------------|---------------|---------------------|
| LLM Provider | `llm.provider` | `openai` | Check YAML file |
| LLM Model | `llm.model` | `gpt-5-mini` | Check YAML file |
| Embedding Model | `baseline_params.embedding_model` | `text-embedding-3-large` | Check YAML file |
| Retrieval Top-K | `baseline_params.retrieval_top_k` | `10` | Check YAML file |

### 1.2 Startup Log Verification
After starting, verify the configuration is printed correctly:
```
[SimpleMem] === Configuration ===
[SimpleMem] LLM Provider: openai
[SimpleMem] LLM Model: gpt-5-mini
[SimpleMem] Embedding Model: text-embedding-3-large
[SimpleMem] Retrieval Top-K: 10
[SimpleMem] Resume: false
[SimpleMem] =======================
```

**If this log is missing or shows wrong values:**
- Check YAML config file path is correct
- Check `baseline_params` section exists
- Verify no hardcoded defaults are overriding config

### 1.3 Environment Variables
Verify environment variables are loaded:
```bash
# Check .env file exists
ls -la generation/simplemem/.env

# Verify API key is set
grep "AZURE_OPENAI_API_KEY" generation/simplemem/.env
```

## 2. Implementation Architecture

### 2.1 Core Components
| Component | File | Description |
|-----------|------|-------------|
| Index | `generation_tce/tce.py::SimpleMemIndex` | LanceDB + BM25 hybrid index |
| Retrieval | `generation_tce/tce.py::_retrieve_context` | RRF fusion retrieval |
| Adapter | `adapters/simplemem.py` | Config to implementation bridge |
| Config | `configs/experiments/tce/simplemem.yaml` | Default configuration |

### 2.2 Key Implementation Details

**Indexing Strategy:**
- Incremental indexing: each log indexed one at a time
- LanceDB for vector storage (3072 dim for text-embedding-3-large)
- BM25 for lexical search
- Reciprocal Rank Fusion (RRF) for hybrid scoring

**Concurrency Policy:**
- `checkpoint_parallelism: forbidden` - Must process checkpoints sequentially
- `within_checkpoint_parallelism: forbidden` - Must process keys sequentially
- Reason: Incremental indexing changes state between operations

**retrieve_context Interface:**
```python
def retrieve_context(
    cp: Dict[str, Any],
    memory_pool: List[Dict[str, Any]],
    target_keys: List[str],
    task_text: Optional[str] = None,           # Required for Task C
    retrieval_top_k_override: Optional[int] = None,
) -> Dict[str, Any]:
```
- `task_text` is REQUIRED for proper Task C retrieval
- Without `task_text`, fallback uses Task A style query which may not find Task C evidence

## 3. Health Checks

### 3.1 Database Health
```bash
# Check LanceDB exists and has correct structure
python -c "
import lancedb
db = lancedb.connect('generation/simplemem/results/{user_id}/prediction/simplemem_db')
table = db.open_table('logs')
print(f'Rows: {len(table)}')
print(f'Schema: {table.schema}')
"
```

Expected output:
- Rows: Number of indexed logs
- Schema: id (string), log_id (string), text (string), vector (fixed-size list), log_data (string)

### 3.2 Indexing Progress
Monitor indexing progress in logs:
```
[SimpleMem] Indexing log log_00123...
[SimpleMem] CP cal_quarterly_001: indexed 5 new logs, total: 185
```

### 3.3 Retrieval Verification
Check retrieval is working:
```bash
python -c "
from generation.simplemem.generation_tce.tce import SimpleMemIndex, SimpleMemConfig
config = SimpleMemConfig(db_path='generation/simplemem/results/test_v4/prediction/simplemem_db')
index = SimpleMemIndex(config)
index.load_from_existing()
results = index.hybrid_search('test query', top_k=5)
print(f'Retrieved {len(results)} results')
for r in results[:3]:
    print(f'  {r.get(\"app_log_id\")}: score={r.get(\"retrieval_score\", 0):.4f}')
"
```

## 4. Task Execution Verification

### 4.1 Task A (State Completion)
Expected log output:
```
[SimpleMem] CP cal_quarterly_001: indexed X new logs, total: Y
SIMPLEMEM-TCE checkpoints: 100%|████████| 1/1
```

Verify output:
```bash
python -c "
import json
data = json.loads(open('generation/simplemem/results/{user_id}/prediction/tce_results.json').read())
pred = data['predictions'][0]
print(f'Task A keys: {len(pred.get(\"snapshot_state\", {}))}')
print(f'Complete: {pred.get(\"metadata\", {}).get(\"_checkpoint_complete\")}')
"
```

### 4.2 Task C (Apply Service QA)
Prerequisite: `enable_rq3_apply_service_qa: True` in config or adapter

Expected behavior:
- `task_text` parameter is used for retrieval (not fallback)
- Retrieval query should match benchmark's `rq3_apply_service_qa.keys[].items[].retrieval_query`

Verify Task C execution:
```bash
python -c "
import json
data = json.loads(open('generation/simplemem/results/{user_id}/prediction/tce_results.json').read())
pred = data['predictions'][0]
ra = pred.get('rq3_apply_answers', {})
print(f'Task C keys: {len(ra)}')
# Check evidence exists
total_items = sum(len(v.get('items', [])) for v in ra.values())
with_evidence = sum(1 for k, v in ra.items() for i in v.get('items', []) if i.get('evidence'))
print(f'Total items: {total_items}, with evidence: {with_evidence}')
"
```

## 5. Troubleshooting

### 5.1 Common Issues

| Symptom | Possible Cause | Solution |
|---------|---------------|----------|
| `401 Unauthorized` | API key not loaded | Check `.env` file exists and has correct key |
| `dimension mismatch` | Wrong embedding model | Verify `embedding_model` matches actual model used |
| Empty evidence for Task C | Missing `task_text` parameter | Check `_retrieve_context` accepts `task_text` |
| Resume not working | DB path mismatch | Verify `simplemem_db` directory exists |
| Slow indexing | Network latency | Normal for embedding API calls |

### 5.2 Debug Commands

**Check if process is running:**
```bash
ps aux | grep simplemem
```

**Kill stuck process:**
```bash
pkill -f "simplemem_test.yaml"
```

**Check API connectivity:**
```bash
python -c "
from openai import OpenAI
client = OpenAI()
print(client.embeddings.create(model='text-embedding-3-large', input=['test']))
"
```

**Inspect LanceDB contents:**
```bash
python -c "
import lancedb
db = lancedb.connect('generation/simplemem/results/{user_id}/prediction/simplemem_db')
table = db.open_table('logs')
rows = table.search().limit(5).to_list()
for r in rows:
    print(f'{r[\"id\"]}: {r[\"text\"][:50]}...')
"
```

### 5.3 Resume Behavior

When `resume: true`:
1. LanceDB is loaded from existing path
2. `indexed_log_ids` is restored
3. Only new logs (not in `indexed_log_ids`) are indexed
4. Checkpoint predictions are skipped if `_checkpoint_complete: true`

To force fresh run:
```bash
rm -rf generation/simplemem/results/{user_id}/prediction/tce_results.json
rm -rf generation/simplemem/results/{user_id}/prediction/simplemem_db
```

## 6. Performance Expectations

### 6.1 Timing (per checkpoint, ~180 logs)
| Operation | Expected Time |
|-----------|---------------|
| Indexing (first run) | 5-10 minutes |
| Indexing (resume, 0 new logs) | < 1 second |
| Retrieval (per query) | 1-3 seconds |
| LLM generation (per key) | 3-10 seconds |

### 6.2 Resource Usage
- Memory: ~500MB - 2GB depending on log volume
- Disk: ~100MB per 1000 logs (LanceDB + BM25 index)
- API calls: 1 embedding per log, 1 LLM call per key

## 7. Output Files

| File | Description |
|------|-------------|
| `tce_results.json` | Main prediction output |
| `simplemem_db/` | LanceDB vector index |
| `debug_tce/*.json` | Debug artifacts (if enabled) |

### 7.1 Prediction Structure
```json
{
  "predictions": [{
    "checkpoint_id": "cal_quarterly_001",
    "snapshot_state": {...},      // Task A predictions
    "evidence": {...},             // Task A evidence
    "rq3_apply_answers": {...},    // Task C predictions
    "metadata": {
      "_checkpoint_complete": true,
      "simplemem_indexed_count": 180,
      ...
    }
  }]
}
```

## 8. Evaluation

Run evaluation after generation:
```bash
python -m eval.eval_tce \
  --benchmark path/to/tce_benchmark.json \
  --prediction generation/simplemem/results/{user_id}/prediction/tce_results.json \
  --output generation/simplemem/results/{user_id}/eval/eval_results.json
```

Key metrics:
- `snapshot_value_f1_mean_on_expected_mean` - Task A value accuracy
- `rq3_apply_answer_point_score_mean_mean` - Task C answer quality
- `evidence_recall` - Evidence retrieval quality