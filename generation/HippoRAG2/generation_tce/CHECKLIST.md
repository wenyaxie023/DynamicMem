# HippoRAG TCE Implementation Checklist

> **最后更新**: 2026-03-28  
> **状态**: 完整实现验证 ✅

---

## 1. 整体实现架构

### 1.1 核心文件

| 文件 | 作用 | 关键职责 |
|------|------|----------|
| `configs/experiments/tce/hipporag2.yaml` | 配置文件 | 定义 LLM、Embedding、运行参数 |
| `generation/adapters/hipporag2.py` | 适配器 | 配置 → 实现的参数映射 |
| `generation/HippoRAG2/generation_tce/tce.py` | 主实现 | HippoRAG 初始化、TCE 流水线执行 |
| `generation/HippoRAG2/HippoRAG/src/hipporag/` | HippoRAG 库 | 图构建、检索、OpenIE |
| `tce_core/pipeline.py` | TCE 流水线 | Checkpoint 处理、Key 预测 |

### 1.2 数据流向

```
┌─────────────────────────────────────────────────────────────────────┐
│  YAML Config (hipporag2.yaml)                                       │
│  ├── llm.provider, llm.model, llm.max_workers                       │
│  ├── baseline_params.hipporag_dir                                   │
│  ├── baseline_params.online                                         │
│  ├── baseline_params.embedding_model (新增)                         │
│  └──────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│  Adapter (hipporag2.py)                                             │
│  └── 提取 extras 参数，调用 run_generation()                         │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│  TCE Implementation (tce.py)                                        │
│  ├── 初始化 LLMClient (生成用)                                       │
│  ├── 初始化 HippoRAG (检索用)                                        │
│  ├── 创建 BaseConfig (temperature=0)                                │
│  └──────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│  HippoRAG Library                                                   │
│  ├── HippoRAG 初始化 → llm_model_name, embedding_model_name         │
│  ├── OpenIE → 实体/三元组抽取                                        │
│  ├── 图构建 → igraph                                                │
│  ├── EmbeddingStore → LanceDB                                       │
│  └──────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│  TCE Pipeline (run_pipeline)                                        │
│  ├── Checkpoint 循环 (串行，online mode)                             │
│  ├── Key 预测 (Task A: State Completion)                            │
│  ├── Task C (Apply Service QA)                                      │
│  ├── 输出 → tce_results.json                                        │
│  └──────────────────────────────────────────────────────────────────┘
```

---

## 2. 配置参数检查清单

### 2.1 必需配置参数

| 参数 | 配置位置 | 默认值 | 说明 |
|------|----------|--------|------|
| `llm.provider` | YAML top-level | `openai` | LLM 提供商 |
| `llm.model` | YAML top-level | - | **必需**: LLM 模型名称 (如 `gpt-5-mini`) |
| `llm.max_workers` | YAML top-level | 1 | LLM 并发数 |
| `baseline_params.hipporag_dir` | YAML baseline_params | - | **必需**: HippoRAG 输出目录 |
| `baseline_params.online` | YAML baseline_params | `true` | Online 模式 (增量索引) |
| `baseline_params.embedding_model` | YAML baseline_params | - | **必需**: Embedding 模型名称 |

### 2.2 可选配置参数

| 参数 | 配置位置 | 默认值 | 说明 |
|------|----------|--------|------|
| `runtime.max_checkpoints` | YAML runtime | None | 最大 checkpoint 数 |
| `runtime.debug` | YAML runtime | false | 调试模式 |
| `baseline_params.openie_cache_path` | YAML baseline_params | 自动推断 | OpenIE 缓存路径 |
| `baseline_params.retrieval_top_k` | YAML baseline_params | 5 | 检索 top-k 数量 |
| `baseline_params.checkpoint_workers` | YAML baseline_params | 1 | Checkpoint 并发 (必须=1) |
| `baseline_params.within_checkpoint_workers` | YAML baseline_params | 1 | Key 并发 (必须=1) |
| `runtime.enable_rq3_apply_service_qa` | YAML runtime | true | 启用 Task C |
| `runtime.rq3_apply_items_per_key` | YAML runtime | 1 | Task C 每个问题数 |
| `runtime.enable_change_reasoning` | YAML runtime | false | Task B (当前无数据) |

### 2.3 环境变量

| 变量 | 来源文件 | 用途 |
|------|----------|------|
| `AZURE_OPENAI_API_KEY` | `.env` | API 密钥 |
| `AZURE_OPENAI_BASE_URL` | `.env` | API 端点 (OpenAI 兼容) |
| `OPENIE_MAX_WORKERS` | Shell export | OpenIE 并发数 |

---

## 3. 启动日志验证

### 3.1 应该看到的日志

启动时应显示以下关键信息:

```
[INFO] Initializing HippoRAG TCE Pipeline
[INFO] Configuration:
  - LLM Model: gpt-5-mini
  - Embedding Model: text-embedding-3-large
  - Provider: openai (Azure)
  - Base URL: https://xxx.openai.azure.com/openai/v1/
  - Temperature: 0.0 (TCE requirement)
  - Online Mode: true
  - Retrieval Top-K: 5

[INFO] Initializing HippoRAG from <hipporag_dir>...
[INFO] Found OpenIE cache: <cache_path>
[INFO] OpenIE cache contains X pre-processed logs
[ONLINE] Starting with EMPTY graph (no data leakage)
[ONLINE] Using fresh embedding stores at <online_working_dir>
```

### 3.2 日志检查命令

```bash
# 检查启动日志是否包含模型名称
head -50 <run_log_path> | grep -E "LLM Model|Embedding Model|Model:"

# 检查是否使用正确的 embedding
grep "Using embedding model" <run_log_path>

# 验证 online mode
grep "ONLINE" <run_log_path>
```

---

## 4. Online Mode 验证

### 4.1 关键特性

| 特性 | 验证方法 | 正确表现 |
|------|----------|----------|
| 空 graph 启动 | 查看日志 `[ONLINE] Starting with EMPTY graph` | 无数据泄露 |
| 增量索引 | 查看日志 `[ONLINE] Indexing X new logs for CP` | 每个 CP 只索引新 logs |
| Fresh EmbeddingStore | 查看日志 `[ONLINE] Using fresh embedding stores` | 使用独立索引目录 |
| 时序保证 | 检索结果只包含 ≤ checkpoint 时间 | 无未来数据 |

### 4.2 Online Mode 禁止项

❌ **禁止并行 checkpoint 处理**: `checkpoint_workers=1` (必须)
❌ **禁止并行 key 处理**: `within_checkpoint_workers=1` (必须)
❌ **禁止预加载 graph**: Online mode 必须从空 graph 开始
❌ **禁止并行索引**: 索引必须按时间顺序串行

---

## 5. TCE 任务输出验证

### 5.1 输出文件格式

```json
{
  "predictions": [
    {
      "checkpoint_id": "cal_quarterly_001",
      "snapshot_state": {  // Task A
        "key1": { "value": "...", "evidence": [...] },
        "key2": { "value": "...", "evidence": [...] }
      },
      "rq3_apply_answers": {  // Task C
        "key1": { "items": [{ "qa_id": "q1", "answer": "...", "evidence": [...] }] }
      }
    }
  ]
}
```

### 5.2 输出验证命令

```bash
# 检查 checkpoint 数量
python3 -c "
import json
with open('tce_results.json') as f:
    data = json.load(f)
    preds = data.get('predictions', [])
    print(f'Checkpoints: {len(preds)}')
    for p in preds:
        cp = p.get('checkpoint_id')
        task_a = len(p.get('snapshot_state', {}))
        task_c = len(p.get('rq3_apply_answers', {}))
        print(f'  {cp}: TaskA={task_a}, TaskC={task_c}')
"

# 检查 evidence 提取
python3 -c "
import json
with open('tce_results.json') as f:
    data = json.load(f)
    for p in data['predictions']:
        for k, v in p['snapshot_state'].items():
            ev = v.get('evidence', [])
            if ev: print(f'{k}: has evidence ({len(ev)} items)')
"
```

---

## 6. 问题排查指南

### 6.1 启动失败

| 错误 | 原因 | 解决方案 |
|------|------|----------|
| `HippoRAG directory not found` | hipporag_dir 不存在 | 检查路径，确保先运行 indexing |
| `No OpenIE cache found` | 缓存文件不存在 | 运行 indexing 生成缓存，或等待实时抽取 |
| `API key not found` | .env 未加载 | 确保 `generation/HippoRAG2/.env` 存在 |
| `embedding_model not in config` | 配置缺少参数 | 添加 `embedding_model: text-embedding-3-large` |

### 6.2 运行时问题

| 问题 | 诊断方法 | 解决方案 |
|------|----------|----------|
| 索引速度慢 | 检查 `OPENIE_MAX_WORKERS` | 设置更高的并发 (如 8) |
| API 限流 | 检查日志中的 error code | 降低并发，等待重试 |
| 检索无结果 | 检查日志 `[DEBUG] Raw docs retrieved: 0` | 验证 graph 是否正确构建 |
| 进程卡住 | `ps aux | grep tce.py` | 检查是否有死锁，重启 with `--resume` |

### 6.3 检索质量问题

| 问题 | 诊断方法 | 解决方案 |
|------|----------|----------|
| Evidence 缺失 | 检查输出 `evidence: []` | 验证 OpenIE 提取质量 |
| 值预测错误 | 对比 ground truth | 调整 retrieval_top_k |
| 空回答率高 | 统计 `answer: ""` 数量 | 检查检索覆盖度 |

### 6.4 配置验证命令

```bash
# 验证 YAML 配置完整性
python3 -c "
import yaml
with open('configs/experiments/tce/hipporag2.yaml') as f:
    cfg = yaml.safe_load(f)
    print('LLM:', cfg['llm'])
    print('Baseline:', cfg['baseline_params'])
    required = ['hipporag_dir', 'online', 'embedding_model']
    for r in required:
        if r not in cfg['baseline_params']:
            print(f'WARNING: Missing {r}')
        else:
            print(f'  {r}: {cfg[\"baseline_params\"][r]}')
"

# 验证环境变量
echo "API Key set: ${AZURE_OPENAI_API_KEY:-NOT_SET}"
echo "Base URL: ${AZURE_OPENAI_BASE_URL:-NOT_SET}"
```

---

## 7. OpenIE 缓存机制

### 7.1 缓存文件

| 文件 | 格式 | 作用 |
|------|------|------|
| `openie_results_ner_{llm_model}.json` | JSON | 实体/三元组抽取结果 |
| `llm_cache/*.sqlite` | SQLite | API 调用缓存 (HippoRAG 内部) |

### 7.2 缓存验证

```bash
# 检查缓存文件
ls -la generation/HippoRAG2/outputs/*/openie_results_ner_*.json

# 验证缓存内容
python3 -c "
import json
with open('openie_results_ner_gpt-5-mini.json') as f:
    data = json.load(f)
    print(f'Cached logs: {len(data.get(\"docs\", []))}')
"
```

---

## 8. 预期运行时间

| 场景 | Checkpoints | 预估时间 | 瓶颈 |
|------|-------------|----------|------|
| 测试运行 | 3 | ~30-60 分钟 | OpenIE + 检索 |
| 完整运行 | 30+ | ~2-4 小时 | 增量索引累积 |
| 有缓存 | 任意 | 快 50-70% | OpenIE 缓存命中 |

---

## 9. 运行命令

### 9.1 标准运行

```bash
# 设置环境变量
export OPENIE_MAX_WORKERS=8

# 运行 (使用配置文件中的模型)
python -m generation.run_tce --config configs/experiments/tce/hipporag2.yaml

# 限制 checkpoints 测试
python -m generation.run_tce --config configs/experiments/tce/hipporag2.yaml --max-checkpoints 3
```

### 9.2 后台运行

```bash
nohup python -m generation.run_tce \
  --config configs/experiments/tce/hipporag2.yaml \
  > logs/hipporag2_run.log 2>&1 &

# 监控日志
tail -f logs/hipporag2_run.log
```

---

## 10. 代码修改记录

### 10.1 本次修复内容

| 修改 | 文件 | 原问题 | 修复方案 |
|------|------|--------|----------|
| Embedding 从配置读取 | `tce.py` | 从目录名推断，有 fallback | 直接从 config 读取，无 fallback |
| LLM model 日志 | `tce.py` | 未打印 LLM model | 启动时打印完整配置 |
| Config 参数 | `hipporag2.yaml` | 缺少 embedding_model | 添加 embedding_model 参数 |
| Adapter 参数 | `hipporag2.py` | 未传递 embedding_model | 添加参数传递 |

---

## 附录 A: 配置模板

```yaml
# configs/experiments/tce/hipporag2.yaml (完整版)
users:
  - 001_user_001

runtime:
  baseline: hipporag2
  resume: false
  debug: true
  save_prompt_and_raw: true
  max_checkpoints: 3

data:
  benchmark: data_construction/generated_outputs/gemini_3_flash_preview/{user_id}/tce_benchmark.json
  app_logs_path: data_construction/generated_outputs/gemini_3_flash_preview/{user_id}/app_log_large.json

output:
  prediction_path: generation/HippoRAG2/results/{user_id}/prediction/tce_results.json

llm:
  provider: openai
  model: gpt-5-mini
  max_workers: 1

baseline_params:
  hipporag_dir: generation/HippoRAG2/outputs/{user_id}_large/gpt-5-mini_text-embedding-3-large
  online: true
  embedding_model: text-embedding-3-large
  retrieval_top_k: 5
  openie_cache_path: generation/HippoRAG2/outputs/{user_id}_large/openie_results_ner_gpt-5-mini.json
  checkpoint_workers: 1
  within_checkpoint_workers: 1
```

---

## 附录 B: 关键常量

| 常量 | 值 | 来源 | 说明 |
|------|-----|------|------|
| `temperature` | 0.0 | TCE 要求 | 确保可复现 |
| `checkpoint_parallelism` | forbidden | TCE 要求 | 避免数据泄露 |
| `within_checkpoint_parallelism` | forbidden | TCE 要求 | 避免数据泄露 |
| `embedding_dim` | 3072 | text-embedding-3-large | 向量维度 |
| `embedding_dim` | 1536 | text-embedding-3-small | (已弃用) |

---

**维护者**: MemBench Team  
**文档版本**: v2.0