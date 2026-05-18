
# TODO: 防止模型偷看未来信息 (Prevention of Future Information Leakage)

## 🎯 目标 (Goal)
在 HippoRAG 评估过程中，确保模型无法访问位于当前 Checkpoint 时间戳之后的“未来信息” (Graph/Logs)，从而保证评估的公平性和真实性。

## 🛠️ 实现方案 (Implementation Approaches)

我们计划通过以下两种方式之一来实现这一目标：

### 1. 在线同步测试 (Online Synchronous Testing)
*   **机制**: 在评估循环 (Evaluation Loop) 中，边预测边索引。
*   **流程**:
    1.  初始化时只加载空图或初始状态。
    2.  到达每个 Checkpoint 时，识别并增量索引 (Incremental Indexing) 该 Checkpoint 之前新增的日志。
    3.  进行预测。
    4.  重复步骤 2-3 直至结束。
*   **优点**:
    *   **零存储开销**: 不需要保存任何中间状态的图文件。
    *   **灵活性**: 随时可以运行，不需要预处理。
*   **缺点**:
    *   **耗时增加**: 每次评估都需要实时跑索引（虽然 OpenIE 结果可复用，但图更新仍需时间）。
    *   **并发限制**: 难以像离线版那样大规模并发 (如 10 User 并行)，受限于单机资源。

### 2. 离线保存 (Offline Saving / Snapshotting)
*   **机制**: 预先跑一遍索引流程，在每个 Checkpoint 处保存图的快照 (Snapshot)。
*   **流程**:
    1.  运行 `run_index_snapshots.py`（需开发）。
    2.  遍历所有日志，每遇到一个 Checkpoint，就将当前的 `graph` 和 `embeddings` 完整保存到磁盘的一个子目录（如 `cp_10/`）。
    3.  评估时，根据 Checkpoint ID 直接加载对应的图文件进行检索。
*   **优点**:
    *   **评估极快**: 评估时只有检索开销，没有索引开销。
    *   **纯净隔离**: 每个 Checkpoint 的图是物理隔离的，绝对保证无泄漏。
*   **缺点**:
    *   **存储爆炸**: 143 个 Checkpoint * 1.5GB/个 * 10 User ≈ 2TB+ 存储空间。
    *   **管理复杂**: 产生大量小文件和目录。
