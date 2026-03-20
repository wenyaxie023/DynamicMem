# TCE Full-Dict vs Per-Key Audit (2026-02-26)

Scope:
- Baseline: `rag`
- User: `001_user_001`
- Files compared:
  - `generation/rag/results/001_user_001/prediction/tce_results_topk5_queryv2_10_fullcmp.json`
  - `generation/rag/results/001_user_001/prediction/tce_results_topk5_queryv2_10_perkey.json`

Findings:
1. Checkpoint coverage is aligned.
- fullcmp checkpoints: `10`
- perkey checkpoints: `10`
- common checkpoints: `10`
- unmatched checkpoints: `0`

2. `metadata.predict_per_key`标记正确。
- fullcmp flags: `{False}`
- perkey flags: `{True}`

3. 同一 checkpoint 下目标 key 集一致。
- key-set mismatch checkpoint count: `0`

4. 补充观察（预测值层面，不是正确率）：
- fullcmp 与 perkey 预测值逐 key 完全相同比例：`58 / 77 = 0.7532`

Conclusion:
- 现有数据满足公平对比前提：checkpoint 对齐、key 集一致、运行模式标记可区分。
- 可以在此基础上进行 item-level 时序分析（changed vs unchanged、per-key trend）。
