# Temporal Checkpoint Evaluation Developer Manual

Status: active
Owner: DynamicMem team
Last Updated: 2026-04-20

## 1. Purpose & Scope
本手册定义 `Temporal Checkpoint Evaluation`（TCE）的稳定开发规范。
后续与 TCE 相关的需求、实现、评审、验收，应先对齐本文件。

Goals:
- 统一 TCE 任务边界、输入输出契约和指标语义。
- 统一 state/QA 质量控制与准入门槛。
- 提供可审计的验收门槛，减少“文档-实现漂移”。

Out of Scope:
- 不覆盖 `End-to-End QA` 的细节协议（由独立协议维护）。
- 不承载 baseline 级执行步骤（执行流程见 runbook）。

Execution Runbook:
- `docs/runbooks/tce_execution_runbook.md`

Contributor-facing generation/adapter contract:
- `docs/protocols/tce_generation_and_adapter_contract.md`

## 2. Evaluation Landscape (DynamicMem)
DynamicMem 当前评测协议分为两类：
1. `Temporal Checkpoint Evaluation`（TCE，本文件）
2. `End-to-End QA Evaluation`（全上下文，3 levels x 6 sublevels）

## 3. Task Contracts

### 3.0 Task-Contract Versioning
- benchmark / prediction / eval artifact 顶层应带：
  - `task_contract_version`
  - `research_frame_version`
  - `canonical_research_doc`（若有）
- supported contract families:
  - `taskabc_v1`
    - 历史冻结合同
    - 对 `2026-04-13` 之前生成且缺少显式 `task_contract_version` 的旧 artifact，读取方默认按 `taskabc_v1` 解释
  - `taskabc_v2`
    - 当前 active 合同
    - 新构建 benchmark / task packs / prediction / eval 默认写入该值
- compatibility rules:
  - `state_completion_pack` / `change_tracking_pack` / `rq3_apply_service_qa` 这些 pack 名称在 `v1` 和 `v2` 中保持不变
  - 合同升级首先通过 top-level `task_contract_version` 区分，而不是通过 pack 名字或文件名中的 `RQ1/RQ2/RQ3`
  - 历史文件名若仍保留 legacy `rq1/rq2/rq3` 前缀，解释时以 artifact 内部 metadata 和当前协议文档为准
- current research-facing mapping under `taskabc_v2`:
  - `RQ1 = State Reconstruction`
    - primary benchmark object: `Task A`
  - `RQ2 = State Updating`
    - no standalone Task B in the active v2 contract
    - primary analysis object: `Task A` on `changed` vs `unchanged` states
    - `changed` 的定义：与前一个 checkpoint 都存在且 current-state value 发生变化的 key
  - `RQ3 = Personalization Utility`
    - primary benchmark object: `Task C`
    - `Task C - Task A` 可以作为 utility diagnostic gap，但不是 Task C 的定义本身

### 3.1 Checkpoint Definition
- checkpoint 是时间切片下的评测单元。
- 每个 checkpoint 都有统一的目标 key 集与 ground truth 状态视图。
- 在 TCE 中，所有任务均以“从起点到该 checkpoint 的可见记忆”为输入边界。

Execution semantics (code-aligned):
- `data_construction/build_tce_benchmark.py` 在 benchmark 构建阶段直接产出 sampled checkpoints（默认 `calendar + quarterly`）。
- sampled checkpoint 的采样信息在 benchmark 内落地：
  - 顶层 `sampling_strategy`
  - 每个 checkpoint 的 `sampling.mode / sampling.params`
- `generation` 阶段默认直接消费 benchmark 中的 checkpoint；若 benchmark 已标记 `sampling_strategy.stage=benchmark_build`，运行时采样参数应忽略。
- `generation.run_tce_batch` 的 YAML 模板渲染至少支持 `{user_id}`；当前协议也支持 `{run_id}`、`{run_name}`。
- 建议将 `runtime.experiment_name` 保持为稳定实验名，将具体运行实例隔离放在 `runtime.run_id`；默认 `run_id=main` 便于 resume，若要新开一轮独立 run，可显式设为自定义标签。

Execution stages (current protocol):
1. Raw benchmark build:
   - 只产出 sampled checkpoints、`expected_snapshot_state`、`state_observability`
   - 不做 state validate，不出任务题包
2. Independent state validation:
   - 入口：`data_construction/build_tce_state_validation.py`
   - 产出：`state_questionability`、`validated_snapshot_state`、`state_validation_summary`
3. Prebuilt task packs:
   - 入口：`data_construction/build_tce_task_packs.py`
   - 产出：`state_completion_pack`、`rq3_apply_service_qa`
   - `taskabc_v1` 额外产出：`change_tracking_pack`
4. Generation / evaluation:
   - 优先消费 prebuilt packs
   - 若 benchmark 缺 pack，则 fallback 到 legacy raw benchmark 逻辑
5. Optional post-TCE final-memory QA:
   - 可在最后一个 executed checkpoint 完成后，复用该 checkpoint 的 memory state 运行一轮完整 QA
   - QA artifact 必须继续遵循 `docs/protocols/qa_generation_and_eval_contract.md`
   - 该阶段属于 TCE generation 的可选附加阶段，不要求修改原有 `generation.run_qa` / QA adapter 流程

### 3.1a Baseline Concurrency Policy
TCE generation 必须显式声明 baseline 的并发策略，至少区分两个维度：
- `checkpoint_parallelism = allowed | forbidden`
- `within_checkpoint_parallelism = allowed | forbidden`

Definitions:
- `checkpoint_parallelism`
  - 是否允许不同 checkpoint 同时生成预测
- `within_checkpoint_parallelism`
  - 是否允许同一个 checkpoint 内，不同 key / question 同时生成预测

Code-aligned policy:
- `rag`
  - `checkpoint_parallelism = allowed`
  - `within_checkpoint_parallelism = allowed`
- `oracle`
  - `checkpoint_parallelism = allowed`
  - `within_checkpoint_parallelism = allowed`
- `icl`
  - `checkpoint_parallelism = allowed`
  - `within_checkpoint_parallelism = allowed`
- `hipporag` / `hipporag2`
  - `checkpoint_parallelism = allowed`
  - `within_checkpoint_parallelism = allowed`
- `amem`
  - `checkpoint_parallelism = forbidden`
  - `within_checkpoint_parallelism = allowed`
- `memoryos`
  - `checkpoint_parallelism = allowed`
  - `within_checkpoint_parallelism = allowed`
- `mem0`
  - `checkpoint_parallelism = allowed`
  - `within_checkpoint_parallelism = allowed`
- `zep`
  - `checkpoint_parallelism = forbidden`
  - `within_checkpoint_parallelism = forbidden`
- `simplemem`
  - `checkpoint_parallelism = forbidden`
  - `within_checkpoint_parallelism = forbidden`
- `letta` / `memgpt`
  - `checkpoint_parallelism = forbidden`
  - `within_checkpoint_parallelism = forbidden`

Unified TCE execution route:
- all baselines must execute through one shared `tce_core` orchestrator
- the orchestrator must drive three explicit phases:
  1. `prepare_checkpoint_state`
  2. `retrieve_context_for_query`
  3. `answer_query`
- baseline differences are expressed only through backend hooks, not through separate pipeline contracts
- `letta` / `memgpt` are not protocol exceptions; until they migrate to the required snapshot-builder route, they should be treated as pending compliance work rather than carve-outs
- legacy `retrieve_context(...)` callback is not part of the current TCE protocol contract
- Task A answer prompting must use the pack-authored `question_text` directly as the visible query text.
- Shared Task A prompt builders must not prepend extra wrapper sections such as `[Task]`, `Instruction:`, `Query:`, or `Checkpoint time:` ahead of that `question_text`.
- Shared Task A prompt builders should place explanation of output requirements before the JSON template, and do not need a trailing catch-all `Rules` block.
- explicit-retrieval baselines must consume shared `QuerySpec.retrieval_query_text` directly; baseline-local regeneration or fallback of retrieval query text is not part of the current protocol.

Shared generation config contract:
- cross-baseline generation knobs must not live in `baseline_params`
- canonical shared sections are:
  - `runtime`
    - `user_id`
    - `task_selection`
    - `checkpoint_workers`
    - `within_checkpoint_workers`
    - `save_every_generation_keys`
    - `enable_change_reasoning`
    - `enable_rq3_apply_service_qa`
    - `rq3_apply_save_prompt_and_raw`
  - `llm`
    - answer-model provider / model / decoding settings
  - `retriever`
    - shared explicit-retriever backend settings such as provider / model / batch size
    - this includes retrieval / indexing embedding models for baselines such as `rag`, `amem`, `hipporag2`, and `MemoryOS`
  - `retrieval`
    - shared top-k settings such as `top_k`, `rq3_apply_top_k`, `final_qa_top_k`
  - `final_qa`
    - `enabled`, `path`, `output_path`, `save_prompt_and_raw`
- `baseline_params` is reserved for backend-specific knobs only
- `user_id` is not a backend-specific knob:
  - batch runs must treat the selected user as shared runtime state
  - `generation.run_tce_batch` should inject shared `runtime.user_id` from the chosen batch user
  - direct single-user configs may set `runtime.user_id` explicitly
  - `baseline_params.user_id` is not part of the current protocol
- `task_selection` is a shared runtime knob:
  - supported values:
    - `all`
      - default behavior
      - execute the contract-default task set for the benchmark
      - under `taskabc_v2`, this currently means `Task A + optional Task C`
    - `task_c_only`
      - diagnostic execution mode for prebuilt Task C-only benchmark artifacts
      - runtime must skip Task A generation instead of treating missing `state_completion_pack` as an error
      - runtime still consumes pack-authored `rq3_apply_service_qa` items and shared `QuerySpec.retrieval_query_text`
  - `task_selection` must live under `runtime`, not `baseline_params`

Implementation requirements:
- generation runtime 必须将 requested worker counts 与 effective worker counts 一并记录到 prediction metadata。
- 当某 baseline 的 policy 为 `forbidden` 时，运行时必须把对应 worker 数强制降为 `1`，不能只靠文档约定。
- 对 stateful memory baseline，只有在“每个 checkpoint 的 context 是只读且 per-question answering 不会改写共享 memory state”时，才允许开启 `within_checkpoint_parallelism`。
- 对所有 baseline：
  - 必须按 benchmark checkpoint 顺序 ingest user app logs 或等价地推进 memory state
  - canonical app-log payload 必须是 raw app log object 本身
  - builder ingest 必须直接消费 raw app log object 本身
  - 若 baseline 的原生 builder 不接受 raw event objects，则允许使用 lossless backend transport wrapper：
    - 若 backend 接受单条 note / document / text entry，则每条 raw app log object 可以直接投影成一条 lossless text entry
    - 若 backend 明确要求 dialogue turns，则每条 raw app log object 可以投影成两条 dialogue turn
      - turn 1 必须是 `User: {raw_json}`
      - turn 2 必须是 `Assistant: [ingested]`
      - `raw_json` 必须是该 raw app log object 的 lossless serialization
      - `[ingested]` 必须是 literal acknowledgement，不得承载额外语义内容
    - 该 wrapper 只是一种 backend transport；source lineage 仍必须能精确回到原始 `app_log_id`
  - shared inline-memory rendering 应直接承载 backend retrieval 阶段返回的 memory content
  - 对 stateful baseline，checkpoint snapshot 的 retrieval-visible memory state 可以是从 raw app logs 推导出的 derived memory units；shared prompt 中的 inline memory 应直接使用这些 backend-derived memory units 的 lossless serialization，而不是强制 rematerialize 成 raw app log object
    - 若 backend memory unit 保留 raw-log lineage，则调试 metadata 应尽可能保留这些 source ids 以支持审计
    - 若 backend memory unit 不保留 raw-log lineage，则 adapter 不应伪造或臆造 raw app log 映射
    - adapter 不得在 backend 官方 retrieval payload 之外额外加入本地包装说明、重写标签或二次改写的 rematerialized view
  - 对采用 backend-native retrieval-content mode 的 baseline，shared prompt 直接看到 backend 官方 `retrieve_context(...)` 返回内容是 canonical 行为；inline memory 应是该 retrieval payload 的 lossless serialization
    - debug metadata 仍应尽可能保留 backend memory ids、payload counts，以及在可恢复时的 raw app log ids/source ids
  - 除上述 derived memory unit 例外外，baseline 可为 backend transport 做 lossless serialization / wrapper，但不得在这些阶段前额外增加、删除、重命名、摘要化或改写 payload 语义字段
  - 每个 checkpoint 的 retrieval 只能读取该 checkpoint 对应的 prepared state
  - 回答阶段不得把 Task A/B/C 的问答行为写回共享 memory state
  - `baseline_params.enable_change_reasoning=true` 是 Task B generation 的规范开关；Task C 继续由 `enable_rq3_apply_service_qa` 控制
  - generation runtime 默认信任 pack-first benchmark；Task C pack 的完整性与合法性应在 task-pack build 阶段保证，不应在 baseline runtime 再引入 pack-missing gate
  - 若 baseline 无法证明 query path 为只读，则必须把 `checkpoint_workers` 与 `within_checkpoint_workers` 都强制降为 `1`
  - 对 agent-memory baseline，若 backend 暴露 native retrieval+answer API（例如 `ask()`）：
    - shared visible task prompt 仍必须来自 `QuerySpec.answer_query_text` 对应的 shared agent-memory prompt builder
    - backend-native retrieval 的 driving question 必须来自 `QuerySpec.retrieval_query_text`
    - `retrieve_context_for_query(...)` 必须执行或 materialize 可复用的 backend retrieval state，并在 `debug_metadata` 中写出可审计的 retrieval metadata（至少包括 `retrieval_query` 以及 backend-exposed retrieval ids / source ids when available）
    - `answer_query(...)` 可以复用该 retrieval state 调用 backend-native `ask()` / answer generator，但不得 silently regenerate a different retrieval question
- 对所有 stateful baseline：
  - 必须采用统一的 checkpoint-snapshot 路线；不再存在 test-phase 内推进 builder 的允许路径
  - build phase 与 test phase 必须是两个正式相位
  - build phase 负责顺序 ingest raw app logs，并产出：
    - required persisted checkpoint artifacts（checkpoint snapshot / `manifest.json` / snapshot-owned builder-resume metadata when needed）
    - optional local builder progress / runtime bookkeeping
  - build phase 可以额外维护 full-corpus 的 local per-log preprocessing cache（例如 OpenIE / embedding 结果），并允许在该 cache 构建阶段使用并行
  - 这类 preprocessing cache 不是 retrieval-visible memory state；它只能作为后续 builder replay 的本地输入
  - 若使用 preprocessing cache，checkpoint memory state 仍必须按 confirmed prefix 顺序 replay / materialize；不得先基于全量 future logs 直接建成可检索图或可检索集合
  - test phase 才调用 shared `run_pipeline(...)`
  - `prepare_checkpoint_state(...)` 在 test phase 中只能定位并加载当前 checkpoint 的 persisted snapshot / collection；不得推进 builder
  - standalone baseline entrypoints 必须使用 canonical path knobs；不保留 compatibility alias，也不允许从 `checkpoint_dir` 一类旧 builder-state 路径隐式推导 snapshot 读取路径
  - 若 baseline 维护本地 builder progress，它只可记录 confirmed progress；遇到 ingest 结果不确定时必须标记 uncertain，而不是回写更早 confirmed index
  - resume 默认以 latest confirmed snapshot bundle 为准；builder 所需的 authoritative resume metadata 应随 snapshot bundle 一起持久化，而不是只存在于本地 runtime 目录
  - 本地 builder progress 若存在，只能作为运行时 bookkeeping / 调试辅助，不应成为 builder/ingest resume 的唯一 authoritative source
  - 当前尚未迁移到该路线的 stateful baseline 应标记为 pending compliance work，而不是协议例外

### 3.1b Shared Point Scoring Contract
vNext 的答案质量评估统一采用 `point` 作为评分单元。

Canonical point object:
- `point_id`
- `point_type`
  - `field | micro`
  - legacy reader/eval compatibility may still tolerate `list_item`, but new pack build must not emit it
- `polarity`
  - `positive | negative`
- `point_text`
- `target_path`
  - `field` 必填
  - path-bound `micro` strongly recommended
- `reference_value`
  - `field` 必填
  - `micro` 不要求提供；默认只保留 atomic `point_text`

Path note:
- 对 scalar current-state value，canonical field point 仍可使用 `target_path = "current_value"`。
- evaluator 必须把 `current_value` 解释为该 root blob 的 alias，而不是要求 prediction 额外包成 `{"current_value": ...}`。
- list-like value 在构造 `scoring_points[]` 之前，必须先按顺序预展开成 indexed field paths（例如 `schedule_dates.0`, `schedule_dates.1`, `memberships.0.name`）。
- 若某个 indexed leaf 是 complex string，则它应被视为一个 path-bound complex field，并生成绑定到该 indexed path 的 `micro` atomic facts，而不是再引入单独的 `list_item` point type。
- 对 root-level descriptive string（例如带 parenthetical qualifier、descriptor、or other sentence-like detail 的 scalar string），pack build 应与 nested complex string 保持一致，优先生成 `micro` atomic facts，而不是退化成单条 binary field-equality point。

Terminology note:
- 在 schema 中统一沿用 `point` / `scoring_points` 命名，避免字段名反复变化。
- 在 prompt、manual review、讨论中，所有用于打分的 scoreable meaning unit 统一称为 `atomic fact`。
- `micro` 不是另一套对象；它只是 `point_type` 的一种，表示 statement-like atomic fact。
- 对任意 atomic fact：
  - `point_text` 是 evaluator-facing 的完整评分要求
  - `field` point 仍使用 `reference_value` 作为 canonical gold anchor
  - `micro` point 默认只保留 atomic `point_text`；若该 atomic fact 绑定到某个 source subpath，推荐同时保留 `target_path` 以便 judge 看到对应的 `predicted_value`

### 3.1c Task-Level Ground-Truth Resolution (Code-Only)
`validated_snapshot_state` 是 task-agnostic 的 validated state view；在进入具体 task pack 之前，必须先经过一层轻量、纯代码的 task-level ground-truth resolution。

Why this layer is required:
- data construction 允许在 state 发生变化时使用 delta / transition-like 表示
- 因此 `validated_snapshot_state` 中的某个 key，未必能直接作为 Task A / Task C 的“当前状态”
- 同理，Task B 的 `before/after` 也未必能直接从 raw validated object 原样读取

Transition-style validated state:
- 若某个 validated state object 的顶层 key 集是 `{"from", "to"}` 的子集，视为 transition-style state
- 该判断是程序规则，不依赖 LLM

Task A current-state projection:
- 非 transition-style state：
  - `task_a_ground_truth = validated_state_value`
- transition-style state：
  - `task_a_ground_truth = validated_state_value.to`
- 若无法得到 `task_a_ground_truth`（例如只有 `from`、没有 `to`），该 key 必须从 `state_completion_pack` 过滤掉
- under `taskabc_v2`, Task A pack build must additionally remove excluded derived / bookkeeping fields before authoring `answer_template`, `retrieval_query`, and `scoring_points`
  - current exclusion set:
    - `schedule_date`
    - `schedule_dates`
    - `priority`
  - rationale:
    - these fields are not part of the intended explicit current-state benchmark object in v2
    - leaving them inside Task A would inflate slot count and pollute the overall score with derived/date-expansion details

Task B before/after resolution:
- transition-style current validated state：
  - `before` 优先取当前 validated state 的 `from`
  - 若当前 `from` 缺失，可回退到上一个 checkpoint 已解析的 current state
  - `after` 必须取当前 validated state 的 `to`
- 非 transition-style current validated state：
  - `before = previous checkpoint resolved current state`
  - `after = current checkpoint resolved current state`
- 若 `before` 或 `after` 任一无法解析，该 key 必须从 `change_tracking_pack` 过滤掉
- 若解析后的 `before == after`，该 key 不应进入 Task B changed set

Task C current-state projection:
- Task C 只允许基于“当前 checkpoint 可解析的 current state”出题
- 因此 Task C 必须复用 Task A 的 current-state projection
- 若 current state 无法解析，该 key 必须从 `rq3_apply_service_qa` 过滤掉

Implementation requirements:
- 这层 task-level validation 必须是 deterministic、code-only 的；不能委托给 LLM
- task pack builder 应记录被过滤 key 的 `reason_codes`，用于人工审阅与回溯
- `validated_snapshot_state` 本身不因 task-level validation 而回写修改；过滤只发生在 task-pack build 阶段

Shared atomic-fact scoring semantics:
- 所有 point 的单点评分统一为 `0/1`。
- `1`
  - 该 atomic fact 被命中 / 满足。
- `0`
  - 该 atomic fact 未被命中，或与 gold atomic fact 冲突。

Negative-point semantics:
- 负向点与正向点共用同一二值标尺。
- 对负向点：
  - `1 = 明确未违背禁忌 / 约束`
  - `0 = 明确触犯禁忌或忽略硬约束`

Aggregation:
- item/key 内：所有 points 等权平均
- checkpoint 内：所有 item/key 等权平均
- summary：所有 checkpoint 等权平均

### 3.2 Task A: State Completion (Cloze)
Objective:
- 评估模型是否能补全 `observable + valid` state item 的当前值。

Input Contract:
- checkpoint 前可见记忆（logs / retrieval context）
- target key set（优先由 `state_completion_pack.keys` 提供；旧 benchmark fallback 到 raw benchmark）
- value template（优先由 `state_completion_pack.keys[*].answer_template` 提供）
- retrieval query（优先由 `state_completion_pack.keys[*].retrieval_query` 提供；query 只承载 item-specific 问题，不承载通用 instruction）
- checkpoint scope 由 shared `tce_core` 注入到 `QuerySpec` 与执行元数据；baseline 不得在本地 wrapper 中私自补一套额外 checkpoint 语义

Generation granularity contract:
- Task A 的 state completion generation 必须按 `state_key` 独立执行。
- 每个 `state_key` 都必须使用自己的：
  - retrieval query
  - retrieval context
  - final answering prompt
  - raw model output record
- checkpoint-level combined retrieval / combined prompt for multiple Task A keys 不属于当前 TCE 协议路径。
- prediction metadata 应以 `per_key_retrieval[*]` 作为 Task A retrieval 审计主记录。

Output Contract:
- `snapshot_state[key] = predicted_value`
- `evidence[key] = [{"app_log_id": str, "evidence_content": str}, ...]`

Evaluation Scope:
- 新协议：仅评估 `state_completion_pack.keys`
- fallback：旧 benchmark 仍评估 `observable + valid` keys

Scoring Sources:
- slot-level LLM judge（only semantic route）
  - 每个 `state_completion_pack.keys[state_key]` 必须带 `scoring_points[]`
  - simple field：一个 field = 一个 `field` point
  - list-like attribute：必须先展开成有序 indexed field paths；每个 ordered leaf = 一个 slot
    - 例如 `schedule_dates.0`, `schedule_dates.1`
    - judge 看到的 `predicted_value` 必须来自同一 indexed path，不做跨位置 best-match
  - 对 structured atomic values（例如 `schedule_dates`、`days_of_week`、`start_time`）做 point scoring 时，必须先做 type-aware normalization，再按 exact match 判断；不得因为共享 token（例如同一年份）给 partial credit
  - complex statement（例如 preference `statement`）：
    - 在 benchmark / task-pack 构造阶段生成 `micro` atomic facts
    - 每组 generated atomic-fact set 应优先是 `1..3` 条 atomic facts
    - 对 LLM 生成的 atomic-fact set，必须做 source-grounded validation
    - atomic facts 应优先表达稳定的语义要求，而不是脆弱的 exact surface form
    - 不得生成多个 slot 反复围绕同一个过细实体、型号、规格、参数或 exact wording 卡分
    - exact identifier 只有在它本身就是 canonical memory fact，或缺少该 identifier 会失去必要区分度时，才可作为 atomic fact 的核心约束
    - 否则应优先改写成更稳健的 semantic slot
    - validation 至少检查：
      - point set 是否相对原始 state field 发生 drift
      - 是否杜撰或引入 source field 中不存在的新含义
      - 每条 atomic fact 是否足够 atomic，能够独立在 shared `0/1` 标尺上评分
      - 是否因为不必要的 exact model / spec / parameter wording 变得过于 brittle
    - 若 validation 失败，必须触发 rewrite；若多次 rewrite 后仍失败，必须回退到更保守的 safe rubric
  - eval 时不得再把这些 points 交给 deterministic string matcher 直接打主分

### 3.5 Optional Final-Checkpoint QA Extension
Objective:
- 在不改变原 QA runner 的前提下，允许 baseline 复用 TCE 最后一个 executed checkpoint 的 memory state，再额外回答一轮完整 QA 问题集。

Contract:
- 该阶段是 optional post-hook，不属于 Task A/B/C 主 prediction artifact。
- final QA retrieval 必须只读取最后一个 executed checkpoint 对应的 snapshot / collection / manifest entry。
- final QA 提问过程同样不得污染共享 memory state；若 backend query path 可变，必须使用隔离副本。
- final QA 输出必须是独立 QA prediction artifact，遵循 `docs/protocols/qa_generation_and_eval_contract.md`。
- 原有 `generation.run_qa`、`generation/qa_adapters/*` 与 QA baseline contract 不因该扩展而改变。
- 对没有显式 query-time retrieval 的 agent-memory baseline（例如 `letta` / `memgpt`）：
  - 不应暴露 retrieval-only top-k config
  - `final_qa_retrieval_top_k` 不适用
  - Task C 的 `rq3_apply_retrieval_top_k` 也不适用

Recommended implementation shape:
- 在 baseline 的 TCE adapter 中，先完成标准 TCE Task A/B/C prediction。
- 然后取最后一个 executed checkpoint：
  - 复用同一组 shared hooks：
    - `prepare_checkpoint_state`
    - `retrieve_context_for_query`
    - `answer_query`
  - 以 QA question text 作为 retrieval query
  - 将输出写入独立 `*_final_qa.json` artifact

Resume semantics:
- final QA 可以复用 TCE 的 `resume` 语义单独跳过已完成 QA items。
- final QA artifact 不得回写进主 TCE prediction JSON，也不得作为 evaluator 依赖字段嵌入 TCE 结果。
  - eval 必须把 `scoring_points[]` 当作 slot 集合，交给 LLM judge 对每个 slot 独立判 `correct=true|false`
  - slot-level eval 的 canonical split contract：
    - program-owned `slot_context`
      - 由程序提供给 judge，也由程序负责落盘
      - common fields:
        - `point_id`
        - `point_type`
        - `polarity`
        - `predicted_value`
      - type-specific fields:
        - `field` / legacy `list_item`
          - 保留 `reference_value`
          - 默认不重复保留 `point_text`
        - `micro`
          - 保留 `point_text`
          - 默认不重复保留 `reference_value`
      - optional fields:
        - `target_path`
        - `slot_group`
        - `list_index`
    - LLM-owned `judgments`
      - 只允许：
        - `point_id`
        - `analysis`
        - `correct`
      - `analysis` 必须在 output schema 中先于 `correct`
    - aggregate score
      - 由程序把 `correct` 映射到 `1/0` 后计算
  - 对 simple `field` point：
    - judge 输入至少包含：
      - `point_id`
      - `point_type`
      - `target_path`
      - `reference_value`
      - 从 prediction 中按 `target_path` 抽出的 `predicted_value`
  - 对 `micro` point：
    - judge 输入至少包含：
      - `point_id`
      - `point_type`
      - `point_text`
      - `predicted_value`
      - 若该 `micro` 绑定到某个 subpath，推荐同时提供 `target_path`
  - 每个 slot 的 judge 输出只允许：
    - `point_id`
    - `analysis`
    - `correct`
  - `snapshot_point_score_mean_on_expected` 的定义改为：
    - 对每个 key，把其所有 slot judgment 的 `correct` 映射到 `1/0` 后等权平均；再对 keys 平均
- 结构化 continuity / diagnostic 指标（exact/f1/accuracy）
- evidence 指标分两组并列报告：
  - id-based metrics（precision/recall/f1/exact）
  - content-based metrics（当前版本至少要求 non-empty / paired-with-id 等结构化指标）

Metric meanings (Task A):
- Primary metrics:
  - `snapshot_point_score_mean_on_expected`
    - 对每个 expected `state_key`，对其 `scoring_points[]` 做 slot-level LLM judge；把 `correct` 映射到 `1/0` 后等权平均；再对 keys 平均。
    - 这是 Task A 的主语义指标。
  - `snapshot_value_f1_mean_on_expected`
    - 对每个 expected `state_key`，把 expected value 与 predicted value 展开成叶子文本并做 token-level F1；再对 key 平均。
    - 它是 continuity / diagnostic 指标，不再承担主语义解释。
- Secondary metrics:
  - `snapshot_evidence_recall_mean_on_expected`
  - `snapshot_evidence_precision_mean_on_expected`
  - `snapshot_evidence_f1_mean_on_expected`
    - 这些按 `app_log_id` 比较 predicted evidence 与 gold evidence 的重合程度，再对 key 平均。
    - 这些属于 auxiliary evidence 指标，不是所有 baseline 的强制能力要求。
    - `rag` / `hipporag2` 应尽量填写非空 `app_log_id`；其他 baseline 可留空。
  - `snapshot_evidence_app_log_id_nonempty_rate_mean_on_expected`
  - `snapshot_evidence_content_nonempty_rate_mean_on_expected`
  - `snapshot_evidence_content_with_id_rate_mean_on_expected`
    - 这些是 evidence payload 的结构质量指标，不衡量语义 correctness，只衡量 evidence object 是否填得完整。
- Diagnostic metrics:
  - `snapshot_value_accuracy_on_expected`
    - 对每个 expected key 的 value 做 strict exact equality，统计正确率。
    - 这是一个很严格的指标，适合作为诊断，不应单独代表总体质量。
  - `snapshot_exact_match`
    - 整个 checkpoint 的 snapshot 是否完全一致。
    - 这是最严格的 checkpoint-level 诊断指标。

### 3.3 Task B: Change Tracking & Attribution (Cloze)
Legacy note:
- this section describes the frozen `taskabc_v1` Task B contract
- under active `taskabc_v2`, standalone Task B is not part of the benchmark contract and should not be built by default

Objective:
- 评估模型是否能预测 changed keys 的最近两次状态，并给出变化归因。

Input Contract:
- checkpoint 前可见记忆
- changed key set（优先由 `change_tracking_pack.keys` 提供；旧 benchmark fallback 到 raw diff）
- retrieval query（优先由 `change_tracking_pack.keys[*].retrieval_query` 提供；query 只承载 item-specific 问题，不承载通用 instruction）

Output Contract:
- `change_analysis[key].before`
- `change_analysis[key].after`
- `change_analysis[key].change_reason`
- `change_analysis[key].evidence = [{"app_log_id": str, "evidence_content": str}, ...]`

Evaluation Scope:
- 新协议：仅评估 `change_tracking_pack.keys`
- fallback：旧 benchmark 仍评估 `observable + valid + changed` keys

Scoring Sources:
- slot-level LLM judge for state reconstruction
  - `before_scoring_points[]`
  - `after_scoring_points[]`
  - `before` / `after` 的 point 生成规则与 Task A 完全一致
- slot-level LLM judge for change reason
  - `reference_change_reason`
    - Task B 的 canonical gold reason text。
    - 必须直接来自上游 data construction 的原始 `change_reason`，不能由 `before/after` 现推。
  - `change_reason_scoring_points[]`
  - `change_reason_scoring_points[]` 必须从 `reference_change_reason` 生成
  - `change_reason` 一律用 `micro` points
  - eval 时，Task B 不再使用 whole-change `1..5` LLM rubric judge
  - `before_scoring_points[]`、`after_scoring_points[]`、`change_reason_scoring_points[]` 都必须按 slot-level `correct=true|false` 独立评判
  - main eval JSON 必须使用 split canonical structure：
    - `change_slot_eval_by_key[state_key].before`
    - `change_slot_eval_by_key[state_key].after`
    - `change_slot_eval_by_key[state_key].state_predict`
    - `change_slot_eval_by_key[state_key].change_reason`
    - 每个 group 都只包含：
      - `score_0_1`
      - `slot_count`
      - `slot_context`
      - `judgments`
- evidence 指标分两组并列报告：
  - id-based metrics（precision/recall/f1）
  - content-based metrics（当前版本至少要求 non-empty / paired-with-id 等结构化指标）
- Task B answer scoring 不得把 evidence ids 混入 answer-quality route

Metric meanings (Task B):
- Primary metrics:
  - `change_state_predict_point_score_mean_on_changed`
    - 对每个 changed key，把 `before_scoring_points[] + after_scoring_points[]` 的 slot-level LLM judgments 合并后做等权平均；再对 changed keys 平均。
    - 这是 Task B 的 state reconstruction top-line 指标。
  - `change_reason_point_score_mean_on_changed`
    - 对每个 changed key 的 `change_reason_scoring_points[]` 做 slot-level LLM judge；把 `correct` 映射到 `1/0` 后等权平均；再对 changed keys 平均。
    - 这是 Task B 的 reason-quality top-line 指标。
  - `change_before_after_f1_mean_on_changed`
    - 对每个 changed key，分别计算 `before` 和 `after` 的 value F1，再做均值；随后对 changed keys 平均。
    - 它是 continuity / diagnostic 指标。
- Secondary metrics:
  - `change_evidence_recall_mean_on_changed`
  - `change_evidence_precision_mean_on_changed`
  - `change_evidence_f1_mean_on_changed`
    - 基于 `app_log_id` 的 changed-evidence 对齐质量。
    - 这些属于 auxiliary evidence 指标，不是所有 baseline 的强制能力要求。
  - `change_evidence_app_log_id_nonempty_rate_mean_on_changed`
  - `change_evidence_content_nonempty_rate_mean_on_changed`
  - `change_evidence_content_with_id_rate_mean_on_changed`
    - evidence object 的结构完整性指标。
- Diagnostic metrics:
  - `before_point_score_mean_on_changed`
  - `after_point_score_mean_on_changed`
    - 只用于 checkpoint / item-level 诊断，不是 public top-line summary。
  - `change_before_after_correctness_mean_on_changed`
    - strict exact equality 下，`before` 与 `after` 是否都完全匹配；仅作诊断。

### 3.4 Task C: Personalized Service Completion
Objective:
- 评估模型是否能把已知 state 用于个性化服务决策。

Implementation status note:
- `taskabc_v1` continues to use the legacy apply-QA-style Task C contract
- `taskabc_v2` uses structured proactive personalized-service completion
- v1/v2 compatibility is controlled by top-level `task_contract_version`, not by pack name

Input Contract:
- checkpoint 的 Task C pack（`rq3_apply_service_qa`）
- retrieval context（优先按 apply item 的 `retrieval_query` 查询）
- apply QA 输入 state 必须来自 `validated_snapshot_state`
- generation runtime 对每个 `state_key` 只消费一个 apply item；Task C 不提供 runtime item-count override
- `taskabc_v2` 下，Task C build-time construction uses a strict canonical mapping:
  - `habits -> Habit-Conditioned User Communication`
  - `preferences -> Preference-Conditioned Filtering Parameter Completion`
  - `attributes -> Attribute-Conditioned Action Configuration`

Output Contract:
- `taskabc_v1`:
  - `rq3_apply_answers[state_key].items[*].answer`
  - `rq3_apply_answers[state_key].items[*].evidence = [{"app_log_id": str, "evidence_content": str}, ...]`
- `taskabc_v2`:
  - `rq3_apply_answers[state_key].items[*].answer` for `user_communication`
  - `rq3_apply_answers[state_key].items[*].output` for `information_request_construction` / `action_configuration`
  - `rq3_apply_answers[state_key].items[*].evidence = [{"app_log_id": str, "evidence_content": str}, ...]`

Scoring Sources:
- slot-level LLM judge for answer atomic facts
  - 每个 apply item 必须带：
    - `qa_id`
    - `answer_scoring_points[]`
    - `gold_memory_evidence_app_log_ids`
  - `taskabc_v1` items additionally carry the legacy QA fields:
    - `service_category`
    - `question`
    - `reference_answer`
    - `qa_validation`
    - `atomic_fact_validation`
  - `taskabc_v2` items carry the structured service-completion fields:
    - `service_family`
    - `scenario`
    - `task_instruction`
    - `item_validation`
    - `scoring_validation`
  - `taskabc_v2` `user_communication` items additionally carry:
    - `reference_answer`
  - `taskabc_v2` `information_request_construction` / `action_configuration` items additionally carry:
    - `output_template`
    - `reference_output`
  - `taskabc_v2` structured families must define a family-appropriate structured service object:
    - both must be top-level JSON objects
    - `output_template` replaces every required output leaf with `"<fill>"`
    - `reference_output` has the same exact shape as `output_template`
    - they do not need to mirror the raw `state_value` schema
    - they must preserve every non-derived source leaf value exactly once
    - required output leaves must appear in the same order as the source leaves so paired slot scoring stays deterministic
    - a raw copy of the source state schema is not a valid v2 service object
    - for `preferences -> Preference-Conditioned Filtering Parameter Completion`, the v2-contracted source input is statement-only:
      - if the underlying preference state includes `statement` plus auxiliary `signals`, only `statement` is shown to the answering assistant
      - only that contracted `statement` field participates in paired slot scoring
  - `taskabc_v2` structured-family `answer_scoring_points[]` must be positive-only paired field points:
    - `point_type = "field"`
    - `polarity = "positive"`
    - each point must include:
      - `source_field_path`
      - `output_field_path`
      - `target_path`
      - `reference_value`
  - every non-derived leaf in the v2-contracted `state_value` must appear in at least one scoring point
  - `answer_scoring_points[]` is the formal scoring contract for both v1 and v2
  - `answer_scoring_points[]` 缺失、为空或 schema 非法时，Task C eval 必须视为 pack/protocol violation 并直接失败
  - eval 时，Task C 不再使用 whole-item `1..5` judge
  - main eval JSON 必须使用 split canonical structure：
    - `rq3_apply_slot_eval_by_item[item_id]`
    - 每个 item 只包含：
      - `state_key`
      - `qa_id`
      - `score_0_1`
      - `slot_count`
      - `slot_context`
      - `judgments`
- evidence 指标分两组并列报告：
  - id-based correctness metrics
  - content-based structural metrics

Metric meanings (Task C):
- Primary metrics:
  - `rq3_apply_answer_point_score_mean`
    - 对每个 `(state_key, qa_id)` 的 `answer_scoring_points[]` 做 slot-level LLM judge；把 `correct` 映射到 `1/0` 后平均；再对 apply items 平均。
    - 这是 Task C 的主 answer metric。
- Secondary metrics:
  - `rq3_apply_key_coverage`
    - 有 apply items 的 key 中，模型是否都产出了对应回答。
  - `rq3_apply_evidence_precision`
  - `rq3_apply_evidence_recall`
  - `rq3_apply_evidence_f1`
    - 基于 `gold_memory_evidence_app_log_ids` 与 predicted evidence ids 的独立 evidence correctness 指标。
    - 这些属于 auxiliary evidence 指标，不是所有 baseline 的强制能力要求。
  - `rq3_apply_evidence_app_log_id_nonempty_rate`
  - `rq3_apply_evidence_content_nonempty_rate`
  - `rq3_apply_evidence_content_with_id_rate`
    - 这些是 evidence payload 的结构质量指标，不是 answer correctness 指标。

Unified answering-prompt contract:
- Task A / Task B / Task C:
  - Task A uses pack-authored `question_text`
  - Task B uses pack-authored `question_text`
  - Task C v1 uses pack-authored `question`
  - Task C v2 `user_communication` uses pack-authored `scenario + task_instruction`
  - Task C v2 structured families use pack-authored `scenario + task_instruction + output_template`
  - Task C v2 retrieval query should be the same pack-authored `scenario + task_instruction` string used to drive the item
  - Task C v2 retrieval query should not prepend extra wrapper sections such as `Service family:`, `Retrieval objective:`, or `Required output fields:`
  - Task C v2 final output format must be:
    - `user_communication`: `{"answer": <natural-language response>, "evidence": [...]}`
    - structured families: `{"output": <filled object>, "evidence": [...]}`

Evidence object contract (all three tasks):
- `app_log_id`
  - exact app log identifier from the provided user memory when available
  - 若 baseline 无法稳定把 retrieval payload 回溯到原始 app log，则允许留空字符串
- `evidence_content`
  - short quoted or closely paraphrased supporting snippet from the same log
  - 应简短、局部、直接支撑答案，不允许长段摘要
  - 必须与同一个 evidence object 内的 `app_log_id` 对齐

Task C v2 service-interface taxonomy:
- `Habit-Conditioned User Communication`
  - user-facing proactive communication actions
- `Preference-Conditioned Filtering Parameter Completion`
  - paper-facing name for the preference family
  - the assistant fills structured filtering parameters for a downstream retrieval / screening / recommendation step
  - this task does not ask for a final recommendation; it asks for the value of filtering fields that will later guide a downstream information system
- `Attribute-Conditioned Action Configuration`
  - the assistant fills structured configuration parameters for a downstream tool / workflow / executable service action
  - this task is about how a service should be configured or executed, not about what information should be retrieved

Runtime compatibility note:
- the stable internal Task C v2 family ids remain:
  - `user_communication`
  - `information_request_construction`
  - `action_configuration`
- for backward compatibility, the internal id `information_request_construction` now corresponds to the paper-facing preference task name `Preference-Conditioned Filtering Parameter Completion`

Task C v2 synthesis-prompt contract:
- Task C v2 generation prompt must be selected by `service_family`, not shared across all families
- `user_communication`, `information_request_construction`, and `action_configuration` each require their own family-specific synthesis prompt with targeted definitions, constraints, and example
- family-specific prompt examples should match the canonical service interface of that family rather than relying on one generic example across all state types
- each family-specific synthesis prompt should declare its own canonical source `state_type` exactly, rather than describing `state_type` as a generic multi-way choice
- the builder should inject `service_family` programmatically after synthesis; the family-specific LLM output does not need to generate that field

Task C v2 item validity rules:
- `user_communication`: output is one short natural-language assistant response
- `structured-completion`: structured families output a bounded JSON object
- `full-field dependency`: all required non-derived state fields are needed for the gold output
- `schema-groundedness`: structured families use a family-appropriate service object instead of a raw state copy
- `point-pairability`: structured families score one paired source/output field at a time; `user_communication` uses point-based natural-language scoring against `reference_answer`
- `unique gold`: under the provided context and schema, the gold output is stable enough for deterministic slot scoring

Task C v2 scoring target:
- primary metric:
  - `user_communication`: point-based correctness against `reference_answer`
  - structured families: required structured slot accuracy
- evidence metrics remain auxiliary and separate from answer correctness

## 4. Validation Policy (State + QA)

### 4.1 State Validate（Mandatory, Two-Level）
State validate 必须使用两层：

L1 Rule (deterministic):
- `empty_value`
- `noisy_value`
- `askable_fields` 可提取
- 对包含 `schedule_dates` 的 habit state，必须检查基于 evidence 的日期覆盖度，而不是只检查 `schedule_dates` 是否存在
  - evidence 日期来自该 checkpoint 的 `state_observability.evidence_app_log_ids`
  - 需要把 evidence log 的实际日期与目标 `schedule_dates` 做覆盖比较
  - 若 evidence 未能 100% 覆盖目标 `schedule_dates`，必须在 L1 直接拦截

L2 LLM Validate (mandatory):
- 判断该 state 的证据是否充分到“可出题且可作答”
- 输出字段必须包含：
  - `is_questionable`
  - `reason_codes`
  - `field_verdicts`（field-level 判定，列表）
    - `field_name`
    - `reason_analysis`
    - `is_valid`
  - `validator_version`
- 若 `state_observability[state_key].last_change_reason` 存在，则必须并行执行 `change_reason` metadata validation：
  - 该 validation 不属于 `field_verdicts`
  - 该 validation 不得进入 `validated_field_paths` / `validated_state_value`
  - 该 validation 的目标是判断 canonical gold `change_reason` 是否被 evidence logs 支持，并可作为后续 Task B 的 gold reason source
  - 输出字段：
    - `exists`
    - `reason_analysis`
    - `is_valid`
    - `reason_codes`

Final Decision:
- 仅当 L1 + L2 均通过，state 才可进入 apply QA 生成。

Stage output contract:
- `state_questionability[state_key]` 必须包含：
  - `is_questionable`
  - `l1_is_questionable`
  - `l2_is_questionable`
  - `reason_codes`
  - `askable_fields`
  - `validated_field_paths`
  - `dropped_field_paths`
  - `validated_state_value`
  - `field_verdicts`
  - `change_reason_validation`（optional；仅当 `state_observability.last_change_reason` 存在时要求）
  - `validator_version`
  - `validation_source`
  - `validation_identity`
- `validated_snapshot_state`
- `state_validation_summary`

State validation reuse policy:
- state validate 独立于 QA pack build 执行
- 入口：`data_construction/build_tce_state_validation.py`
- 复用只在同一次 state validation 运行内跨 checkpoint 生效
- 复用命中条件：
  - `state_key` 相同
  - `validation_identity.evidence_signature` 相同
  - 当前 raw state 在历史 `validated_field_paths` 上投影后，与历史 `validated_state_value_signature` 相同
  - `validator_version` 和 prompt version 相同
- 只要 evidence 集合增长或变化，必须重新做 L2 validate
- invalid 结果也允许复用，但前提仍是 evidence signature 和 projected validated value 未变

Validation identity fields:
- `state_key`
- `validated_state_value_signature`
- `evidence_signature`
- `validator_version`
- `prompt_version`
- `change_reason_signature`
- `change_reason_prompt_version`

State-evidence association and recomputation:
- benchmark build 阶段会先建立 `state -> evidence` 关联：
  1. `events[*].evidence_for_states` 指明某个 event 为哪些 state / fields 提供证据；
  2. Stage 3 app logs 将其写入每条 app log 的 `golden_evidence`；
  3. `build_tce_benchmark.py` 在时间推进过程中，把 `golden_evidence` 聚合到每个 checkpoint 的 `state_observability[state_key]`。
- `state_observability[state_key]` 是 state validate 的唯一证据入口，当前至少包含：
  - `evidence_app_log_ids`
  - `last_app_log_id`
  - `evidence_count`
  - `provenance_evidenced_fields`
  - `last_change_reason`（若上游 chain 提供 canonical gold `change_reason`）
- state validate 不会复用旧 checkpoint 的 evidence payload；
  每个 checkpoint 都会根据该 checkpoint 自己的 `state_observability` 重新计算：
  - `evidence_signature`
  - `evidence_logs`
- 当前实现中，`evidence_app_log_ids` 是“截至当前 checkpoint 的累计证据集合”，不是“仅支撑当前值的最小新证据集合”。
  因此对于同一个 state：
  - 较早 checkpoint 可能只看到支撑 `A1` 的 evidence；
  - 较晚 checkpoint 若 state 变为 `A2`，通常会看到 `A1` 的历史 evidence 加上新增的 `A2` evidence。
- 该机制不应引入 future leakage；
  因为 evidence 只会来自“checkpoint 时间点之前已进入 `state_observability` 的 app logs”。

### 4.2 QA Validate + Rewrite + Discard（Mandatory）
- Task C 必须拆成两个严格分离的阶段：
  1. `qa_generation -> qa_validation/rewrite`
  2. `atomic_fact_generation -> atomic_fact_validation/rewrite/fallback`
- apply QA 默认每个 state key 生成 1 条（`pair_count_per_key=1`，可通过配置覆盖）。
- rewrite 最多 2 次。
- rewrite prompt 必须注入失败规则（负例上下文）。
- apply QA pack 只允许从 `validated_snapshot_state` 生成。
- Task C validator 的三层职责必须严格分离：
  - `QA structural check`
    - 只判断 generated QA object 是否在结构上可处理
    - 不负责判断 question quality
    - 不负责判断 atomic-fact semantic quality
  - `LLM semantic validation`
    - 只判断 question 是否值得作为 Task C item 保留
  - `atomic-fact validation`
    - 只判断 scoring points 是否忠实、原子、可评分
- 所有 Task A / B / C 的 generated atomic-fact set 必须共享同一套 rubric-validation taxonomy：
  - point-level fail reasons：
    - `unsupported`
    - `drift`
    - `not_atomic`
    - `redundant`
    - `over_specific`
  - set-level fail reasons：
    - `coverage_gap`
- `over_specific` 的定义：
  - atomic fact 依赖不必要的 exact model / spec / parameter / exact wording，导致评分 brittle，但没有带来新的稳定语义覆盖
- `redundant` 与 `over_specific` 的边界：
  - `redundant` 关注多个 points 是否在重复同一语义
  - `over_specific` 关注某个 point 是否因为过细实体或表述而不必要地变脆
- semantic criteria 只保留以下三项：
  - `personalization_necessity`
    - The user's state must materially affect the best answer.
  - `service_decision_quality`
    - The item must ask for a real assistant/service decision rather than a fact lookup, pure classification, raw state recall, or explicit A/B menu.
    - 对 `habit` state，schedule-grounded service execution / conflict handling / batching / silencing / defer-escalate / routine-protection policy 仍可视为有效。
    - 该 criterion 只应在 item 退化成 event-name lookup、timestamp/date lookup、纯 state restatement、或没有真实 assistant-action layer 的 one-step arithmetic 时失败。
  - `answer_groundedness`
    - The gold answer plus atomic facts must stay grounded in information explicit in `state_value` or `question`.
    - `reference_answer` 若引入新的 concrete facts、parameters、product specs、thresholds、tactics、metrics、或 background knowledge，则该 criterion 必须失败。
- Apply QA validator output contract:
  - `semantic_criteria`
    - 列表长度必须为 3，且顺序固定为上述三项
    - 每项必须包含：
      - `criterion`
      - `analysis`
      - `pass`
- `qa_validation.is_valid` 必须继续由程序计算，不委托给 LLM 决定。
- Task C 的 QA structural check 只保留最小 QA-shape 校验：
  - `service_category` 必须存在且非空
  - `question` 必须存在且非空
  - `reference_answer` 必须存在且非空
- 以下内容不属于 Task C QA structural check：
  - question 长度
  - second-person wording
  - question mark presence
  - service category surface style
  - gold evidence 是否存在
- rationale:
  - question quality 归 `semantic validation`
  - atomic-fact semantic quality 归 `atomic-fact validation`
  - `gold_memory_evidence_app_log_ids` 由前序 validated-state / evidence pipeline 保证，不在该层重复判 invalid
- 程序侧必须将以下情况判为 invalid：
  - 缺 semantic criterion
  - semantic criterion 重复或顺序错误
  - 未知 semantic criterion
  - `analysis` 为空
  - 任一 semantic criterion `pass=false`
- atomic-fact validation 仍然必须判 invalid 于以下情况：
  - `answer_scoring_points[]` 缺失、为空或 schema 非法
  - 任一 atomic fact semantic validation 失败
  - 任一 atomic-fact set-level validation 失败
- QA 失败路径：
  - QA rewrite 只允许重写 `service_category` / `question` / `reference_answer`
  - QA 若超过 2 次仍失败：
    - 标记 `qa_validation.manual_review_required=true`
    - 记录到 `discarded_items`
    - 不得再进入 atomic-fact generation
- atomic-fact 失败路径：
  - atomic-fact rewrite 只允许重写 atomic facts；不得回写 QA
  - 若 atomic facts 超过 2 次仍失败：
    - 保留 QA
    - 使用 A/B-style safe fallback：直接用原始 `reference_answer` 生成 1 条 atomic fact
    - `atomic_fact_validation.used_safe_fallback=true`
    - 不得因为 atomic-fact 失败而回退到整题 QA rewrite

Stage 2 task-pack validation pipeline (all tasks):
- 统一顺序必须是：
  1. `Stage 1 state validation`
  2. `Stage 2 task-ground-truth resolution + task-pack build`
  3. `Stage 2 task-pack validation`
  4. rewrite / safe fallback / discard
- Task A:
  - pack build 前必须先做 current-state projection；对 transition-style state 只允许使用 `to`
  - `question_text` 与 `retrieval_query` 来自 deterministic cloze template，question quality 由程序模板保证
  - 若 `scoring_points[]` 中包含 LLM 生成的 `micro` atomic facts，则该 atomic-fact set 必须做 validation + rewrite
  - 若 generated atomic-fact set 多次 rewrite 后仍失败，safe fallback 必须直接使用原始 gold 句子本身，不能再改写或压缩原句
- Task B:
  - pack build 前必须先解析 task-level `before/after`
  - `question_text` 与 `retrieval_query` 来自 deterministic change template，question quality 由程序模板保证
  - `before_scoring_points[]` / `after_scoring_points[]` 中的 LLM 生成 `micro` atomic-fact set 必须做 validation + rewrite
  - `change_reason_scoring_points[]` 必须从 `reference_change_reason` 生成，并对该 gold reason text 做 validation + rewrite
  - `reference_change_reason` 必须直接来自 `state_observability.last_change_reason`
  - `reference_change_reason` 只有在 `state_questionability[state_key].change_reason_validation.is_valid=true` 时才可被消费
  - 若 `reference_change_reason` 缺失，或其 Stage 1 `change_reason_validation` 未通过，该 key 必须从 `change_tracking_pack` 过滤掉
  - 若 `before` / `after` / `change_reason` 的 generated atomic-fact set 多次 rewrite 后仍失败，safe fallback 必须直接使用对应原始 gold 句子本身，不能再改写或压缩原句
- Task C:
  - pack build 前必须复用 Task A 的 current-state projection；无法解析 current state 的 key 不得出题
  - 对 unchanged Task C state，必须默认复用前一个 checkpoint 的整套 apply pack：
    - `question`
    - `reference_answer`
    - `answer_scoring_points`
  - Task C 的 canonical reuse policy 为 `key_value_signature`
    - cache key 必须至少绑定：
      - `state_key`
      - `validated_state_value_signature`
      - `evidence_signature`
      - `prompt_version`
    - 若 `validated_state_value_signature` 与 `evidence_signature` 均未变化，则不得重新出题
  - `reuse_scope=none` 不属于正式协议路径；仅允许历史 artifact 兼容读取，新的正式 build/config 不得使用
  - LLM 只生成 QA item（`service_category` / `question` / `reference_answer`），不得在此阶段生成 rubric
  - QA item 必须先做 question-quality validation + rewrite
  - 只有 QA 通过后，才允许生成 `answer_scoring_points[]`
  - Task C atomic-fact generation 的 canonical grounding input 必须同时包含：
    - `state_value`
    - `question`
    - `reference_answer`
  - Task C atomic facts 不得只根据 `question` 或 `reference_answer` 的 surface wording 生成；必须受 user state 约束
  - Task C 的 atomic-fact generation / validation / rewrite / fallback 必须与 Task A / B 的 atomic-fact 协议对齐
  - TODO（future milestone）：
    - Task C `gold_memory_evidence_app_log_ids` 应升级为 QA-item-specific minimal evidence
    - 当前 milestone 不修改 evidence specificity；仍沿用现有 state-level evidence contract
- 任何 task 的 generated atomic-fact set 若多次 rewrite 后仍失败：
  - 必须回退到更保守的 safe rubric，或按 task-specific contract 直接 discard 当前 item

### 4.3 Hard Gates（Strict）
以下门槛任一不满足，pack 状态必须标记 `blocked`，不得进入主实验对比：
1. 每个 `questionable state` 至少保留 1 条有效 apply QA。
2. 全局 `discard_rate <= 30%`。
3. 每轮人工抽样 `>= 10 states`（eyeball set）。

人工审核推荐执行方式：
- 当目标是验证 Stage 1 + Stage 2 的细节质量，而不是只做轻量 eyeball 时，推荐使用分层 manual-review pack。
- 该 pack 应按“验证目的”拆分，而不是只输出一个混合样本文件，至少拆为：
  - `stage1_l1_fail`
  - `stage1_l2_fail`
  - `stage1_pass`
  - `stage2_state_completion`
  - `stage2_change_tracking`
  - `stage2_apply_accepted`
  - `stage2_apply_discarded`
- 每个 purpose-specific pack 应同时提供：
  - machine-readable sample set（JSON）
  - quick scan sheet（CSV）
  - fillable review sheet（CSV）
- 该分层 manual-review pack 的执行细节与产物路径由 runbook 维护。

## 5. Data Contracts (Benchmark / Prediction / Eval IO)

### 5.1 Benchmark Minimum Fields
- top-level `task_contract_version`
- top-level `research_frame_version`
- `checkpoints[].checkpoint_id`
- `checkpoints[].expected_snapshot_state`
- `checkpoints[].state_observability`
- `checkpoints[].state_questionability`
- `checkpoints[].validated_snapshot_state`
- `checkpoints[].state_completion_pack.keys[state_key].scoring_points`
- `checkpoints[].rq3_apply_service_qa.keys[state_key].items[*].answer_scoring_points`
- `checkpoints[].rq3_apply_service_qa.keys[state_key].items[*].gold_memory_evidence_app_log_ids`
- `taskabc_v1` additionally requires:
  - `checkpoints[].rq3_apply_service_qa.keys[state_key].items[*].service_category`
  - `checkpoints[].rq3_apply_service_qa.keys[state_key].items[*].question`
  - `checkpoints[].rq3_apply_service_qa.keys[state_key].items[*].reference_answer`
  - `checkpoints[].rq3_apply_service_qa.keys[state_key].items[*].qa_validation`
  - `checkpoints[].rq3_apply_service_qa.keys[state_key].items[*].atomic_fact_validation`
- `taskabc_v2` additionally requires:
  - `checkpoints[].rq3_apply_service_qa.keys[state_key].items[*].service_family`
  - `checkpoints[].rq3_apply_service_qa.keys[state_key].items[*].scenario`
  - `checkpoints[].rq3_apply_service_qa.keys[state_key].items[*].task_instruction`
  - `checkpoints[].rq3_apply_service_qa.keys[state_key].items[*].output_template`
  - `checkpoints[].rq3_apply_service_qa.keys[state_key].items[*].reference_output`
  - `checkpoints[].rq3_apply_service_qa.keys[state_key].items[*].item_validation`
  - `checkpoints[].rq3_apply_service_qa.keys[state_key].items[*].scoring_validation`
- `taskabc_v1` additionally requires:
  - `checkpoints[].change_tracking_pack.keys[state_key].before_scoring_points`
  - `checkpoints[].change_tracking_pack.keys[state_key].after_scoring_points`
  - `checkpoints[].change_tracking_pack.keys[state_key].reference_change_reason`
  - `checkpoints[].change_tracking_pack.keys[state_key].change_reason_scoring_points`

### 5.2 Prediction Minimum Fields
- top-level `task_contract_version`
- top-level `research_frame_version`
- `predictions[].checkpoint_id`
- `predictions[].snapshot_state`
- `predictions[].evidence`
- `predictions[].change_analysis`（若开启 change 任务）
- `predictions[].rq3_apply_answers`（若开启 apply 任务）
- `predictions[].metadata.concurrency_policy`
- `predictions[].metadata.requested_checkpoint_workers`
- `predictions[].metadata.requested_within_checkpoint_workers`
- `predictions[].metadata.effective_checkpoint_workers`
- `predictions[].metadata.effective_within_checkpoint_workers`

### 5.3 Eval Output Minimum Fields
- top-level `task_contract_version`
- top-level `research_frame_version`
- `summary`
- `checkpoints[]`
- `prediction_alignment`
- `summary.snapshot_point_score_mean_on_expected_mean`
- `summary.rq3_apply_answer_point_score_mean_mean`
- `checkpoints[].snapshot_slot_eval_by_key`
- `checkpoints[].rq3_apply_slot_eval_by_item`
- `taskabc_v1` additionally requires:
  - `summary.change_state_predict_point_score_mean_on_changed_mean`
  - `summary.change_reason_point_score_mean_on_changed_mean`
  - `checkpoints[].change_slot_eval_by_key`
- 注：
  - Task A continuity 指标仍保留 `value_f1`
  - under `taskabc_v2`, `RQ2` should be derived from Task A changed-vs-unchanged slices rather than from standalone Task B metrics
  - Task C vNext 不再要求 option-style summary 字段
  - TCE eval 不再要求 whole-state / whole-change `1..5` LLM judge 输出
  - judge prompt / raw output / request-level audit payload 不得写入 main eval JSON
  - judge prompt / raw output / request-level audit payload 只允许写入单独 audit artifact

## 6. Acceptance Gates

### 6.1 Document Consistency
1. 主手册中不允许保留任何草稿标记标签。
2. 本文档必须包含固定章节：
   - Purpose & Scope
   - Task Contracts
   - Validation Policy
   - Data Contracts
   - Acceptance Gates
3. `docs/runbooks/tce_execution_runbook.md` 存在且与本文互链。
4. `docs/protocols/tce_generation_and_adapter_contract.md` 存在且与本文互链。

### 6.2 Spec-Implementation Consistency Spot-Check
1. `state_questionability` 字段与本文契约一致。
2. `validated_snapshot_state` 与 `state_questionability[*].validated_state_value` 一致。
3. 若 `state_observability.last_change_reason` 存在，则 `state_questionability.change_reason_validation` 也必须存在。
4. `state_completion_pack/change_tracking_pack/rq3_apply_service_qa` 与本文契约一致。
5. `max_rewrites=2` 与本文一致。
6. point schema / metric names 与本文一致。

### 6.3 Minimal Executable Validation
1. 10-state pack 构建流程可执行（见 runbook）。
2. RAG 单 checkpoint 流程可执行（见 runbook）。
3. eval 输出包含预期指标簇（见 runbook）。
4. contributor-facing generation contract 与 runner/config/tests 一致（见 `docs/protocols/tce_generation_and_adapter_contract.md`）。

### 6.4 Stage-Wise Acceptance

Stage 1: State Validation
- Definition:
  - 输入：raw benchmark（`expected_snapshot_state` + `state_observability`）
  - 处理：L1 deterministic + L2 field-level LLM validation + optional `change_reason` metadata validation
  - 输出：`state_questionability`、`validated_snapshot_state`、`state_validation_summary`
  - 执行语义：
    - Stage 1 允许把部分完成结果直接落回 output artifact
    - 默认应支持“每处理固定数量 state 增量落盘一次”（当前默认 5）
    - resume 的唯一来源是上一次 Stage 1 输出文件本身，不额外维护独立进度文件
- Acceptance:
  1. 启动时必须能报告：
     - `pre_validate_count`
     - `after_l1_count`
     - `l1_filtered_count`
  2. 运行完成后每个 checkpoint 必须写出：
     - `state_questionability`
     - `validated_snapshot_state`
     - `state_validation_summary`
     - 若 `state_observability.last_change_reason` 存在，则对应 `state_questionability.change_reason_validation`
  3. `state_validation_summary` 至少包含：
     - `pre_validate_count`
     - `after_l1_count`
     - `after_l2_count`
     - `after_l1_l2_count`
     - `reused_count`
     - `computed_count`
  4. 必须产出：
     - `*_state_validate_counts.csv`
     - `*_state_validated_eyeball.csv`
  5. 复用规则必须满足：
     - evidence 变化 => 重新 validate
     - evidence 不变且 validated projection 不变 => 可复用
  6. 增量保存 / resume 必须满足：
     - output artifact 中已存在的 `state_questionability[state_key]` 在 resume 时必须跳过，不得重复请求 L2
       - 例外：若该 key 现在需要 `change_reason_validation`，但保存结果中缺失该 block，则允许只补算该 metadata validation
     - 已保存结果在后续 checkpoint 仍可进入复用缓存
     - 每次增量保存时必须同步刷新：
       - `validated_snapshot_state`
       - `state_validation_summary`
       - `*_state_validate_counts.csv`
       - `*_state_validated_eyeball.csv`
- Blockers:
  - 缺失 `validated_snapshot_state`
  - `state_questionability` 与 `validated_snapshot_state` 不一致
  - L2 批量失败导致无法生成可审阅结果

Stage 2: Prebuilt Task Packs
- Definition:
  - 输入：Stage 1 完成后的 validated benchmark
  - 处理：
    - Task A: 生成 `state_completion_pack`
    - Task B: 生成 `change_tracking_pack`
    - Task C: 生成 `rq3_apply_service_qa`
  - 输出：带 pack 的 benchmark
- Acceptance:
  1. Stage 2 输入必须已经包含：
     - `state_questionability`
     - `validated_snapshot_state`
  2. Task A pack 验收：
     - 仅从 `validated_snapshot_state` 出题
     - `state_completion_pack.keys[*].answer_template` 结构与 validated state 对齐
     - under `taskabc_v2`, `answer_template` / `scoring_points` 不得包含 `schedule_date` / `schedule_dates` / `priority`
     - `state_completion_pack.keys[*].scoring_points` 存在且 schema 合法
     - 若 `scoring_points` 中包含 LLM 生成的 `micro` atomic facts，则必须经过 point-set validation；失败时必须触发 rewrite 或 safe fallback
  3. Task B pack 验收（仅 `taskabc_v1`）：
     - 仅对相邻 checkpoint 的 validated state 交集且值变化的 key 出题
     - `change_tracking_pack.keys[*]` 不得包含“仅新出现/仅消失但不在交集”的 key
     - `before_scoring_points / after_scoring_points / change_reason_scoring_points` 均存在且 schema 合法
     - `reference_change_reason` 必须存在，且 `change_reason_scoring_points` 必须由该 gold reason text 生成
     - `reference_change_reason` 必须对应 Stage 1 已通过的 `change_reason_validation`
  4. Task C pack 验收：
     - 仅从 `validated_snapshot_state` 出题
     - 每个可问 state 至少保留 1 条有效 QA item
     - 每条 accepted item 含 `answer_scoring_points`
     - 每条 accepted item 含 `gold_memory_evidence_app_log_ids`
      - 每条 accepted item 含 `validation`
  5. Stage 2 产物必须包含：
     - `state_completion_pack`
     - `rq3_apply_service_qa`
     - `taskabc_v1` additionally includes `change_tracking_pack`
- Blockers:
  - 用 raw benchmark 直接构建 Stage 2
  - Task A/B scope 未对齐 validated state
  - apply QA `discard_rate > 30%`
  - 某个 `questionable state` 没有任何 accepted apply QA

Stage 3: Generation / Evaluation
- Definition:
  - 输入：Stage 2 benchmark + baseline prediction pipeline
  - 处理：
    - generation 优先消费 prebuilt packs
    - evaluation 优先按 pack scope 评测
  - 输出：
    - prediction artifact
    - eval artifact
- Acceptance:
  1. Generation:
     - Task A 优先消费 `state_completion_pack`
     - Task A 必须按 `state_key` 独立 retrieval + prompt + answer，不得把多个 state-completion keys 合并成一个 checkpoint-level retrieval/prompt
     - Task B 优先消费 `change_tracking_pack`
     - Task C 消费 `rq3_apply_service_qa`
  2. Prediction contract:
     - `snapshot_state`
     - `evidence`
     - `change_analysis`
     - `rq3_apply_answers`
  3. Evaluation contract:
     - `summary`
     - `checkpoints`
     - `prediction_alignment`
     - point-score metrics存在
  4. fallback 兼容：
     - benchmark 缺 pack 时，旧 benchmark 仍可运行
- Blockers:
  - generation 未优先消费 packs
  - evaluation scope 未切到 pack keys
  - prediction/eval contract 回退到 legacy 字段

## 7. Canonical References
默认读取顺序：
1. 本文档：`docs/protocols/temporal_checkpoint_evaluation_developer_manual.md`
2. 执行手册：`docs/runbooks/tce_execution_runbook.md`
3. Prompt/Retrieval checklist：`docs/protocols/tce_prompt_retrieval_checklist.md`
4. QA 协议补充：`docs/protocols/qa_generation_and_eval_contract.md`
5. 代码入口：
   - State validation: `data_construction/build_tce_state_validation.py`
   - Task pack build: `data_construction/build_tce_task_packs.py`
   - Apply-only wrapper: `data_construction/build_tce_rq3_apply_pack.py`
   - Generation: `tce_core/pipeline.py`
   - Evaluation: `eval/eval_tce.py`
6. Contributor-facing generation contract:
   - `docs/protocols/tce_generation_and_adapter_contract.md`

## 8. Change Control
- 任何影响任务定义、schema、metric 的改动必须先更新本手册。
- 文档更新后再改实现，再补测试与验收记录。
- 若文档与代码冲突，以“已合并代码 + 最新手册修订”仲裁，并在 24h 内补齐一致性。

Historical artifact note:
- `results*/` 与 `generated_outputs/` 下允许保留旧命名历史产物，不作为当前协议入口。
