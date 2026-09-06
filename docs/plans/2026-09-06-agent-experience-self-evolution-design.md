# Frozen Agent 外部经验自进化设计

## 状态

- 文档状态：Implementation authorized
- 日期：2026-09-06
- 当前阶段：核心代码已实现，尚未执行模型、环境、测试或实验
- 正式运行状态：未启用

本文描述并冻结第二条实验路线。仓库允许以下两条可审计工作流：

```text
Baseline → SFT → GRPO → Evaluation
Frozen Actor → Offline Experience Distillation → Experience-Augmented Agent Loop
             → Offline Experience Evolution → Evaluation
```

本次实现不授权启动训练、合并模型或运行 Final-200 Clean 全量评测。

## 1. 背景与研究问题

当前项目通过 LoRA SFT 建立购物工具使用能力，再通过在线 GRPO 优化约束满足。
候选新路线不继续更新模型参数，而是把 Teacher 和 Agent 自身执行轨迹沉淀为
外部程序性经验，在推理期间按任务和状态检索、临时注入。

自进化对象因此不是模型权重，而是：

- 结构化经验内容；
- 经验适用条件和召回策略；
- 经验的证据、效用统计与生命周期；
- 一次运行采用的冻结经验库快照。

本路线需要回答两个不同命题：

1. **替代性命题**：Teacher 经验能否让 Frozen Raw Base 获得一部分原本由
   SFT/GRPO 带来的能力。
2. **持续优化命题**：在 Frozen SFT 已掌握工具协议的前提下，能否不再更新参数，
   仅通过外部经验实现稳定、可审计、可回滚的持续改进。

主实验采用 Frozen SFT，优先验证第二个命题。Frozen Raw Base 作为探索性对照，
不预设经验能够补足底层工具格式、合法动作和终止能力。

## 2. 设计目标与非目标

### 2.1 目标

- 从真实 Teacher/Agent trajectory 中提炼任务无关、可执行的短经验。
- 在不修改 ShopSimulator 工具协议的情况下接入推理期 Agent Loop。
- 依据当前任务、页面阶段和失败信号进行事件驱动召回。
- 保证经验注入不暴露 Gold 商品、隐藏环境字段或评测答案。
- 完整记录“为什么召回、注入了什么、消耗多少 token、产生什么效果”。
- 使用 Rule Gate、Reward v3 和 Rubric/Judge 对候选经验分层验证。
- 经验晋升必须通过配对开发集实验，并支持回滚和复现。
- Final-200 Clean 与经验抽取、调参、晋升完全隔离。

### 2.2 非目标

第一版明确不做：

- 更新 Actor、Reward Model 或检索模型的参数；
- 把完整 Teacher trajectory 直接作为正式经验库内容；
- 给购物 Actor 增加 Bash、CLI、MCP 或 Recall 工具；
- 让 Actor 自主决定是否调用经验检索；
- 在一次正式运行中同步修改 Active Experience Store；
- 使用 Final-200 Clean 生成、选择或调优经验；
- 用 LLM Judge 分数替代 Reward v3 的确定性终局判定；
- 自动把未经开发集验证的反思写入正式经验库。

Skill、CLI 和 MCP 可以在核心机制验证后作为分发层增加，但不属于第一阶段的
实验创新点。

## 3. 当前仓库基础与限制

### 3.1 可复用能力

现有实现已经具备：

- `evaluation/rollout.py`：完整的模型—工具—环境推理循环和原始轨迹记录；
- `evaluation/trajectory.py`：将原始轨迹规范化为稳定的 Actor-visible 事件流；
- `environment/actions.py`：动作合法性与 Action Guard；
- `environment/projection.py`：不破坏可操作目标的 observation 投影；
- `environment/context.py`：按完整工具调用组进行上下文压缩；
- Reward v3：确定性终局类型、硬门槛、证据覆盖和约束分项；
- Rubric/Judge：对搜索、候选利用、证据核验、决策和终止过程进行诊断；
- `evaluation/artifacts.py`：JSON/JSONL 原子写入和 fail-closed 校验；
- `evaluation/manifest.py`：运行版本、数据哈希和产物 Schema 的可复现清单。

经验系统应复用这些边界，不重新实现第二套环境客户端、Reward 或评测协议。

经验注入启用时必须同时启用上下文压缩。新路线采用两层策略：固定 system prompt、
原始 query、最新完整工具交互和当前经验块原样保留；更早的完整交互组交给固定配置的
DeepSeek V4 Flash 生成带证据引用的结构化摘要。摘要请求使用 temperature=0、
top_p=1、thinking=false，响应经过严格 Schema、event ID 和逐字 evidence quote
校验，并按输入 hash 缓存。相同输入优先复用缓存，保证恢复运行时结果可控、可复现。

### 3.2 可用的 Teacher 经验源

`data/sft_pure_v4/all.jsonl` 当前有 1,192 条 DeepSeek V4 Flash 轨迹：

- task ID 唯一；
- 与 Final-200 Clean 零重叠；
- 均通过 `gold_purchase` 选择条件；
- 包含 user、assistant tool call 和 Actor-visible tool observation；
- 属于 SFT 清洗后的 action-only transcript。

这些行不再包含原始 `steps`、完整 `terminal_result`、失败采集记录和逐条 Reward
审计信息。因此它们可以支持第一版成功策略抽取，但不能单独支持失败因果分析。
来源标签必须写成 `curated_teacher_gold`，不能伪装成仍然保有完整 Reward 证据的
raw rollout。

### 3.3 正负样本限制

Teacher Gold 只能说明某种行为曾与成功共同出现，不能证明：

```text
不采用该行为一定失败
```

以下经验必须等待新的 Agent rollout 或找回原始失败轨迹后再生成：

- 错误购买的关键成因；
- 过早放弃与最大步数耗尽的分叉点；
- Guard rejection 后的恢复策略；
- 两种冲突策略之间的因果优劣；
- 某条经验对成功率的真实增量。

## 4. 总体架构

系统分成三个运行层：

| 层 | 输入 | 输出 | 运行方式 |
|---|---|---|---|
| 经验沉淀 | Teacher/Agent trajectory | Candidate Experience | 离线异步 |
| 经验使用 | 当前任务与 Actor-visible 状态 | Experience Bundle | 在线同步、只读 |
| 经验进化 | 新轨迹与候选经验 | 新 Active Store 快照 | 离线周期性 |

完整数据流：

```text
Teacher Gold Transcript ─┐
                         ├─→ 规范化/阶段切分
Agent Success/Failure ───┘          │
                                    ↓
                        Rule + Reward + Judge 诊断
                                    │
                                    ↓
                         多轨迹比较与经验抽取
                                    │
                                    ↓
                         Candidate Experience Store
                                    │
                         去重 / 冲突 / 定向 A/B
                                    │
                                    ↓
                           Active Store Snapshot
                                    │
            ┌───────────────────────┘
            ↓
Task/State → Recall Event → 检索 → 临时注入 → Frozen Agent Loop
                                                   │
                                                   ↓
                                             新完整轨迹
```

正式运行加载一个不可变的 Active Store 快照。运行过程中只追加 telemetry，不修改
快照；新经验只能在运行结束后的离线进化阶段产生。

## 5. 数据隔离与任务划分

### 5.1 数据角色

| 数据 | 角色 | 是否允许生成经验 | 是否允许调参 |
|---|---|---:|---:|
| `data/sft_pure_v4/all.jsonl` | Teacher seed | 是 | 否 |
| GRPO train task pool | Agent 经验发现 | 是 | 是，仅发现规则 |
| GRPO validation task pool | Store 回归集 | 否 | 否 |
| Final-200 Clean | 最终评测 | 否 | 否 |

实现阶段应从现有 GRPO train task pool 通过固定 seed 生成两个只含 task ID 的清单：

- `experience_discovery`：用于产生成功/失败 rollout 和候选经验；
- `experience_candidate_dev`：用于候选级定向 A/B，永远不参与经验抽取。

现有 GRPO validation 保留为经验库级回归集。划分清单只引用现有任务，不复制或
改写任务内容。Manifest 必须记录 seed、任务数、task-ID hash、源文件 hash，以及
与 SFT、GRPO validation、Final-200 Clean 的交集检查结果。

### 5.2 Final-200 隔离

Final-200 Clean 的禁止事项包括：

- 不用于生成 query embedding 的统计归一化或词表调优；
- 不用于选择 Top-K、阈值、注入 token 预算；
- 不用于候选经验晋升或冲突裁决；
- 不在评测后把其 trajectory 反写到经验库；
- 不依据其 bad case 手工改写经验。

最终评测开始前必须冻结 Actor、Experience Store、Retriever 配置、Prompt、环境、
Reward 和随机种子清单，并写入 run manifest。

## 6. 轨迹规范化与决策窗口

### 6.1 阶段切分

经验抽取不直接对整条 trajectory 做一次总结。规范化事件流先划分为：

```text
task_understanding
search
candidate_screening
detail_verification
option_selection
pre_purchase
termination
error_recovery
```

阶段由工具名、页面类型、页面迁移、selected option、Guard event 和终局状态
确定，不让抽取模型自由发明阶段。

### 6.2 决策窗口

围绕关键动作构造局部窗口：

```text
任务的公共约束摘要
+ 前两个 Actor-visible observation/event
+ 当前 assistant decision/action
+ 当前 tool result
+ 后两个 action/event
+ 终局 Rule/Reward 标签
+ 可选的对比轨迹窗口
```

窗口只能包含 Actor 当时可见的 observation。`audit_only_raw_observation`、隐藏目标、
Gold ASIN、私有 task facts 不得进入经验抽取 Prompt 的行为证据区。

### 6.3 配对比较

高价值失败经验优先来自同一任务的配对：

```text
Frozen Agent failure vs Frozen Agent success
Frozen Agent failure vs Teacher success
Without-experience failure vs With-experience success
```

比较器定位第一处具有行为意义的分叉，而不是从终局向前随意挑选一个动作。分叉
必须关联到可执行的未来条件，例如“多规格商品购买前未重新核验 variant 价格”。

## 7. Experience Card 合同

### 7.1 Card Schema

每行经验使用以下逻辑结构：

```json
{
  "schema_version": "shopping-experience-card-v1",
  "experience_id": "exp-verify-option-price-001",
  "revision": 1,
  "status": "candidate",
  "type": "verification",
  "phase": "pre_purchase",
  "trigger": {
    "events": ["pre_purchase"],
    "page_types": ["product_detail"],
    "required_constraint_tags": ["budget"],
    "predicates": ["has_multiple_options", "final_price_unverified"]
  },
  "guidance": [
    "先选定影响价格的目标规格，再以完整 variant 的当前价格判断预算。"
  ],
  "anti_patterns": [
    "不要把搜索卡片的价格直接当作最终规格价格。"
  ],
  "verification_checks": [
    "购买前已观察到选定规格对应的当前价格。"
  ],
  "scope": {
    "environment_version": "shopsimulator-environment-v2.1",
    "reward_version": "shopsimulator-reward-v3",
    "observation_version": "shopping-observation-v2",
    "tool_version": "shopping-tools-v2",
    "categories": []
  },
  "evidence": {
    "source_kind": "paired_agent_rollouts",
    "supporting_trajectory_ids": [],
    "contradicting_trajectory_ids": [],
    "support_count": 0
  },
  "provenance": {
    "extractor": "...",
    "extractor_revision": "...",
    "prompt_version": "experience-distill-v1",
    "created_at": "..."
  },
  "supersedes": null,
  "content_hash": "..."
}
```

### 7.2 枚举约束

`type` 第一版只允许：

- `search_strategy`
- `candidate_comparison`
- `verification`
- `recovery`
- `termination`

`phase` 和 `trigger.events` 必须来自本文冻结枚举。自由文本只允许出现在 guidance、
anti-pattern、verification check 和 evidence 引用中。

### 7.3 内容约束

每条经验必须：

- 描述“何时适用、该做什么、如何确认完成”；
- 使用短的祈使句，最多三条 guidance；
- 不包含 ASIN、商品标题、具体任务答案或 Gold 字段；
- 不把 Reviews 当作型号、官方功能、规格或价格的确定证据；
- 不放宽当前系统 Prompt、Action Guard、工具 Schema 和预算门槛；
- 不声称未被 supporting trajectory 支持的因果关系；
- 对没有反例分析的 Teacher-only 经验保留较低证据等级。

### 7.4 哈希与可变统计分离

`content_hash` 对以下语义字段的 canonical JSON 计算 SHA-256：

```text
type, phase, trigger, guidance, anti_patterns, verification_checks, scope
```

不把 status、时间、使用次数和效果统计放入 `content_hash`。运行期 utility 写入独立
telemetry，避免“被召回一次”就改变经验内容哈希。

Active Store Manifest 另行记录完整经验文件 SHA-256、经验 ID/revision 集合、索引
后端和索引哈希。

## 8. Experience State View

Retriever 不能直接读取环境内部对象，而是接收最小化的只读状态：

```python
class ExperienceStateView:
    task_id: int
    public_query: str
    constraint_tags: tuple[str, ...]
    phase: str
    page_type: str
    current_product_id: str | None
    seen_product_ids: tuple[str, ...]
    candidate_set_hash: str
    unverified_constraints: tuple[str, ...]
    selected_options: tuple[str, ...]
    last_tool_name: str | None
    last_guard_reason: str | None
    repeat_action_count: int
    executed_steps: int
    remaining_steps: int
```

允许字段只能从 query、Actor-visible history、Action Guard 公共错误和执行计数导出。

明确禁止读取：

- target/gold ASIN 和目标 variant；
- 环境隐藏字段和私有 task facts；
- 未投影的 audit raw observation；
- 当前未结束任务的 Reward 类型、分项或未来结果；
- 当前任务 Judge 结论；
- Final-200 标签或历史 bad-case 人工答案。

未能从可见证据确定的状态必须标为 unknown，不能用 Gold 数据补齐。

## 9. Recall Event 与状态签名

第一版只支持六类事件：

| Event | 触发条件 | 去重键 |
|---|---|---|
| `task_start` | reset 后、第一次模型生成前 | task ID |
| `search_stagnation` | 搜索未产生新候选，或近期出现重复/近重复搜索 | query 与 candidate-set hash |
| `candidate_opened` | 首次进入一个新商品详情页 | product ID |
| `pre_purchase` | 当前详情页可购买，且规格/证据状态发生变化 | product + selected-options hash |
| `guard_rejection` | 最近一次调用被 Action Guard 拒绝 | guard reason + page signature |
| `pre_finish` | 可合法结束且探索停滞，或剩余步数进入终止区间 | exploration signature |

一次事件只在对应状态签名变化后重新检索。否则复用当前 Experience Bundle，避免
每一步重复检索、经验抖动和 Prompt Cache 损耗。

`pre_purchase` 和 `pre_finish` 是状态触发，不要求提前知道模型下一步一定会调用
`buy_now` 或 `finish_without_purchase`。

## 10. 检索合同

### 10.1 Provider 接口

购物 Actor 不新增工具。Harness 在模型生成前同步调用：

```python
class ExperienceProvider(Protocol):
    def retrieve(
        self,
        *,
        task: Mapping,
        state: ExperienceStateView,
        event: str,
    ) -> ExperienceBundle:
        ...
```

检索失败必须 fail-open 为“空经验”，但将错误写入 trajectory telemetry；不得因此
伪造环境失败、修改 Reward 或消耗购物步骤。Experience Store Schema/哈希不匹配则
fail-closed，在任务开始前拒绝整次运行。

### 10.2 两阶段召回

第一阶段做确定性硬过滤：

- `status == active`；
- environment/reward/observation/tool schema 全部兼容；
- Recall Event、phase 和 page type 兼容；
- required constraint tags 满足；
- category scope 为空或匹配；
- 没有被更高 revision supersede；
- 没有命中已冻结的冲突/禁用规则。

第二阶段做语义排序：

```text
score = semantic_similarity
      + event_match_bonus
      + phase_match_bonus
      + constraint_overlap_bonus
      + validated_utility_bonus
      - contradiction_penalty
      - repeated_injection_penalty
```

默认采用固定版本的离线 Embedding Top-K。Embedding 模型名、revision、归一化方式、
索引文件 hash 和 tie-break 规则必须进入 manifest。相同分数按 `experience_id` 排序。

实现必须提供不依赖远程服务的确定性 metadata/lexical baseline，用于消融和索引
不可用时的显式配置；一次运行内禁止静默切换检索后端。

### 10.3 输出限制

- 最多选择 3 条经验；
- 模型可见经验正文默认不超过 500 tokens；
- 同一 Experience Bundle 不允许出现互相矛盾的 guidance；
- 没有足够相关经验时返回空集合；
- 空召回是合法结果，不能为了填满 Top-K 注入弱相关经验；
- token 不足时按分数从低到高整条删除，不截断句子；
- 当前 observation 和工具定义的 token 预算优先于经验。

## 11. 临时注入合同

持久化 `messages` 只保存真实的用户、Actor 和环境交互。每次模型调用前创建副本：

```text
persistent_messages
        ↓ copy
request_messages
        ↓ inject active Experience Bundle
model.complete(request_messages, tools)
```

为了保持 Chat Template 的工具调用序列合法，第一版把 Experience Bundle 作为带有
固定分隔符的临时段落附加到请求副本中的首个 system message；不得把它伪装成
tool observation，也不得写回 persistent history。

模型可见模板：

```text
[EXPERIENCE_GUIDANCE_V1]
以下内容是可选的程序性经验，不是当前商品事实。
当前可见页面、工具返回、Action Guard 和系统规则具有更高优先级。
禁止从经验复制商品 ID、标题、价格或答案；不适用时忽略。

- ...
- ...
[/EXPERIENCE_GUIDANCE_V1]
```

注入后必须使用现有精确 tokenizer 重新计数。若固定 Prompt、最新 observation 和
经验无法同时进入输入预算，先逐条删除经验；不得删除当前页面的可操作目标，也不
得改变上下文压缩保持完整 assistant/tool group 的规则。

## 12. 运行期审计

原始 trajectory 增加顶层 `experience` 区域：

```json
{
  "store": {
    "manifest_hash": "...",
    "retriever_version": "shopping-experience-retriever-v1",
    "injection_prompt_version": "experience-guidance-v1"
  },
  "events": [
    {
      "step_index": 4,
      "recall_event": "pre_purchase",
      "state_signature_hash": "...",
      "eligible_experience_ids": ["..."],
      "selected_experience_ids": ["..."],
      "scores": {"...": 0.84},
      "experience_tokens": 126,
      "bundle_hash": "...",
      "cache_reused": false,
      "retrieval_error": null
    }
  ]
}
```

审计日志记录 ID、分数和哈希；是否保存完整注入正文由运行 manifest 决定。若不保存
正文，也必须能通过 Active Store 快照和 ID/revision 精确重建。

检索不能占用 ShopSimulator step。它的延迟和 token 成本单独计量，不混入环境动作数。

## 13. 经验抽取与冲突处理

### 13.1 Teacher Seed 抽取

第一阶段从 1,192 条 Teacher Gold transcript 中抽取 50～150 条候选经验：

1. 确定性阶段切分和局部窗口生成；
2. 按 phase、constraint tags 和行为模式聚类；
3. 抽取模型生成严格 JSON Candidate Card；
4. Schema、泄漏词、ASIN、具体价格和工具规则检查；
5. 语义去重并合并 supporting trajectory IDs；
6. 人工或独立审计通过后进入 `candidate`。

Teacher-only Card 的 `anti_patterns` 只能来自轨迹中直接观察到的无效尝试与恢复，
不能凭常识反推。没有对比证据时允许为空。

### 13.2 Agent 自身经验

后续 Agent rollout 保留完整 Rule/Reward 结果。经验抽取输入分层提供：

```text
Task public query
+ Actor-visible decision windows
+ deterministic Rule diagnostics
+ terminal Reward v3 result
+ Judge process diagnosis
+ paired comparison window
+ potentially conflicting active experiences
```

Reward 和 Judge 只用于终局后的离线诊断，不得出现在未来运行的经验正文中。

### 13.3 去重与冲突

候选经验进入验证前执行：

- 完全 content hash 去重；
- 同 trigger/scope 下的语义近重复合并；
- guidance 与 anti-pattern 交叉冲突检测；
- 与系统 Prompt、工具 Schema、Action Guard 的规则冲突检查；
- supporting/contradicting evidence 计数；
- category-specific 规则向全局 scope 提升时重新验证。

不能自动解决的冲突保持 `candidate` 并标记 review，不允许同时进入 Active Store。

## 14. 生命周期与晋升门控

经验状态机：

```text
candidate
   │ targeted validation passed
   ↓
validated
   │ full store regression passed
   ↓
active ──→ deprecated
   │          ↑
   └ revision/supersede

candidate/validated ── failed validation ──→ rejected
```

状态变化生成新的 revision 或 promotion record，不覆盖历史证据。

### 14.1 候选级验证

默认验证合同：

- 从 `experience_candidate_dev` 选择 20～50 个符合 trigger 的任务；
- 相同 Frozen Actor、Prompt、工具和环境；
- 有经验/无经验使用相同 sampling seed 配对；
- 随机采样实验默认每任务至少 3 个 seed；确定性实验可以每任务 1 次；
- 目标失败模式显著下降；
- strict success 的配对均值提高，配对 bootstrap 95% CI 下界不小于 0；
- `wrong_purchase` 和 `reward_unverifiable` 的数量均不得增加；
- Guard rejection、repeat loop、max steps 不出现实质性恶化；
- 平均注入不超过 3 条和 500 tokens。

实际阈值必须在查看结果前写入 validation manifest，禁止事后移动门槛。

### 14.2 经验库级回归

把 validated Card 加入完整 Store 后，在冻结的 GRPO validation task pool 上比较
旧/新快照：

- overall strict success 不低于预先声明的 non-inferiority margin；
- 目标类别 strict success 提升；
- wrong purchase、reward unverifiable、基础设施无效轨迹均不增加；
- 无关类别没有集中退化；
- 平均步骤、Guard rejection、检索延迟和 token/成功任务在预算内；
- 所有 Prompt、Store、Retriever 和任务哈希可复现。

只有候选级和 Store 级门控都通过，经验才能进入新的 Active Store 快照。

## 15. Rule、Reward 与 Judge 的职责

三层信号不可混用：

### 15.1 Rule Gate

负责确定性过程事实：

- 工具和参数是否合法；
- Action Guard 是否拒绝；
- 是否重复动作或搜索；
- 页面状态迁移是否合法；
- 是否发生上下文或 observation 投影错误；
- 是否正常结束。

### 15.2 Reward v3

负责终局效用和严格成功：

- `gold_purchase` 且 `reward_valid=true` 才是 strict success；
- wrong/partial/alternative purchase、early abstain、repeat loop、max steps、
  reward unverifiable 分栏保留；
- 基础设施无效轨迹不产生伪学习结论。

### 15.3 Rubric/Judge

负责解释行为过程：搜索、候选利用、证据核验、决策质量和终止效率。Judge 可以
帮助定位经验目标，但不能单独决定经验晋升，也不能覆盖确定性 Reward 结果。

## 16. 目录与产物规划

当前代码布局如下；`data/experience/` 与 `outputs/experience/` 中的运行产物尚未生成：

```text
src/shopping_grpo/experience/
  contracts.py          Experience Card/Bundle/State View 严格校验
  segmentation.py       轨迹阶段和决策窗口
  distillation.py       候选经验抽取输入与输出
  store.py              快照加载、哈希、版本和冲突检查
  retrieval.py          metadata filter 与可插拔排序
  injection.py          临时 Prompt 注入与 token 预算
  events.py             Recall Event 和状态签名
  runtime.py            trajectory-local 召回缓存与审计
  compaction.py         DSV4 Flash 证据化上下文压缩
  validation.py         配对结果与晋升门控
  promotion.py          两级验证后的正式晋升
  task_split.py         固定 seed 的 discovery/dev 划分

data/experience/
  seed.jsonl            审计通过的 Teacher seed 经验
  active.jsonl          当前冻结 Active Store
  manifest.json         Schema、来源、哈希和兼容范围

configs/
  experience.json       检索、Top-K、token、DSV4 与事件配置
  experience_validation.json  冻结候选/Store 验证门槛

outputs/experience/<run_id>/
  manifest.json
  source_windows.jsonl
  candidate_experiences.jsonl
  retrieval_events.jsonl
  trajectories.jsonl
  validation_results.jsonl
  promotion_decisions.jsonl
  experience_utility.json
  summary.json
```

JSONL 是经验事实源。Embedding/SQLite 索引是由 manifest 固定配置后生成的可重建
产物，不作为唯一事实源提交。

## 17. 实验矩阵

主实验：

| ID | Actor | Experience Store | 目的 |
|---|---|---|---|
| E0 | Frozen Raw Base | none | Raw Base 基线 |
| E1 | Frozen Raw Base | Teacher seed | 测试对 SFT 的部分替代性 |
| E2 | Frozen SFT | none | 主基线 |
| E3 | Frozen SFT | Teacher seed | 验证外部经验增益 |
| E4 | Frozen SFT | Teacher + self-evolved | 验证持续自进化 |

关键消融：

| Ablation | 对照 | 回答的问题 |
|---|---|---|
| A1 | 相似 Teacher 原轨迹 few-shot | 抽象 Card 是否优于完整轨迹 |
| A2 | 只按 task query 召回 | 状态化 Recall Event 是否必要 |
| A3 | 随机/错配经验 | 增益是否只是 Prompt 变长 |
| A4 | 静态全部经验 | Top-K 检索是否减少冲突和 token |
| A5 | 无 anti-pattern/verification | Card 各字段是否有实际贡献 |

所有主比较必须使用相同 Actor checkpoint、采样参数、任务顺序、seed、环境版本和
最大步数。

## 18. 指标

### 18.1 执行结果

- Reward v3 strict success；
- purchase success 和各 reward type；
- done rate；
- wrong purchase / reward unverifiable；
- Guard rejection；
- repeat action/search、max steps；
- 平均执行步骤；
- 五维 Judge 分数，但不计算综合总分。

### 18.2 经验系统

- Recall Precision@K；
- 空召回正确率；
- 平均 eligible/selected 数；
- 冲突经验率；
- 各 Recall Event 分布；
- 平均经验 token 和总输入 token；
- 检索 P50/P95 延迟；
- 单条经验 targeted success delta；
- 单条经验 guard/wrong-purchase delta；
- token/strict-success；
- Store revision 的净胜/负任务数。

Precision@K 需要独立的任务—经验相关性标注或冻结规则，不能用 Retriever 自己的
分数定义“相关”。

## 19. 失败模式与保护措施

| 风险 | 保护措施 |
|---|---|
| Prompt stuffing | Top-3、500-token 硬限制，允许空召回 |
| Gold/评测泄漏 | State View 白名单、ASIN/答案扫描、Final-200 blind guard |
| 成功样本幸存者偏差 | Teacher Card 降低证据等级，失败规则等待配对轨迹 |
| 错误反思自我强化 | candidate/validated/active 两级 A/B 门控 |
| 经验互相冲突 | 同 trigger/scope 冲突检测，未解决不得 active |
| 商品事实过时 | Card 只保留程序性策略，不保存商品事实 |
| 检索抖动 | Recall Event + state signature 缓存 |
| 上下文挤占 observation | 精确计数，先删除低分经验，永不裁剪动作目标 |
| 无法归因增益 | 配对实验、注入事件日志、随机错配消融 |
| Store 无限增长 | 语义去重、utility 衰减、supersede/deprecate |
| 评测后污染 | 正式评测 Store 只读，Final-200 trajectory 禁止回流 |
| 索引不可复现 | 固定模型 revision、索引 hash、tie-break 和 backend |

## 20. 分阶段实施计划与验收

### Phase 0：合同冻结

产出：

- 本文评审通过；
- 明确是否更新仓库单工作流合同；
- 冻结 Experience Card、State View、Recall Event 和 Manifest Schema；
- 冻结任务划分和 Final-200 隔离规则。

验收：只做静态 Schema/泄漏测试，不调用模型或环境。

### Phase 1：Teacher Seed

产出：

- 决策窗口生成器；
- 50～150 条 Candidate Card；
- 去重、冲突和泄漏报告；
- 经审计的 seed/active 快照。

验收：所有 Card 可追溯到源 trajectory；无 ASIN、具体答案、Reward/Gold 文本泄漏；
同输入可重建相同快照哈希。

### Phase 2：只读检索与临时注入

产出：

- ExperienceProvider；
- 六种 Recall Event；
- Top-K 和 token 控制；
- trajectory experience telemetry；
- Experience disabled 时与原 Harness 行为等价。

验收：不改变工具 Schema，不消耗环境步骤，不污染 persistent history；并发 trajectory
不共享可变检索状态；空经验和检索错误路径可测试。

### Phase 3：Teacher Experience 有效性

产出 E2/E3 和 A1～A4 开发集结果。

验收：在预先冻结的门槛下确认经验是否带来可重复净增益。未通过时停止，不进入
自动自进化。

### Phase 4：Agent 自身经验与晋升

产出：

- 成功/失败配对；
- Candidate → Validated → Active 流程；
- promotion decisions 和 Store rollback；
- E4 开发集结果。

验收：新 Store 通过候选级定向验证和完整回归，且能根据 manifest 回放晋升决策。

### Phase 5：冻结最终评测

只有用户显式授权后才能执行 Final-200 Clean。运行前打印并保存：

- Actor checkpoint hash；
- Experience Store/Index hash；
- Retriever 和 injection prompt version；
- task dataset hash；
- Environment/Reward/observation/tool contract；
- sampling config 和 seeds。

最终结果同时保留完整 trajectories、retrieval events、Judge requests/results、逐任务
评估和汇总，不再只提交无法反算的静态均值。

## 21. 第一版最终范围

第一版以验证以下最小闭环为结束条件：

```text
Teacher Gold Transcript
Agent/Teacher Multi-rollout ─→ 多轨迹联合分析
                              ↓
Teacher Gold Transcript ────→ 结构化 Experience Card
→ Frozen SFT 的事件驱动召回
→ 临时注入
→ DSV4 Flash 证据化上下文压缩
→ 完整审计轨迹
→ 开发集有/无经验配对比较
```

第一版成功不等于完成“自进化”。只有 Agent 新轨迹能够产生候选经验，并经过两级
验证形成新的 Active Store 快照后，才宣称系统具备外部经验自进化能力。

## 22. 已实现代码与配置

本次实现只落代码和静态合同，没有启动 ShopSimulator、模型 API、测试、训练或评测。

### 22.1 在线 Agent Loop

- `src/shopping_grpo/experience/contracts.py`：Experience Card、State View、Bundle 与
  严格字段/泄漏/哈希验证；未声明字段一律拒绝。
- `src/shopping_grpo/experience/store.py`：按文件哈希和行数加载不可变 Active Store。
- `src/shopping_grpo/experience/events.py`：八阶段状态推导与六类 Recall Event。
- `src/shopping_grpo/experience/retrieval.py`：确定性硬过滤、lexical/embedding 排序、
  稳定 tie-break 和 Top-K。
- `src/shopping_grpo/experience/injection.py`：把经验临时注入请求副本，按 Actor 的精确
  tokenizer 计算增量 token；persistent history 不变。
- `src/shopping_grpo/experience/runtime.py`：trajectory-local 状态签名缓存、召回与完整
  experience telemetry。
- `src/shopping_grpo/evaluation/rollout.py`：在现有 Harness 的每次模型生成前接入上述
  Runtime；未传 `--experience-config` 时保留原路径。

### 22.2 DSV4 Flash 上下文压缩

- `src/shopping_grpo/experience/compaction.py` 只压缩旧的完整 assistant/tool groups。
- 固定 system prompt、原始 query、当前 Experience Guidance、最近完整交互均逐字保留。
- 摘要必须逐项回传 source event ID，并为每个事实提供源事件中的连续逐字 quote；
  Schema、引用、商品 ID 与隐藏字段检查失败时 fail-closed。
- 请求固定 `temperature=0`、`top_p=1`、`thinking=false`；缓存键绑定模型名、
  `model_revision`、prompt version、source hash 和记录上限。
- append-only cache 保存首个有效结果，恢复运行时相同输入直接复用；trajectory 记录
  source/summary hash、被压缩事件、token 变化与 provider metadata。

配置位于 `configs/experience.json`。代码只读取下列环境变量名，不在配置或 manifest
中写入真实凭证：

```text
EXPERIENCE_DSV4_BASE_URL
EXPERIENCE_DSV4_API_KEY
EXPERIENCE_COMPACTION_CACHE   # 可选，覆盖默认 cache 路径
```

若 provider 提供不可漂移的部署 revision，应把 `model_revision` 改成该精确 revision，
并用稳定的 `provider_id` 标识具体服务方；二者只进入审计与缓存键，不作为 API
凭证。

### 22.3 多 trajectory 经验沉淀

- `scripts/prepare_experience_tasks.py`：固定 seed 产生互斥的 discovery/candidate-dev
  task 清单，并记录 GRPO validation、Final-200 和 SFT overlap。
- `scripts/collect_experience_rollouts.py`：同一任务可配置多次 Teacher 或 Student
  rollout；每个 attempt 使用显式 sampling seed，同一 seed 可用于有/无经验配对；
  JSONL 可恢复追加，manifest 记录 Actor revision、采样参数与输出哈希。
- `src/shopping_grpo/experience/segmentation.py`：按冻结阶段构造 `before=2/after=2`
  Actor-visible 决策窗口。
- `src/shopping_grpo/experience/distillation.py`：优先把同 task/同 query 的 Teacher
  success、Student success/failure 与重复 Student rollout 放入同一分析组，再补充相似
  query；候选经验必须引用组内真实 trajectory ID。
- `scripts/distill_experiences.py`：支持 action-only Teacher Gold、带真实终局的 raw
  Teacher 和 Student rollout；输出 Candidate Card、联合分析 audit 与 manifest。
- `src/shopping_grpo/experience/validation.py` 与
  `scripts/validate_experience_results.py`：按 task + sampling seed 对齐候选级和 Store 级
  A/B，使用冻结 bootstrap/安全/token 门槛生成 validation result 与 promotion decision；
  默认门槛位于 `configs/experience_validation.json`。
- `scripts/build_experience_validation_store.py`：把指定 Candidate 临时投影为
  `validation_only` Store 以解除“未 active 就无法测试”的环形依赖；正式配置默认拒绝
  加载该角色，验证运行必须在配置副本中显式设置 `allow_validation_only=true`。
- `scripts/promote_experience_store.py`：只有候选定向验证和 Store 回归两个 gate 都通过，
  才能构建新的只读 Active Store 快照。

### 22.4 尚未发生的事项

- `data/experience/` 下的任务划分、rollout、候选经验和 Active Store 均需显式执行后
  才会产生；仓库当前没有伪造这些实验产物。
- 单元测试文件已补充，但依照本次要求未运行，因此不能把它们表述为“已通过”。
- 正式 E0～E4、消融、候选晋升和 Final-200 结果仍为空；Final-200 继续要求用户另行
  显式授权。
