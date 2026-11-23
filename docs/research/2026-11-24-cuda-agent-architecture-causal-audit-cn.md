# 2026-11-24｜CUDA Coding Agent 逐步架构、真实失败归因与 DeepSeek API 核查

> 依据 `feat/gpu-native-coding-harness-20261123` 当前源代码，以及 Tang 上 2026-11-23 两轮真实 Qwen3-8B-AWQ h001 四臂试验。本文区分**代码事实**、**实验观察**和**待检验假设**；绝不把 DeepSeek API 连通性当成新 Agent 成功率。

## 1. Agent 边界和与旧 Workflow 的区别

新 Repository Coding Harness 的实际 LangGraph 是 `START→Model→Tool→(Model 或 END)`。它没有独立 Planner/Progress/Critic 节点，也没有跨阶段固定流程。LLM 每次只能选一个 `HarnessDecision(action,reason)`，ToolHost 再执行安全工具；仅当合法 `finish` 返回 `OK` 才正常完成，否则到步数限制停机。

旧 HardSet D/E Agent 的图是固定的 `investigate→diagnose→patch→public_selfcheck→finish`，失败候选可以返回 `patch`，而可选 E 分支使用 LLM 决定 functional probe。**旧 Workflow 并不是“没有 LLM”；D 仍有模型诊断和模型补丁。** 过去 DeepSeek D/E 的 3/3 不能与新 ReAct Harness 使用 Qwen3-8B 的 0/8 直接比较，模型/工具/预算都不同。

核心代码：
- `src/gpu_agent/cli.py`：`harness repo-run`、`harness sweep-repo`。
- `src/gpu_agent/harness/ablation.py`：冻结模型/预算/Git/HardSet，四臂 `legacy_single,legacy_batch,evidence_single,evidence_batch`；保留全体失败，D 是单独不可等价参考。
- `src/gpu_agent/harness/repo_runner.py`：Session、Provider、RunStore、Journal、Resume 的连接器。
- `src/gpu_agent/harness/runner.py`：真正 LangGraph 的 Model 和 Tool 两节点，`AgentLoopState` 只有 history、decision、observation、turns、stop、reason。
- `src/gpu_agent/harness/repo_tools.py`：`RepositoryWorkspaceTools`、动态合法工具、公开候选检查、结束授权。
- `src/gpu_agent/harness/worktree.py`：Git SHA-pinned 隔离工作树、路径/符号链接校验、源码精确更新。
- `src/gpu_agent/harness/context.py`：历史压缩、行号渲染。
- `src/gpu_agent/agent/provider.py`：带物理调用审计的 OpenAI/DeepSeek/vLLM Responses 适配层。
- `src/gpu_agent/harness/repo_verification.py`：Agent 结束后**仅控制器可执行**的实验性隐藏验证。

## 2. 任务一轮从开始到结束逐步展开

| 步骤 | 执行器/函数 | 输入 | 关键输出及安全约束 |
|---|---|---|---|
| 0 | `create_benchmark_plan` | task/arm/预算/clean Git base | 注册 h001/h002/h003 与任务顺序，未出模型结果前冻结 |
| 1 | `preflight_paired_experiment` | 0600 evaluator secret、模型 API 配置 | 不可用则拒绝整个实验，0 GPU/LLM |
| 2 | `PinnedGitWorkspace` | Git base_sha | 生成分离 worktree；禁止更改测试、Harness、隐藏 evaluator；多文件编辑总数受限 |
| 3 | `RepositoryWorkspaceTools` | worktree, public_case_id | 控制器 RunStore、当前 source/diff、public_check 计数和 model tool authority |
| 4 | `OpenAIResponsesProvider` | API endpoint/model、killable worker、调用预算 | 每次物理请求记 STARTED/COMPLETED/FAILED/UNCERTAIN、usage，禁止自动回放不确定副作用 |
| 5 | `AgentLoopState` | history=[], turns=0 | Graph START→model→tool；无专门 plan/diagnosis/milestone |
| 6 | `snapshot()` | 当前源码、回执、剩余步数、权限状态 | `SessionSnapshot`：goal、functional requirement、source、last5 actions、allowed_tools 等 |
| 7 | `choose_harness_v1` | JSON `untrusted_data.snapshot` + 可信 Prompt | Responses JSON Schema 输出单个 `action + reason`；DeepSeek 强制 `reasoning=none` |
| 8 | `invoke` | model action | 动态权限、路径限制、是否为新候选；返回 `OK/REJECTED/FAILED` |
| 9a | `repo_read_file` | path/start/lines | 读取实际源码，输出**带行号**的文本；相同范围 fingerprint 会被拒绝 |
| 9b | `repo_replace_snippet/many` | exact old_text/new_text | 旧内容必须出现且仅出现一次；Batch 中每个片段必须唯一且不重叠；成功才原子落盘 |
| 9c | `repo_git_diff` | 当前 worktree | 返回有界 Diff，重复 Diff 被拒绝 |
| 9d | `repo_run_public_checks` | 已成功编辑且未重复的候选 | 仅注册单个 Kernel、全新候选、检查预算内，执行真实 Docker CUDA/数值/四 Sanitizer |
| 10 | `on_event / on_checkpoint` | tool/status/observation | 原子 RunStore、公有哈希链 Journal、checkpoint，之后走 graph conditional |
| 11 | `FinishTask` | 模型决定停止 | 仅当 Diff 非空且同一注册候选公开检查通过，才能 finish；否则继续或预算耗尽 |
| 12 | 外部 `verify_repository_hardset_candidate` | 已结束的会话 + 独立 secret | 严格验证来源、Diff、原始候选，隐藏 9 数值尺寸 + 四 Sanitizer，**非模型工具，非正式签名 holdout** |

## 3. 为什么“模型只关注下一步”是部分真实、部分误诊

`run_tool_agent` 的 model node 每轮没有持久模型 Planner，仅调用 `tools.snapshot` + `model.choose_harness_v1`。是否反复读取是模型决定，但以下 Harness 机制增加了失败风险：

1. **显示行号污染 edit anchor**：`PinnedGitWorkspace.read()` 自带行号；legacy 模式将这个带号片段塞给 `compact_public_context`，后者又用 `source_with_line_numbers()` 统一编号，出现双层编号风险。evidence_first 模式在 raw source 前加说明性 header 再进行行号渲染，屏幕显示行号与真实文件行号可能错一行。
2. **编辑协议过于苛刻**：`replace_exact` 必须 `text.count(old_text)==1`；换行被合并、复制的行号、缩进改变都会得到相同通用 ValueError。没有 `expected_source_sha256 + start_line/end_line + replacement_lines` 的可信范围编辑。
3. **上下文损失**：最近仅 5 次详细工具反馈，每条最多 900 字符；已执行工具名、结果能保留，但旧历史只被压缩。旧历史去重规则匹配 `read_file` 等名称，没有完整覆盖 `repo_read_file` 等新工具。
4. **仅限制重复工具的执行，不阻止参数层面的重复决策**：动态合法工具 Schema 可以排除 `repo_list_files` 等工具类型；但即使完全读过同一个文件，`repo_read_file` 类型仍合法，模型重复请求会先消耗模型回合再在控制器被 `REDUNDANT_READ` 拒绝。
5. **缺失完整进度状态**：AgentLoopState 无明确 hypothesis、diagnosis_done、proposed_patch、failed_edit_reason、next_gate、validation_reserve。文本 `BUDGET_ALERT` 并非强制规划算法。
6. **标识语义不严谨**：`SessionSnapshot.source_hash` 当前是 `sha256(base_git_sha)`，不是实时源码内容哈希；`current_candidate_sha256` 实际是工作树 Diff 的文本哈希。安全的独立校验使用真实源码/候选哈希，但对模型和编辑协定是概念混淆。
7. **轨迹观测不足**：历史记录只有 `tool/status/content`；模型每轮的完整工具参数不全写入可回放 public trace，checkpoint 主要保留最近一次 decision。Provider 保存请求元数据及 output_hash，不等于持久完整推理内容。

## 4. 已实际观察到的失败与不能证明的事情

实测模型为 **Qwen3-8B-AWQ (4-bit AWQ)**，不是所谓“7B”；运行在 Tang RTX 4090 vLLM，h001，`max_steps=6`，同一轮四臂×两轮：

| 阶段 | 物理模型请求 | Tokens | 成功修复 | Public CUDA 次数 | 实际失败模式 |
|---|---:|---:|---:|---:|---|
| Round 1 | 24 | 46,986 | 0/4 | 0 | 重复 `repo_list_files` / 尚无候选就选择 GPU check |
| Round 2 | 24 | 58,752 | 0/4 | 0 | 重复 `repo_read_file`，`repo_replace_snippet` 和 `repo_replace_many` 均旧源码匹配失败 |
| 合计 | 48 | 105,738 | **0/8** | **0** | 没有进入真实 CUDA 验证的候选 |

Round 2 evidence_batch 的连续 6 步：read OK → read duplicate REJECTED → read OK → replace_many REJECTED → replace_snippet REJECTED → replace_snippet REJECTED。它未触发 GPU 公开检查，也不能算 GPU 测试失败。

- legacy_single 末次 `old_text` 含阅读器行号 `17`，真实文件 `text.count(old_text)==0`。
- evidence_single/batch 末次 `old_text` 把多行 C++ 代码合成一行，`text.count(old_text)==0`。
- h001 的数值错误是已计算最大值却对 `expf(x)` 和 `expf(own)` 没有减最大值；至少两个位置都需要正确更新。
- `6` 步预算很紧；每次 REJECTED 仍计一回合；模型未能取得三个必需成果——有效编辑、public verification、合法 finish。

完整原始报告及哈希：[2026-11-23 Qwen3 真实失败轨迹](../repair-log/2026-11-23-qwen3-8b-actual-failure-trajectories.md)。

这支持**当前具体模型+运行时组合失败**；不足以证明只因为模型参数小，也不足以说明新增模型一定会成功。

## 5. DeepSeek v4-pro 凭据核查、API 真实请求和限制

Tang 原有 operator-private `0600` 配置中确有 `OPENAI_BASE_URL=https://api.deepseek.com`，`OPENAI_MODEL=deepseek-v4-pro` 和非空 API 密钥。未打印或提交密钥。

- 首次连接/权限真实测试：`responses.create(store=false, reasoning=none)` 返回 `completed`，`usage.total_tokens=22`，输出文本非空。**表明凭据和 endpoint 可用**。
- 尝试在原 repo 上启动完整 DeepSeek 四臂配对：远程执行安全检查拦截，两次均未启动试次。**不是 DeepSeek 失败，也不是 0/4 模型成绩**。
- 另一次独立模型公开源码补丁诊断：官方 Responses 返回 `completed`、`usage.total_tokens=1,233`，但 SDK `output_text` 为空、JSON 解码失败，未得到可评价候选。不能解释成“DeepSeek 不懂 CUDA”；未确认该次响应正文是否处在其他字段或截断路径。
- 后续一次不同格式单轮代码诊断被远程安全检查拦截。不能据此编造新 DeepSeek 自主修复成绩。

此外，当前 Repo Model Adapter 的 `invoke_sdk` 对 DeepSeek **固定** `reasoning={"effort":"none"}`，官方文档明示这意味着关闭思考模式。DeepSeek 的高推理强度是一项必须单独设置/消融的实验因素；更换模型时不能忽略 thinking/采样及 Token 预算差异。

相关官方说明： https://api-docs.deepseek.com/guides/thinking_mode/ 与 https://api-docs.deepseek.com/api/create-response/

## 6. 必须分层验证的因果顺序

| 关卡 | 保持不变 | 改动 | 有效判据 |
|---|---|---|---|
| A：隔离 Code Understanding | 同一 CUDA 源码、题目、单轮模型预算 | 仅模型/推理强度 | 是否提出**同时修正分母和分子**的正确修改；不计 Agent 成功 |
| B：隔离编辑工具 | 同一模型+Prompt+任务+工具预算 | exact snippet vs SHA 绑定的 line-range edit | 合法补丁写入率、失败代码分类、修改质量 |
| C：隔离任务管理 | 同一模型+同一编辑接口和 GPU Oracle | ReAct-only vs Progress State/Planner+ReAct | 首次有效补丁轮次、重复动作/总成本、public/strict 成功 |
| D：隔离模型能力 | 同一 Harness+同一任务和协议 | Qwen3-8B-AWQ vs DeepSeek-v4-pro | 同样验收标准下的逐案例成功率及 Tokens/墙钟/GPU 成本 |
| E：独立泛化 | 研发阶段方法已冻结 | h002/h003 + 未见过的其他错误类型 | 不调参的 independent strict/holdout 成绩 |

建议的最小新工具是：`repo_replace_range(path,expected_source_sha256,start_line,end_line,new_lines)`，并检查真实文件 SHA、行号范围、变更差异、受保护路径及候选 Diff。拒绝时输出结构化 `LINE_RANGE_STALE / INVALID_CONTEXT / INVALID_PATCH / OUT_OF_SCOPE`，不要模糊替换或把有错误的补丁自动默默修好。

增加 `TaskProgress`：当前故障假设、证据引用、已尝试并验证的候选、失败类型、剩余模型/GPU 预算和下一验收条件。Planner 只在初始化或关键失败后重新规划，Executor 仍自主选工具。每一项提升都要与单独编辑协议改进进行消融，否则不知道改善来自 Planner 还是新 Tool。

**判定原则**：模型可以解释 Bug 但不能成功写入 Diff，优先排查编辑协议；强模型能生成 Diff 但在 Loop 中浪费预算，优先排查上下文和规划；换模型后在其他条件相同的独立任务上稳定成功，才能有依据说模型能力是核心因素。
