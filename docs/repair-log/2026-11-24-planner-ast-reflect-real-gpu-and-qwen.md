# 2026-11-24｜CUDA Coding Agent：Planner / AST / Reflect 实施与真实实验

## 1. 交付范围（代码已进入 GitHub PR #16，尚未合并）

这次升级保留原先的 ReAct 作为可复现实验基线，不改动旧 HardSet D/E 的固定 Workflow，新增三个独立能力：

**Planner + Executor：** 在 `src/gpu_agent/harness/runner.py` 中使用 LangGraph `START → planner → model → tool → (model / reflect / END)`。Planner 由真实模型独立输出包含 `inspect/edit/public_check/finish` 的四个里程碑，每个定义目标与可验证成功证据。Executor 仍通过 `choose_harness_v1` 自主选择下一项合法工具；进度标记来自**控制器实际 tool observation**，不是 LLM 自报。在源码更改后自动将旧 public_check 标记无效；只有 `finish` 被工具批准才结束。Planner 的结构化简要目标**不是**完整 CoT 推理链或真正 ToT 搜索，当前只生成一次全局计划，不应宣传为动态多候选树搜索。

**tree-sitter CUDA AST 编辑：** 采用锁定版本 `tree-sitter==0.25.2` / `tree-sitter-cuda==0.21.2`。构建已注册 CUDA Kernel 的 AST 表达式/声明节点表（`node_id` 绑定当前源码 SHA、语法节点类型与字节范围），`repo_replace_syntax` 接受 1–4 个 `node_id/replacement_code`，对受保护路径、当前源码 SHA、节点合法性、内容无操作、重叠及新代码 parser errors 做 fail-closed 验证，然后原子写回 SHA-pinned 工作树。通过 Node ID 定位，因此**不再需要模型复制包含显示行号/换行的 old_text**。这仅保证语法与来源条件，数值正确性仍交给真实 CUDA GPU 检查与独立严格验证。

**Reflect + 压缩：** 在执行达到半程（但还没有结束且还未 Reflect）时，调用真实模型的 `reflect_repository_task`，输出简要 `observed_failure/avoid_next/next_priority`，并将结果纳入后续 Snapshot。压缩器对重复拒绝按类去重，最近 AST 节点 JSON 保留完整有界 3900 字符，Canonical RunStore / Journal / Checkpoint **一字不删**。Reflect 不拥有修改工作区与隐藏验证的权限。

**关键修复：** 修正了 `LLMCallGate` 对只用 `plan` 的 Coding Agent 无故预留未来 diagnose/patch 两次请求的错误。Repository session 显式 `reserve_repair_stages=False`，旧诊断 Workflow 仍维持原预留策略。为已注册 CUDA Kernel 的 AST 工具参数加入 JSON Schema literal path，不能再让模型误写错误文件路径。新 flag、计划与反思字段均写入 Checkpoint、Journal 和公开指标，恢复时校验模式不变。

### 主要模块与运行接口

| 模块 | 实际代码 | 行为 |
|---|---|---|
| Plan / Reflect 数据 | `src/gpu_agent/harness/models.py` | `TaskPlan / PlanMilestone / TaskProgress / TaskReflection` |
| LangGraph 节点 | `src/gpu_agent/harness/runner.py` | Planner → Executor Model → Tool → Reflect → Model |
| 已验证进度 | `src/gpu_agent/harness/planning.py` | 工具观察更新里程碑、生成模型可见压缩记录 |
| 可信模型协议 | `src/gpu_agent/agent/provider.py`, `agent/prompts.py` | Planner/Reflect 单独真实调用、受控工具 JSON Schema |
| AST Parser + Rewrite | `src/gpu_agent/harness/syntax_tools.py` | 确切 AST node_id、源码 SHA、CUDA 语法重新解析 |
| 工作区/工具 | `src/gpu_agent/harness/worktree.py`, `repo_tools.py` | 隔离 Git Worktree 原子编辑、动态合法工具 |
| CLI / 控制对照 | `src/gpu_agent/cli.py`, `harness/ablation.py` | `repo-run --hierarchical --syntax-editing` 和 `sweep-repo --architecture-study` |
| 恢复/指标 | `src/gpu_agent/harness/recovery.py`, `metrics.py` | 绑定模式和结果，防切换、保留未成功轨迹 |

## 2. 工程测试与真实 CUDA

- 当前 GitHub 专项回归覆盖 AST stale SHA、重复/非法/重叠节点、parser 拒绝、动态工具权限、Planner/Executor/Reflect 节点、Checkpoint 模式一致性及工具成本门禁。
- 在 Tang 上冻结 commit `f32cbbf97ef8f030e464febd2ce3cbcb191b5b9f` 后，运行相关 **49 CPU 测试 passed**，证据在该冻结目录；GitHub HardSet 专项 CI 在这一系列提交中持续验证。
- **真实 CUDA Docker**：测试替身顺序 `repo_inspect_syntax → repo_replace_syntax (2 nodes) → repo_run_public_checks → finish`，4 工具动作、0 真实模型调用；公开数值通过且四类 Compute Sanitizer 均为 `CLEAN`，public RunStore 子运行含 57 项全部公开可见的 artifact。
- **独立操作者 experimental Strict V2**：同一 h001 candidate SHA256 `88dd297b8a0bc53dc3b93b7d1a4678551346b4855d7139b4edaddecef0070ba4`，返回 `VERIFIED_FIXED / ALL_REQUIRED_CHECKS_PASSED`；标准定义检查 9 组隐藏数值输入与四项 sanitizer；私有 RunStore 91 项 evaluator-only artifact。公开 verdict projection SHA256 为 `221e0bf21588a87a98924275475b4fbcf1aa10237905a53f37d6be2639273c5a`。**这不是签名生产级 holdout，亦不是真实 LLM 自主修复成功。**

## 3. 两批真实 Qwen3-8B-AWQ Agent 失败轨迹（不能选择性删掉）

实验均为 h001、实际 vLLM 本地 Qwen3-8B-AWQ；每臂最多 8 个 Tool Turns，模型调用审计和 public GPU 检查独立记录。第一批冻结 SHA `9fce49150eb88adbd9fddc6b5718f58268e44fc4`；第二批修复模型预算及 AST 路径 Schema 后冻结 SHA `e4ab3fdc88e3a35365d52712392a4d8c33465301`。

| 试次 | ReAct + exact | ReAct + AST | Planner/Reflect + AST |
|---|---|---|---|
| 第一批物理模型调用 | 8 | 8 | 8（提前拦截） |
| 第一批 Tokens | 22,554 | 19,688 | 24,253 |
| 第一批结果 | 未公开验收 | 未公开验收 | ProviderError、未公开验收 |
| 第二批物理模型调用 | 8 | 8 | 10（旧预算 Bug 已修正） |
| 第二批 Tokens | 22,177 | 24,329 | 30,530 |
| 第二批结果 | 未公开验收 | 未公开验收 | 未公开验收 |
| 第二批真实 CUDA public check 次数 | 0 | 0 | 0 |

**第一批的实际故障**：exact 组最后一轮才写入候选、无轮次做 CUDA 验证；AST 组反复引用错误 Kernel 路径；分层组成功进入 Planner+Reflect，但重复 AST inspection，提交的 AST replacement 曾被判 `CUDA_SYNTAX_NOOP`。此前 `LLMCallGate.reserve(plan)` 每次隐式预留两次并不存在的 diagnose/patch，导致只使用了 8/10 次允许请求就报 ProviderError。

**第二批的实际故障**（修复了上述两项运行时 Bug 后）：exact 组出现重复读和多次无法匹配的字符串替换；AST 组一次成功取得语法表、随后重复请求相同 AST inspection，最终耗尽 8 个 Turn；分层组完整使用 10 次模型请求（含 Planner/Reflect），在第 4 个 Turn 提交 `CUDA_SYNTAX_NOOP`，半程 Reflect 的输出为 `observed_failure=CUDA_SYNTAX_NOOP` / `next_priority=inspect`，反而继续反复 inspection，最终无可公开验收的候选。**三个真实模型配置均为开发集负结果，不得冒充成功率提升。**

- 第一批 `paired-report.json` SHA256: `777a9a59ca03990910f2d7c115c7b2a647996c645ede4d8439b31400da28dd1a`
- 第二批 `paired-report.json` SHA256: `f76d3211f8e3b97f05fb4cd184b1bbfaf3708590668c00b328d89cd63adb48c6`
- 实验目录在 Tang：`/home/you/gpu-native-harness-validation-20261124/qwen3-architecture-9fce491/` 和 `qwen3-architecture-budget-bound-e4ab3fd/`。完整原始模型调用元数据、工具拒绝、各次实例与 Checkpoint 持久化保存（不上传 evaluator secret）。

### 如何理解负结果

- AST 解析定位解决了源码旧文本复制的**接口脆弱性**，不保证 8B 模型会选择正确语义修改；
- Planner 确实跑了，Reflect 也确实跑了，但反思内容可能**强化错误的下一步决策**，不能将模型自述的 Reflect 当作外部校验；
- 三个 arm 每个任务仅运行一次，且第一/第二批代码不同、模型采样未声明固定，不构成统计性、跨分布的因果增益；
- 控制器 Tool-steps 相同，Planner+Reflect 额外消耗两次模型请求，因此必须分别对比 **Tool Steps / 物理模型请求 / Tokens / GPU 成本**，不能假装预算成本完全相同。

## 4. 复现方式与下一道门槛

```bash
# 只读、零 API/GPU 消耗的架构对照预检：
gpu-agent harness sweep-repo --repository /frozen/repo \
  --output /external/new-experiment --cases h001,h002,h003 \
  --architecture-study --repetitions 1 --max-steps 8 \
  --max-model-calls 10 --max-public-checks 2 \
  --evaluator-secret-file /operator-private/0600-secret.bin

# 真实执行需操作者明确允许：
# ...同一命令追加 --execute --allow-paid-calls --allow-real-gpu
```

下一步研究问题不是“是否有 Planner/Reflect/AST 模块”，而是：**模型一旦获取了完整 AST 源码证据，如何让下一轮从无效重复读取转向真正的新候选？** 应研究可验证的阶段约束、语义正确性反馈、Reflection 质量门禁与可持续上下文 evidence pinning。然后使用多任务（包括 h002/h003 和分布外 CUDA 缺陷）、多个模型与固定成本预算做真实独立对照。当前不应写“Agent 成功率提升”或“tree-sitter 确保正确修复”。

