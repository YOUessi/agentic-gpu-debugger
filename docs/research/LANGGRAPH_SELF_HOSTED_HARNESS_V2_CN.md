# 自研 LangGraph Agent Runtime + GPU-Native Harness（V2 架构与验收）

**决策**：保留 LangGraph 作为 Graph/State 基础，**自研 Session、Context、Tool Registry、Permission、Workspace、Audit 和 GPU 任务能力**。Codex、Claude Code、DeepSeek Harness 用于阅读源码和技术对照，不作为本项目任务的黑盒执行引擎。

## 1. 对照三家开源设计：学什么，不复制什么

| 参考 | 确认的实际设计 | 自研对应层 |
|---|---|---|
| Codex | Agent 交替进行模型推理和工具调用；高度重视可复用前缀、缓存和长上下文压缩 | 简洁的 LangGraph Model → Tool → Model Loop；\`context.py\` 确定性公共证据压缩 |
| Claude Code | 工具权限、允许/询问/拒绝模式、项目配置；工具失败通过后续模型回合处理 | \`registry.py\` 的风险分类和操作员授权门禁 |
| DeepSeek Harness | Session 持久化事件、Tool Registry、Model Adapter、Agent Loop 作为可组合服务 | \`journal.py\` 事件日志，\`provider.py\` 审计 Provider，\`hardset_tools.py\` GPU Tool Host |

上游资料：
- [OpenAI：Unrolling the Codex Agent Loop](https://openai.com/index/unrolling-the-codex-agent-loop/)
- [Claude Code 权限与工具控制](https://code.claude.com/docs/zh-CN/agent-sdk/permissions)
- [DeepSeek Harness 架构说明](https://github.com/deepseek-ai/deepseek-harness/blob/master/docs/architecture.md)
- [LangGraph 官方状态与 Checkpoint 设计](https://docs.langchain.com/oss/python/langgraph/overview)

重要区别：DeepSeek Harness 使用 Cordis 的“everything is a plugin”；我们自研不需要复制其 TS/Cordis 实现，只学习 Tool Registry、Session Log 和组件边界。LangGraph 不是对外主业逻辑；GPU 修复能力依旧来自我们自己的 CUDA 执行后端、PatchScope、Sanitizer、Numeric Oracle 和 Verifier。

## 2. V2 分层架构

\`\`\`text
User CLI / Dashboard (future conversation interface)
         |
         v
GPU-Native Session Runtime (self-owned)
  - SessionJournal: model_started / model_selected / tool_started / tool_completed
  - ContextView: current kernel source + recent observations + compact historical evidence
  - Budget: per-turn model/tool/GPU bounds
         |
         v
LangGraph: START -> model_decision -> tool_execution -> [model_decision | END]
         |                |
         |                +--> Typed Tool Registry + Permission Policy
         |                    list_files / read_file / search_code
         |                    functional_probe / apply_patch / replace_snippet
         |                    run_public_checks / finish
         |                |
         |                +--> SourceSnapshot + PatchScope + Docker CUDA backend
         |                         |
         |                         +--> public CPU numerical oracle and 4 Sanitizers
         |
         +--> public RunStore + hash-chained Journal
                    |
      (after agent finishes only)
                    v
        Operator-only hidden strict verifier
\`\`\`

- **Tool 不负责判断最终正确性。** \`finish\` 必须有当前候选已通过公开 CUDA 检查的事实；随后外层 Controller 独立判断隐藏输入。模型不能调用隐藏 Verifier。
- **模型不拥有任意 Shell。** Tool Registry 采用白名单，权限由操作员配置，PatchScope 只允许修改被登记的 \`kernel.cu\`。未来通用 Shell 必须单独设计沙盒、审批和资源上限。
- **允许自主重新规划。** 任意失败后可以再读文件、探测、替换片段或提交统一 Diff；LangGraph 不强制诊断/补丁阶段顺序。
- **Context 不是完整原始日志。** 全量证据保留在 RunStore；模型只接收确定性、受字符预算约束的当前源码、最近 5 条记录和高价值旧证据摘要。拒绝将隐藏测试数据塞入摘要。

## 3. 本次从真实失败中修复的机制

真实 DeepSeek 在旧 V1 Harness 首轮 \`h001\` 上 14 次工具动作、61,063 Tokens，最后一次 \`apply_patch\` 被拒绝，没有做 GPU 修复验证。

现已补：
1. \`context.py\`：传入当前源码带行号，压缩之前的工具历史；失败和拒绝反馈优先保留，不臆造实验结果。
2. \`editing.py\`：\`replace_snippet(old_text,new_text)\` 仅对当前源码中的唯一匹配片段生效；工具把结果转为相对**原始冻结源码**的 Diff，仍由原来的 PatchScope 校验。不允许直接编辑 Test/Verifier。
3. \`registry.py\`：注册全部工具及风险类别，\`ToolPermissionPolicy\` 支持 ALLOW/DENY/APPROVAL_REQUIRED；无外部批准的编辑不能被模型自我批准。
4. \`journal.py\`：创建专属 \`0600\` JSONL，会话事件有递增序号、前向哈希链、fsync；不确定的付费模型或 GPU 副作用被标记为人工对账，不自动重放。
5. \`runner.py\`：模型与工具开始前写入 Write-Ahead 事件、实际完成后单独提交；每次事件同步记录到公开 RunStore。不会把 Session 的成功状态当作私有验证成功。

**明确限制**：目前没有跨进程透明 Resume、任意 Git Worktree、多文件编辑、真实终端/MCP、Subagents 或基于模型的语义 Compaction。这些还不能算完成。持久日志与失败安全判定是恢复机制的先决条件，但不是恢复本身。

## 4. 面试可展示的原创工程能力

核心贡献不是一个 LangGraph 图，而是模型-工具调用契约、对工具误用的安全门禁、上下文预算、精确代码编辑、外部副作用去重审计、私有验证隔离和真实性能评测设计。

可以现场展示以下测试与演示：
- 真实模型基于不同源码和公开证据自主选工具，而非固定阶段。
- 工具重复读取无收益时，控制器拦截；从上下文摘要中仍能找到历史数值失败证据。
- 模型错误提案、越权路径、非法命令、未经批准修改均不能执行。
- 对被拒补丁可以改用唯一精确片段替换，再执行真实 GPU 编译/数值检查/四种 Sanitizer。
- 不成功就输出 INCONCLUSIVE/NOT_VERIFIED；日志不会重放不确定工具或隐藏验证。
- 用同样的真实 CUDA 任务比较固定 Workflow、原 Harness V1、Context-aware Harness V2 与 Codex Baseline 的正确率、模型 Tokens、工具数量、GPU 耗时和成功修复成本。

## 5. 验收和后续优先级

本分支 V2 仅在原有 HardSet V2 注册的三类 CUDA 案例上启用。任何新 SHA 都需要单独的 Ruff、mypy、pytest 和 Tang RTX 4090 验证；**2026-11-21 旧 Workflow 的 3/3 结果不能自动当作 V2 Harness 成绩。**

后续工程门禁依次为：
- V2 真实模型有独立验证成功的修复，并保存全部失败轨迹。
- Context 控制与 snippet 工具的消融：比较无压缩/有压缩、只 Unified Diff/允许 exact replace 的执行成本与正确率。
- Git Worktree 和文件修改权限、命令审批与取消、工具 Registry 插件化 / MCP 接口。
- 真正的 LangGraph 持久 Checkpointer + 幂等性保证后的 Resume；人工审批 Interrupt。
- CLI/Web 的流式事件、用户可见文件 Diff、独立失败报告。
- 独立 OOD 多算法与多次重复实验，不能凭少量公开题目宣称 Agent 优于规则系统。
