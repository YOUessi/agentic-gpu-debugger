# GPU-Native Coding Agent Harness：Codex / Claude Code / DeepSeek Harness 风格重构

## 为什么重构

HardSet V2 虽然已经在真实 DeepSeek + RTX 4090 上完成三类 CUDA 修复，但原流程是 \`investigate → diagnose → patch → public_selfcheck → strict_verify\` 的有界 Workflow。模型仅在其中若干节点做决定，并未持续掌握工具选择和工作空间。

本方案将 **通用模型决策循环 + GPU 专用工具/权限 + 审计/会话** 分开，从架构上转为可以扩展的 Coding Agent Harness。参考：
- [OpenAI Codex Agent Loop](https://openai.com/index/unrolling-the-codex-agent-loop/)
- [Codex as a Platform](https://developers.openai.com/blog/codex-as-a-platform)
- [Claude Code 权限文档](https://code.claude.com/docs/en/permissions)
- [DeepSeek Harness Architecture](https://github.com/deepseek-ai/deepseek-harness/blob/master/docs/architecture.md)

## 系统结构

\`\`\`mermaid
flowchart TB
  User[用户目标 / CLI / 后续 WebUI] --> Session[Agent Session + 上下文 + 预算]
  Session --> Loop{LLM 选择下一个动作}
  Loop --> ToolRegistry[类型化 Tool Registry]
  ToolRegistry --> Policy[权限/预算/参数检查]
  Policy --> Workspace[隔离工作空间 + 文件搜索与编辑]
  Policy --> CUDA[CUDA Docker: 编译 / 功能探测 / Sanitizer]
  Workspace --> Evidence[公开工具结果 / 不可变审计]
  CUDA --> Evidence
  Evidence --> Session
  Loop -->|finish| Gate[必须已有通过的公开 GPU 候选]
  Gate --> Private[仅 Operator 调用独立隐藏验证]
  Private --> Result[报告 + 补丁哈希 + 运行轨迹]
\`\`\`

**关键点：LangGraph 只有 Model → Tool → Model 的循环**，不强制执行特定节点顺序。下一步是读取代码、搜索、测试、提交补丁、验证还是停止，全部由模型在受控能力集内决定。Operator 掌握工具授权和最后的严格验证权。

## 第一批真实代码：Harness V1 垂直切片

| 能力 | 实现 | 现阶段限制 |
|---|---|---|
| Model Loop | \`src/gpu_agent/harness/runner.py\` | 单 Agent、单 GPU 任务，轮次有界 |
| Strict ToolCall schema | \`harness/models.py\` | 一次只选择一个动作 |
| 文件读/搜 | \`read_file\`、\`list_files\`、\`search_code\` | 已注册 workspace 中的 3 个已知文件；无任意宿主文件访问 |
| CUDA 故障探测 | \`functional_probe(size)\` | 15 个注册输入形状上限，调用预算有界 |
| 代码修改 | \`apply_patch(unified_diff)\` | 只能修改原有 \`kernel.cu\`，提交基于冻结源码的 Diff；重复候选拒绝 |
| 自主验证 | \`run_public_checks\` | 真实 Docker GPU + CPU Oracle + 四种 Compute Sanitizer；失败反馈回模型 |
| 结束门禁 | \`finish(summary)\` | 只有当前候选通过公开检查才可完成；严格验证在循环外执行 |
| LLM | 复用 \`OpenAIResponsesProvider\`、独立 Prompt/Wire Schema | 支持当前已连接的 DeepSeek 及兼容接口；不新建不受审计的 SDK |
| 运行轨迹 | 原有 RunStore + \`harness/v1/event-*.json\` 与 \`trace-*.json\` | 尚不是安全的跨进程 Checkpoint/Resume |
| 安全 | 严格 Schema、Workspace 隔离、GPU 独占锁由外部执行者管理 | 不对不可信恶意代码提供硬隔离保证；Docker 共用宿主内核 |

当前重点是**可工作的单 Agent Coding Harness**，不是宣称复制了 Codex 或 Claude Code 的全部能力。旧版 HardSet 签名/verifier 和 A–E 比较不被本分支覆盖。

### 操作员示例

配置项目现有 Model Provider、RTX 4090 CUDA Docker 环境，并把隐藏验证密钥放在仓库外仅操作员能读的 \`0600\` 文件：

\`\`\`bash
gpu-agent harness run \
  --repository "$PWD" \
  --case h001 \
  --goal "Read the CUDA kernel, reproduce the numerical bug, fix it and verify it." \
  --max-steps 16 \
  --max-model-calls 24 \
  --allow-paid-calls \
  --evaluator-secret-file /operator/private/secret.bin \
  --output /new/run/path
\`\`\`

Agent 可以自行选择如下工具顺序（**只是示例，不是硬编码流程**）：

\`\`\`text
list_files
read_file(kernel.cu)
search_code("expf")
functional_probe(33)
apply_patch(diff)
run_public_checks
finish
\`\`\`

可能另一轮直接先探测、再读源码，或在测试失败后重新搜索、读取变更、生成第二份补丁。所有工具权限/预算仍由 Controller 强制，而不由模型输出的理由决定。

## 必须继续补齐到 Codex / Claude Code 级别的能力

1. **任意代码仓库工作空间**：Git worktree 的真正多文件读写、精确 Diff、符号搜索与版本控制。扩大范围时保持测试、构建系统、私有证据不可由模型篡改。
2. **受控持久 Shell**：在隔离容器中运行审核命令，提供权限申请；不能把本机 Desktop Commander 或 Host Shell 无限制暴露给模型。
3. **持久会话**：checkpoint、可恢复的 turn/工具调用 Ledger、uncertain external call 不重放、事件流与上下文压缩。
4. **模型无关/可插拔**：DeepSeek、其他兼容模型，功能探测和修补工具分别作为插件/Skills/MCP 服务；保留统一 Trace 格式。
5. **UI**：现有 React/Next.js/FastAPI Dashboard 改为真正的聊天式 Agent Console，实时显示行动、工具日志、待批准操作、文件 Diff 和验证进度。
6. **可选子代理**：Researcher/Debugger/Verifier Research 分配隔离 worktree；**私有 evaluator 永远不是可委托子代理工具**。
7. **Agent 质量评测**：单个模型自动修复成功不等于 Agent 优于规则。用复杂、多文件、OOD CUDA bug 对比同模型、同权限、同预算的固定 Workflow。

## 验收判据

- 必须有实时模型选出的非固定顺序工具轨迹、合法代码修改、真实 GPU 工具运行及最终独立验证。
- 至少一个人工预制补丁不能被模型直接读取；错误诊断或错误候选应能从公开反馈再规划。
- 记录模型物理请求和 Token、工具数量、失败原因、最终候选 SHA；保留所有失败。
- 完整 CI + Tang 真实 GPU acceptance + 私有日志隔离通过才称当前 slice 通过。
- 未实现通用 Shell、跨仓库交互、会话恢复、子代理之前，**只能称为 GPU Harness V1，不称为 Codex 功能对等**。

这一设计避免第三个固定 \`investigate → diagnose → patch\` Workflow，也避免把一个上千行的“综合 Agent 框架”空壳称作实际能力。
