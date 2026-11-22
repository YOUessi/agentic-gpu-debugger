# 2026-11-23 · GPU Coding Harness V1 首次真实模型试跑（保留失败）

## 当前证据与界限

本项目新实现的通用 Model → Tool → Model LangGraph Loop，确实接入了已有的 `deepseek-v4-pro` 审计通道，并在 Tang RTX 4090 Laptop GPU 上执行了第一轮真实工具决策。此次结果**没有修复成功**，不得把它与 2026-11-21 HardSet 固定 Workflow 的 3/3 实验混淆。

- 实际运行 SHA：`d3c3e02ed3991b730cf7fcb41e7a0195ec067655`。
- 固定源码 GitHub tar SHA256：`23e202f7c7b299285422ea9e1cc5549a2921fae05bc81b73ea8f273da16a1d99`。
- 案例：`h001` Softmax32，目标为检查源代码、复现数值问题、修复并验证。
- 真实 DeepSeek 物理调用：**14**，全部 `COMPLETED`，**61,063** 已记录 Tokens。
- Agent 调用了 14 次工具；最终状态：`INCONCLUSIVE / STEP_BUDGET_EXHAUSTED`。
- 没有候选源码通过 PatchScope，没有独立隐藏验证，不能计算为自主修复成功。
- 原始 `result.json` SHA256：`2a49e2f4359b9430cb16b5fed8d2a8400c00cd57246f593b1cc09fcf42887396`。
- Tang 原始根目录：`/home/you/gpu-native-harness-validation-20261123/live-harness-h001-d3c3e02/`，公开 RunStore ID `9f6b34407517478b8753b80f471d0c96`。未发现私有/隐藏 Artifact 泄漏到公开 RunStore。

## 实际模型决策顺序（不是固定流程）

| Turn | Tool | 结果 |
|---|---|---|
| 1 | `list_files` | 成功 |
| 2-4 | `read_file` | 成功，但多次重复或相似源码读取 |
| 5 | `functional_probe` | 请求了未经注册的尺寸 8，被控制器拒绝 |
| 6 | `functional_probe` | 成功运行 |
| 7-9 | `read_file` | 冗余读取 |
| 10 | `search_code` | 成功 |
| 11 | `functional_probe` | 成功运行 |
| 12 | `search_code` | 成功 |
| 13 | `read_file` | 冗余读取 |
| 14 | `apply_patch` | 模型识别出缺少 max-subtraction，但生成的 Diff 未通过控制器校验；没有轮次用于修正 |

这与旧 Workflow 有重要区别：模型实际决定每次工具调用，且错误方向可由公共工具反馈指出。**但开放的工具选择也暴露了自主 Agent 效率问题**：重复读代码、猜测非法参数、直到轮次终点才开始修复。

## 问题处理（后续提交，不挪用旧成功成绩）

- `e249f9921fa06c0772a767fdea7785c55d7477d4`：在 `SessionSnapshot` 显式标注允许的探测尺寸，拒绝无收益的重复 read/search，扩大历史工具返回可见长度，公共审计保留有界拒绝原因；真实模型 Prompt 明确强调足够证据后迅速尝试 Patch。
- `09fa00f2b566fbd18f9ba59575e0bc97b05b2fb3`：修复严格类型检查中复用不同形状 cache key 的问题。
- Tang `e249f99` 离线相关测试：**14 passed**；固定 tar SHA256 `3c89f6e11c13aeb10dadff5583476ee29504f54def494445913c45909929abbd`。
- 对 `e249f99` 的真实模型试跑**没有启动成功**：远程执行请求在到达 Tang 前遭执行环境安全检查拦截。这不是 DeepSeek API 拒绝，不得算作模型失败、成功或 GPU 实验。
- 目前最新 `09fa00f2` 尚未获得新的真实模型端到端成绩；CI 状态应按其 SHA 单独检查。

## 研究和工程结论

第一版 Coding Harness 让 LLM 在受控工具空间里持续自行决定动作，而不是被固定的诊断/补丁阶段强制路由。**真实模型确实使用了这条 Loop，但首轮没有修好任务**。这比“在旧 Workflow 上加个 Codex 标签”更接近真正 Harness；同时说明上下文控制、行动去重、预算、工作空间和工具权限本身就是 Coding Agent 的核心研究问题。

接下来必须在最新冻结 SHA 上验证正常的 GPU Patch → Public Checks → Finish → Private Verification；再扩展受控 Git Worktree、多文件编辑、被批准的容器 Shell、持久 Session/Resume、MCP/Skills 和聊天式工具执行 UI。完成前应称为 CUDA 特化 Harness V1，不是与 Codex/Claude Code/DeepSeek Harness 功能对等。
