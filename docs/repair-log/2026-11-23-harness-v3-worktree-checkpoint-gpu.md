# 2026-11-23｜Agent Harness V3：Git Worktree 与持久 Checkpoint 验收

## 固定运行版本与目标

冻结运行源码 **`764c6c257b170348fab3f05ce4197e527b7b9e05`**。

本轮属于 PR #17（堆叠于 PR #16）的**运行时安全与工程闭环验收**，不是新模型算法或 Agent 智能性 benchmark。核心变化包括：

- 使用自研 `PinnedGitWorkspace` 从固定 Git SHA 创建分离 Worktree，多文件编辑严格限制在可信源代码集合。检查整个 Worktree 的 staged、unstaged、untracked 与外部重写漂移，阻止模型直接修改测试、隐藏 Verifier 或运行时。
- `RepositoryWorkspaceTools` 通过同一个 LangGraph Agent Loop 暴露 `repo_list_files`、`repo_read_file`、`repo_replace_snippet`、`repo_git_diff`、`finish`。该路径仅生成 **`DIFF_STAGED_UNVERIFIED`**，不能声称 CUDA 修复通过。
- `SessionJournal` 在模型、工具开始和完成时独立 fsync 审计；`TurnCheckpointStore` 只在 `tool_completed` 后存储全量 LangGraph State，绑定 Source SHA、Public RunStore、Journal SHA 和当前候选/Worktree 哈希。
- `SessionJournal.reopen`、`PinnedGitWorkspace.attach_existing`、`RepositoryWorkspaceTools.reopen` 提供受限重新挂接原语，但**没有把自动恢复全部外部副作用的跨进程 Resume 当作已完成能力**。

## A. Tang 离线验收

- 使用 GitHub 固定 commit tarball，在 Tang Python 3.11 / 现有 project venv 测试。
- tarball SHA256：`b78cb03f903696256ded415620ed0279d2232f41c524d4b898cc38da21dc7764`。
- `pip check`：No broken requirements found。
- `pytest`：**171 passed in 2.60s**。覆盖多文件工作空间、受限工具、上下文、会话日志、SQLite 回合 Checkpoint、已知 HardSet 诊断与 Verifier 接口等。
- `mypy --strict src/gpu_agent`：**Success: no issues found in 110 source files**。
- 新增 `gpu-agent harness inspect-checkpoint` CLI：已验证帮助命令可启动。

本轮修复过一个类型错误：LangGraph Tool node 返回的全量 State 必须标注为 `AgentLoopState`，不能错误标注为一般 `dict[str, Any]`；修复后严格检查通过。

## B. Tang 真实 RTX 4090 GPU + 持久 Checkpoint 连通性

采用明确的 `ScriptedToolChoice` **模拟模型工具选择**，但以下工具和验证是真实执行：

```text
list_files
replace_snippet: denominator += expf(x)  -> denominator += expf(x - maximum)
replace_snippet: expf(own) / denominator -> expf(own - maximum) / denominator
run_public_checks: CUDA Docker build + numeric Oracle + 4 Sanitizers
finish: 当前候选公开检查通过后才允许退出模型循环
[独立外层 Verifier] 9 个非公开数值输入 + 4 个 Sanitizers
```

原始结果：
- `status=VERIFIED_FIXED / ALL_REQUIRED_CHECKS_PASSED`、`private_verification=true`。
- `case=h001`，5 次控制工具调用，候选 SHA256 `88dd297b8a0bc53dc3b93b7d1a4678551346b4855d7139b4edaddecef0070ba4`。
- **模型物理请求 0 / 记录 Tokens 0**：因为用的是模拟工具选择替身，不是新的真实 DeepSeek 修复证明。
- Public RunStore：`caf94a68f93742e6a4b3720168e23143`。父级 5 个工具事件、20 个 Model/Tool 起止 Trace 事件，私有/隐藏/evaluator Artifact 泄漏 **0**。
- `session-events.jsonl`：22 个不可变哈希链事件，读取 `FINISHED`，没有可重放的未确认动作。
- `turn-checkpoints.sqlite3`：最新 Tool Turn = **5**；绑定 Journal Tool Completion SHA `69864b0649f67a914998dc9aa592df22ed86c306f49608efcde2bab9bfdb3fc2`，State SHA `a76815f53eb447ade19fe4607234d9ac92d06553a72cbae9fa5dadeca52c36b5`。
- Checkpoint 决策：`manual_rehydration_candidate=false`、`automatic_replay_permitted=false`，原因是本次会话已正常完成；不能为了做恢复实验重放已成功运行。
- `result.json` SHA256：`af753b06f3f2e4b0a37c65f7c1a174127f08131e43440db7fc7d466cb6e5f9bc`。
- Journal 文件 SHA256：`8cc5d5b7f9c3abee6306fc2ad4f15cb91964d846d9a087ac2055b94b660696fc`。

Tang 路径：`/home/you/gpu-native-harness-validation-20261123/scripted-harness-v3-gpu-764c6c2/`。GPU 检查后 RTX 4090 Laptop 显存约 204 MiB、无剩余任务容器。

## C. 科学与能力边界

- **已经证实**：这个固定源码上的自研 LangGraph Runtime，在脚本模型控制下能完成 Source Edit → GPU Check → 独立严格验证，并在每轮持久化不会自动重放的 Checkpoint。多文件 RepoHost 的纯 CPU 集成也通过测试。
- **没有证实**：V3 真实 DeepSeek 自动修复率、V3 比 Workflow 更省 Tokens、真实跨进程后无重复 API/GPU 的自动 Resume、任意项目跨文件 GPU 编译与测试、MCP/Skills、授权容器 Shell。
- 2026-11-21 的 DeepSeek V1 Workflow 3/3 与 2026-11-23 V1 Harness 0/1 都绑定**不同冻结源码**，不能挪用为本提交的成功率。
- PR #16 是并行推进中的基线分支，因此本轮功能创建堆叠 PR #17，避免非快进覆盖。正式合并前仍需处理基线变更和完整 GitHub CI；Tang 针对性测试不等于全项目 CI 通过。

## 下一批验收

1. 真实挂起的安全 Repo Session 在 Journal 静止边界 **reopen → 验证源和 diff 哈希 → 恢复状态 → 继续工具调用**，并确保证书检查失败时零副作用。
2. 将现有 CUDA Build/Sanitizer 工具挂到经批准的 Worktree 的隔离执行 Host，而不是在模型侧提供任意 Shell。
3. 新源码 V3 真正的 DeepSeek Agent 端到端实测，记录模型决策、修补、公开错误反馈与隐藏最终验证，比较是否改善先前 14 次工具调用和 61,063 Tokens 的无效循环。
