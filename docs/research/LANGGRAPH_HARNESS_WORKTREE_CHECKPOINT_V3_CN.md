# 2026-11-23｜LangGraph 自研 Coding Agent：多文件 Worktree 与副作用安全 Checkpoint

## 研发目标

在现有 PR #16 基础上，继续构建自有 Codex/Claude Code/DeepSeek Harness 风格运行时，不依赖它们作为黑盒 Agent 引擎。用 LangGraph 管理 Model → Tool → Model 状态循环，由自己实现工作空间、工具权限、上下文和外部副作用管理。

**这是新的工程能力，不等于已经证明新 Harness 比旧 Workflow 更聪明。**

## 1. 可执行的多文件 Coding Agent

此前的单文件 HardSet 工具集只允许查看或修改预注册的 \`kernel.cu\`，拥有真实 Docker CUDA 验证。

现在新增两个**隔离工具子系统**：

- \`harness/worktree.py\`：operator 在 Git SHA 上创建独立 detached worktree；原仓库不被模型直接编辑；只允许已登记的源代码文件改动。拒绝路径穿越、绝对路径、符号链接、隐私文件、测试、原有 Verifier 和 Agent 运行时；限制最多编辑文件数。保存 Git Diff SHA，禁止意外出现的外部修改或新文件。
- \`harness/repo_tools.py\`：向模型提供 \`repo_list_files / repo_read_file / repo_replace_snippet / repo_git_diff / finish\`。这套工具**没有** Shell、执行编译、调用私有 Verifier 的能力。权限与状态由 Controller 控制。
- \`harness/repo_runner.py\`：复用现有 LangGraph \`run_tool_agent\` 与受审计的 DeepSeek Provider。模型真正决定源码读取、编辑和检查差异的顺序。
- \`harness/models.py\` 和 \`agent/provider.py\`：统一类型化工具决策结构，不向模型暴露 Host shell；Repo Prompt 与 CUDA Prompt 分版本。
- \`gpu-agent harness repo-run --repository ... --base-sha ... --goal ... --output ... --allow-paid-calls\`：有明确付费许可的命令入口。

**禁止把多文件工作空间的 \`DIFF_STAGED_UNVERIFIED\` 报告为 \`VERIFIED_FIXED\`。** 该子系统目前仅生成可审查的 Diff，不运行原仓库的任意编译、测试或 GPU 命令。任意项目的隔离执行必须另行接入批准的 Docker Tool。

## 2. 持久会话与安全恢复边界

此前的 \`SessionJournal\` 仅有哈希链事件和只读恢复状态，尚无 LangGraph 执行状态的物理存储。

新增 \`harness/checkpoints.py\`：

1. 模型调用开始前记录 \`model_started\`，收到合法结构化动作后记录 \`model_selected\`。
2. 工具执行前记录 \`tool_started\`，确实完成后记录 \`tool_completed\`，均 fsync。
3. **只有收到工具完成记录后**，才将 LangGraph 全量 Agent 状态（含 Decision、Observation、History、Turn、Stop）存入 SQLite 0600 Checkpoint。
4. Checkpoint 绑定源 Git SHA/源码 SHA256、公开 RunStore ID、Journal 事件 SHA256 和当前工作区 Diff SHA256。任何单步序号不连续都拒绝。
5. 使用 \`gpu-agent harness inspect-session\` 和 \`gpu-agent harness inspect-checkpoint\` 验证日志与最近持久化状态。
6. 已完成会话不可自动重新执行；未确认的外部 API/GPU 调用需人工核对；工作空间 Diff 漂移与损坏日志直接拒绝。

这**尚不是完全自动 Resume**。真正跨进程继续工作还需要工具状态重建、Provider Invocation Ledger 的幂等对账、Git Worktree 重新绑定以及 LangGraph Checkpointer/interrupt 的安全接入。不能因为磁盘上有 SQLite 就声称“断点续跑已完成”。

### 命令示例

\`\`\`bash
gpu-agent harness repo-run \\
  --repository /path/to/authorized/repository \\
  --base-sha FULL_FROZEN_GIT_SHA \\
  --goal "Inspect the CUDA kernel and helper and propose a scoped repair" \\
  --max-steps 18 --max-model-calls 24 \\
  --allow-paid-calls \\
  --output /new/isolated/run

gpu-agent harness inspect-session \\
  --journal /new/isolated/run/session-events.jsonl

gpu-agent harness inspect-checkpoint \\
  --database /new/isolated/run/turn-checkpoints.sqlite3 \\
  --journal /new/isolated/run/session-events.jsonl
\`\`\`

上面的 Repo-run 产物永远只能叫 \`DIFF_STAGED_UNVERIFIED\`。原 HardSet GPU Harness 仍保有真实 CUDA 修复与严格验证接口，二者的能力边界和安全隔离不能混为一谈。

## 3. 这批提交的测试门禁

- \`tests/unit/test_gpu_harness_worktree.py\`：实际初始化临时 Git 仓库、创建 SHA 固定 Worktree、跨两文件编辑、Git Diff、原仓库无修改、保护 Test/Evaluator 文件、符号链接和路径越界、超预算、检测外部漂移和未知文件。
- \`tests/unit/test_gpu_harness_repo_agent.py\`：模拟模型输出，执行真实的 LangGraph 多文件工具循环；验证两个文件修改、原仓库仍干净、Journal/Checkpoint 完整、实际 Provider Wire Marker。
- \`tests/unit/test_gpu_harness_checkpoint.py\`：日志绑定、SQLite 内容哈希、工作区 Diff 匹配、故障恢复判定、模糊模型/GPU 调用不自动重放、伪造/篡改拒绝。
- 原 \`tests/unit/test_gpu_harness_runtime.py\` 及 HardSet CUDA 相关回归测试全部保留。

**禁止使用模拟模型的测试结果证明真实 LLM 修复能力。** 每条 GPU 原生及模型请求的实验报告必须固定到实际运行的源码 SHA，并同时记录失败。

## 4. 下一阶段

1. 对冻结 SHA 完成 Python 3.11/3.12 严格类型检查、离线测试和 Tang 验收；修复所有门禁。
2. 保持新 Worktree 独立权限的前提下，为通用源码任务接入真实可控的编译/测试工具和显式批准（不允许 Host Shell）。
3. 设计 LangGraph Checkpointer 与 Agent/Journals 协同的无重放 Resume，实测中断/恢复而不重复付费请求与 Docker GPU 副作用。
4. 实际运行真实 DeepSeek 新 Harness，并与旧 Workflow 按冻结案例进行 matched-budget 比较；不盲目声称 Agent 优于规则。
