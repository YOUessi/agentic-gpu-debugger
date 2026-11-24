# 2026-11-23 · 自研 CUDA Coding Harness：Git 工作区与可恢复会话

## 目标与明确的能力边界

继续实现 GPU-native Coding Agent 的自研 LangGraph + Model → Tool → Model Harness，而不是直接套用 Codex/Claude Code 的进程。**本批实现的是仓库级代码操作的安全暂停、重启与审计**，不等同于新增一次真实 DeepSeek 自动修复成绩。

- 当前多文件能力：在指定 40 位 Git commit 的独立 detached worktree 内，读取受限文件、精确修改已有源码、检查 Git diff、结束并交给人工复核。
- 当前恢复能力：只允许在**完整工具动作已经结束**时保存的私有 checkpoint 上恢复，不重新执行已完成的工具动作，也不会重置模型调用上限或工具步数上限。
- 当前 GPU 能力：HardSet 单案例 Harness 已可连接隔离 GPU public checks 与独立 strict verifier。**仓库级多文件入口暂不提供编译、GPU 测试、任意 Shell 或私有验证工具**；其结束状态只能是 `DIFF_STAGED_UNVERIFIED`，不得宣称已修好 CUDA。
- 旧成果不能移植：已报告的脚本模型 RTX 4090 成功、旧固定 workflow 真实模型 3/3 与本批仓库级 Resume 不是同一个运行/模型成绩。

## 自研运行时（已落地）

```text
User goal + frozen base Git SHA + operator budgets
                    |
            RepositoryDecisionModel
                    |
       LangGraph model -> tool loop
                    |
          ToolPermissionPolicy
                    |
          RepositoryWorkspaceTools
                    |
       PinnedGitWorkspace (host owns Git)
                    |
             source-only edits
                    |
    Worktree full-tree integrity guard
                    |
    public RunStore + private SessionJournal
                    |
 private session-checkpoint.json (only after completed tool)
                    |
         explicit --resume / final result
```

`PinnedGitWorkspace` 在每次编辑、读取、列出和 Diff 前验证整个工作区：Git HEAD 不得漂移，未授权 tracked/staged/untracked/ignored 文件不得变化，受控修改文件的 SHA256 必须与控制器最后一次原子写入一致。重新挂载还检查 Git common dir 身份，阻止将工作区替换为另一个仓库。符号链接与路径穿越全部拒绝。

`SessionJournal` 采用独占写锁、0600 私有事件文件、顺序号和 SHA256 哈希链。模型调用/工具执行的 `started` 与 `selected/completed` 分别落盘。恢复前验证 Journal 最后一条是完整 `tool_completed`、没有悬而未决的外部副作用，且 checkpoint 绑定同一 Journal head、源码 commit、目标任务 hash、公共 RunStore ID、已修改文件 hash、Diff hash、执行预算及模型工具历史。

`run_tool_agent` 的 `on_checkpoint` 在每次工具完成后保存完整状态（`history/turns/decision/observation/stop/reason`）。`SESSION_PAUSED` 只代表可恢复暂停，不返回正式 `result.json`；`FINISHED` 后的结果和公共记录不允许再次执行。恢复继续剩余轮次，不提供额外的 18/24 次调用额度。

**安全说明**：哈希链可检查意外损坏和非一致性修改，但并非外部签名或跨权限安全边界；同一 OS 用户若能同时伪造所有私有数据则仍然不受保护。关键执行器、模型计费调用和 GPU 测试的外部副作用不可简单用 checkpoint 自动重试。崩溃恰好发生在“工具已完成但 checkpoint 尚未持久化”的不确定窗口时故意拒绝自动恢复；应人工检查/隔离旧会话、再启动新会话，禁止静默重放。

## 操作示例（需要真实模型凭据时才启用 `--allow-paid-calls`）

```bash
# 示例：操作者显式选择待修改仓库、冻结提交和输出目录。
REPO=/absolute/path/to/target-git-repo
BASE_SHA=$(git -C "$REPO" rev-parse HEAD)
GOAL='Inspect and modify the registered CUDA source; do not touch tests'

# 首次执行：请求模型按工具自行决策，在第 5 个工具轮次后主动暂停。
gpu-agent harness repo-run \
  --repository "$REPO" --base-sha "$BASE_SHA" \
  --goal "$GOAL" --output /absolute/private/output \
  --max-steps 18 --max-model-calls 24 \
  --pause-after-turns 5 --allow-paid-calls

# 查看可审计的恢复状态。处于危险/不完整状态时只检查、不重放。
gpu-agent harness inspect-session \
  --journal /absolute/private/output/session-events.jsonl

# 明确恢复：目标、SHA、max-steps 和 max-model-calls 必须与原值一致。
# 编辑文件数预算也保持默认值（或用 Python API 指定与初始运行相同的值）。
gpu-agent harness repo-run \
  --repository "$REPO" --base-sha "$BASE_SHA" \
  --goal "$GOAL" --output /absolute/private/output \
  --max-steps 18 --max-model-calls 24 \
  --resume --allow-paid-calls
```

注意：如果 Agent 在第 5 次工具之前成功 `finish`，它会直接结束而不是暂停；在第 5 次工具遇到模型或执行器异常，则不可把异常状态误判为正常暂停。

## 对照测试、没有声称的结果

本批加入 `tests/unit/test_gpu_harness_worktree.py` 与 `tests/unit/test_gpu_harness_repo_agent.py` 的新回归案例，包括：

1. 同一原始 Git checkout 不受 Agent 的修改影响；多文件修改与 Diff 可验证。
2. 未授权修改测试文件、外部重写已审批的 CUDA 文件、未跟踪文件、路径穿越、符号链接、越权工具等必须拒绝。
3. 脚本模型进行暂停→新进程式恢复→完成，已执行工具不能重复，恢复后总步数与 RunStore 轨迹保持一致。
4. 恢复预算不一致、文件 hash 不匹配、Journal 仍有 `tool_started` 未完成、合法但额外的 Journal append 都必须拒绝。

GitHub HardSet V2 native-contract validation 在 `fc6a9d16597bab00873a0c04a2b61b62fc8f1fe7` 快照上 **169 passed**（Python/CPU/模拟模型，不是 GPU/真实模型）；随后代码风格清理的提交需要以各自最新 CI 结果独立确认。

## 还未解决：下一阶段的 CUDA Agent 目标

1. 将仓库级工具空间与**只读构建探测 + 被批准的 Docker GPU 公共检查**对接，限定命令白名单、输入与网络权限；实现候选文件快照绑定。
2. 给真实 DeepSeek 做新的冻结源码复验：检查冗余读取、无效工具调用、晚提交补丁、预算耗尽；记录完整调用序列、Tokens、失败和真实 GPU verifier，不从旧成功次数推断。
3. 在外部副作用可去重/人工协调之前，不承诺“任何进程崩溃都无缝恢复”，更不把私有检查结果反馈给模型。
4. 与旧固定 Workflow 在同一案例、成本预算、随机种子和独立隐藏集上对照，验证自研 Harness 是否真正改善任务完成率与成本。
