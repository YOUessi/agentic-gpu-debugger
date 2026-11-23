# 2026-11-23 · 自研仓库级 Agent：绑定注册任务的真实 CUDA 公开 GPU 检查

## 问题与设计约束

原有 `run_repository_coding_session` 支持 LangGraph 自主模型选工具、多文件受限编辑、Git Worktree、SessionJournal 和显式 Resume，但是 **finish 只代表已暂存 Diff**，不能直接验证 CUDA 数值语义。原有 HardSet 单文件 Agent 已有 `check_candidate_publicly()`：复用固定 CPU Oracle、CUDA Docker、Numeric Check 和 4 个 Compute Sanitizer，无需引入模型选择 shell 命令的能力。

本次改造采用受限的 **Registered Public Case Capability**，而不是将任意多文件修改假装可以直接交给 HardSet Oracle：

- 只有操作者明确传 `--public-case h001|h002|h003`，仓库 Agent 才暴露 `repo_run_public_checks`；
- 操作者所选 Case 必须和冻结 Git SHA 下的 `benchmarks/hardset_v2/manifest.json`、原始 mutant 的字节哈希与公共 harness 一致；
- 工具无参数，模型不能更换样例、给出自定义 CUDA 命令/测试代码、修改 GPU flag、绕过 Compute Sanitizer，也不能读取独立隐藏验证；
- 必须恰好只编辑该 Case 的 `benchmarks/hardset_v2/<case>/mutant/kernel.cu` 一个文件；**任何第二个文件被改动，都拒绝调用 HardSet 单文件 checker**；
- 后端再次验证受控文件的 SHA256 和整个 Git Diff，记录 candidate hash、Diff hash、public RunStore 子 run；
- 一次失败会反馈 `BUILD_FAILED`、`NUMERIC_MISMATCH`、`SANITIZER_FAILED` 等受限公共字段；修改后可再次测试，不可在同样 candidate 上烧掉重复检查；
- 默认公开检查预算为 2，最大为 3；`finish` 只允许当前完整 Diff 与真实公开检查通过的 Diff 相同；
- 即使 **public_checks_passed=true**，`RepoHarnessOutcome.status` 仍是 `DIFF_STAGED_UNVERIFIED`，**不能**宣称已经通过独立 Strict Verifier 或隐藏测试。

通用多文件任务仍可使用 `repo_list_files`、`repo_read_file`、`repo_replace_snippet`、`repo_git_diff`、`finish`，但没有对应的测试契约时不能借用 HardSet 的数值正确率。通用多文件编译/测试需要下一阶段独立设计受限目标白名单与可复验的测试清单。

## 完整过程

```text
Operator: frozen SHA + goal + --public-case h001 + budget
              |
          LangGraph model
              |
      repo_read_file / repo_replace_snippet
              |
         PinnedGitWorkspace
        (HEAD, all files and edits checked)
              |
      repo_run_public_checks (no arguments)
              |
           Controller
    |- exact registered case/source path
    |- only one edited kernel file
    |- candidate SHA + full Diff SHA
    |- candidate uniqueness + check budget
    \- check_candidate_publicly()
                |
       IsolatedGPUBackend (Docker)
          |- NVCC build
          |- fixed public sizes and CPU Oracle
          \- four Compute Sanitizer tools
                |
      Public RunStore child evidence
                |
       ToolObservation pass/fail
                |
      Model may revise candidate
                |
  finish only if latest source/diff passed
                |
  DIFF_STAGED_UNVERIFIED
  (independent verifier remains outside the loop)
```

## 实际运行（需要已配置的 CUDA/Docker 和模型）

```bash
REPO=/absolute/path/to/agentic-gpu-debugger
BASE_SHA=$(git -C "$REPO" rev-parse HEAD)
gpu-agent harness repo-run \
  --repository "$REPO" --base-sha "$BASE_SHA" \
  --goal "Diagnose and fix the registered h001 Softmax32 kernel" \
  --public-case h001 --max-public-checks 2 \
  --max-steps 18 --max-model-calls 24 \
  --output /absolute/new/private/h001-repo-session \
  --allow-paid-calls

# 若启动时显式使用 --pause-after-turns 并正常暂停，
# 使用完全相同的 SHA、目标与调用预算显式 --resume。
```

不能复用现有输出目录覆盖证据。脚本化测试替身在 CPU CI 中不产生真实模型物理请求；只有在真实模型配置下运行上述入口并记录独立产物，才能声称发生了真实 DeepSeek 自主修复。

## Resume 与证据安全

- 每次 `tool_completed` 后 checkpoint 持久化额外字段：operator case、check budget、已尝试 candidate hashes、最后检查的 candidate/diff/public RunStore ID、是否通过。
- 恢复前检查 hash-chain Journal、冻结 Git SHA、RunStore 主 run 状态和公共检查子 run 的类型、父 ID、完整 artifact 及 pass/fail；不能仅凭 checkpoint 上的布尔值恢复为通过。
- 已完成的 CUDA 检查不会因为进程重启自动再次执行，也不会额外赠送检查预算；尚未明确完成的容器/付费模型外部副作用不允许无条件重放。
- 对外仍只开放 public 工具结果；独立隐藏验证结果从未进入该工具词表。

## 验证矩阵与当前限制

`tests/unit/test_gpu_harness_repo_public.py` 使用**模拟 GPU 检查器**、真实 Git Worktree、真实 RunStore 和真实 LangGraph Loop，验证：检查成功、失败后修改、重复候选拒绝、第二个文件污染拒绝、没有操作者选择时工具隐藏、提前 finish 拒绝、暂停恢复检查不会重跑、公共校验文件损坏拒绝恢复。

以上是 **CPU 零费用测试**，并不是 GPU 通过率实验；CI 需以对应 commit 的工作流结论为准。尤其真实 `check_candidate_publicly()` 的 Docker 运行结果必须在 Tang RTX 4090 Laptop GPU 上冻结源码单独留存，不得挪用上一轮 scripted Harness V2 的成功数据。

下一批：将注册 Case 的 GPU 调用做受控真实模型复测、复用之前失败 h001 的诊断轨迹，比较旧规则 Workflow、现有 HardSet agent 和新 repository Harness 的成功率、Token、工具冗余、修复延迟与成本，保留所有失败。
