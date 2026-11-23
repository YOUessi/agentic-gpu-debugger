# 2026-11-23 · Repository Harness 注册 HardSet h001：首次真实 Docker GPU 双轮公开检查

## 本次实际完成了什么

在 Tang 的 **NVIDIA RTX 4090 Laptop GPU** 上，以冻结 GitHub 源码运行新增 `repo_run_public_checks`：模型动作由**脚本化替身**给出，实际 CUDA 构建、Numeric Oracle 与 Compute Sanitizer 在隔离 Docker 后端执行；并非模拟 GPU，也不是 DeepSeek 自主修复。

**真实 GPU 链路已通过**：第一轮仅修复 Softmax 分子时，公开数值检查正确报告 `NUMERIC_MISMATCH`；第二轮修复分母后，公开数值检查及 `memcheck`、`racecheck`、`initcheck`、`synccheck` 均通过（四项 `CLEAN`）。在 `finish` 后保留 `DIFF_STAGED_UNVERIFIED`，因为独立隐藏验证**没有运行**。

## 冻结源码与运行环境

- GitHub 仓库：YOUessi/agentic-gpu-debugger，分支：feat/gpu-native-coding-harness-20261123。
- **实际运行 GitHub 源码提交**：`786bc35725edbfd4cd1a3dcaa94589cef9d72c21`。
- GitHub codeload 归档 SHA256：`79c22d213f5be72257541ca5ffd463aad1b45e16c6dfdf014f506588004261e9`；归档大小 2,319,257 bytes。
- Tang 在独立目录解压归档后重新 `git init` 并 commit（仅为 SHA-pinned Worktree 验收）：本地 Git SHA=`09b756df102eabb737f580d5f1cd2465be5f527b`。**本地 Git SHA 不等于原始 GitHub commit；两者由固定归档哈希关联，不要混淆。**
- 代码文件 523 个；独立 Python 3.11 环境按 `requirements.lock` 安装（镜像下载而非修改锁定版本），`pip check`：No broken requirements found。
- Tang 验收源码路径：`/home/you/gpu-native-harness-validation-20261123/repo-public-gpu-786bc357/source`。
- 独立运行结果根目录：`/home/you/gpu-native-harness-validation-20261123/repo-public-gpu-786bc357/run-h001-public-smoke`。保留在 Tang 本地，未将二进制、Docker 私有对象或模型密钥上传到仓库。

## 实际工具序列与 GPU 证据

| 指标 | 实测 |
|---|---|
| Agent 决策来源 | 固定 ScriptedPublicCudaAgent，**不是付费/真实 LLM** |
| 工具动作 | 6 次：read → replace(分子) → public check → replace(分母) → public check → finish |
| 真实 GPU 公开检查 | 2 次，先正确检出失败，后通过 |
| 模型物理调用 / Tokens | 0 / 0（测试替身） |
| 独立隐藏验证 | **未运行** |
| 最终状态 | `DIFF_STAGED_UNVERIFIED`，公开检查通过但未完成独立验证 |
| RunStore public parent ID | `dd4b207acc4b4f27a2d9c16fa6a6ac29` |
| Journal | 26 条，`recovery_disposition=FINISHED` |

### GPU 检查 1：正确识别未完全修复

- public child Run ID：`2ec09dd15087467bb935a892161102d5`，状态 `FAILED`。
- candidate SHA256：`992264e8d9c65ba504357399a1d8229fe03a6396f215f0e226da5618ea9c9f73`。
- 公开反馈：`NUMERIC_MISMATCH`，`passed=false`；16 项公开工件。
- 此次未进入全部 Sanitizer，因为数值 Oracle 已发现问题。**这一失败不应被删除或计为通过。**

### GPU 检查 2：真实公开验证通过

- public child Run ID：`622daacdeb03468c9d4b855c4dd5bbe0`，状态 `COMPLETED`。
- candidate SHA256：`88dd297b8a0bc53dc3b93b7d1a4678551346b4855d7139b4edaddecef0070ba4`。
- 公开反馈：`PUBLIC_CHECKS_PASSED`，`passed=true`；57 项公开工件。
- 已读取四份独立 Sanitizer result.json，均为 `check_outcome=CLEAN`。经 RunStore 检查两次子运行的工件 `visibility=public`，没有私有评测工件进入这两个公开运行。

## 证据完整性与哈希

- stdout 摘要 SHA256：`24b7715f5727edafaa0d0515ff063c19ff8527559c88eeebd4bb34c5415a20c5`。
- `result.json` SHA256：`a0abbba9e414457eadd62f09afdb4c88bb245f7a442dd1650e8495dd9afb7a09`。
- `session-events.jsonl` SHA256：`6d8566e58ae7a5118437236f8f18c4cdba787351bc42af10923c4e2d60cc25b2`。
- `session-checkpoint.json` SHA256：`3a9a026ad094470973876dc473f097dc45dbec35c64c30eab28126c1171284fa`。
- 对应机器可读记录见 [`2026-11-23-repo-registered-gpu-smoke.json`](2026-11-23-repo-registered-gpu-smoke.json)。

## 复现

```bash
R=/home/you/gpu-native-harness-validation-20261123/repo-public-gpu-786bc357
# 为保留现有实验，重新复现时必须指定一个新的输出目录。
"$R/venv/bin/python" -I "$R/source/scripts/run_repository_hardset_public_gpu_smoke.py" \
  --repository "$R/source" \
  --output "$R/run-h001-public-smoke-second" \
  --allow-real-gpu
```

以上命令在有 Docker NVIDIA Runtime、已锁定 CUDA 镜像及 RTX 4090 资源时可用于重新验收，**不得**把运行程序预期输出当作再次实测的成绩。

### 与研究目标的差距

本次验证的是 **自研 LangGraph Harness → 受控 ModelTool host → 真 CUDA GPU 构建/反馈 → 修订 → 公共检查** 这一工程闭环；尚无本版本上的真实 DeepSeek 自主动作选择实验，不能说 Agent 比旧规则 Workflow 更好。下一阶段需要在同一冻结源码与预算上做真实模型、多种案例与独立隐藏评测，比较修复率、动作冗余、Tokens 和成本。通用多文件测试仍未建立合法的数值 Oracle，只能在明确的注册 Case 范围内判断通过。
