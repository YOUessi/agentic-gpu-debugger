# 2026-11-23｜自研 LangGraph Harness V2 首批验收（明确区分模拟模型与真实 GPU）

## 版本与目标

固定执行代码：`4c619084fddcafc06f8aa6cfa6f5b5372fa10f91`，对应 GitHub PR #16 的自研 Runtime V2。

本批验证的是 **Context 压缩、Tool Registry / Permission Policy、逐片段改代码、Model → Tool → Model 轨迹、SessionJournal 与独立 Strict Verifier 的工程闭环**。这不是新的真实 DeepSeek 修复成功实验，模型侧使用了确定性的 `ScriptedToolChoice`，物理模型调用数为 **0**。

## A. Tang CPU 离线回归

- Tang 已存在 Python 3.11/项目冻结 venv；原生 CUDA/GPU 驱动独立。
- 源码由 GitHub 固定 commit tarball 解压，SHA256：`c48fd7ecb9b3dce188a3df4daf47a69923947746fafc8cc19f25fb735da86d94`。
- 运行目标：`tests/unit/test_gpu_harness_runtime.py`、`test_gpu_coding_harness.py` 及已有 HardSet/Agent/LLM/Verifier 相关单元测试。
- **155 passed in 0.97s**；`pip check` 为 `No broken requirements found`。
- 修复后的 `replace_snippet` 最终仍提交受限 `kernel.cu` Unified Diff；不授权 shell 命令或隐藏答案访问。

## B. Tang NVIDIA RTX 4090 Laptop GPU 原生闭环（脚本化模型）

统一运行入口为 `run_gpu_coding_harness(..., model_factory=ScriptedToolChoice)`，仅用脚本化决策来验证真实后端的连通性，**没有使用任何 API 凭证**。

工具动作序列由测试替身给出：

1. `list_files`；
2. `replace_snippet`：`denominator += expf(x)` → `denominator += expf(x - maximum)`；
3. `replace_snippet`：`expf(own) / denominator` → `expf(own - maximum) / denominator`；
4. `run_public_checks`：真实 GPU Docker 编译、CPU 数值 Oracle、4 种 Compute Sanitizer；
5. `finish`：仅在当前候选通过公开 GPU 检查后允许。

结果：
- `status=VERIFIED_FIXED`、`private_verification=true`、`tool_calls=5`；
- 实际候选 SHA256：`88dd297b8a0bc53dc3b93b7d1a4678551346b4855d7139b4edaddecef0070ba4`；
- 独立实验性 Strict Verifier：**9 组隐藏数值输入 + 4 项 Sanitizer 检查通过**；
- 原始 `result.json` SHA256：`ec3b7ab81b5b4a3b67d68badee7a1441a6318bcd3044780279bdbabffaa3288b`；
- 公开 RunStore：`bbffc52cb9ab4d8bb490929d00ab1da5`；
- 公开 `harness/v1/event-*`：5 个工具事件；
- 逐步同步 `harness/v2/trace-*`：20 个 Model/Tool 起止事件；
- Hash-chain `session-events.jsonl`：22 条（含 Session 头尾）；`recovery_disposition=FINISHED`；
- Parent Public RunStore **没有**隐藏/私有/evaluator Artifacts；
- `model_physical_calls=0`、`model_tokens=0`（**仅因为使用测试替身**）。

Tang 原始目录：`/home/you/gpu-native-harness-validation-20261123/scripted-harness-v2-gpu-4c61908/`。

## C. 解释与仍未完成

**工程上**：证明了新建的 Context/权限/精确编辑/事件日志系统，已连接原来的隔离 CUDA GPU 后端、Numeric Oracle 和独立 Verifier；两个逐片段编辑可在真实 GPU 上完成提交、验证与最后的隐私隔离。

**研究上**：不能将这次测试算作真实 DeepSeek Agent 修复成功，也不能继承旧 Workflow 的 3/3 成绩。新的工具选择质量、Context 减少 Token 的实测幅度、被拒补丁后的真实 LLM 修订质量、跨任务泛化、自动恢复均未通过验收。

另外，当前的 `SessionJournal` 虽能保留 Write-Ahead 状态并阻止不确定操作自动重放，**还不是 LangGraph 完整的跨进程持久 Checkpointer/Resume**，后者需要真正的副作用幂等保障。

后续真实模型实验必须在新增的冻结源码 SHA 上单独记录模型调用、工具决策、候选、成功/失败、Tokens 和每个验证门禁；失败不能丢弃或重抽样。
