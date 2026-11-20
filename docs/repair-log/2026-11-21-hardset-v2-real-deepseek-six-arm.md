# HardSet V2｜真实 DeepSeek CUDA 自主调试 Agent 验收（2026-11-21）

## 本次实际完成了什么

这轮**不是脚本化模型**。模型使用 `deepseek-v4-pro`（`api.deepseek.com`），针对真正运行失败的三份 CUDA kernel，依次执行 LangGraph 调查、真实 GPU 功能探测、结构化诊断、**模型自己生成统一 Diff**、受控 GPU 公开自检、最后由独立的实验性严格验证器处理隐藏输入和四种 Compute Sanitizer 检查。

固定 **运行源码 SHA**：`ca01ec6deb02ecade2a23e21ded52552d9409f83`；Tang NVIDIA RTX 4090 Laptop GPU，隔离 Docker CUDA 执行。三项原始任务都是**已在开发中使用的 HardSet V2 公开案例**，不是未见训练数据的盲测。每个案例每臂只执行一次模型尝试，不允许根据失败重新抽样。

**研究含义：已经证明完整的 Agent 能够由真实 LLM 自主产生正确补丁并修复三种不同 CUDA 功能故障。尚未证明 Agent 比规则更好，也不代表大规模泛化成功。**

## 六单元真实 LLM 对照

| CUDA 任务 | D 规则探测+LLM 修复 | E 自主探测+LLM 修复 | D 模型请求 / Tokens | E 模型请求 / Tokens |
|---|---|---|---|---|
| h001 · Stable Softmax32 | VERIFIED_FIXED | **VERIFIED_FIXED** | 2 / 4,661 | 5 / 9,726 |
| h002 · Tail RMSNorm32 | VERIFIED_FIXED | **VERIFIED_FIXED** | 2 / 4,273 | 5 / 9,395 |
| h003 · Two-stage Group Center32 | VERIFIED_FIXED | **VERIFIED_FIXED** | 2 / 4,489 | 5 / 9,720 |
| **总计** | **3/3** | **3/3** | **6 / 13,423** | **15 / 28,841** |

D 和 E **使用相同的真实 DeepSeek 模型进行诊断和补丁生成**；区别在于 D 固定选择公共探测输入 `32,33,65`，而 E 让模型在 LangGraph 中根据已观察证据选择 `FunctionalProbeAction(size)` 或 `finish_probe`。所以 D 是强规则**调查基线**，不是完全不调用 LLM 的规则补丁生成器。

E 的 15 次模型请求细分为：**9 次 plan、3 次 diagnose、3 次 patch**，所有请求已完成，无这六个运行内的模型异常。E 有 8 次实际 GPU Probe（h001 2，h002 3，h003 3），D 每题 3 次合计 9 次。E 模型规划导致总体模型 Token 开销约为 D 的 **2.15 倍**，并没有获得更高严格修复率。

六个候选全部一次生成、一次公开 GPU 自检通过、一次独立验证通过；没有用失败试验换随机种子的行为。原始 `api_cost_usd` 是 `null`（费用尚未依据真实费率审定），**不能填 0 美元**。

## 模型真正修改了什么

核对 E 模式候选 `kernel.cu` 的 Git-independent SHA256 和受控执行产物：

- **h001 / Softmax32**：真实模型将 `denominator += expf(x)` 修改成 `denominator += expf(x - maximum)`，并将 `out[i] = expf(own)/denominator` 修改成 `expf(own - maximum)/denominator`；补丁 SHA256 `88dd297b8a0bc53dc3b93b7d1a4678551346b4855d7139b4edaddecef0070ba4`。
- **h002 / RMSNorm32**：将错误的 `sum_sq / 32.0f` 改为 `sum_sq / static_cast<float>(valid)`，处理最后不足 32 个元素的分组；SHA256 `2a6b46543d50286749e7eb310fa1ed630db93c6d32dd25693c38ca2dc607b567`。
- **h003 / 两阶段 Kernel**：将第一阶段错误的 `total += a[start + j]` 改为 `total += a[start + j] + b[start + j]`；SHA256 `dd2f7ba2889c6870416f3aabca6bcdf18b16a00fea0cbe1fd1124ea3b17519e4`。

三份 E 候选源码 SHA256 都与仓库中的正确实现一致，但**正常版本没有作为 LLM 输入**。在真实运行中，Agent 只能看到错误源码、功能要求、已经运行的公开 Probe 证据及其诊断结果；独立隐藏输入和正确源码不在模型上下文中。这里的“补丁源码等于 Clean 参考”是**事后验证结果**，不是下发给模型的答案。

## 真实验收门禁

每个运行：

1. `investigate`：模型真实选择公共 GPU 功能 Probe；证据经 `HardsetFunctionalProbe` 和控制器记录，拒绝 Shell/路径/未注册尺寸。
2. `diagnose`：LLM 生成结构化根因和源码行，引用真实公开 `evidence_ref_id`。
3. `patch`：LLM 输出只允许 `kernel.cu` unified diff；PatchScope、哈希及大小约束由控制器执行。
4. `public_selfcheck`：真实编译运行候选，受控数值 Oracle 检查并执行四种 Sanitizer。
5. `independent_verify`：**在 Agent 模型循环退出后**由 controller 在单独 evaluator RunStore 执行 9 组 HMAC 派生的非公开输入 + 4 类 Sanitizer；没有私有结果回流给模型。

六个运行共 **54 组隐藏数值测试、24 次独立 Sanitizer 结果记录**，均通过。每份 parent public RunStore 检查都不存在隐藏/私有/evaluator Artifact。六份模型调用均留有独立、可核验的物理 Invocation 记录，状态均 `COMPLETED`。

## 原始证据与机器路径

Tang 原始根目录：

`/home/you/gpu-agent-autonomous-validation-20261121/`

- 代码：`source-ca01ec6/`，必须用 `git rev-parse HEAD` 验证。
- E 模型结果：`live-deepseek-h001-ca01ec6/`、`live-deepseek-h002-ca01ec6/`、`live-deepseek-h003-ca01ec6/`。
- D 对照：`live-deepseek-h001-D-ca01ec6/`、`live-deepseek-h002-D-ca01ec6/`、`live-deepseek-h003-D-ca01ec6/`。
- 汇总：`real-deepseek-six-arm-summary.json`（SHA256 `474d5a962aae4c2933d9322ca582bb24938b809252dfd22a63538ca2c6fef4f2`）。
- 所有单元的 `investigation/public/<run_id>/` 有真实模型的 Started/Completed Invocation、诊断、补丁 hash、探测与公开自检反馈；`strict/evaluator/runs/` 保存私有原始轨迹，仅供授权审核。

随本报告提交的 [机器可读六单元汇总](2026-11-21-hardset-v2-real-deepseek-six-arm.json) 含每单元报告 SHA256、原始 run_id、Token、调用细分、输出哈希、私有检查数量，以及无私有数据泄漏审查结果。**不会将 evaluator 密钥、隐藏输入或私有运行日志上传 GitHub。**

## 可重现命令（仅在有 CUDA Docker、已授权真实模型和本机私有 evaluator key 的环境）

```bash
# 使用固定代码版本；禁止在未确认版本时把结果归到本次 SHA
git rev-parse HEAD

# Agent E：模型自己选择工具、诊断和修补
gpu-agent benchmark run-hardset-agent \
  --repository "$PWD" --case h001 --mode E \
  --max-probes 3 --max-candidates 2 --max-llm-calls 15 \
  --allow-paid-calls \
  --evaluator-secret-file /operator-only/private/secret.bin \
  --output /new/path/h001-agent

# Rule D：固定调查顺序，但仍由同一个 LLM 诊断/修补
gpu-agent benchmark run-hardset-agent \
  --repository "$PWD" --case h001 --mode D \
  --max-probes 3 --max-candidates 2 --max-llm-calls 15 \
  --allow-paid-calls \
  --evaluator-secret-file /operator-only/private/secret.bin \
  --output /new/path/h001-rule
```

输出目录必须不存在，模型密钥只能由宿主受信任环境提供，不能复制到脚本、README、模型上下文或 public RunStore。

## 仍然没有证明的内容

- **本次只是一轮公开开发集每臂每题 1 次尝试，不能说明统计显著性或跨模型泛化。** HardSet 的故障类别被公开功能要求提示，且 v1 Patch Prompt 包含一些通用的 CUDA 错误检查建议；正式外部 OOD 评测应冻结中立 Prompt 并审核模型训练集污染风险。
- **E 没有超过 D**。同为 3/3，D 明显减少模型调用。后续要针对真正有搜索价值的复杂 Bug 证明证据驱动规划优于高质量规则，而不是在简单分母上继续增加模型请求。
- 当前独立验证器是 **HardSet V2 实验版**，不是 v1 完整的签名生产 Release；安全/性能/跨文件任务和持续恢复仍需各自验收。
- 本报告只标记在固定源码上的实际结果。后续仅写文档的 Git 提交不能继承为“新代码已运行 GPU”。

**结论：完整 CUDA Agent 自主调用和修补闭环已经在真实 LLM + 真实 GPU 上得到正向证据。项目的研究阶段目标（优于强规则基线和多家族独立泛化）尚未达到。**
