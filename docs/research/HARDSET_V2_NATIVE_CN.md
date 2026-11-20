# HardSet V2：三种数值与跨 Kernel 故障的原生 GPU 验收

**状态：开发中的真实 CUDA 原生验收集。不能将下述“设计任务”或离线测试称为模型已修复成功。**

本模块扩展 [GPU Debugger Benchmark V2](GPU_DEBUGGER_BENCHMARK_V2_CN.md)。它使用旧系统已经具有的受控 CUDA C++ / JSON 向量协议、\`IsolatedGPUBackend\` Docker GPU 执行、四种 Compute Sanitizer 和不可变 RunStore，不触碰隐藏 Oracle，不调用模型，不改现有 A–E 公开实验分母。

## 为什么不直接加入旧 \`diverse-registry.json\`

现有 \`validate-diversity\` 在 mutant 侧要求某种明确的 \`SanitizerTool.FINDING\`。这适合 OOB/race/init/sync，却无法检测**纯数学功能错误**：Sanitizer 全 CLEAN，CPU 参考和实际输出不同。新 HardSet 因此保留独立的审核及原生运行入口，不伪造 sanitizer finding 来让旧代码通过。

## 当前三个新题

| Case | 算子与正确语义 | 故障机制 | 为什么值得给 Agent |
|---|---|---|---|
| \`h001\` | **Softmax32**：对每个最多 32 元素的组计算 \`softmax(a[i]+b[i])\`，必须先减去组内最大值再算指数 | mutant 直接 \`expf(logit)\`，对高 logits 发生溢出、产生 NaN/Inf；可在小幅输入下偶然无异常 | Sanitizer 无法单靠内存/同步诊断数值稳定性，必须理解 \`logsumexp\` 数值语义 |
| \`h002\` | **RMSNorm32**：\`x[i]=(a[i]+b[i]); out[i]=x[i]/sqrt(sum(x²)/valid + 1e-5)\` | mutant 在最后不足 32 个元素时仍除以 32，而非该组有效长度 \`valid\` | 需要不规则长度、尾组及数值测试；许多原生长度恰好为 32 倍数时会漏检 |
| \`h003\` | **Two-stage Group Center**：第一个 Kernel 计算组均值，第二个 Kernel 从每个 \`a[i]+b[i]\` 中减去组均值 | mutant 第一阶段只累加 \`a[i]\`，漏掉 \`b[i]\`，两个 Kernel 同样能够编译/正常运行 | 修复必须分析跨 Kernel 数据流，而不是仅看一行越界 |

这些全是**自编写、可审计的源码模板**；mutant 由确定性、一致性检查过的字符串替换生成。生成器 \`scripts/generate_hardset_v2.py\` 不执行模型输出、不下载第三方程序、不接受用户自报 Shell。

本阶段使用原有 \`n, a, b\` 的严格输入/输出协议，以验证新的 CPU Oracle 和 GPU 功能证据流程。**它尚不是任意维度 tensor、真正多文件仓库级 CUDA 修复**，不能替代后续 CUTLASS / FlashAttention 任务适配层。

## 独立 CPU Oracle / 故障证据

- Python CPU Oracle 在 \`src/gpu_agent/benchmark/hardset_v2.py\` 中实现，不调用或读取运行时 CUDA 候选进行“参考”。
- 所有输入在受控 JSON 域上验证，\`n\` 严格正整数，\`a,b\` 为等长有限浮点列表。
- \`Softmax32\` 使用数值稳定 \`exp(x-max)\`；\`RMSNorm32\` 用有效长度统计；两阶段 Center 先计算**两数组合成**的组均值。
- 检验形状包括 \`1,31,32,33,65,129,257\`，即空尾、单元素尾、不规则最后一个组和跨 block 情形。使用浮点容差 \`atol=rtol=3e-5\`，保证阈值在真实 GPU 结果出现前冻结。
- 每个 clean 源码在全部尺寸上必须通过 CPU Oracle，并对长度 65 的输入分别通过 \`memcheck/racecheck/initcheck/synccheck\`，且 instrumented 程序结果数值仍正确。
- 每个 mutant 必须**实际编译并运行**，有至少一个预声明输入通过独立 Oracle 判定功能错误（可以是数值 mismatch 或 C++ harness 拒绝非有限输出）；不得以源码 diff 猜测成功。
- 每次编译/运行/Sanitizer 的真实日志写入当前任务的不可变 RunStore，并生成单独的 \`report.json\`；已有输出目录禁止覆盖。

## 本地及 Tang 验证

首次生成或重放（\`--verify\` 只审查，**不修改**已有文件）：

\`\`\`bash
python scripts/generate_hardset_v2.py --verify
python -m pytest -q tests/unit/test_hardset_v2.py
\`\`\`

有受控 CUDA Docker 和 RTX 4090 的 Tang 环境上：

\`\`\`bash
gpu-agent benchmark validate-hardset \
  --repository "$PWD" \
  --output /path/to/new/hardset-v2-native
\`\`\`

\`report.json\` 只有在三个案例 clean + mutant 的原生检查均成功时才报告 \`passed: true\`。即便通过，该结论仍是 **\`REAL_NATIVE_GPU_ACCEPTANCE_ONLY_NOT_AGENT_FIX\`**，不是 Agent 已自动修复，也不是发行版本的 blind holdout 分数。

## 必须进一步完成的工作

1. 将三类 **纯功能故障** 接到新的 Agent 动作空间：可以主动调用受控 CPU Numeric Oracle / 对抗形状测试（而不仅仅 sanitizer）；强规则 D 与 Agent E 均允许使用同样工具。
2. 可信注册 CPU Oracle、case 的 SHA256、独立 evaluator 种子，严格隔离模型与参考实现；不能直接复用旧 \`VerificationTruth\` 对象冒充已经批准的新算法。
3. 在代码和工作流冻结之后开展 D / E baseline / E adaptive 实际 GPU+LLM 多种子配对，对照源错误、补丁范围、严格修复、Token 与 GPU 开销；**离线成功不代表 Agent 优于 D**。
4. 多文件 CUDA 工程、Stream/Event、KernelBench/CUDABeaver 外部基准仍需另加可信 Adapter，不运行元数据中的任意 Shell。
