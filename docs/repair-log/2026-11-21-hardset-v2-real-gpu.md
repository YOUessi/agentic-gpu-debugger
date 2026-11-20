# 2026-11-21：HardSet V2 新任务真实 RTX 4090 GPU 验收

## 证据级别

**范围：自编 CUDA HardSet 的原生功能/数值与 Sanitizer 验收，尚不是 Agent 自动修复、盲测、已注册的独立严格 Verifier 或生产 Release。** 使用审定并冻结的项目 CUDA 源码、隔离 Docker GPU 执行和 CPU Oracle；没有模型 API 调用。

### 固定代码和环境

- GitHub 仓库：\`YOUessi/agentic-gpu-debugger\`
- 正式加难版完整源 SHA：\`b20645fad5ebe8bc6194066315facb7161647a62\`
- 前一版真实 GPU 通过源 SHA：\`b4df9dae673e3797807aa04060c74ffebc50375f\`，其 Softmax Mutant 在所有 7 个输入上都失败，之后为提高任务区分性增加混合 easy/stress 输入，而不是模型跑出来不理想后调参。
- 最初编译失败快照：\`650b986c61ea2fb60b1ad5a24def3cd8004e1a87\`，\`h001\` clean/mutant 未通过编译，NVCC 明确报 \`CUDART_INF_F\` 未定义；其他两题原生验收通过。该失败证据保留在独立 \`native-evidence\`，不是被覆盖为成功。
- 计算设备：Tang NVIDIA RTX 4090 Laptop GPU，Ada SM89，项目 \`IsolatedGPUBackend\` Docker GPU，无模型 API。
- 源码注册元数据 SHA256：\`ebe0abf0378f89be88040abe418d4db3eff5315b7900f8d2c7527de00298e063\`。
- 验收使用固定 \`SIZES = (1,17,31,32,33,63,64,65,127,128,129,256,257,511,513)\`，\`NumericOracle\` 的 \`atol=rtol=3e-5\`；在比较前声明输入长度、数学语义和容差。

## 3 个真实任务的逐项验收

| 任务 | Clean 数值 Oracle | Clean 四种 Sanitizer + 数值 Oracle | Mutant 通过输入 | Mutant 失败输入 | HardSet 原生准入 |
|---|---:|---|---|---|---|
| \`h001\` Stable Softmax32 | **15/15** | 四工具 CLEAN、结果正确 | 1、17、31、32、63、64、127、128、129、256、511 (**11**) | 33、65、257、513 (**4**) | **PASS** |
| \`h002\` Partial RMSNorm32 | **15/15** | 四工具 CLEAN、结果正确 | 32、64、128、256 (**4**) | 1、17、31、33、63、65、127、129、257、511、513 (**11**) | **PASS** |
| \`h003\` Two-stage Group Center32 | **15/15** | 四工具 CLEAN、结果正确 | 32 (**1**) | 1、17、31、33、63、64、65、127、128、129、256、257、511、513 (**14**) | **PASS** |

总计 Clean **45/45** 组 GPU 数值执行通过，Clean 四工具检查 **12/12** 检出为 CLEAN 且工具运行输出正确。Mutant **19/45** 个输入正确、**26/45** 个输入被数值 Oracle/输出契约拒绝。这里的 Mutant “PASS” **不是修复成功**，只是用于证实一部分普通输入无法暴露故障；真正的 HardSet 目标是同时存在至少一个实际正确样本及一个实际失败样本。

- \`h001\` 的极端 logits 表明未减最大值的 \`expf\` 会产生溢出/非有限输出，普通 logits 则可能正确。
- \`h002\` 在完整 32-lane 组内不报错，部分组才暴露不正确的归一化分母。
- \`h003\` 在 32 元素的第二输入平均值恰为零时，跨 Kernel 漏算 \`b\` 被掩盖，其余尺寸暴露功能误差。
- 这些都是**真实 CUDA 编译与执行结果**；只检查 Sanitizer 无法发现三者的核心数值/语义错误。

## 真实文件路径和 SHA256

以下文件在 Tang 保留原始内容，没有覆盖失败版本：

- Final \`report.json\`：\`/home/you/gpu-agent-hardset-v2-20261121/native-evidence-b20645f/report.json\`，SHA256 **\`a65606e706c98f8fd4dbb236f476e340ac063b9c541057a22b03e215b656640b\`**。
- Final native stdout：\`/home/you/gpu-agent-hardset-v2-20261121/native-b20645f.log\`，SHA256 **\`77ee47dca285d664144fcc626c0b87a38850841db447abd5ec564245c895db3d\`**。
- 前版本 \`b4df9da\`：\`native-evidence-b4df9da/report.json\` SHA256 **\`041aac870e4961815ff0e7167e096d1f037bf3599a4abb9497559afd1172b3f1\`**。
- 初始失败 \`650b986\`：\`native-evidence/report.json\`、RunStore 中 \`CUDART_INF_F\` NVCC 编译报错均保留。
- 每个 clean/mutant 有独立的 RunStore ID 和原生编译、程序运行、Sanitizer 工具原始 artifacts；完整原生结果与模型 API 证据分开。
- 最新 \`b20645f\` 离线本地 \`pytest tests/unit/test_hardset_v2.py\` 为 **83 passed**；CI 的相同 SHA 应独立查看，不用其他提交的结果替代。

## 研究结论和硬性剩余项

可以确认：**已实现三类独立、比 OOB 更复杂的数学/跨 Kernel 故障，数值 Oracle 和边界输入在真实 GPU 上能够可靠区分 clean 与 mutant；Docker 隔离/四 Sanitizer 对 clean 源码没有可见退化。**

不能确认：LangGraph/Agent 是否能够自动从这些功能故障恢复；原有 \`PublicTask\`/独立 \`VerificationTruth\` 尚只覆盖已注册七种 Oracle；HardSet 原生通过不自动授予 \`VERIFIED_FIXED\`。此外尚未建立 BF16/FP16、多维 tensor、多 Stream Event、CUTLASS/GEMM 性能保真或 CUDABeaver 36 候选的本机执行适配。

下一批应该增加**规则与 Agent 公平共享的受控 \`functional_probe\` 工具**（不要由 Agent 自选命令或执行外部 checker），然后为这些 HardSet 算法注册版本化公共功能规则及 evaluator-owned 隐藏输入。必须先冻结 oracle/输入/候选权限/源码 SHA，再对 D/E/E_ADAPTIVE 进行多种子真实配对 GPU+LLM 实验。未获得配对结果前，不宣称 Agent 优于规则，也不将原生准入数放进 Agent StrictFix 分母。
