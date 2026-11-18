# GPU Debugging Agent Benchmark V2 — 研究与工程设计（2026-11-19）

**状态：设计 + 可执行的元数据候选选择器。外部数据集未完成本项目的 GPU 适配/严格验证；新增复杂任务没有被宣称为已运行。**

## 0. 为什么必须扩展

仓库现有 \`case_0001\`–\`case_0016\` 基于同一 Vector Add 类模板的大量故障注入；\`case_0017\`–\`case_0022\` 引入 6 个额外工作负载。当前 \`PublicTask\` 只支持七种 Oracle 语义以及 \`n,a,b → float32 vector\` 数据协议；模型补丁只修改 \`kernel.cu\`。这非常适合受控正确性实验，却不是充分代表真实 GPU 工程的任务空间。

2026-11-14 的真实 DeepSeek + RTX 4090 五臂探索（[原始报告](../docs/repair-log/2026-11-14-five-arm-real-gpu-llm-comparison.md)）只有两道开发集公开任务各执行一次：固定规则 D 2/2、E_ADAPTIVE 2/2；后者调用模型 8 次而 D 只要 4 次。不能据此证明 Agent 优势；也不应靠单纯增加相似 OOB 用例证明算法改进。

## 1. 对标的公开数据资源与用途

| 来源 | 真实任务与特点 | 在本项目的定位 | 适配约束 |
|---|---|---|---|
| [CUDABeaver](https://github.com/UMN-APEX-Lab/CUDABeaver) | 真实损坏的 CUDA 工作区、构建/测试命令；当前 **固定 Git 提交**的 \`manifest.json\` 为 **199** 实例、11 task families；编译/逻辑/内存/性能/超时五类。论文实验数据集数量可能不同。 | **第一优先：Debugging Benchmark 外部候选集** | 需要适配原生脚本、CUDA/Python/CUTLASS/KernelBench 差异，完全不信任 manifest 中的任意命令 |
| [KernelBench](https://github.com/ScalingIntelligence/KernelBench) | PyTorch → CUDA/DSL 的生成与优化任务，含单算子、融合算子、模型级任务。 | 提供**复杂算子和外部功能语义**，不直接把 generation pass@k 当 debug score | 转换为具有真实 bug/固定 starting point 的 repair 任务，并独立校验 |
| [KernelBench-Verified](https://github.com/facebookresearch/kernel_bench_verified) | 多分布隐蔽正确性测试、精度/TF32 基线与显存效率，特别防止投机性优化。 | **复用评测原则**，加强 Oracle、性能退化门禁 | 不复制其隐蔽测试到 Agent 上下文；重新冻结适合本硬件的基线 |
| [GPUVerifyBenchmarks](https://github.com/mc-imperial/GPUVerifyBenchmarks) | GPUVerify 使用的 CUDA/OpenCL kernel；涉及 intra/inter-group race 等并发行为。 | 多线程/同步复杂样例的**静态分析和任务来源参考** | 需要 CUDA 编译版本兼容、功能 Oracle 与真实 GPU failure 复验；不能直接当可执行 repair 评测 |
| [NVIDIA CUDA Samples](https://github.com/NVIDIA/cuda-samples) | 官方 CUDA 功能示例，包含复杂运行时和多 GPU 特性。 | 开发领域知识及跨模块修复任务的参考素材 | 官方也明确提醒 Samples 不构成全面验证/性能 Benchmark |

外部基准所有题目都属于**公开可见数据**。即便从未在我们项目里用过，也不能声称它们对被测基础模型是“训练集外未知”的。正式研究保留经独立冻结的 evaluator-controlled 任务集，并对基础模型数据泄漏问题作明确限制。

### 当前外部候选源固定

- GitHub repo: \`UMN-APEX-Lab/CUDABeaver\`
- Git commit: \`991f0df14367e64907b9244295d5997e7717ffc3\`
- \`manifest.json\` Git blob: \`91d66049492e08765933af0b04cfe19a350454a3\`
- 官方 manifest：199 cases（当前 pinned 提交）；类别分布 \`compile_error=76\`、\`logic_error=45\`、\`memory_crash=32\`、\`perf_broken=41\`、\`timeout=5\`。数据集自带的 \`difficulty=hard\` 只是发布方标签，不是我们实际 GPU 性能测定。
- 原文与数据标记 CC BY 4.0；不直接复制第三方代码、私有参考和运行脚本进本项目。
- **重要**：原项目的参考性能缓存包括 Hopper/Blackwell 等 GPU；Tang 使用 Ada \`sm_89\`。禁止把外部 GPU 的 timing/speedup 直接用于本机评分。

## 2. 新 Benchmark 三层，而非只增加单一维度题

### B0 — 现有回归集（22 个公开案例，不纳入新任务泛化分数）

确保旧 Agent/Rule/Verifier、四种 Sanitizer、RAG 与 Patch Scope 不退化。当前 16 个 Vector Add 系列和六个 Diversity Case 都作为 **Regression**，不伪装成独立的新题。

### B1 — Self-authored HardSet（目标 24 个**新增、真实故障**，待实施）

至少覆盖以下四种完全不同的任务特征：

1. **单一根因但复杂语义**：LayerNorm/RMSNorm、Softmax、二维 Convolution、CSR SpMV、Fused GEMM+Activation、Tile Transpose，配不规则形状与高精度 Oracle。
2. **跨 Kernel 的证据链**：两阶段归约、Scan + Scatter、Histogram 初始化与合并、不同 CUDA Stream 的 H2D/Kernel/D2H 时序，要求读懂多个调用点而不靠单条 sanitizer 字符串。
3. **多根因任务**：一个地址越界 + 一个不影响常规输出的隐藏同步缺陷；修复第一个后需重新运行测试发现第二个。必须两个错误均修好才算成功。
4. **性能不退化的正确性修复**：避免添加整卡同步或 CPU fallback、序列化 Kernel、把 Tensor Core/GEMM 切为慢速标量代码；只有严格功能验证通过，才能比较性能。

候选任务设计表（**设计草案，不是已生成/已验证的 runnable cases**）：

| 代号 | 工作负载 | 注入问题与典型难点 | 需要的判定 |
|---|---|---|---|
| H01 | Block / Warp reduction | partial warp mask、末组 31/33/65 元素 | 分组 Oracle + sync 检查 |
| H02 | Shared-memory tiled transpose | 越界与错误的二维寻址 | 多种宽高、stride、memcheck |
| H03 | Segmented inclusive scan | 阶段 barrier 缺失、分段边界重置错误 | racecheck 重复运行 + prefix Oracle |
| H04 | RMSNorm FP16/BF16 | epsilon 与累积精度错误，Sanitizer CLEAN | FP32/FP64 参考与 dtype tolerance |
| H05 | Softmax | 极端 logits 溢出、全负值、尾部 mask | 数值稳定、NaN/Inf 检查 |
| H06 | Convolution | padding/stride/dilation 组合下的错误索引 | 非对称形状、随机参数 |
| H07 | Weighted histogram | 初始化缺失 + 非确定性竞争 | initcheck + 多 seed Oracle |
| H08 | Stream pipeline | 缺 Event/Wait 导致读到过期结果 | 异步执行 trace + 多轮一致性 |
| H09 | Multi-kernel reduce | 第二阶段读未初始化 partial sums | initcheck + 两阶段依赖验证 |
| H10 | Vectorized load/store | \`float4\` 不对齐、尾部误写 | memcheck + 对齐/长度 fuzz |
| H11 | CSR sparse-matvec | 空行、无序索引、极端 nnz 分布 | CSR reference + 跨形状 |
| H12 | Fused GEMM + activation | 正确结果但补丁退化为串行实现 | CPU/PyTorch Oracle + 性能保真 |

目标至少 24 个独立算法-根因组合，不通过给同一模板改常数或复制另一套文件名虚增分母。实际验收必须执行干净参考、错误样本、全部严格检查并保存独立 SHA/provenance。

### B2 — CUDABeaver 外部真实修复任务（先挑选 12 development + 24 frozen-public-eval **候选**）

使用仓库现有的 [元数据候选选择器](../../scripts/plan_debugger_benchmark_v2.py) 和 [固定策略](../../benchmarks/v2/cudabeaver_selection_policy.json)：

\`\`\`bash
# 先固定上游提交并取得其 manifest.json；不得使用浮动 main。
python scripts/plan_debugger_benchmark_v2.py \
  --manifest /path/to/cudabeaver-pinned/manifest.json \
  --output /path/to/new/cudabeaver-candidate-selection.json
\`\`\`

脚本在本地验证 Git Blob 标识、总数、实例 ID / basename / 格式 / 分布、唯一性；\`development\` 与 \`frozen_public_eval_candidates\` 按 **base stem（同一问题的多个变种）严格互斥**。选择优先平衡 task family，再按预声明随机 SHA 破平局。

配额：开发 12 = 编译 2、逻辑 3、内存 3、性能 3、超时 1；冻结公开候选 24 = 编译 5、逻辑 6、内存 6、性能 5、超时 2。这个 36 仅是 **metadata candidate**，不算 runnable、validated 或得分分母。受限硬件兼容性失败要保留剔除原因，在模型运行前按预声明算法确定替补；不得观察模型成绩后删题。

## 3. 从单 \`kernel.cu\` 升级为真实 Debug Agent 必须补充的能力

目前 \`ApplicationService\` 强制 \`PublicTask\` + \`vector_api.h\`、\`n,a,b\` float32 协议；\`RuleRouter\` 首先进行 memcheck；\`Agent\` 权限只允许固定 CUDA 调查动作。这些边界不能靠直接加载外部 \`build_command\` 绕开。

在新的 **Benchmark V2 受信执行面** 单独增加：

- \`BuildEvidence\`：合法编译参数和结构化 \`nvcc\` 错误；编译没通过时能够继续修复，不能先要求 memcheck。
- \`FunctionalEvidence\`：可控输入的参考输出比较；Sanitizer 全 CLEAN 但数值不对时必须保持可调查。
- \`SanitizerEvidence\`：对应官方四工具；Racecheck 主要报告 **共享内存** hazard，不能据它无报错证明所有 global-memory 竞争都不存在。
- \`TemporalEvidence\`：多 Stream、Event 顺序、超时、异步执行与多次重复结果。
- \`PatchScopeV2\`：allowlisted 项目内少数受信任可编辑文件；必须阻止改测试、Oracle、reference、build authority 和 evaluator 控制。
- \`PerfGate\`：正确性先验 + 同卡同精度基线 + warmup / 多次重复 + p50/p95 与显存；不允许丢掉优化结构的“退化修复”当高质量成功。

**外部 \`input.json\` 的 \`build_command\` / \`test_command\` 是不可信数据，不得作为 Agent 可执行的 Shell。** 只有人工/静态批准的任务家族适配器可将其编译成预注册执行计划。当前 Docker 不是 hostile multi-tenant 沙箱，仍不运行恶意未知 CUDA 程序。

## 4. 对照实验的严格协议

同一个冻结源码 SHA、同一模型版本、同一 GPU、同一 NVIDIA 知识索引、同一 Patch 上限、同一公开反馈级别、同一运行预算。主比较：

- D：\`LangGraph + deterministic RuleRouter\`（保留原固定规则 D；若扩大动作空间，D 与 E 必须获得**相同**合法工具）。
- E0：\`LangGraph + LangChain baseline planner\`。
- E1：\`LangGraph + LangChain adaptive-v1\`。
- 消融：E1 分别关闭强制步骤跳过、证据充分结束、异常降级，以判断减少调用究竟来自哪里。

按 **算法家族/根因类别** 先分层再分配；case×arm 在固定 seed 下排序，至少进行 3 个独立模型种子/尝试，失败完整保留，严禁动态补抽。离线/真实 GPU/付费 API 三类证据严格区分。

记录下列结果：
- **Strict Verified Fix Rate**：全部指定公开输入、额外隐藏输入、Sanitizer 和 Patch Scope 均通过的单元数 / **所有预声明单元**；无法启动与超时必须留在分母。
- **Family macro success**：每个真实任务家族平均后再宏平均，避免同模板海量变种主导总分。
- **Failure taxonomy**：编译拒绝、数值错误、race、uninitialized、超时、重复/无进展、输出/证据格式拒绝。
- **Cost**：实际物理 LLM calls、input/output/cached tokens、未知 usage、经审定的美元费用（未审定保持 null），GPU 工具调用数、总耗时。
- **Repair quality**：有效修复证据、补丁是否只改允许范围、是否引入性能/精度/显存退化、是否靠特定测试硬编码。
- **Agent advantage**：配对样本上的 \`Δ VerifiedFix\`、\`Δ Cost\`；同时比较 matched-budget 和 cost-at-equal-quality，不因 E 多花预算而鼓吹 Agent 优势。

统计采用按 **原问题 base stem / task family** 的配对 bootstrap 或离散配对检验；重复模型采样并不等价于增加独立任务数。先给点估计和区间，不因不足 24 个样本就谈“显著优于”。

只有新版本在预先冻结、未参与策略选择的案例上表现出可靠收益，才考虑改默认 planner；否则仍保留确定性 D，并明确哪些任务值得使用 Agent。

## 5. 交付/验收门禁（不按日期划分）

1. **元数据完整性：** 生成 12+24 base-stem-disjoint candidates，记录 upstream Git Commit、manifest blob、SHA256、seed；脚本故意损坏数据时 fail closed。
2. **原生可运行性：** 逐项在 Ada \`sm_89\` 和项目受限执行后端上验证，自动拒绝不兼容任务并记录原因；保持临时 GPU 输出与模型结果隔离。
3. **可信判定：** 参考实现、输入、checker、性能门禁经审核；干净参考应该 PASS，损坏实例应该暴露目标故障；错误但恰好通过旧测试的样例需被找出。
4. **Agent 真实执行：** 扩展受控工具空间和多文件 Patch Scope，通过注入故障与隔离测试，不允许模型修改验证器；再跑同 SHA 的 D/E 对照。
5. **独立评测：** 新的冻结候选集经过完成后，只用于最终报告，不拿其失败重新改 Prompt 再宣称仍是独立测试。需要更多样本则另建明确标识的新版本。

**当前实际交付范围：候选计划与数据完整性门禁；不把未执行的新题写为已经达到 Benchmark V2。**
