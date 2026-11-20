# HardSet V2：受控 Functional Probe + 独立严格验证（开发通路）

**状态**：实验性 V2 组件；旧 A–E 冻结评测的工具、评分与数据分母均不变。

上一批 HardSet V2 已经在 Tang 的 RTX 4090 GPU 上证明三类数学/跨 Kernel Mutant 可以通过部分普通输入、在边界或极端输入下失败。这一批补上受控功能探测和单独评估器，使未来 D / E Agent 能使用**同一种合法工具**，而非直接得到独立参考或隐藏答案。

## 1. 受控 Functional Probe

- 数据源仍限定为仓库锁定的 \`h001\` Softmax32、\`h002\` RMSNorm32、\`h003\` 两阶段归约 Mutant。
- 模型或规则只能提交 \`{"action_type":"run_functional_probe","size":33}\` 一类**结构化**请求；没有路径、命令、测试程序、可执行参考或隐藏文件字段。
- size 必须来自项目预定义的 15 种公开尺寸；一次会话中重复调用、超出固定预算、未知 source、未知 case 均拒绝。失败的工具调用不由控制器自动重试。
- 控制器调用隔离 Docker GPU + 完整独立的 CPU reference，只向 Agent 返回公开 \`PASS\`、\`NUMERIC_MISMATCH\`、\`OUTPUT_INVALID\`、\`PROGRAM_FAILED\`，以及尺寸、输入 hash、RunStore 证据引用。**不会提供期望输出的具体数值**。
- 执行产物及原始日志保存在 public RunStore；当底层 GPU 不可用时失败封闭，不能把设备故障误判成算法错误。
- 对照研究 \`run_hardset_probe_graph()\` 同一节点注册 D 和 E 两种策略：D 固定探测 32/33/65，E 由显式注入的合规 planner 适配器选择尺寸。每个请求最终都经过同一个 \`HardsetFunctionalProbe.invoke()\` 权限/预算门禁。**当前 E planner 连接仅完成类型化适配接口与离线脚本模拟，尚未接入付费 LLM 的真实决策循环**。

操作员可运行（不调用模型）：

\`\`\`bash
gpu-agent benchmark probe-hardset --repository "$PWD" \
  --case h001 --size 32 --size 33 --output /path/to/new/public-probe
\`\`\`

## 2. 独立实验性严格验证（尚不是 v1 signed release）

\`run_strict_hardset_verification()\` 只接受对原始 Mutant 的 \`kernel.cu\` unified diff。它首先复验原 Mutant SHA256，并使用项目现有 \`apply_generated_candidate\` 对 patch 做范围/同步一致性拒绝；非法 diff 不创建 evaluator RunStore。

通过审核的候选仅在 **evaluator visibility** 的 \`IsolatedGPUBackend\` 下重新编译和运行。其隐藏测试：

- 使用与公开 \`SIZES\` **没有交集**的固定长度集合 \`(2,7,30,34,66,79,130,258,514)\`；
- 每个输入数据由 evaluator-only 32 字节以上的密钥通过 HMAC-SHA256 派生；同一任务和密钥可重放，值不会进入 public RunStore；
- 逐长度执行独立 CPU 数值 Oracle；全部数值通过之后仍需四种 NVIDIA Sanitizer 的真实 CLEAN 结果和程序功能正确；
- 输出只返回 \`VERIFIED_FIXED\` / \`NOT_VERIFIED\` / \`INCONCLUSIVE\` 与固定原因码，公开 Artifact 不携带隐藏数值/具体测试轨迹。**这仍是实验性 V2 verifier，不会伪装成旧版本签名发布的 \`VERIFIED_FIXED\`。**

安全的操作员 CLI（密钥仅在 Tang 安全本地、\`chmod 600\`，不进入仓库）：

\`\`\`bash
gpu-agent benchmark verify-hardset \
  --repository "$PWD" --case h001 \
  --patch /path/to/controller-approved.diff \
  --evaluator-secret-file /path/to/private/secret.bin \
  --output /path/to/new/verification
\`\`\`

此 CLI 仅为操作员诊断/验收，不开放为模型调用工具；不得将 evaluator root、秘密或私有日志传给 Planner。用户输入的 Patch 不能绕过代码哈希、PatchScope 和隔离执行约束。不能将这个验证器称为严格的安全多租户隔离，GPU Docker 仍共享宿主内核与驱动。

## 3. 目前验收与剩余工作

本分支本次提交包含独立回归测试：D/E 共用工具预算、合法/非法动作、重复拒绝、受控输入哈希、禁止隐藏数据进入公开投影，以及未授权补丁失败前不创建评估目录。GitHub CI 与 Tang 实机结果单独归档。

在正式声称完整 Agent 修复闭环之前仍需要：
1. 通过旧项目可信 \`LLMProvider\` 调用通道接入 **HardSet V2 专用**的模型 \`run_functional_probe\` 选择 schema/提示词，不增加未审计的 API Client；同时完成证据引用校验。
2. 把真实 Agent Patch Candidate 与受限 V2 验证器关联，限定每任务最大候选/重新调查/超时，保存修订轨迹；不能从严格验证器获取精确隐藏失败数据后调整本轮补丁。
3. 将新验证用例集冻结为一个未参与模型策略调优的固定集合；D/E/E_ADAPTIVE 进行等预算、多种子真实 GPU+LLM 配对实验，报告全部失败与 Token、耗时。
4. 当数值差异被证实且新工具能提高 Agent 真实决策效果时再扩充多文件 Patch Scope 和 CUDABeaver Adapter。

**本批完成的是可用的共享 Probe 与 V2 私有验收通路，不是已证明 Agent 优于规则系统。**
