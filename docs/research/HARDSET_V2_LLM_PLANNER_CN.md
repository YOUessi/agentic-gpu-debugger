# HardSet V2：真实 LLM Function Probe Planner 接入（开发实验）

2026-11-21 新增 **受审计的真实模型 Planner 适配器**，与上一个 HardSet V2 Functional Probe + 私有验证器版本衔接。本次只实现调查阶段，不宣称已经完成模型生成/多轮修复，也不能作为正式签名 benchmark 分数。

## 1. 为什么不新建 LLM Client

复用 \`OpenAIResponsesProvider._call(kind="plan")\`。原项目已有经过审查的：
- 一次物理请求一条 Started/Terminal Invocation Audit，包含 Prompt 版本、Token Usage 与统一的响应 Schema；
- LLMCallGate 预算、全程超时、失败/不确定传输不自动重试，格式无效仅允许有限重试；
- 隔离 Python Worker 中真正的官方 SDK 请求（可停止），\`store=False\` 的受控响应协议，保护 API Key 不进入 RunStore。

新增 \`hardset_probe_contract=hardset-v2\` 仅在**显式调用** \`plan_hardset_v2\` 时选择 \`HardsetProbePlanOutput\`，不修改已有 \`WIRE_MODELS["plan"]\` 及历史 Agent 的 PlannerOutput。新 Schema 只允许：

\`\`\`json
{"action":{"action_type":"run_functional_probe","size":33}}
\`\`\`

或

\`\`\`json
{"action":{"action_type":"finish_probe"}}
\`\`\`

不接受任意 Shell、URL、源码 Patch、测试脚本或秘密路径。受信任 Prompt 版本 \`hardset-v2-functional-probe-2026-11-21-v1\`。模型看见的输入仅为公开 mutant 的 \`kernel.cu\`、公开功能要求、合法 size 列表、剩余次数和已完成的公开数值探测结果。没有隐藏 CPU reference 或私有输入。

## 2. D/E 实际执行

同一个 LangGraph \`run_hardset_probe_graph\`：

- D：不调用模型，固定探测 32/33/65。
- E：通过同一开销与 GPU 权限下的 \`OpenAIResponsesProvider.plan_hardset_v2(snapshot)\` 选择下一次 size，或结束调查。操作员必须显式 \`--allow-paid-calls\`，且 \`max-llm-calls\` 给下一步状态和有限重试留下空间。
- **D 和 E 的所有 GPU 调用都只能通过相同的 \`HardsetFunctionalProbe.invoke\`**；工具拒绝重复/未知 size 和超出预算的请求。没有自动调用隐藏 Verifier 的路径。

操作员使用命令（不打印 API Key，API/模型必须预先在 Tang 私有环境中配置）：

\`\`\`bash
# 无模型费用的规则调查
gpu-agent benchmark investigate-hardset \
  --repository "$PWD" --case h001 --mode D --max-probes 3 \
  --output /path/to/new/D-h001

# 显式允许真实模型调用的 E 调查
gpu-agent benchmark investigate-hardset \
  --repository "$PWD" --case h001 --mode E --max-probes 3 \
  --max-llm-calls 8 --allow-paid-calls \
  --output /path/to/new/E-h001
\`\`\`

最终 \`investigation.json\` 只保存公开 probe 结果、是否发现功能失败、模型物理调用次数与普通证据摘要。模型请求 Audit 留在 **public** RunStore 中；不允许读取 evaluator/private RunStore。

## 3. 当前研究局限

本次 **不是** Agent 完整修复任务：模型做的是「选择真实 Functional Probe 的试探输入」，而 \`diagnose → propose_patch → public self-check/revise → independent strict verify\` 尚待为 HardSet V2 设计独立受控版本。实验时需将「能否定位到功能失败」与「是否修复成功」明确分开；禁止把模型选中 n=33 算作 GPU Kernel 修好。

还需：针对 h001/h002/h003 做 D/E 多随机种子配对；对模型输出不确定性保留真实调用日志；按同一 SHA 冻结真实费用/Token；不得拿此前的公开两题自动修复成功率充当这三题的 benchmark 成绩。
