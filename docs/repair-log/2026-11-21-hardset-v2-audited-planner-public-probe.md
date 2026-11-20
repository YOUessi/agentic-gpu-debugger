# 2026-11-21 · HardSet V2 LLM Planner Adapter / D vs Scripted-E GPU 验收记录

## 实验边界

**这次验证的是受控 Functional Probe 的 D/E LangGraph 工具路径和模型接口的离线协议，不是 DeepSeek 真正执行 Agent 探索或修复，更不是 Agent 优于规则的研究结论。**

- 固定代码 SHA：\`470086ca799fac1d95c7b8855659b6fa76af96ad\`。
- 机器：Tang，RTX 4090 Laptop GPU SM89；隔离 Docker GPU 后端，同一共享 GPU 锁。
- 使用已在上一批证明数值错误可复现的 \`h001\` Softmax、\`h002\` RMSNorm、\`h003\` 多 Kernel Center。
- 三个 \`Rule D\` 使用一样的固定探测序列 \`32,33,65\`；三个 \`Scripted E\` 在不同的 LangGraph Planner node 中注入**人工固定的**一次 \`65\` 探测与后续 \`finish_probe\`。这不是模型智能选择，不能评价工具效率的算法优势。
- 实际模型调用 **0**；E 真实 DeepSeek 的付费调用被执行工具的安全检查阻止，**没有**产出 E-LLM 实验。已通过固定 MockPort 验证代码的 bounded LLM wire schema、模型审计/预算记录，真实 provider 兼容性待进一步独立验收。
- 使用该 source SHA 的 \`tests/unit/test_hardset_v2.py\`、\`test_hardset_v2_probe.py\`、\`test_hardset_v2_llm.py\` **122 passed**，0 skipped；完整 CI 另行查看固定 SHA 状态。

## Tang 真实 GPU 调查轨迹

| Case | 规则 D 探测结果（n32 → n33 → n65） | Scripted E 探测结果（n65） | 两者都发现错误 |
|---|---|---|---|
| \`h001\` Stable Softmax32 | PASS → PROGRAM_FAILED → PROGRAM_FAILED | PROGRAM_FAILED | 是 |
| \`h002\` Tail RMSNorm32 | PASS → NUMERIC_MISMATCH → NUMERIC_MISMATCH | NUMERIC_MISMATCH | 是 |
| \`h003\` Two-kernel Group Center32 | PASS → NUMERIC_MISMATCH → NUMERIC_MISMATCH | NUMERIC_MISMATCH | 是 |

脚本化 E 的 \`decision_steps=2\`（一次工具选择 + 完成）；规则 D 用了 3 次工具。**这是不同的预置序列产生的工具数量差异，不是实际 DeepSeek 自主决策的成本节省。**

### D 公开报告 SHA256

- \`h001\`: \`b7cc7ea87d30f39190e76f2743d15e62ceeb47789ab64b52193b724366a9a266\`
- \`h002\`: \`15ec7e474b4991c5cf1987148910a28cc1c266fad439a02fa5d818ff55de6039\`
- \`h003\`: \`fafc5bb55e4be6b9b44561a0283778bb46ab5b4f49497febb4159a117409f345\`

### Scripted E 公开报告 SHA256

- \`h001\`: \`f9e83ec10c87c9946d0f85c4595d7256fc8b4c1a49bc393fd375507e2801758c\`
- \`h002\`: \`f2763c91f526ff51488743d4297c1b56fc9e845770487e94c1280aad3bc97831\`
- \`h003\`: \`500af8aaa5f1dc9cfc065e709ec464e74f02a23856ff0e74ea6bd7b8834b9728\`

完整公开目录（无私有 Oracle）：

\`/home/you/gpu-agent-hardset-v2-llm-20261121/\`

- \`source-470086c/\`：Git 固定干净检出。
- \`D-h001-470086c/\`、\`D-h002-470086c/\`、\`D-h003-470086c/\`：真实 GPU D 公开日志、RunStore、\`investigation.json\`。
- \`E-scripted-h001-470086c/\`、\`E-scripted-h002-470086c/\`、\`E-scripted-h003-470086c/\`：同一受控工具但显式脚本化 E 的 GPU RunStore、\`scripted-investigation.json\`。
- \`predeclared-h001.json\`：h001 D/E 模型选择试点的预声明；E-LLM 因环境安全门禁阻止 **未执行**，不挪用 scripted 结果。
- \`offline-470086c.log\`：新模型调用协议测试 122/122。

## 代码组件与安全性质

本版 [LangGraph Probe 图](../../src/gpu_agent/benchmark/hardset_v2_graph.py) 只接受 \`FunctionalProbeAction(size)\` 或 \`FinishProbeAction\`，D/E 均交给 [HardsetFunctionalProbe](../../src/gpu_agent/benchmark/hardset_v2_probe.py) 进行工具白名单、预算、预注册尺寸和不可变日志检查。

[模型适配](../../src/gpu_agent/agent/provider.py) 的 \`plan_hardset_v2\` 是通过已有 \`_call(kind="plan")\` / killable Worker / \`LLMCallGate\` / Invocation Log 实现，且仅在明确的 \`hardset_probe_contract=hardset-v2\` 时使用新输出 Schema；旧版 \`WIRE_MODELS["plan"]\` 未修改。失败格式有界重试，未知时不自动重复不确定模型请求。

私有实验性 Verifier 独立存在，不暴露给 Planner。**本批没有生成任何模型 patch，也没有在新 HardSet 三题上开展真实 Agent StrictFix 比较**。如果之后获得平台允许的真实模型运行权限，仍需先对当前 SHA 的 E-LLM 做首次真实调用，不得用先前假模拟结果当作模型成绩。

## 下一步验收

1. 真实 E-LLM（平台批准的调用路径）→ 记录真实 \`run_id\` / \`provider\` 调用记录、input/output/cached tokens、完整 D/E 公开选择与失败，绝不拷贝 API Secret。
2. 把功能数值证据交给模型 \`diagnose\` → 有预算 \`propose_patch/revise_patch\` → 公开自检 → 独立 V2 Verifier；禁用独立验证器对同轮候选反馈回路。
3. 新的、自上游模型训练数据风险明确声明的独立故障家族集；同 SHA / 同预算 / 多种子，与确定性 D 做配对实验。没有预声明独立评测成绩前不能宣称 Agent 优于规则。
