# 2026-11-21：HardSet V2 Functional Probe / 实验性私有验证器真实 GPU 验收

## 证据性质与代码版本

**这是新的 HardSet V2 研发验证器和受控探测工具的真实 GPU 验收，不是既有 v1 签名独立 Verifier 的 Release 结果，也不是 DeepSeek/Agent 修复的实验证明。**

- 冻结开发源码 SHA：\`3e6b32cd9ed427ca94867c507be8261f8bc420a9\`
- 机器：Tang，NVIDIA RTX 4090 Laptop GPU，SM89；严格使用同一预配置的 \`IsolatedGPUBackend\` / Docker、原项目 RunStore。
- 受测案例：\`h001\`，Softmax32 溢出故障。所有输入尺寸在代码中白名单锁定；参考数值不会返回给 Planner。
- 新工具 / 验证器路径：\`src/gpu_agent/benchmark/hardset_v2_probe.py\`、\`hardset_v2_graph.py\`、\`hardset_v2_verifier.py\`。
- 离线测试：\`tests/unit/test_hardset_v2.py\` + \`test_hardset_v2_probe.py\`，Python 3.11，**117 passed**，0 skipped。正式 CI 另独立记录。
- 真实模型调用 **0**，不能解释为 E 模型在 GPU 上自动选择探测或生成修复；当前 D 固定序列 / E 注入合法 planner 的 LangGraph 适配仅由离线测试验证。

## 公开 Functional Probe

操作员调用（模型无 Shell 权限）：

\`\`\`bash
gpu-agent benchmark probe-hardset \
  --repository "$PWD" --case h001 --size 32 --size 33 --size 65 \
  --output /home/you/gpu-agent-hardset-v2-probe-20261121/public-probe-h001-3e6b32c
\`\`\`

| N | 公开工具结果 | 解释 |
|---:|---|---|
| 32 | \`PASS\` | 普通输入数值可通过，错误被掩盖 |
| 33 | \`PROGRAM_FAILED\` | 极端输入触发非有限数值，旧 C++ harness 拒绝非法输出 |
| 65 | \`PROGRAM_FAILED\` | 另一个非规则长度与高数值输入触发故障 |

完整公开响应 \`public-probe-results.json\` SHA256：

\`82f8a88d8fb278e9bf52946f37d6673140e0032b70c2f6875b05be3b488db1c6\`

此时只可以宣称受控 Functional Probe 已实现公开的可复现数值故障发现，不能宣称 Agent 学会了选取高信息增益输入。

## 实验性私有验证器：正向/负向测试

控制器人工构造并提前 scope-check 了两份合法 unified diff：

1. **Known-good 正向对照**：把 h001 的原始 mutant 恢复到干净的数值稳定 Softmax 实现；它是已知正确的人工补丁，**不是模型产出**。独立评估器执行 9 组隐藏输入（与公开输入长度不交叉）+ 4 种 Sanitizer。
   - 结果：\`VERIFIED_FIXED / ALL_REQUIRED_CHECKS_PASSED\`
   - candidate SHA256：\`88dd297b8a0bc53dc3b93b7d1a4678551346b4855d7139b4edaddecef0070ba4\`
   - EVAL 完成记录：9 次隐藏数值结果记录、4 次私有 Sanitizer 结果记录。
   - CLI 日志 SHA256：\`9f6dcd7b72db0d51f808a0637c0334776636815009f79f6e23e0288c60c5250e\`
2. **Known-bad 负向对照**：仅在原始错误源码中增加不影响计算的注释，故障原封不动；同样合法 PatchScope，并非非法 diff。
   - 结果：\`NOT_VERIFIED / HIDDEN_NUMERIC_FAILED\`，CLI 预期返回 1。
   - candidate SHA256：\`12e1cbc4bdbc6ea762ec36b00c4062b46e85ae4acb2d0daa595b37c8d5b59752\`
   - EVAL 完成记录：第 4 个隐藏数值测试发生失败后停止；未执行没必要的 Sanitizer。
   - CLI 日志 SHA256：\`32352c109abbfb39af55609c4b4c70e3bcd7ec2b8c17e0e4c1f553ae5a52af6f\`

## 评估数据隔离检查

每次私有验证都保留 **1 个公开 RunStore + 1 个 evaluator-visible RunStore**。独立检查显示：

- public artifact 名称仅为 \`hardset/candidate-provenance.json\`、\`hardset/strict-verdict.json\`。
- public manifest 不含隐藏输入、隐藏结果、private verification contract、evaluator-only trace 或 secret 文件路径。
- private manifest 含 \`hardset/private-verification-contract.json\`、隐藏数值和 Sanitizer 结果，\`visibility=evaluator\`，不会回传给 Agent 工具。
- evaluator 密钥存放在 Tang 仓库外的 0600 私有文件，**不打印、不提交、不拷贝到报告**。
- 受控 CUDA 容器运行完毕后没有本项目遗留容器，GPU 回到空闲状态；未覆盖 2026-11-21 的旧 HardSet 原生准入证据。

Tang 的独立证据根目录：

\`/home/you/gpu-agent-hardset-v2-probe-20261121/\`

含 source-3e6b32c、public-probe-h001-3e6b32c、strict-good-h001-3e6b32c、strict-bad-h001-3e6b32c 等；*其中 evaluator-private 文件夹不可发布*。

## 下一道研究门禁

- **Agent Planner 尚未使用付费模型**：当前 D/E Graph 可以用同一个受控 \`run_functional_probe\` 接口，但 E 使用了显式注入的合规 planner，未接入已有 \`OpenAIResponsesProvider\` 的受审计 HardSet 专属结构化输出；**不可计作 E Agent 的真实自主多步搜索成绩**。
- 当前实验只验证 h001。h002/h003 需按同一版本测试协议补齐独立正负验证和跨 Kernel 搜索。
- 新实验性验证器目前尚不等于原项目签名 v1 Verifier；要成为正式 strict fix benchmark，必须完成 source/prompt/schema frozen, hidden dataset review, telemetry and release gate。
- 后续优先为 HardSet 的 \`run_functional_probe\` 增加**受审计 LLM 结构化 planner 适配**，然后将模型生成的 scope-limited patch 连接 V2 verifier，再做多种子 D/E/E_ADAPTIVE 配对真实 CUDA 实验。对照应报告全体预声明单元、成本、Tokens、延迟和失败原因，而不只挑成功修复。
