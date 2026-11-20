# HardSet V2：真正的 LangGraph CUDA 自主修复 Agent

## 目标

这不再是单独的 Benchmark 工具。新增的 **Real CUDA Repair Agent** 是一条真正连接 LLM、GPU 工具、补丁修订、独立验证的有状态路径。

- D：规则 Router 选择固定的 Functional Probe；诊断、补丁与修订仍由同一个 LLM 执行。
- E：LLM 选择 Functional Probe；随后 LLM 自主诊断、产生 Patch、根据公开 GPU 自检反馈修订。
- 两者共用源代码、实际 GPU 工具、候选上限、私有 Verifier、LLM 实现和预算，能在后续做 matched-budget 对照。
- 独立 Verifier **不属于 Agent 可调用工具**。Agent 最多根据公开自检反馈迭代；最终 Verifier 仅在调查和修订完全结束、公开检查通过后由外层受控程序调用一次。
- 旧 A–E / Repair V3 的七种算法与签名发布协议不变，V2 是独立研究通路。

## 一、真实 Agent 流程

\`\`\`mermaid
flowchart TD
    A[观测原始有故障的 CUDA 源码] --> B[LangGraph Planner]
    B -->|D 固定规则 / E 真实 LLM| C[受控 Functional Probe]
    C -->|真实 CUDA 执行结果| B
    B --> D[LLM Diagnosis: 源码行+公开证据引用]
    D --> E[LLM Patch: 生成统一 diff]
    E --> F[Controller PatchScope + Hash Check]
    F -->|非法或重复| H{仍有候选预算？}
    F -->|合法| G[公开 GPU 编译+数值 Oracle+四 Sanitizer]
    G -->|失败反馈| H
    H -->|有预算| E
    H -->|无预算| J[INCONCLUSIVE / NOT_VERIFIED]
    G -->|全部公开通过| K[退出模型决策循环]
    K --> L[独立 Evaluator: 隐藏输入+四 Sanitizer]
    L -->|私有通过| M[VERIFIED_FIXED]
    L -->|私有失败| N[NOT_VERIFIED]
\`\`\`

这里的模型是真实 \`OpenAIResponsesProvider\`，复用项目既有的 killable worker、请求记录、Token usage、格式失败有界重试、预算门禁和不可重复不确定调用。模型没有获得任意 CUDA 命令、文件路径、Oracle 真值、验证器密钥或隐藏测试 API。

## 二、实装模块

- \`src/gpu_agent/benchmark/hardset_v2_agent.py\`：主要 LangGraph Agent 状态机，节点 \`investigate\`、\`diagnose\`、\`patch\`、\`public_selfcheck\`、\`finish\`；外层独立严格验证。
- \`hardset_v2_agent_models.py\`：模型诊断、证据引用、公开反馈和最终状态的严格 Schema。
- \`hardset_v2_selfcheck.py\`：模型生成的候选源码实际在 Docker GPU 中编译、运行固定公开尺寸、比较 CPU 独立数值 Oracle，再运行四个 Sanitizer。
- \`hardset_v2_graph.py\` 与 \`hardset_v2_probe.py\`：真实 Functional Probe 工具，D/E 公平共享。
- \`agent/provider.py\`、\`agent/prompts.py\`：使用同一 Model Provider/SDK/审计通道，分别新增 HardSet V2 Diagnose 和 Patch 输出，Prompt 版本独立，完全不动 v1 的默认 Wire Schema。
- \`hardset_v2_verifier.py\`：独立的 private evaluator；HMAC 种子隐藏数据与独立 RunStore 不进入模型。
- \`tests/unit/test_hardset_v2_autonomous.py\`：模拟模型在公开 GPU 自检失败后自动进行第二次补丁；同时验证拒绝非法、重复、伪造的候选和无证据终止。

## 三、在 Tang 上运行真实模型 Agent

需要正式许可真实模型调用以及本机 GPU，切勿把秘密参数复制到公有代码仓库。使用预先配置好的 Provider 环境，不在报告中输出密钥。

\`\`\`bash
# 先冻结 Git SHA、确认 GPU 独占与 Docker 资源
git rev-parse HEAD
nvidia-smi

# E: LLM 自主规划、诊断、生成 patch、公开修复再独立验收
gpu-agent benchmark run-hardset-agent \
  --repository "$PWD" --case h001 --mode E \
  --max-probes 3 --max-candidates 3 --max-llm-calls 20 \
  --allow-paid-calls \
  --evaluator-secret-file /path/to/operator-private/secret.bin \
  --output /path/to/new/e-agent-run

# D: 同一模型诊断和修复，仅 Probe 决策改成固定规则
gpu-agent benchmark run-hardset-agent \
  --repository "$PWD" --case h001 --mode D \
  --max-probes 3 --max-candidates 3 --max-llm-calls 20 \
  --allow-paid-calls \
  --evaluator-secret-file /path/to/operator-private/secret.bin \
  --output /path/to/new/d-rule-run
\`\`\`

运行成功的产物包含：

- \`investigation/public/<run_id>/\`：逐步真实探测、源码快照和 LLM Invocation Audit。
- \`investigation/public/<run_id>/hardset/agent-trace/*.json\`：每一轮调查、诊断证据引用、Patch hash 和 Public Self-check 的因果轨迹。
- \`public-candidate-XX/\`：每轮真实 GPU 候选编译、运行和 Sanitizer 原始证据。
- \`strict/evaluator/runs/\`：私有验证，仅供 operator 审核；不能进入模型上下文。
- \`agent-report.json\`：公开、受限的最终结果、模型调用数、Token 和严格判定。

## 四、当前验收级别

本 PR 只代表代码实现和测试，**不要仅凭一条 PR 或模拟模型用例就宣称模型已经自主成功修复**。必须先通过 CI 类型和回归，再用真实 GPU+真实模型报告至少 D/E 配对尝试，包含所有失败；随后扩大新的算法家族、随机输入和多次模型运行。

完整成功率以真正的独立严格验证为准。模型挑中高价值 Probe、生成貌似合理 Patch、甚至通过公开测试，都不是最终成功。

模型如果在计划阶段就终止、无法提供合法证据引用、生成非法/重复补丁、公开 GPU 自检失败、预算耗尽或最终私有测试失败，均应完整记录，不能静默丢弃或重新抽样。
