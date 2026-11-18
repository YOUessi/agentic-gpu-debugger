# 2026-11-14：LangGraph / LangChain / adaptive-v1 真实 GPU + DeepSeek 五组公开对照

## 结果级别与来源

**实验类型：真实 NVIDIA GPU + DeepSeek 模型调用；公开开发集的探索性小样本。不是盲测、不是独立 Holdout，也不是生产 Release 验收。**

- 实验冻结源码 SHA：`db18b321d5bdf5abfc5624dcfb526e0c738d5c71`（包含 [预声明五臂脚本](../../scripts/compare_framework_orchestration.py)）。
- 机器：Tang；NVIDIA GeForce RTX 4090 Laptop GPU；模型：`deepseek-v4-pro`，`api.deepseek.com`。
- 冻结知识库：CUDA 12.8.1 / Compute Sanitizer 2025.1.0.0；文档索引 SHA256 `6ddc3cf2443ff983f08d0dcae44ca5f68ba28033beeb391a162422f9936ed27a`。
- 公开案例：`case_0019` (Histogram)、`case_0020` (Warp Reduction)，**此前均参与开发实验，无法代表未见分布**。
- 每臂 × 每案例一次，预声明顺序随机种子 `20261114`，一共 10 个独立运行单元；失败不可自动补抽。
- 单单元最大模型调用 12；全部最大调用 60；补丁最多 2 个、重新调查最多 1 次；Repair Memory 关闭；所有臂共用同一源码、模型、知识库和候选限制。
- 实验完成时间：2026-11-14 05:23 (+08:00)，通过 `completed.json` 核对 `complete=true, completed=10, expected=10`。
- 审核日期：2026-11-19。结论来自 Tang 原始文件，不从 CI 或旧实验推断新 GPU 成绩。

## 十个单元（原始执行顺序）

| 序号 | 案例 | 臂 | 严格验证终态 | 模型调用 | 已记录 Token | 耗时秒 |
|---:|---|---|---|---:|---:|---:|
| 00 | case_0019 | E_GRAPH | VERIFIED_FIXED | 7 | 21,404 | 97.22 |
| 01 | case_0020 | E_LEGACY | VERIFIED_FIXED | 6 | 23,232 | 103.14 |
| 02 | case_0020 | D_RULE | VERIFIED_FIXED | 2 | 6,194 | 84.26 |
| 03 | case_0019 | D_RULE | VERIFIED_FIXED | 2 | 6,683 | 87.03 |
| 04 | case_0020 | E_LANGCHAIN | VERIFIED_FIXED | 5 | 15,451 | 93.95 |
| 05 | case_0020 | E_ADAPTIVE | VERIFIED_FIXED | 3 | 9,339 | 88.26 |
| 06 | case_0019 | E_ADAPTIVE | VERIFIED_FIXED | 5 | 14,474 | 95.48 |
| 07 | case_0020 | E_GRAPH | NOT_VERIFIED | 11 | 47,190 | 54.97 |
| 08 | case_0019 | E_LANGCHAIN | VERIFIED_FIXED | 6 | 18,831 | 93.83 |
| 09 | case_0019 | E_LEGACY | NOT_VERIFIED | 4 | 11,520 | 14.95 |

所有模型调用的 token usage 都已记录（各单元 `unknown_usage_calls=0`）；**真实美元费用未进行价格审定，原始字段 `api_cost_usd=null`，不能填写 0**。Token 计数是实际 provider 上报而不是模型估算。

## 五臂结果聚合

| 方法 | 精确定义 | VERIFIED_FIXED | 物理 LLM 调用 | 已记录 Token | 总耗时秒 |
|---|---|---:|---:|---:|---:|
| D_RULE | `D + legacy + native + baseline` | **2/2** | **4** | **12,877** | 171.29 |
| E_LEGACY | `E + legacy + native + baseline` | 1/2 | 10 | 34,752 | 118.09 |
| E_GRAPH | `E + langgraph + native + baseline` | 1/2 | 18 | 68,594 | 152.19 |
| E_LANGCHAIN | `E + langgraph + langchain + baseline` | **2/2** | 11 | 34,282 | 187.78 |
| E_ADAPTIVE | `E + langgraph + langchain + adaptive-v1` | **2/2** | **8** | **23,813** | 183.74 |

合计 **8/10 VERIFIED_FIXED**，51 次物理模型调用、174,318 已记录 Tokens；十单元耗时之和为 813.09 秒（逐单元依次执行），不是每臂的置信区间。

`E_ADAPTIVE` 相比 `E_LANGCHAIN`：2/2 修复数相同，模型调用 8 vs 11（减少 27.3%），Tokens 23,813 vs 34,282（减少 30.5%）。**但 D_RULE 同样 2/2 成功，只需 4 次调用、12,877 Tokens；自适应 Agent 仍未优于规则方案。** 不可因两个成功案例推断统计优势。

LangGraph 与 LangChain 本身主要是编排和组件抽象；不同模型抽样产生的成功数差异不能归因于框架。五臂比较还存在每案例每臂仅一次随机采样的高方差问题。

## 两次真实失败根因

1. `E_GRAPH × case_0020`：初始诊断 `DIAGNOSED` 且根因定位到 `a[i+1]` 越界，但第 1 个补丁编译并正常运行后发生 `NUMERIC_MISMATCH`（未保持按组聚合语义）。控制器经公开功能失败反馈进行 1 次 `REINVESTIGATE`；第 2 个补丁编译失败，`kernel.cu:17` 对不可赋值的 `group_end` 赋值报错（`expression must be a modifiable lvalue`）。`CANDIDATE_LIMIT` 停止，**没有**通过严格验证。累计 11 次模型调用 / 47,190 Tokens，说明“多轮反思”并不自动产生合法补丁。
2. `E_LEGACY × case_0019`：代理未走到补丁阶段，`diagnostic_outcome=INCONCLUSIVE`，limitations 为 `NO_INFORMATION_GAIN`，`NO_REPAIR_SUMMARY`；仅进行了 4 次规划调用，未提交候选，不可将此结果误记为「补丁验证失败」。

`E_ADAPTIVE` 在 case_0020 的实际规划决策序列为 `MANDATORY_MEMCHECK → LLM_PLANNER → EVIDENCE_COMPLETE`，只发 1 次规划请求和 1 次诊断、1 次补丁请求。case_0019 为 `MANDATORY_MEMCHECK → LLM_PLANNER ×3 → EVIDENCE_COMPLETE`，3 次规划 + 诊断 + 补丁共 5 次。两个案例均第一份候选通过公开功能、四种 Sanitizer 检查及独立严格验证；没有因模型自我评价而直接授予成功。

## 原始证据、哈希、完整性

Tang 只读保留：
`/home/you/gpu-agent-integration-validation-20261114/live-five-arm-0019-0020/`

- `predeclared.json` SHA256 `337c2e5e82e6720fce4085083c7e258b774101b377982f93cfe11cd496fa1cc4`
- `results.jsonl` SHA256 `0ea96cbc6c4400293960c7597fd92d12c764f07b3d6c54968899c1f4bf3f1462`
- `completed.json` SHA256 `337cf7bd93d924797f3aab00fe1a6a6b2aa03342e705bf4fa155bf21fd95acba`
- `knowledge-index.json` SHA256 `6ddc3cf2443ff983f08d0dcae44ca5f68ba28033beeb391a162422f9936ed27a`

2026-11-19 已独立核验：10 行唯一 case×arm×repeat 匹配预声明顺序、每行 `source_sha` 相同、每个 `unit-XX/result.json` 与 `results.jsonl` 对应行完全一致，`completed.json` 声明全 10 单元完成。公开 RunStore 位于各自 `unit-XX/public/`；独立 Verifier 私有内容不提交至 GitHub。原始敏感调用内容、密钥、私有 Oracle 未复制到本报告。

## 后续研究门禁

- 单一公共案例每臂一次**不支持**任何统计显著性、复杂 CUDA 泛化或「Agent 优于规则」结论。
- 先构建尚未专门用于策略和 Prompt 调参的新案例；冻结来源、任务规格、工具链、预算及故障类型；按随机种子对 D 与 E_ADAPTIVE 做多次配对真实实验。
- 分离验证三种收益：①跳过强制步骤规划的开销；②真正证据驱动动态调查的能力；③改进 Patch 生成及语义有效性。不能用框架切换或增加工具调用伪装推理提升。
- 报告每个 case×seed 的严格成功、失败类型、API 调用数、Tokens、可核算费用、耗时及 Sanitizer 覆盖，包含所有失败，不补抽、不混合旧 SHA。
- 在完成复验前，`adaptive-v1` 维持 opt-in Draft，不切为默认算法；GitHub 离线 CI 通过不替代新 SHA 真实 GPU 验证。
