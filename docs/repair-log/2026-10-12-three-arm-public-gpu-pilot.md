# 2026-10-12：Repair V2 / 旧 V3 / 新 V3，两个公开算法的真实三臂对照

## 为什么做

基础 Repair V3、取消独立 Sanitizer 次数上限、可信公开证据复用、补丁有效性分析与 Repair Memory，以及 `case_0022` 的重诊断缺口，均已通过 PR #2/#4/#3/#5 依次合入 `design/v2-operator-workflow`。现从未参与本轮 `case_0022` 定向修复的公开算法 `case_0019` Histogram 和 `case_0020` Warp Reduction 出发，做一次**预声明、不会根据结果调整的真实 GPU 探索对照**。

实验不是严格私有 Holdout：这些案例来自项目公开基准，可能已在更早历史评测中出现；模型输入也可能含通用训练知识。每组**仅一次采样**，不能依据 1/2 vs 2/2 宣称统计意义上的成功率提升。

## 冻结实验设置

- 执行脚本：[three-arm runner](../../scripts/compare_repair_three_arms.py)，提交 `1341df2016fba2db9e226fb90016c6cf71083cdd`。
- `V2`：正式合并代码 `1eaac1fdf456a46118ea3aecc16338a9142c29bb`，关闭重新调查。
- `V3_OLD`：历史代码 `e4e7a8d4d069cecb24355e9c521bf7bd121e0870`，启用旧重新调查。
- `V3_NEW`：正式合并代码 `1eaac1fdf456a46118ea3aecc16338a9142c29bb`，启用修复后重新调查。
- 真实 `DeepSeek v4-pro`、Tang NVIDIA RTX 4090 Laptop GPU、CUDA 12.8.1、Compute Sanitizer 2025.1，使用同一 CUDA 官方文档索引。
- **所有臂均为 4 次独立调查 Sanitizer 上限**，每单元最多 3 个补丁候选、1 次重新调查、40 次模型调用、38 Agent steps、600 秒任务截止。Repair Memory 禁用。
- 随机顺序种子 `20261011`，每个 `case × arm` 恰好一次，没有重复、补抽或动态挑选。两个 checkout 的 `kernel.cu`、`input.json`、`task.json` 已逐文件 SHA256 对齐。
- 预声明开始时间：UTC `2026-10-11T16:11:06Z`，相当于 Tang +08:00 的 **2026-10-12 00:11**。原始文件：[预声明](artifacts/2026-10-12-three-arm-public-pilot/predeclared.json)和 [逐单元原始结果](artifacts/2026-10-12-three-arm-public-pilot/results.jsonl)。
- 预声明 SHA256 `2ad590c76907fa66911313cfbd99e0817bdf1704c43f111b322f510fc8cbd5a6`；`results.jsonl` SHA256 `ce8f6974ecfcd259fb725519ce38026a744f9ece35d4b5c24543dd011d916b15`；文档索引 SHA256 `6ddc3cf2443ff983f08d0dcae44ca5f68ba28033beeb391a162422f9936ed27a`。

## 六个真实模型 + GPU 结果

| 次序 | 案例 | 臂 | 严格验证终态 | 候选 | 重新调查 | 物理模型调用 | Tokens | 秒 |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | `case_0019` | `V2` | **VERIFIED_FIXED** | 1 | 0 | 6 | 18,758 | 196.12 |
| 2 | `case_0020` | `V3_OLD` | **NOT_VERIFIED** | 3 | 1 | 11 | 47,591 | 106.75 |
| 3 | `case_0019` | `V3_OLD` | **VERIFIED_FIXED** | 1 | 0 | 7 | 21,393 | 207.94 |
| 4 | `case_0020` | `V3_NEW` | **VERIFIED_FIXED** | 1 | 0 | 5 | 15,750 | 204.65 |
| 5 | `case_0019` | `V3_NEW` | **VERIFIED_FIXED** | 1 | 0 | 8 | 23,909 | 196.33 |
| 6 | `case_0020` | `V2` | **NOT_VERIFIED** | 0 | 0 | 5 | 20,550 | 36.32 |

按版本聚合（每个臂 2 单元）：

| 臂 | VERIFIED_FIXED | 物理模型调用合计 | Token 合计 | 各单元耗时之和（秒） |
| --- | ---: | ---: | ---: | ---: |
| V2 | **1/2** | 11 | 39,308 | 232.45 |
| 旧 V3 | **1/2** | 18 | 68,984 | 314.69 |
| 新 V3 | **2/2** | 13 | 39,659 | 400.99 |

## 失败类别不是一回事

**Histogram / `case_0019`：** 三种实现这次均首候选通过，公开 CPU Oracle 和四种 Sanitizer 全为通过，新 V3 无再调查。调用量 V2 为 6（18,758 tokens）、旧 V3 为 7（21,393）、新 V3 为 8（23,909）；本例既未显示新 V3 成功率优势，也不能说它成本更低。

**Warp Reduction / `case_0020`：**

- V2 本次**根本没有生成候选**。原始运行 `3ac8b2b1e243496c8bee28cc3a879b5e` 的 `diagnosis.json` 是 `INCONCLUSIVE`，limitations 为 `LLM_INVALID_OUTPUT`；真实 provider 在两次诊断尝试中均 `DOMAIN_REJECTED / diagnose_policy_rejected`，因此没有进行补丁或 strict verifier。**不能将其写为“补丁未通过”**。
- 旧 V3 本次确实生成并检查了 3 个候选，三次均普通数值 `NUMERIC_MISMATCH`；只进行一次重新调查，`stop_reason=CANDIDATE_LIMIT`。
- 新 V3 的**第一份**候选就通过公开数值、全部四工具和隔离严格验证 `VERIFIED_FIXED`，模型调用 5 次、15,750 tokens。它不曾调用新增的 V3 多轮重新调查。因此这个成功**不能证明修复后的重新调查策略导致改善**。

`NOT_VERIFIED` 在这里指未产生独立严格验证通过的终态，**不是私有评测器返回的 NOT_FIXED**。这些项目的严格验证内容仍由隔离 evaluator 负责；本报告只保存 verdict 和公开证据，不输出私有输入/答案。

## 结论与下一步

这次真实探索中，新 V3 的成功数为 2/2，相较 V2 和旧 V3 的 1/2 更好，且 Token 总量与 V2 接近、明显少于旧 V3。它说明新 V3 至少**没有在这两个公开算法上复现旧 V3 的相同 Warp Reduction 失败**，值得继续扩大测试。

但仅 2 个公开案例、一次采样，且 V2 的失败属于证据 gate 下的 LLM 诊断输出被拒绝；不能把差异归结为修订功能。下一步需要：

1. 更多此前没有专门用于提示与改进的公开案例，按算法和故障家族平衡，保持预算和顺序冻结。
2. 同一 `case×arm` 的预先设定多次独立采样，报告 Wilson 区间或确切置信界和实际费用。
3. 分别消融 `reinvestigate`、共享证据复用、静态 Patch Effect、Repair Memory，不能把混合改动当单一变量。
4. 重点追查 `case_0020` 的 `LLM_INVALID_OUTPUT` 和旧 V3 的三轮数值错误，区分模型输出格式/引文约束错误与真正修复算法失败。

本轮原始保留路径：Tang `/home/you/gpu-agent-repair-three-arm-20261011-01`，全部 6 个单元的 `public` RunStore、公开 self-check、父 Run、调用审计与严格验证元数据均单独保留；三个代码检出目录分别绑定源 commit，结束时均为 clean；项目 owner 标签 Docker 容器 **0 个**。美元费用未审定，记为未知，不填 0。实验用 Python 进程正常退出 0，只说明脚本六单元收齐，不代表每组都修复成功。
