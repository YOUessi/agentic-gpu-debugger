# 2026-10-09 Repair v3 P0 验收、静态补丁分析与冻结经验记忆

## 背景与范围

用户要求顺序完成：V3 首次失败后的重新调查和二次修复、补丁有效性检查、复杂 CUDA 案例 V2/V3 对照、历史失败经验记忆，最后再整理合并。基线来自 `feat/repair-v3-reinvestigation` 和 `experiment/repair-v3-no-sanitizer-cap-20261009`。独立开发分支 `feat/repair-v3-robustness-memory-20261009`。全部新功能仍只限开发态 public repair；旧 A–E 的 frozen 评测与 private verifier 不改变。

## P0：真实 GPU 两轮闭环验收（已完成）

Tang 上运行 `tests/gpu/test_repair_v3_gpu.py::test_real_gpu_race_failure_reuses_self_check_and_repairs_in_two_candidates`，使用精确代码提交 `957a3efa9f2b1ea86cebc83192a63af66eb6002c`，GPU 真机构建、racecheck、自检与最终独立验证均使用真实 backend，脚本仅控制模型回答。

- 原始案例：公开 `case_0009`，共享内存 write/write 冲突。
- 候选 1：在 `threadIdx.x == 0` 分支把 `slots[0]` 改为 `slots[threadIdx.x]`，语义无变化；功能输出虽正确但 Racecheck `FINDING`。
- 来源匹配的自检证据：同候选源码、相同 public input、兼容工具链且 RunStore hash 匹配时，4 项 Sanitizer 结果可生成当前子 Run 内的引用。失配时不可复用。
- 重新调查：原始候选新鲜 public run + 4 项已验证的自检结果，模型可重新推断（本测试为脚本回答），不重复计算已有工具调用。
- 候选 2：调整线程独占的 shared memory 槽位；数值和四种 Sanitizer 自检通过；独立严格验证 `VERIFIED_FIXED`。
- 实测：**1 passed, 1 deselected, 191.50s**；候选 2 个、重新调查 1 次、复用 4 项公开观察、调查独立 Sanitizer 计数 2。
- 结束后检查进程已退出；本结果仅证明工程链路，不是模型性能改善的证明。RunStore 原始目录：`/home/you/gpu-agent-repair-v3-p0-native-20261009-b/public`；父 Run ID `e2a7076c9443444380709ada2ba92b41`。

早前新增证据复用代码的第一次定向回归失败于子 Run 源码引用的假设（误把 `sources/kernel.cu` 当作固定存储键）；已改为读取 EvidenceBundle 源码快照。其后 **76 项定向回归通过**。新增 GPU 测试的首次预检因兼容键写成 `compute-sanitizer`（规范字段应为 `compute_sanitizer`）失败，修复该夹具后只重跑失败 GPU 用例并通过。全部失败如实保留。

## P1：补丁有效性（已实现，需跨场景对照）

`src/gpu_agent/patch_effect.py` 在没有 GPU 前，只对少量可证明的条件等价替换生成 `PROVEN_LOCAL_NO_OP`，并检查修复是否涉及对应原诊断位置；其余情况返回 `NOT_ESTABLISHED`，避免过度推断。控制器将结果写入 `repair/<round>/patch-effect.json` 并反馈给后续修订，**不能当成 verifier 或替代 GPU**。

## P1：复杂案例对照（已完成）

预先冻结公开 `case_0021`（二维 Stencil）和 `case_0022`（Segmented Scan），V2/V3 各 1 次，共 4 单元；使用真实 DeepSeek 和 Tang GPU，固定种子 20261009、源代码和文档索引 Hash。V3 同时取消独立 Sanitizer 次数上限，因此为混合方案对照，不是重新调查的单因素因果试验。

运行目录 `/home/you/gpu-agent-repair-v3-compare-20261009-01`。启动时首单元为 `case_0022 / V2`；本日志不提前填写最终结论。运行中不注入历史记忆，后续所有实际结果另行逐单元补记。

## P2：冻结公开失败经验记忆（已导出一份真实样例）

从 2026-10-09 首次真实模型失败的公开父 Run `177a6daf715a4b53bfc47eeca11cf831` 提炼单条 `NUMERIC_PASS_RACE_REMAINS`；冻结 corpus hash：`45869ac44ba56aade3be5c8accfde86c981fc4aa326d52e336f173221102aa02`。原始导出路径：`/home/you/gpu-agent-repair-v3-public-memory-20261009-01.json`，文件 SHA256 `3dc25bcdb79f36f85bd2f82b99e8d87d96c9b542b6272f9ab02a5940c46d780a`。

与官方 CUDA 知识库完全分离；经验来自经过来源与候选绑定的 **public failed self-check**，不是模型编造的教训或任意 C++ 规则，也没有隐藏 holdout 输入；冻结索引按需通过 `GPU_AGENT_REPAIR_MEMORY_INDEX` 加载，仅 V3 的 public 开发修复会检索。为避免 V3 额外得到一份历史样例，此次 V2/V3 pilot 显式禁用 Repair Memory。

## 仍需完成

1. GitHub Python 3.11/3.12 完整离线 CI 及安装包验证；已发现 `tests/unit/test_patch_effect.py` 的格式阻断，已修正提交。其它检查以工作流实际结果为准。
2. 四单元真实模型对照最终结果、每单元实际调用量和未知费用状态记录。
3. Repair Memory 的加载/检索复验与工程综述，确认正常结束的任务不会泄漏信息到经验库。
4. PR 评审与合并：不得把未完成的复杂案例对照写成成功；旧 PR #2、后续独立分支保持清晰。


### 2026-10-09 补记：探索对照已完成

固定 4 单元真实模型试验已全部结束：V2 2/2 成功，V3 1/2 成功；`case_0022/V3` 三候选、1 次重新调查后仍失败。详细原因、调用量和原始结果 SHA256 见 [独立对照报告](2026-10-09-v2-v3-complex-pilot.md)。旧的“执行中”段落保留为当时状态记录，不覆盖当时事实。该结果只代表一次探索性试验。
