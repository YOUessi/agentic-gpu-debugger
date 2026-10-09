# 2026-10-10：case_0022 强制第一次补丁失败后，真实 DeepSeek 修复成功

## 实验设计与边界

目的：避免“首候选碰巧成功”，直接检验 V3 v5 在已知首候选失败后，是否能获得当前源码真实 Racecheck 证据、更新诊断并自主提出修订。

**第一份补丁由实验控制器强制注入**（历史 V3 第 1 个失败补丁），**不是 DeepSeek 生成**，也不计入模型物理调用；初始规划/诊断、重新调查规划/诊断、修订补丁均使用真实 DeepSeek v4-pro；GPU 构建、功能检查、四种 Sanitizer 和独立 strict verifier 是原生执行。它是**开发用机制验证**，不用于未经调试案例的成功率统计或说明统计显著提升。

- 执行代码提交：`9ea075c943607b35d9f92c5f60a084b9640258c1`。
- 启动：UTC 2026-10-09 16:08（Tang 本地 +08:00 的 2026-10-10 00:08）。
- 案例：公开 `case_0022`（Segmented Scan）；原始 Kernel SHA256 `b79fd84c71131f94601cc153fa00307e848d73c477267bfd7a6f80384d0394a1`。
- 冻结 CUDA 文档索引文件 SHA256 `6ddc3cf2443ff983f08d0dcae44ca5f68ba28033beeb391a162422f9936ed27a`，经验记忆禁用；最大候选 3、重新调查 1、模型调用 40、任务截止 600 秒。
- [原始预声明条件](artifacts/2026-10-10-case0022-forced-failure/predeclared.json) 绑定首候选源码 hash 和唯一采样；没有“失败就重试”。

## 真实两轮过程

**首候选——强制失败：** 将原始 Kernel 中循环体的 `__syncthreads()` 从写入前移到写入后，SHA256（单文件）`d1fac7bc60f231f4bc041b71ef7b04ece7f13ba8794aa90f7a9cdd504ebecce5`、完整候选 Hash `cf55f41f779456a72097ed94eeaa77ee5b0cd304c7a3e8d20581277931867d05`，与早前真实 V3 失败候选一致。真实自检 `functional=NUMERIC_MISMATCH`，首候选失败。

**重新调查——真实 DeepSeek 与原生 GPU：** 重新调查子 Run `294901f11582489c978fdea7cb404641`，执行动作 `run_memcheck`（CLEAN）→ `run_racecheck`（**FINDING**）→ `retrieve_official_docs` → `finish_diagnosis`。诊断重新引用候选当前 Racecheck 发现的共享内存读写冲突；没有像旧 V3 一样在 CLEAN Memcheck 后提前结束。

**跨轮反馈——实际投递给修订：** 父 Run 的 `repair/1/feedback.json` 含首轮失败、候选 Hash 与 patch excerpt 的 `revision_history`、`diagnosis_scoped_to_latest_candidate=true`。模型在 `public-repair-v3-2026-10-09-v5` 提示下产生下一份补丁。

**第二候选——模型自主生成：** [差异文件](artifacts/2026-10-10-case0022-forced-failure/candidate-02.diff) 在循环中建立读前、读后写前、写后的同步阶段。Hash `7a0186fc6757a292b628b90ad83641deac7f46f79bfdfe13d76b3b420fb3`，与以前 V2 成功 Hash `77c257cd0e6a010ef5bfc7e623783bcc56e8d89246ca92ad196702242fe1ad07` **不同**。公开数值 PASSED，memcheck/racecheck/initcheck/synccheck 均 CLEAN；隔离独立验证 `VERIFIED_FIXED`，reason `ALL_REQUIRED_CHECKS_PASSED`，CLI exit 0。

## 原生数据与资源

| 指标 | 数值 |
| --- | --- |
| 父 public Run ID | `9d9ad8807eab4ee08161ca4a884d2b7a` |
| 第一次注入候选 | 1，非模型补丁 |
| 真实模型调用 | **11**，全部 COMPLETED（8 plan + 2 diagnose + 1 patch） |
| 总 Token | **52,938**；美元成本因未核实费率而未知 |
| 修复候选总数 | **2** |
| 新候选调查 | **1**，Racecheck FINDING |
| 调查工具 | 4 次 Sanitizer、2 次检索；公共最终自检另计 4 种 |
| 最终验证 | **VERIFIED_FIXED** |
| Python 实验内部耗时 | **236.87 秒**；外层约 238.58 秒 |

Tang 原始运行目录 `/home/you/gpu-agent-case0022-forced-first-deepseek-v5-20261009-01`；[结构化试验结果](artifacts/2026-10-10-case0022-forced-failure/result.json) 与 predeclared.json 来自真实工件，并保留公开自检和修订历史。工具检查时，代码 worktree 干净，项目 Docker owner 标签剩余容器 0 个。未导出独立验证器的私人输入/答案、API key 或未经记录的隐藏思维链。

## 结论与下一次研究验证

首次强制失败之后，真实模型确实利用新增 Racecheck 证据完成一次重新诊断与修订，且修订候选通过独立严格验证。**这证明新闭环能工作，但仍不能证明 V3 整体修复率高于旧 V3/V2**，因为首候选由实验强制给定，且没有冻结随机采样的重复对照。应继续使用未见 CUDA 算法、预声明重复规则、等价模型预算的多组实验检验泛化与成本，并保留失败。
