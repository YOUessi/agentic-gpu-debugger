# 2026-10-09：case_0022 V2 成功、V3 失败的证据级根因复盘

## 目标、来源与证据范围

本次**不调用模型重新采样**；直接对齐原始公开 RunStore 工件、真实 CUDA 补丁、自检报告、来源引用、重新调查动作和 provider 调用元数据。调试对象为 `case_0022`（Segmented Scan，128-thread block 原地共享内存迭代），在四单元固定探索测试中的原生结果：

- 冻结执行代码 `e4e7a8d4d069cecb24355e9c521bf7bd121e0870`、真实模型 `deepseek-v4-pro`。
- V2 父 Run `bdbb9c9962e448ac8d38b4c9a39e2ffd`，公开数据 `/home/you/gpu-agent-repair-v3-compare-20261009-01/unit-00/public`。
- V3 父 Run `c55216300e0840eb8d56dec46bfeefbb`，公开数据 `/home/you/gpu-agent-repair-v3-compare-20261009-01/unit-02/public`；唯一重新调查子 Run `d39699314da5439fbb9f802ba4745bc8`。
- 两次运行输入源文件的 `kernel.cu` SHA256 都是 `b79fd84c71131f94601cc153fa00307e848d73c477267bfd7a6f80384d0394a1`。候选源码与单个参考源码之间的 patch 提交 hash 必须区分；详细见同目录的 [审计摘要](artifacts/2026-10-09-case0022-root-cause-audit/trace.json)。
- 所有对照只使用 `public` RunStore，候选差异和自检 `repair/<n>/*`；**没有读取私有验证输入/答案、请求 API key 或模型隐藏思维链**。历史的真实 provider 工件持有 `kind`、`prompt_version`、调用用量与输出 Hash，不存储原样的模型请求体；精确请求输入只能依照保存的公开反馈和源码重构，不能伪称有原始 verbatim wire payload。

## 时间线：每份真实补丁实际做了什么

原始 Kernel 的关键语句如下（示意，保留计算及同步顺序）：

```cpp
__shared__ float tile[128];
tile[lane] = i < n ? a[i] + b[i] : 0.0f;
for (unsigned int offset = 1; offset < 128; offset *= 2) {
    const float value = tile[lane]
        + (lane >= offset ? tile[lane - offset] : 0.0f);
    __syncthreads();
    tile[lane] = value;
}
```

| 模式 / 候选 | patch hash 前缀 | 屏障结构 | 真实公开检查 | 结论 |
| --- | --- | --- | --- | --- |
| V2 第 1 个 | `39f29c4d03cb` | 初始化后增加；循环中改为只在写后有屏障 | 普通 Oracle PASSED，Racecheck FINDING；Synccheck 条件下 NUMERIC_MISMATCH | 失败 |
| **V2 第 2 个** | `77c257cd0e6a` | **初始化后、循环读完写前、循环写后**分别有屏障 | Oracle 和全部 4 种 Sanitizer 检查通过 | **VERIFIED_FIXED** |
| V3 第 1 个 | `cf55f41f7794` | 只将原循环写前屏障移到写后；未加入初始化屏障 | 普通 Oracle NUMERIC_MISMATCH（自检因此没有继续 Sanitizer） | 失败 |
| V3 第 2 个 | **`39f29c4d03cb`** | 增加初始化和写后屏障，但没有写前屏障 | 与 **V2 第 1 个候选 Hash 完全相同**；Racecheck FINDING，Synccheck NUMERIC_MISMATCH | 失败 |
| V3 第 3 个 | `9cb10afaef84` | 使用循环写前屏障，并在循环后新增屏障；循环写后缺少同步 | 普通 Oracle NUMERIC_MISMATCH | 失败 |

共同危险点不是“屏障少一个”这么简单，而是原地共享内存迭代需要分别分析两类顺序：

1. **读取当前阶段旧值 → 写入新阶段值**：为了避免快线程先覆盖某个 `tile[]`，而慢线程还要读取旧值，写前可能需要一个 **读完之后的屏障**（write-after-read，WAR hazard）。
2. **写完当前阶段新值 → 下一阶段邻居读取**：为了保证下一轮读到完整的新值，写后需要一个 **下一轮读取前的屏障**（read-after-write，RAW hazard）。
3. 初始化共享内存后，进入依赖全 block 的第一次读，也须满足初始化对所有线程可见。

V2 的第二候选同时满足了当前程序的上述同步要求，真实工具通过。V3 三个候选始终没有稳定满足这两类约束。不能把此结构直接推广成任意 CUDA 核函数的固定“两屏障模板”；要分析实际线程参与、访问依赖和 block-uniform 性。

## 信息使用链路：到底在哪一层失真？

### A. 初始诊断并非完全失败

V2 和 V3 的初始 `diagnosis.json` 都把故障归类为 `shared_memory_race`，并引用真正的共享内存 Read/Write hazard。二者都倾向推荐把同步放到写后（针对 RAW）。V2 的首候选也失败，**所以不能说 V2 一开始就完全诊断对了，而 V3 一开始诊断错了**。关键区别出现在失败反馈之后的修正能力。

### B. V3 重新调查时，未要求检验正在延续的竞争假设

V3 第一候选公开输出 `NUMERIC_MISMATCH` 后，`RepairCoordinator.decide` 选择 `REINVESTIGATE`；原自检因普通输出错误提前停止，当前候选没有 Sanitizer 结果。子调查真实动作仅为 `run_memcheck` → `finish_diagnosis`，memcheck 为 CLEAN。现有 `missing_evidence` 因 `public_functional_failure=True` 放宽 `tool_finding`，允许 **没有 racecheck** 的诊断完成。

更不一致的是子运行 Planner 最后一次 `finish_diagnosis` 的理由还提出了“可能缺少跨 tile 的累积前缀”等推断，而最后的 `DIAGNOSED` 又继续主张共享内存 race。这些推断没有新的 racecheck 证据支撑。

**关键对照**：Memcheck CLEAN 只排除该工具当前覆盖的内存错误，**不等于** Racecheck CLEAN，不应据此否定原始竞争假设，也不应凭它声明竞态已消除。

### C. 新诊断建议与失败候选源码自相矛盾

子诊断对 `d1fac7bc...`（V3 候选 1）的 `recommended_change` 是“把 `__syncthreads()` 放到 `tile[lane] = value` 后，下一轮读取前”；但 `d1fac7bc...` **已经这样写**。因此 `DIAGNOSED` 只是符合结构化格式与现有证据 gate，不代表其修复建议对当前源码有信息增益。它忽略了读完旧阶段值之前不能允许其他线程覆盖旧值（WAR）的问题。

### D. 第二候选新失败，第三次修订没有新诊断

V3 第二候选源码 `8e2f0acd...` 对应的 patch hash 与 V2 第一候选完全一致；公开反馈含 `functional=PASSED`、`racecheck=FINDING`、`synccheck_functional=NUMERIC_MISMATCH`。但 `max_reinvestigations=1` 已用完，控制器走 `REVISE_PATCH`，**沿用诊断自另一个失败源码** `d1fac7bc...`。旧的 `public_repair_feedback` 确实包含最新自检和 `diagnosis_source_sha256`，也包含当前候选源码，并未把所有结果彻底丢失；问题是模型无法看到**完整的历史失败序列**，也没有明确判断当前诊断是否仍适用。

第三补丁又回到了另一个不满足完整同步约束的结构，公开数值再次失败。

### E. Prompt 及模型随机性使单次因果归因不成立

V2 的 8 次物理模型请求全部为 `m3-2026-10-01-v12`；V3 起初 7 次同版本，子调查和后续修订 5 次变为 `public-repair-v3-2026-10-09-v4`。因此原试验不是“仅打开重新调查开关，其余完全一致”，不能从一个样本断言额外诊断必然有害。

## 修复方案（与审计结果一一对应）

### 改动 1：当前候选必须有匹配此前故障假设的工具证据

仅对**开发态 V3** 生效：若发生当前候选可验证的公开数值失败、之前已有有效诊断（如 `shared_memory_race`），而 memcheck CLEAN，则在 `finish_diagnosis` 之前要求当前候选至少有相应的 `racecheck` 结果。其他支持的对应关系为 `barrier_misuse→synccheck`、`uninitialized_memory_read→initcheck`。这是**证据采集约束**，不是把旧诊断作为当前有效引用；若工具不可用，必须如实停止为 `INCONCLUSIVE`，不能凭空猜测。

代码：`src/gpu_agent/agent/policy.py` 的 `followup_sanitizer_for_prior_hypothesis` / `missing_evidence`、`rule_router.py` 的 D 路由、`models.py` 的类型枚举。

### 改动 2：V3 每轮反馈保留来源已定界的完整公开失败历史

控制器在每个失败候选后记录本次 candidate kernel SHA256、限定长度的公开 patch excerpt、每项真实公开 checks 与静态 Patch Effect 结果。下一轮 `public_repair_feedback.revision_history` 逐轮保留，并新增 `diagnosis_scoped_to_latest_candidate`，当新诊断依赖旧版源码时明确标记。**不能使用隐藏 verifier 结果，也不改变 V2 的公开摘要合同。**

### 改动 3：V3 提示要求分析 RAW+WAR，两者缺一不可

仅修改 `REPAIR_INSTRUCTIONS`（修订为 `public-repair-v3-2026-10-09-v5`）：把“单个写后屏障未必保证旧值读取完成”作为一般性 CUDA 同步推理规则；不得重复已经存在于当前源码的推荐修改；修订时比较前几轮公共失败而非仅最后一次检查。没有把某个案例的正确补丁写入 Prompt，也不自动跳过 GPU 测试。

## 验证边界及后续

- 本地 124 项定向回归（含新增的来源-工具证据契约）通过；Ruff/Format、strict mypy（80 个源文件）全部通过。上述数值是对本分析分支一次修复后的实测，不等于原完整 CI。
- 必须使用独立公开 GPU 输入对“数值失败后 racecheck 必须真的运行”做物理验证，随后才值得对 V3 做预声明的新案例或验证性回归。
- 即使随后在 `case_0022` 上成功，也只能视为**被用于发现缺陷的开发案例修复**，不能把它计入未见案例的成功率提升；正式对照应选新案例、冻结代码、控制相同模型和采样预算，保留所有失败。

本报告支持的主结论是：**V3 在首次失败后允许仅凭数值失配和 CLEAN memcheck 完成竞争类重新诊断，产生了对已存在修改的重复建议；在下一次故障模式变化时，控制器使用了旧源码诊断与仅一轮自检反馈。** 这两处证据链漏洞已明确可通过代码和回归验证改进。不能从个例判断所有 CUDA 问题均应重新调查，或推广未经验证的同步修复模板。


## 补充：针对原始遗漏证据的真实 GPU 测试（2026-10-09）

测试代码：`tests/gpu/test_repair_hypothesis_gpu.py`。从公开 `case_0022` 原始源码，严格重建历史 V3 第一候选（将循环中的 `__syncthreads()` 从写前移到写后）。只对 prior diagnosis family 使用构造的 `shared_memory_race` 结构，其余是真正的 CUDA 原生构建、运行、公开 Oracle、memcheck 和 racecheck。

Tang 运行结果：公开 Oracle **NUMERIC_MISMATCH**，native memcheck **CLEAN**，native racecheck **FINDING**；修订后的 `missing_evidence` 在已有 CLEAN memcheck 时返回 `racecheck_outcome`，控制器禁止直接 `finish_diagnosis`，而 D Router 提出 `run_racecheck`。**1 passed in 8.83s**、真实模型请求 **0**。原生公共 Run ID `f883e8e94b584e32be3dbc5d5e8e4d92`；独立目录 `/home/you/gpu-agent-case0022-hypothesis-native-20261009-01/public`。这验证了**同一候选确实存在旧调查漏掉的 Racecheck 事实**。

## 补充：一次真实模型开发态单次重试（与冻结探索分开）

在修复证据门禁和 feedback 历史后，使用 Tang 实机 `deepseek-v4-pro` 再次运行公开 `case_0022`，但仅作为**已参与设计的开发案例回归**，绝不回填或改写此前的四单元对照数据。执行代码 commit `6a9a06de4c91e6714d696a8dd183cc94e1a7d13b`；保持正式模型配置、官方文档索引和同一公开输入，关掉 Repair Memory，候选上限 3、重新调查上限 1、模型最多 40 次、总截止 600 秒，仅一次真实模型采样。

实际：父 Run ID `053d36993ff647fab62e2ccd50009666`，**首候选** `PUBLIC_CHECKS_PASSED`，四个 Sanitizer CLEAN，独立验证 **VERIFIED_FIXED**，exit 0，模型请求 **7** 次，合计 **20,209 tokens**（input 18,488；output 1,721；cached 8,704 属于 input 子集），约 **197.91s**。候选 Hash：`77c257cd0e6a010ef5bfc7e623783bcc56e8d89246ca92ad196702242fe1ad07`，**与历史 V2 第二个成功补丁完全相同**。最终审计 Run ID `eaeb634725868e1130c5511d967ef732`。本地 RunStore `/home/you/gpu-agent-case0022-feedback-v5-live-20261009-01`；原始 CLI 日志 SHA256 `e1df3176efac7ea35a19682e38473783e104cd1872e0b60fc1dbde7f24f679ab`。

至关重要：本次调用组成是 **5 plan + 1 diagnose + 1 patch**，`prompt_version` 全部为原版 `m3-2026-10-01-v12`。由于**第一份补丁就成功**，本次完全没有调用新增的 V3 v5 再调查或跨轮修订能力。因而这个新成功**不构成新增机制改善模型修复率的证据**，反而说明单次样本之间的首次补丁波动较大。此结果应与旧 V3 三候选失败并存，未来需要在新未见任务上预声明多次重复和独立消融。

[本次物理 GPU 证据和真实模型回放 JSON](artifacts/2026-10-09-case0022-root-cause-audit/development-replay.json) 包含代码和结果 hash；不会将隐藏输入或未经授权的 API 响应写入公开文档。
