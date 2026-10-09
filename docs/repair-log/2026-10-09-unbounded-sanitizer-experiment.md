# Repair v3：单次真实试用后的实验性 Sanitizer 预算解锁（2026-10-09）

## 背景

2026-10-09 的一次真实 DeepSeek + Tang 4090 调试试验，在 `case_0009` 首个补丁失败后触发了重新调查。
初始调查调用了 3 次 Sanitizer；重新调查先运行 memcheck，耗尽原来 4 次的总调查额度，
导致后续有必要的 racecheck 被拒绝。记录和失败补丁见
[首次真实试用](2026-10-09-repair-v3-live-use.md)。

本实验只改变**开发 Repair v3** 的调查 Sanitizer 计数策略，使它不再受到一个单独的
4 次次数上限约束。整个运行依旧保留原有的 Agent steps（38）、模型物理请求（40）、
总任务 deadline（600 秒）、候选数量（默认 3）、重新调查次数（默认 1）以及
隔离 GPU/Oracle/Private holdout 安全契约。

## 实现范围

- `AgentBudget.max_sanitizer_calls=None` 表示无单独次数上限；默认仍然为 4。
- `AcquisitionUsage` 接受真实记录的超出 4 次的采集数，不放松计数真实性。
- `BudgetLedger.reserve` 与 `decide_action` 在最大次数为 None 时不拒绝调查，
  仍根据动作、阶段、证据、步数、时限、模型额度与去重规则授权。
- `RepairPolicy.unbounded_sanitizer_calls` 默认 `False`。
- CLI 新增 `repair --reinvestigate --unbounded-sanitizer-calls`；不带
  `--reinvestigate` 时拒绝该开关。此策略仅在 public non-evaluation V3 的
  `ApplicationService` 生效，正式 A–E 实验、`diagnose` 与 V2 默认行为不变。

## 验收范围

- 默认预算 4，默认 V2 和普通 V3 行为不变。
- 在选择该模式时，真实次数能超过 4 并被完整保存，且不能绕过其他预算。
- 同一失败候选重新调查可调用原先被拒绝的 racecheck；不为达到成功目的放宽修复验证。
- 最终真实 GPU + DeepSeek 结果单独记录，无论成功失败均保留。
- 未授权无限制模型费用或无穷执行；这里「无上限」特指没有独立 Sanitizer 次数硬上限。

## 第一轮：实现、回归与失败修正

为了不改变冻结的 A–E 和默认 V2，本次只在 Repair v3 显式使用
`--unbounded-sanitizer-calls` 时取消独立调查次数上限；其余限制保持不变。
代码提交 `243e1d7`，随后独立修复新测试夹具的
`verdict` 返回字段遗漏（`0ecd7b2`）、Ruff 未使用导入和格式配置差异
（`9312d88`、`c13f0de`、`b3e9891`），最后修复
`RuleRouter` 对可空预算字段作整数比较导致的 strict mypy 错误
（`c005936`）。这些回归失败没有作为 GPU 或模型失败来统计。

在 Tang **独立 Git worktree** 中对最终代码提交
`c005936d8f3524669ec509049c1b1543c9b11a3f` 验证：

- 四组定向 pytest：**71 passed in 15.60s**，退出 0，涵盖旧预算、
  无独立 Sanitizer 上限、CLI、重新调查续接、旧 Agent 行为。
- Ruff lint：**All checks passed**；Ruff format：**78 files already formatted**。
- `mypy --strict src/gpu_agent`：**77 source files，no issues**。
- `pip check`：**No broken requirements**。
- 无本实验代码版本的完整 1,578 项 CI 结果；不能冒充已进行全量回归。

## 第二轮：真实模型与真实 GPU 修复（成功）

### 1. 预期与输入

用户要求移除导致首次真实失败的 4 次 Sanitizer 调查次数限制，验证能否修复。
使用已有 `DeepSeek v4-pro` 模型和 Tang RTX 4090 Laptop GPU，对
`benchmarks/public/case_0009/public_input` **仅执行这一次新模型采样**。
选例与上一轮保持相同。上次试验：
[2026-10-09 首次真实试用失败](2026-10-09-repair-v3-live-use.md)。

固定执行提交：`c005936d8f3524669ec509049c1b1543c9b11a3f`；
原始 `kernel.cu` SHA256：
`394dc3f945580df9ea74437ef75b2873848ca1d20522c031c7ef02b6f9cd8c11`；
官方 CUDA 文档索引 SHA256：
`6ddc3cf2443ff983f08d0dcae44ca5f68ba28033beeb391a162422f9936ed27a`。
环境沿用 Python 3.11.16、CUDA NVCC 12.8.93、Compute Sanitizer 2025.1 和已锁定的 GPU Docker 工具链。

专用运行目录位于
`/home/you/gpu-agent-repair-v3-live-20261009-02`；原始 CLI stdout/stderr
合并为 `live.log`，其 SHA256 为
`2ce2e0a1f9bdba211dbaf0b65e3fc1db53c39287370b49f4588e335d87490da8`。
该本地原始文件不与本仓库其他历史运行输出合并。

核心选项（实际在隔离 Python 进程中运行已提交代码，凭据从本机已有配置载入且未回显）：

```bash
gpu-agent repair benchmarks/public/case_0009/public_input \
  --allow-paid-calls --reinvestigate --unbounded-sanitizer-calls \
  --max-candidates 20 --max-reinvestigations 3 --max-llm-calls 40
```

`max_sanitizer_calls=null` 表示**没有单独计数限制**；
这不是无穷付费请求：实际仍受到 600 秒总 deadline、38 个 Agent steps、
40 次模型调用、20 个候选上限和 3 次重新调查上限约束。

### 2. 真实诊断与真实补丁

原始诊断再次正确识别共享内存竞争：`folded_warp_writes` 使用
`threadIdx.x & 15U` 将 32 个线程折叠映射到 16 个共享内存槽，
产生同一槽位被多个线程写入的 write/write hazard。

本次真实 DeepSeek 直接生成不同于上次失败的**第一候选**，见
[完整 unified diff](artifacts/2026-10-09-unbounded-sanitizer-case0009/candidate-01.diff)：

```diff
-    __shared__ volatile float slots[16];
-    const unsigned int slot = threadIdx.x & 15U;
+    __shared__ volatile float slots[32];
+    const unsigned int slot = threadIdx.x;
```

同时将 `block_summary = slots[0]` 改为
`block_summary = slots[0] + slots[31]`。
其主要作用是避免不同线程写入同一个 shared-memory 槽位。
后一处同时改变了 debug/global side effect 的表达式；
当前 `vector-add` 受信 Oracle 未以该变量单独定义额外语义，
因此本次正式结论仅在当前注册任务及验证范围内成立，不外推到任意消费者。

### 3. 实际运行、预算和验证结论

应用 CLI 真实返回 **exit code 0**，远程进程总耗时约 **199.21 秒**。
父诊断 Run ID：
`383757f3c8094d1eb478d749aab2d080`；
第一候选公开自检 Run ID：
`a855014937ee47a8b44d8ff4042350fa`。

| 核验 | 实际结果 |
| --- | --- |
| 首候选 compile | CLEAN |
| 普通 GPU runtime | SUCCESS |
| 公开 CPU Oracle | PASSED，32 个输出全部为 3.0 |
| memcheck | CLEAN |
| racecheck | CLEAN |
| initcheck | CLEAN |
| synccheck | CLEAN |
| 公开 Repair stop reason | PUBLIC_CHECKS_PASSED |
| 再调查次数 | **0**，没有触发 |
| 独立 strict Verification | **VERIFIED_FIXED** |
| 验证原因 | ALL_REQUIRED_CHECKS_PASSED |
| 原始 finding 是否仍存在 | false |
| 新 finding | 0 |
| CLI exit | **0** |

候选 SHA256：
`26746251d6bd4cd2834c98aca272be952a4a692540b2e81c0cf7857c75a31fb8`。
独立验证结果记录的 Evaluator Audit Run ID：
`4e92fd634146c32e914b7d3bc87010d1`；
不将私有测试输入、期望答案或 evaluator 机密复制到 GitHub 公开附件。

最终 `agent/final-budget.json`：`max_sanitizer_calls=null`，
`sanitizer_calls=2`、`rag_calls=1`、`agent_steps=4`、`llm_calls=6`。
固定公开自检另执行 4 种 Sanitizer；独立验证也记录 4 种 required checks。
六次真实模型调用全部为 `COMPLETED`，包括
**plan 4、diagnose 1、patch 1**。
合计 input **16,239**、output **1,426**、total **17,665**、
cached **6,912** tokens；cached 属于输入统计的一部分，不再加算。
API 真实计费金额没有审核后的费率绑定，记为 **unknown**，不能写 0。

运行结束后，Tang 的被测 Git worktree 仍在精确提交 `c005936`，
`git status --porcelain` 为空；Docker 项目 owner 标签的剩余容器数为 **0**。
[结构化结果摘要](artifacts/2026-10-09-unbounded-sanitizer-case0009/result.json)
和 [模型用量](artifacts/2026-10-09-unbounded-sanitizer-case0009/provider-usage.json)
与原始本地 RunStore、CLI 输出交叉核对。

### 4. 结论、限制与下一步

**这次确实修复成功，并通过独立严格验证；但不是因为用了超过 4 次
Sanitizer，而是第一次补丁就正确，实际只用了 2 次调查工具，也没有触发 V3
重新调查。**

此前同案例不同一次真实采样：10 次模型调用、31,744 tokens，第一补丁无效，
重新调查时 4/4 Sanitizer 限额耗尽，最终失败。两次采样的模型/Prompt
具体调用轨迹不同，并且此次提高了候选及调查上限；不能把两次结果简单当成
同随机种子、同控制变量的 V2/V3 因果比较。

后续若要证明取消上限或新 Agent 修复闭环是否提高成功率，需要在冻结模型、
任务集合、预算与采样规则后设计成对重复实验，同时保留失败输出与每次成本。
本轮没有继续调用模型、没有追加抽样、没有合并实验分支；冻结 A–E 和
私有验证权限边界仍然保持不变。

## 追加门禁（2026-10-09）

基础 Repair v3 已在 PR #2 合并到 `design/v2-operator-workflow`。
本实验继续保持独立 PR #4；增加 D 模式 `RuleRouter` 在 4 次采集后仍允许未做过的
Racecheck（仅显式取消独立额度时）的回归，同时确认旧默认在 4/4 后会停止。
该提交也触发 GitHub 两套锁定依赖的完整 CI 验证，确保不会只以 71 项定向测试宣布通过。
