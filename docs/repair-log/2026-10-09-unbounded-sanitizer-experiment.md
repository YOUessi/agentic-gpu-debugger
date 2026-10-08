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
