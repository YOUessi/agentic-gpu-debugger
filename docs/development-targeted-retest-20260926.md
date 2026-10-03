# 2026-09-26 开发集失败项定向复测

保留 e80ce75 原始结果，不恢复原超时请求、不覆盖记录、不与新结果拼接。
本轮仅选择原两项格式失败和六项超时，各重新执行一次；未重跑原成功单元。

## 修订与检查

- cbddff5：对 extra_forbidden 明确反馈“返回实例而非 Schema，移除额外字段”。
  没有删除模型字段来强行通过校验，重复错误仍被拒绝。相关 71 项测试通过。
- 472a1ee：单次请求时限可通过 GPU_AGENT_LLM_TIMEOUT_SECONDS 显式配置，
  默认仍为 60 秒，受任务剩余时间约束；记录配置值。此次复测指定 120 秒。
  没有增加自动传输重试、总调用次数或任务时限，也没有美元上限。
- 超时相关组合检查初次 125 passed、1 failed：新测试漏填必需的 attempt 字段；
  补全测试后定向 9 项通过。Ruff、格式检查和相关源文件 strict mypy 通过。
  未宣称重新执行过全量离线套件。

## 实际结果

格式项（cbddff5）：

- case_0012 / B / repeat 1：INCONCLUSIVE，模型声明无法判断，未生成补丁。
- case_0002 / E / repeat 2：VERIFIED_FIXED。

超时项（472a1ee）：

- case_0010 / C / repeat 0：NOT_FIXED。候选只扩大共享数组，保留折叠下标，
  racecheck 仍检出冲突；没有人工修改候选。
- case_0011 / B / repeat 1：INCONCLUSIVE，模型声明无法判断，未生成补丁。
- case_0008 / E / repeat 2：VERIFIED_FIXED。
- case_0014 / E / repeat 1：VERIFIED_FIXED。
- case_0015 / C / repeat 2：VERIFIED_FIXED。
- case_0013 / E / repeat 2：VERIFIED_FIXED。

共 35 次物理调用、107842 个记录 token，全部调用有 usage，本轮无调用错误。
费用未在该定向脚本中计算，不能据此记成零。每个候选使用 full 验证模式。
五项 VERIFIED_FIXED、两项无法判断、一项 NOT_FIXED，并非八项全通过。

原始文件：

- /home/you/gpu-agent-schema-v9-regression/results.jsonl
- /home/you/gpu-agent-timeout-v9-regression/results.jsonl

各目录内有 selection.json、completed.json 和独立运行产物。
这是一轮失败项回归，不是新版本完整评测；其成功率不能作为总体成功率。
两项原格式失败本轮首次输出就合规，未触发新增重试提示，不能把改善归因于提示。
超时未复现也不证明延长期限是改善原因，更不证明网络或服务端根因已消失。
其余原 NOT_FIXED 的补丁逻辑问题没有因本轮修改而自动解决。
