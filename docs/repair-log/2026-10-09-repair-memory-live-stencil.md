# 2026-10-09：冻结 Repair Memory 跨算法真实模型试用

## 目标与验收约束

使用真实 DeepSeek v4-pro、Tang RTX 4090 Laptop GPU、现有真实 CUDA 检查以及源于两个**公开失败**任务的冻结历史经验索引，对未参与历史经验提炼的 `case_0018`（一维 Stencil）试用一次 Repair V3。证明：控制器是否实际检索历史经验、模型是否收到来源约束的提示、公开自检与独立严格验证是否能够完成。只执行一次模型采样，不补抽；**不是**记忆有无效果的 A/B 试验。

执行源码提交 `d2747edd4518a8c116dbc6abf518bbed891beacf`；任务原始 `kernel.cu` SHA256 `d9df4d683f5b09c31454b9f196ec7d49c3cddb94aa9df07d8d3ff9ac1f828434`。public `input.json` SHA256 `7841a87fc3f8300dca145438998e8e5a142044d2e0286cea4d99887ac7e35a7c`，`task.json` SHA256 `2b26ffe710ffd8a50a0aa5cc8467d1397570c8a9840df70fb53e59359fbe5c53`。

本次使用的 [合并冻结经验索引](artifacts/2026-10-09-repair-memory/merged_public_failures_20261009.json) 内部 corpus SHA256：
`502269e68014aade52a5c36ffd6f2cddcb23ba3793cd79ac4f15cfbb60a70031`；Tang 原始索引文件 SHA256 `e200097a23425767ca5678462c044d2e0286cea4d99887ac7e35a7c`。索引只有四条历史失败经验，来源 Run 分别为 `177a6daf715a4b53bfc47eeca11cf831` 与 `c55216300e0840eb8d56dec46bfeefbb`；不含 `stencil-cpu-v1` 本案例。

## 实际检索与使用

父诊断 Run `68a3c235f4d14668a106db50035bb4d4` 的 `repair/experience-retrieval.json` 明确写入以上 corpus hash，实际召回三条：两条 `BLOCK_BARRIER_EDIT_FAILED`、一条 `NUMERIC_PASS_RACE_REMAINS`，均标识为 `HISTORICAL_PUBLIC_FAILURE_NOT_AUTHORITATIVE`。这些经验只作为补丁生成的低权重历史提示，不属于当前 GPU 的直接观测或官方规范，也没有读取私有 verifier 材料。

模型生成首个候选：在 Stencil 的共享内存 tile 主体写入后增加 `__syncthreads()`，之后保留 halo 写入及邻域计算，具体差异见[本次候选补丁](artifacts/2026-10-09-repair-memory-stencil/candidate-01.diff)。

## 原生 GPU 验收结果

| 检查 | 实际结果 |
| --- | --- |
| 模型 | DeepSeek v4-pro，6 次物理请求全部 COMPLETED（plan 4、diagnose 1、patch 1） |
| 候选 | 首候选一次通过，重新调查 0 次 |
| 公开编译和 runtime | CLEAN / SUCCESS |
| 公开 CPU Oracle | PASSED |
| memcheck / racecheck / initcheck / synccheck | 全部 CLEAN |
| 独立严格验证 | **VERIFIED_FIXED**，`ALL_REQUIRED_CHECKS_PASSED` |
| CLI 退出码 | 0 |
| 实际总时间 | 约 186.16 秒 |
| 模型 Token | input 16,962、output 1,370、total **18,332**，cached 7,296（已包含在 input） |
| 费用 | 未核实，未知，不填写 0 |

候选 SHA256 `eac35ef138aabc7dd559589ae1eebb8391b66b988a8134d48d09b8fefc293cbb`；独立验证审计 Run `ab4ee97ae77ddaf58ee01aeb7f4c534d`。运行后的精确代码 checkout 没有未提交修改，带项目 owner 标签的残留 Docker 容器数量为 0。

Tang 独立实验目录 `/home/you/gpu-agent-repair-v3-memory-live-20261009-01`；原始 `live.log` SHA256：
`199fe5eeed489f72e25e92eb6df2c58fe7d648843dfa90f66e61d96ba00b11be`。GitHub 提供的是经核对的公开概要、候选差异和经验索引，不冒充完整原生日志，且不复制私有输入、模型密钥和第三方请求响应标识。

## 限制与后续

**这次证明了历史经验真正进入模型修复输入，并且任务通过独立验证，不能证明经验提示促成成功。** 因为没有同种条件、同一组随机采样下的 memory-off 对照，不能归因为 Repair Memory 的增益。现在的四条记忆也远不足以代表完整 CUDA 错误模式分布。后续应冻结新开发案例集，按有记忆/无记忆做随机顺序重复评估，报告修复率和调用成本，并严禁把评估任务写回同批次经验库。
