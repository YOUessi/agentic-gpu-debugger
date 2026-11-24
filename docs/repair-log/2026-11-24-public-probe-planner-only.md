# 2026-11-24 · AST 分页、公开功能 Probe、Oracle 隔离与 Planner-only 真实对照

## 源码与审计边界

本批实验固定仓库 Git 提交 **`473bbc21c09d689b9860b48d19cf08ac259ece00`**，Tang 的检出工作树为干净状态。代码仍在 Draft PR #16 中，未合并。新增功能：

1. `syntax_page()` 返回 `schema_version/source_sha256/total_nodes/offset/next_offset/has_more/nodes`。每页有界且是**完整 JSON**，所有节点可遍历，不再删除后续节点冒充完整结果。
2. `repo_probe_public(phase,input_size)` 仅允许注册 h001/h002/h003、预定公开尺寸 **32/33/65/129/257**，阶段由 Operator 控制器决定。before 使用原始 mutant，after 使用实际候选 SHA。Oracle 与预期输出只在控制器内部使用；模型得到的是大小/状态/区域/输入哈希/候选哈希/公开 run ID。
3. `PinnedGitWorkspace` 将 Oracle、tests、evaluation、reference、controller、execution 与相关源码从模型读取/写入路径过滤，注册任务的 `repo_read_file` 再限制到该案例的 mutant Kernel。
4. 新 `--planner-only-study`：只比较 `react_syntax` 与 `hierarchical_syntax`，固定模型、基础 SHA、任务、AST 工具、输入上下文、公开 Probe、动作上限、实际模型调用上限及 GPU 验证条件，仅切换 Planner/Reflect。Planner 多出的模型调用**单独报告**，不假装实际消耗相同。
5. 持久化记录 `effective_patch` / `first_edit_turn` / `public_check_reached`，主分析只看有效补丁率、首次有效编辑轮次与最终公开 Checker 到达率。

## 验收 A：真正 ToolHost + Tang RTX4090 CUDA

使用脚本化决定性的 Test Double（**不是物理大模型调用**），ToolHost 内真正执行 Docker CUDA 编译与数值测试：

| 时点 | 源码候选 SHA256 | 固定公开 Probe: n=33 | 解释 |
|---|---|---|---|
| before | `47e69f940ca6c8b9ab287b5ee5c9aed7973f4d1eeb7ac41b7e855a01609c9bce` | **PROGRAM_FAILED** | 原 mutant 在该公开输入下运行失败，不能事先假定只会数值不一致 |
| after | `88dd297b8a0bc53dc3b93b7d1a4678551346b4855d7139b4edaddecef0070ba4` | **PASS** | 两个 AST 节点更新后，当前候选通过同一输入 |
| full public | after 的同一 SHA | **PUBLIC_CHECKS_PASSED** | 真实完整公开 CUDA 数值+sanitizer 检查 |

首个有效编辑在工具 Turn **3**，最终公开检查在 Turn **5**，Turn 6 `finish`，物理 LLM 调用 **0**。进程退出码 **0**。原始审计摘要 SHA256：`783bf249e066f417a3db0f542bc8c8d67c2521c53ce8fdd2b5dda5117306f506`。

已在 Tang 同步运行相关 **54 项 CPU 集成测试 passed**（旧冻结 SHA `145a6e7`），当前最终冻结 SHA `473bbc21` 上 Ruff 检查、格式检查全部通过，GitHub HardSet 专项回归 **219 passed**。完整 GitHub CI 状态独立追踪，不把“专项通过”写成全 CI 已通过。

## 验收 B：真实 Qwen3-8B-AWQ 的 Planner-only 二臂对照

固定配置：h001，`evidence_first`，`syntax_mode=True`，`allow_batch_edits=False`，Tool Steps 上限 **12**，每试次物理模型调用上限 **16**，公开 Probe 上限 **4**，最终公共 Checker 上限 **2**。两组均来自相同 Git SHA、同一 Qwen3-8B-AWQ 本地 vLLM 模型。这是**真正物理模型自主选择工具**，不使用脚本化模型。

| 主指标 | 无 Planner：ReAct+AST | 有 Planner：Planner/Reflect+AST |
|---|---:|---:|
| **有效补丁率** | **0/1** | **0/1** |
| **首次有效编辑轮次** | 无 | 无 |
| **最终公开 Checker 到达率** | **0/1** | **0/1** |
| 修改前/后局部公开 Probe 执行 | 1 次 before | 0 次 |
| 工具动作（非 LLM 请求） | 12 | 12 |
| 实际模型请求 | 12 | 14（含 Planner/Reflect） |
| 严格修复结果 | 未进入隐藏验证 | 未进入隐藏验证 |

结果 **0/2 修复**，必须保留，并不能证明 Planner 有益，也不能断言 Planner 因果性有害；每组仅 1 次开发任务，没有独立种子、跨任务检验。Tokens 有审计记录，但根据预注册分析标准，本阶段不以其作为主要结论。

### 失败轨迹关键事实

- ReAct+AST：最初两次 AST Inspect 使用无法匹配的函数符号，第 3 次取得合法完整 JSON page，此后多次重复请求同一页被 `REDUNDANT_SYNTAX_INSPECTION` 拒绝。第 9 次模型动作调用了一次合法 **before** Probe。最后的 `repo_replace_syntax` 使用一个不存在或过期的 node_id，遭 `CUDA_SYNTAX_NODE_NOT_FOUND_OR_STALE` 拒绝。
- Planner/Reflect+AST：首次 Inspect 获得有效分页。之后重复 Inspect 和多次 `repo_replace_syntax` 的 `CUDA_SYNTAX_NOOP`（修改文本等于原节点文本），另有 `CUDA_SYNTAX_NODE_NOT_FOUND_OR_STALE`。Planner 与 Reflect 确实执行，Controller Progress 仍显示 `active_gate=edit`，没有生成有效候选，更没有到达 full Checker。
- 以上都是真实模型工具决策失败；并非编译器或 GPU Checker 对合法补丁的失败。**不能以“脚本化 GPU 成功”代替“模型自主修复成功”。**

## 对本轮六项要求的结论

- AST 安全分页、ToolHost 真实执行、修改前/后 public Probe、Oracle 与评测源码隔离：**工程链路通过**。
- Hierarchical/Syntax 与 Planner-only 的统一 sweep：**完成接入，二臂真实 Qwen3 对照已运行并保存所有失败**。
- 三项优先指标：**有效补丁率与公开检查到达率仍为零**；首次有效编辑未发生。故暂不在这份研究结论里报告 Strict Fix 增益或 Token 优劣。
- 下一道应攻克的是：**模型取得 AST Node page 后为什么无法从 Inspect 推进到有效 Edit**。需继续检验节点证据钉住、有效编辑参数约束以及停止重复读取的 Controller policy，然后在完全相同的两臂条件下再测。当前没有训练 Agent，也没有证明自主决策质量已提升。

原始数据 SHA256：
- 真实 Qwen3 Pair：`4d7f06755011a504fff3a9bae51b7b9ebc750a6d1cf44b7d68934f2d3fb7fb49`
- 冻结试验计划：`c64119abe91f4650aa4e74d2b8c6ab59375468cd66c6eb23838f39e2ab116274`
- 真实 GPU before/after 功能 Probe：`783bf249e066f417a3db0f542bc8c8d67c2521c53ce8fdd2b5dda5117306f506`

私有实验文件存于 Tang：`/home/you/gpu-native-harness-validation-20261124/public-probe-verified-473bbc/`。包括公共/私有 RunStore、tool trails、物理模型调用审计与每试次持久化结果；Evaluator Secret 仅在 Tang 私有 0600 文件中，**没有上传 GitHub**。机器可读汇总参见 [实验摘要](2026-11-24-public-probe-planner-only-experiment.json)。
