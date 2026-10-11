# e80ce75 收尾记录（2026-09-25）

这是一份工程收尾记录，不是发布通过证明。冻结 checkout、360 个评测结果和失败批次不改写；开发工作区的后续修复不冒充冻结版本的一部分。

## 已完成的证据复核

- 开发集 `1bc6af1b1acb4940a2c0829fc5d274aa`：240 个单元。
- 完成的 holdout `1a218a09a0414b459a732b8b9fd030bc`：120 个单元。
- 保留的失败 holdout `60a614169aaa40c998497257c3fb3159`：未完成任何单元，原始 artifact 已逐项读取并核对哈希。
- 三批调度签名有效，代码/配置绑定一致，corpus cutoff 均为 24。
- 8 个暂存私有案例的四个源码文件和输入文件与已注册清单逐字节哈希一致；未改原始私有树。
- 使用冻结版本的 `HoldoutController.validated_evaluation` 成功核验全部 120 条原生证据链。这不等于独立评分或 release gate 通过。

机器可读审计：`/home/you/gpu-agent-release-e80ce75/closeout-evidence-audit.json`。

## 开发集唯一验证 INCONCLUSIVE 的定位

公开记录：case_0011 / D / repeat=2；diagnosis `1972794b4e2aabbcc2c5bb41e58f039b`，candidate `77c1b420b3f8f42b7ee0a20dfb78558e`，verification `68156360a2e6b6cd69b19759ef0b4902`。

候选补丁在 `cudaMalloc(&scratch, ...)` 之前插入 `cudaMemset(scratch, 0, sizeof(float))`；此时源码中的 scratch 仍为 nullptr。该补丁存在确定的分配/初始化顺序错误。公开验证投影记录 runtime FAILED、memcheck FINDING、initcheck TOOL_ERROR，最终原因码 SANITIZER_TOOL_ERROR。

这不能作为“系统错误地放行修复”的证据，也不能断言运行器基础设施有故障。没有读取隐藏输入或改写旧判定；没有给模型补写正确补丁。工具错误的更底层机制不由此静态检查推定。

## 本轮确认并修复的发布工具缺陷

冻结版本 release collect-evidence 产生 FAILED run `768c13c11b1b44cba516612aeff4ba14`，在启动 pytest 之前拒绝真实 allowlist。原因：参数化 node ID 包含 `Invalid __global__` 等带空格文本，而正则不允许空格。

开发工作区的修复只增加 ASCII 空格；保留对换行、控制字符和 shell 元字符的拒绝。node ID 只用作比较数据，没有 shell 执行。新增测试直接加载真实版本化 allowlist，同时测试非法字符；相关 30 项测试通过，严格 mypy 通过。

未修改冻结版本，也未把该失败 run 改成成功。单独运行冻结版本的 18 项真实验收，输出 `supplemental-release-tests.log`、`supplemental-pytest-evidence.json` 和 `supplemental-pytest-junit.xml`，作为补充测试，不伪装成原控制器的正式通过证据。

实际结果：18 passed、0 failed、0 skipped，耗时 151.52 秒。收集的 18 个 node ID 与冻结 allowlist 完全相同，导入的 gpu_agent 来自冻结 checkout。1341 个非验收测试被 deselect，没有重跑全量离线回归或模型评测。

## 尚不能签署发布

- 独立评审方尚未提供 canonical 120-record HoldoutLabelPackage，不能代填或伪造盲评。
- e80ce75 原发布控制器存在上述 allowlist 缺陷；修复后的工具不能悄悄当作未修改的 e80ce75。
- 正式发布选择、清单和 gate 尚未通过。后续必须明确采用修订工具的验收政策或下一版本方案，不能修改既有实验绑定来绕过要求。
