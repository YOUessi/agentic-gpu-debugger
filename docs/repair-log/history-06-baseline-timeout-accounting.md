# 历史回补 06：完整批次、真实失败分类、超时和费用记录

回补日期：2026-09-29。依据 `docs/evaluation-report.md`、e80ce75 收尾文档、
20260926 定向复测说明、费用记录，以及此次只读核对的 completed 文件和 Git 差异。
不打开 evaluator 私有内容，不把这次文档补录写成重新做了实验。

## 第一轮：holdout 注册成功，但运行所需输入没有放在预期位置

### 问题与原因

e80ce75 的首次 holdout run `60a614169aaa40c998497257c3fb3159` 启动失败，完成零单元。
注册允许共享 harness/input 布局，而执行要求 kernel 邻接输入，原预检查只确认
目录存在，未在开始整个批次前检查所有实际输入。

### 应该与实际如何修复

不覆盖失败批次，不修改私有源码内容；后续用 hash 相同、evaluator-only 的执行快照
满足布局。开发提交 ab9ef0c 在预留别名/单元/签名调度前，先核对所有选择案例的
kernel hash 和 `_public_input`。本次已核对 CLI 差异，修复的是检查时机和完整性，
不是通过修改输入期望来绕过注册。

### 结果与限制

冻结版本后续 holdout `1a218a09a0414b459a732b8b9fd030bc` 完成 120 单元。原收尾记录
称暂存输入与注册 hash 一致，失败批次保留、120 条原生链核验通过。本次引用该审计，
没有重读私有树；这不等于重新完成独立人工评分，也不把后续预检代码算入 e80ce75。

## 第二轮：批次完成必须解释真实结果，不能只报“跑完”

### 实际结果

冻结 e80ce75 开发 run `1bc6af1b1acb4940a2c0829fc5d274aa` 完成 240 单元。
开发 VERIFIED_FIXED 为 139：A 4、B 7、C 43、D 44、E 41，各模式分母 48。
另有 81 模型声明无法判断、11 NOT_FIXED、6 超时、2 终态输出契约拒绝、1 验证无法判断。

公开汇总中的 holdout 为 72 fixed、9 NOT_FIXED、39 无验证结论（含一次超时），
各模式 fixed 为 A 3、B 6、C 20、D 22、E 21，各分母 24。这里只使用已公开汇总，
未引用或分析私有案例内容。这组数字不支持“E 已经优于 D”的结论。

### 为什么必须分类

无法判断可能是模型遵守证据边界，不自动等于代码 bug；NOT_FIXED 表示生成的补丁
没解决缺陷；输出契约拒绝与传输超时又是不同阶段。139/240 也不能混入后续选择性
重跑的成功结果。三次 repeat 不是三个独立案例。

### 实际处理

`tools/dev_failure_inventory.py` 只读公开开发记录，区分单元结局与调用状态，
按每次调用最后状态计数，不把 STARTED 和 COMPLETED 重复统计。完整开发批次
643 次物理调用；绑定历史费率的已知费用 USD 2.986159，六个超时单元费用未知。
这是已知部分，不能说完整账单只有这个数。本次不重新汇总所有公开产物。

## 第三轮：一个验证 INCONCLUSIVE 不能草率归咎运行器

### 现象、已确认原因和未确认部分

case_0011/D/repeat2 的公开候选在 `cudaMalloc(&scratch, ...)` 之前执行
`cudaMemset(scratch, 0, sizeof(float))`，而当时 scratch 仍是 nullptr。
这能确认候选存在分配/初始化顺序错误。公开投影显示 runtime FAILED、memcheck
FINDING、initcheck TOOL_ERROR，最终 SANITIZER_TOOL_ERROR。

### 应该怎么处理及实际结果

应如实保留无法判断而非强行判修好，也不能因 TOOL_ERROR 就断言基础设施坏了。
原收尾审计核对公开候选，没有人工代写补丁、没有读取隐藏输入，未改旧结论。
更底层工具错误机制未由静态检查证明，本篇保持未知。

## 第四轮：真实参数化测试被验收工具当成非法 node ID

### 问题与原因

冻结 collect-evidence run `768c13c11b1b44cba516612aeff4ba14` 在 pytest 启动前失败。
真实 allowlist 的参数化 node ID 含 `Invalid __global__` 等空格，而正则不允许空格。
这是工具输入语法过严，不是 GPU 测试本身失败。

### 实际修复和测试

82379fa 仅允许 ASCII 空格，仍拒绝换行、控制字符和 shell 元字符；node ID 是比较
数据，不拿来拼 shell。新增测试读取真实 allowlist。原记录为相关 30 项通过、mypy 通过。
未改冻结 checkout，未把失败 run 改成成功。

另对冻结版本单独执行允许列表的 18 项 GPU/隔离测试，18 passed、0 failed、0 skipped，
151.52 秒；1341 项非验收测试 deselect。保存 supplemental-release-tests.log、
supplemental-pytest-evidence.json、supplemental-pytest-junit.xml。这是补充证据，
不是原 collect-evidence 控制器突然通过。历史正式 gate 未通过仍如实保留，但不应
转化成今天用户必须另找评审人才允许修项目的新要求。

## 第五轮：60 秒超时的根因不等于已知

### 问题与分析

六项公开超时接近客户端 60 秒，没有 HTTP 状态/usage。它只能证明客户端期限
到达，不能区分服务端慢、网络挂起或请求是否已计费。盲目自动重试可能重复发送。
应记录未知，允许显式配置请求期限，并仍受任务剩余时间约束。

### 实际修改

472a1ee 增加 `GPU_AGENT_LLM_TIMEOUT_SECONDS`，默认 60，有限正数且不超过 600。
STARTED 和终态记录 request_timeout_seconds；旧记录不补造。没有自动传输重试，
没有增加总调用次数或任务总时限，没有美元上限。开发复测显式设为 120 秒。

### 离线失败与下一轮修正

超时组合检查第一次 125 passed、1 failed，原因是新增测试漏填必需 attempt 字段。
补齐 fixture 后定向 9 项通过，Ruff、格式检查和相关类型检查通过。没有重复重跑
全部已成功测试。完整命令及历史 stdout 未在本次恢复，数字来自原复测文档。

## 第六轮：八项定向真实复测——五项修好，不是八项全部成功

### 执行范围

仅原两项格式失败和六项超时，各新跑一次，旧成功单元不重跑。格式批次绑定 cbddff5，
超时批次绑定 472a1ee；本次读取 completed.json 确认分别为 2 和 6 completed_units。
每个候选用 full 验证，旧记录不覆盖，不与基线拼接。

### 逐项结果和解释

- case_0012/B/repeat1：模型声明无法判断，无候选；不是格式拒绝，也不是修好。
- case_0002/E/repeat2：VERIFIED_FIXED。
- case_0010/C/repeat0：NOT_FIXED，候选仅扩大共享数组，仍保留折叠下标，racecheck
  继续检出冲突。这是该次模型补丁逻辑没有消除冲突，不能通过改测试让其成功。
- case_0011/B/repeat1：模型声明无法判断，无候选。
- case_0008/E/repeat2、case_0014/E/repeat1、case_0015/C/repeat2、case_0013/E/repeat2：
  VERIFIED_FIXED。

35 次物理调用，107842 个记录 token，均有 usage，本轮无调用错误。
两项格式单元首次输出就合规，未触发新增重试；超时六项没有再次超时，但不能证明
延长时限是原因，更不能证明外部故障永久消失。一次新请求成功不是旧失败记录作废。

公开产物在 `/home/you/gpu-agent-schema-v9-regression/` 和
`/home/you/gpu-agent-timeout-v9-regression/` 的 selection、results、completed 文件。
本轮只核对完成文件和现有说明，没有读取目录下 evaluator 内容或重新生成补丁。

## 第七轮：费用为空不是免费，不能为补账重跑模型

### 问题和方案

定向脚本起初没有算费用，记录空值容易被误解为零。应从已存 usage 只读补算，
核对调用身份/hash/大小，同一调用只计一次，未知费率或 usage 保持未知。

### 实际修改与验证结果

87fa492 增加 usage_accounting、历史补算工具和脚本逐调用 usage/accounting 记录。
显式费率只用于估算，不影响运行权限或设置金额边界。历史八项输入 99609、输出
8233 token，使用历史输入 1.32/输出 3.96 美元每百万且明确不计缓存优惠，得到
USD 0.16408656，格式部分 0.03946932、超时部分 0.12461724。
这是估算，不是当日定价或实际账单。原 cost_usd=null 文件未覆盖，未查询余额或调用 API。
费用相关测试包含在维护轮 137 passed 中，不能另加为独立总数。

## 本篇最终状态

工程已修复的部分与模型仍失败的部分必须分开。历史 NOT_FIXED、无法判断、未知
费用都保留；没有“全部实验完美”的结论。本轮仅补录，不启动新的验收、人工评分、
GPU 实验或付费调用，也不擅自替用户扩大项目目标。
