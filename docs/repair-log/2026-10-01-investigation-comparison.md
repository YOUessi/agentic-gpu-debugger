# 2026-10-01：第二层D/E收益对照执行

用户明确要求开始第二层。按既定方案8个公开案例、D/E各3次，共48单元；
max_candidates=3、共享40调用、120秒请求、600秒任务，无美元上限或余额查询。
仅开发比较，不是holdout，不拼接旧实验，不以E必须胜出为通过条件。

## 第一轮：执行准备

使用failure-aware-execution技能。新增独立compare_repair_modes.py，不修改
原单次比较脚本或生产策略。合并读取两个公开案例注册表，启动前核对源码和
输入hash、兼容知识库、实际导入代码。每单元独立public/verification目录，
避免跨单元扫描累积；调用前记录attempt，结果fsync，意外异常保留中断信息
并停止，不自动回放已发请求。正常模型/自检失败按服务结果保留继续下一单元。

先测试日程和新增算法验证接通，再冻结独立源码快照与知识库，保存selection。
目前尚未发API调用；后续追加真实路径、检查结果和启动状态。

## 第二轮：预检查结果与实际启动

日程及新增算法验证路径：5 passed、55 deselected，10.72秒。完整命令为：

```bash
env -u PYTHONPATH PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/you/conda_env/agentic-gpu-debugger/bin/python -m pytest tests/unit/test_repair_comparison_schedule.py tests/unit/test_verification_engine.py -k 'predeclared_48 or new_algorithm' -q
/home/you/conda_env/agentic-gpu-debugger/bin/ruff check scripts/compare_repair_modes.py tests/unit/test_repair_comparison_schedule.py
git diff --check
```

首次Ruff发现输出行过长、测试import排序，修正后通过。没有修改模型策略或
GPU判断标准，未重跑旧成功实验。新算法入口检查为离线模拟的engine/replay
路径，不冒充本轮真实GPU结果；既有新算法原生验收记录仍独立保留。

独立目录：/home/you/gpu-agent-layer2-20261001-qll5d2。
source中建立本地快照提交081d17ffe3208373822d6a85f7fca9b095c228ea，env为独立
editable环境（未替换主Conda）。只复制明确项目文件，不复制密钥、旧实验或
claude依赖包。原工作区未commit/push。
runtime hash：e87fa6b81682d6bf01918b826cbb4b754d4e7ff976b17f6c2365e5d7fb9feb00。
prompt：m3-2026-09-30-v11；corpus hash：
15c53142f8949ba1d852a3227fd11460060adec359fda9ee0cf42c52011b8600。
selection.json在首个API调用前保存全部48项日程、输入hash和不含key的配置。
脚本每单元检查clean commit、runtime hash、只读知识库字节；禁止中途换版本。

加载既有/home/you/.config/agentic-gpu-debugger/provider.env，不打印key，unset
GPU_AGENT_CORPUS_FAMILY_ROOT，并设置GPU_AGENT_LLM_TIMEOUT_SECONDS=120。
执行命令（stdout/stderr由tee写到experiment.log）：

```bash
env -u PYTHONPATH /home/you/gpu-agent-layer2-20261001-qll5d2/env/bin/python /home/you/gpu-agent-layer2-20261001-qll5d2/source/scripts/compare_repair_modes.py --repository /home/you/gpu-agent-layer2-20261001-qll5d2/source --knowledge-index /home/you/.codex/worktrees/a352/agentic-gpu-debugger/.cache/gpu-agent/knowledge/20260929-v6/index.json --output /home/you/gpu-agent-layer2-20261001-qll5d2/results
```

进程已启动，工具session为8840，第一项case_0018/D/repeat0，diagnosis run
b0f07ffb25b0425e90fce240ef53e531。启动状态核查见两次provider调用已经COMPLETED，
任务处于PATCH_GENERATING。这里只确认真实请求已发出，不宣称单元修复通过。

每项结果保存results/unit-NN/result.json，并追加results/results.jsonl；只有
48项都走完才写completed.json。意外异常会保存interrupted.json并停止，无
自动重试。中断恢复前应读attempt、provider终态及已保存结果，不可直接重新
启动整批。结果目录存在时脚本拒绝覆盖。当前仍在运行，尚无收益结论。

## 第三轮：用户询问进度时核对完成结果

读取experiment.log末尾、全部results.jsonl和completed.json，确认48项全部
完成，冻结提交仍为081d17ffe3208373822d6a85f7fca9b095c228ea。此次只读核查，
没有重新启动实验或调用模型。统计脚本由一次性shell Python执行，未另存脚本。

D：24项中18 VERIFIED_FIXED、4 REGRESSION_DETECTED、2 PUBLIC_CHECK_UNAVAILABLE，
55次调用、169132 tokens。E：24项中19 VERIFIED_FIXED、3 REGRESSION_DETECTED、
2 PUBLIC_CHECK_UNAVAILABLE，150次调用、452272 tokens。合计205次调用、621404
tokens，已记录usage缺失数0；费用没有核算，不以0费用表述。

7个REGRESSION_DETECTED的reason_code均ORACLE_FAILED，四工具均CLEAN：
case0020/D的3次，case0020/E的repeat0、2，case0017/D的repeat2，以及
case0017/E的repeat1。这是功能输出未通过，不能把Sanitizer清洁等同修好；
具体错误补丁和Oracle失败的根因尚未逐项分析，不在此推定模型或实现责任。

4项PUBLIC_CHECK_UNAVAILABLE均case0017：D的repeat0、1，E的repeat0、2。
公开摘要显示build CLEAN、runtime FAILED、memcheck TOOL_ERROR。仅此不足以
判定是基础设施故障还是候选程序错误导致工具退出，需读取公开原始日志分辨。
本批没有任何单元记录两个以上自检round：因此不能宣称已经验证真实多轮修订
收益。7项独立Oracle失败按协议不回流模型，4项公开工具异常触发停止。

初步结论：E多修好1/24，但调用150对55，约2.73倍；8个已知开发案例各重复3次，
不是24个独立案例，不能据此宣称统计显著或总体优势。尚未完成逐案例配对、
工具/阶段耗时核算及11项未修复原因分析。下一步应先检查4项公开工具异常的
原始日志，确认是否存在阻止修订的分类问题；保留当前冻结结果，不中途改写。

## 第四轮：11项未修复原因的只读调查

用户要求查清原因。本轮仅诊断，不改生产代码、不重跑GPU/API。以results.jsonl
的ordinal定位unit目录，使用tools.dev_failure_inventory._Run读取public manifest
引用的candidate、self-check、run/result及sanitizer/result和日志，逐项核对大小
与hash。未读取verification/evaluator私有目录，Oracle失败仅使用已公开的
最终原因码，并对照既有公开算法说明与公开候选。首次一次性输出过长被截断，
随后改为逐类只输出补丁、退出码和短日志，确认全部11项而非只看一个样例。

### 4项公开自检异常：模型候选错误与项目分类缺陷叠加

一基序号11、34、47、48（case0017，D repeat0/1、E repeat0/2）全部把
`if (i < n)`改成`if (i + 1 < n)`，停止写入最后一个元素。原kernel用
cudaMemset(device_out, 0xff, bytes)填充输出，漏写槽保留非有限浮点值；
harness检测到非有限输出后退出1。公开stderr一致：
`vector harness: kernel output contains a nonfinite value`。

四个普通运行以及memcheck进程都没有超时、取消或截断，tool_error均null。
memcheck原始日志一致：`Target application returned an error`及
`ERROR SUMMARY: 0 errors`，exit_code=1。这是目标程序报错，不是本批证据
显示Docker或GPU运行时不可用，也不能将0 errors解释为应用功能正确。

repair.self_check先将runtime FAILED判为FAILED，继续执行Sanitizer；当解析
结果completed=false/check_outcome=TOOL_ERROR时，将status覆盖为UNAVAILABLE。
repair_candidates只在FAILED时继续修订，故4项都提前停止。保守的Sanitizer
解析不能判CLEAN可以理解，但修复控制层不该因此抹去已确认的候选运行错误。
已确认属于项目回路分类缺陷，触发它的初始补丁同时也是模型修错。

建议修复：区分候选应用失败与真正工具/基础设施异常；对已有完整、可信的
普通运行失败保留可修订状态及stderr。可先修复普通运行再检查Sanitizer，
不必在其失败后把缺少Sanitizer结论冒充通过。真正超时、取消、运行器错误、
截断等仍停止；不能把所有TOOL_ERROR一律当成可修补。需用这4项原日志补回归。

### 7项Oracle失败：只消除非法访问，没有保持预期计算

序号24的case0017/E在末尾补零，而公开功能说明要求循环回绕读a[0]；序号40的
case0017/D直接改为a[i]+b[i]，变成普通加法，丢失移位语义。
case0020的序号12、27、35、42、44保留a[i+1]的移位，只对末尾补零或夹紧，
但公开功能说明是每32元素对同索引a[i]+b[i]求和并广播。它们不再越界，
所以四Sanitizer均CLEAN，却改变/保留了错误的数学语义，最终ORACLE_FAILED。

项目有两个相关缺口。第一，公开self_check只查编译、进程和Sanitizer，没有
功能Oracle；7项因而被认为公开自检PASSED，直接进入最终验证，未触发修订。
这是已明示的设计缺口，不能把最终独立验证失败私自回流模型来掩盖它。
第二，算法语义在docs/case-diversity-CN.md中，但模型输入public_evidence仅有
源码、运行状态、findings和文档检索，patch输入仅public_source、diagnosis、
可选public_repair_feedback；没有独立的任务功能规格字段。损坏源码和Sanitizer
不能唯一确定循环回绕、补零或同索引等正确语义。不能将这7项全归咎为DeepSeek
能力差：实际输入缺少明确的功能约束，模型被迫猜测预期行为。

建议修复：新增公开、版本化的功能契约，让诊断与补丁共享它；提供与该契约
一致的公开样例/功能自检，再让公开失败进入多轮修订。不提供clean源码、
mutation位置、期望补丁或隐藏测试；数学输入输出定义是正常任务需求，不是
泄漏修复答案。最终隐藏验证仍独立，当前48项保留为原版本结果。

### 实际操作与边界

实际使用shell Python只读读取results.jsonl、通过_Run.only/read检查artifact，
输出各失败候选diff、公开check、退出码、tool_error、timed_out/cancelled/truncated。
静态定位repair.py的FAILED→UNAVAILABLE分支、service.py的repair终止和
agent/provider.py::_patch的payload字段。核对公开kernel的0xff初始化及
benchmarks/harness/vector_io.cpp的非有限输出检查，核对公开case-diversity说明。
最初猜测的execution/sanitizer.py、execution/parsers.py路径不存在，rg定位到
evidence/sanitizer.py；这只是只读路径查找错误，不是项目执行故障。

未运行新测试或模型/GPU实验，费用为无新增调用；生产代码未修改。原因已定位，
修复尚未实施。后续应先修分类和公开功能契约/自检，再对受影响失败做定向
验收，而不是直接重跑48项或无限采样。改动后的数据单独存放，不能改写本批。
