# 2026-10-01：运行失败可修订与公开功能自检

## 背景、目标和验收条件

本轮依据第二层48单元实验的11项失败修复项目流程，不要求通过反复抽样把模型变成100%成功。
工作区为 `/home/you/.codex/worktrees/a352/agentic-gpu-debugger`，基线HEAD为87fa492，
已有大量未提交修改，本轮保留它们，不提交或推送。旧实验目录
`/home/you/gpu-agent-layer2-20261001-qll5d2/results` 只读保留。

验收：明确的候选程序失败能够触发修订，真实基础设施失败仍停止；公开功能规格能够进入
模型上下文，公开输入的错误输出不能被自检判为通过；不读取隐藏输入、答案或评测反馈。
先验证离线失败路径及关联回归，再对受影响公开案例开展定向GPU/API验证，不重跑旧成功单元。

## 第一轮

### 问题、证据和原因

具体原始证据见 `2026-10-01-investigation-comparison.md` 第四轮。
case_0017的4项候选未写最后一个输出，程序以非有限数值错误退出。运行阶段已经明确FAILED，
但随后Sanitizer的非完整结果把它覆盖为UNAVAILABLE，修复循环不再修订。这是控制流缺陷，
不能通过把所有Sanitizer工具错误都判为可修复来解决。

另外7项候选通过内存检查却算错数值，最终独立验证报告ORACLE_FAILED。
目前公开自检没有功能oracle，模型也没有收到独立于故障源码的功能规格。
因此这里同时有模型补丁错误及项目未提供需求、未自检功能的设计缺口。

### 应该如何修复

正常执行明确失败时直接返回可修订FAILED，保留stderr；超时、取消、截断和工具故障仍为
UNAVAILABLE。增加与源码哈希绑定的公开task.json，声明算法功能而非故障、修补位置或参考代码；
将其写入公共证据并传给调查、诊断、补丁。控制器使用固定允许的纯CPU参考函数，
只对原有公开输入执行功能比较；独立验证保持不变。未提供公开规格的输入不得声称功能自检通过。
这属于新repair契约，不追溯修改旧评测模式的数据或结论。

### 实际改动与测试

已添加public_task模块、公开task.json与模型功能规格字段，repair更新为v2，prompt更新为v12。
正常运行数值检查及instrumented数值检查只使用本次公开输入。未提供功能规格时标记
PUBLIC_TASK_UNAVAILABLE，不能以工具干净冒充完整功能通过。解析器和独立验证未放宽。

首轮 `env -u PYTHONPATH PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/you/conda_env/agentic-gpu-debugger/bin/python -m pytest tests/unit/test_iterative_repair.py -q`
得到2 failed、14 passed：旧mock缺新增参数；旧测试把无规格的普通程序作为完整自检通过。
分别更新mock接口、明确缺规格必须UNAVAILABLE，新增真正有规格的全通过路径覆盖。
随后同一解释器执行 `-m pytest tests/unit/test_public_repair_correctness.py tests/unit/test_iterative_repair.py -q`
得到2 failed、25 passed。一个真实接线缺陷是backend.prepare重建EvidenceBundle，丢失
新规格；另一个是测试漏数case_0000（公开基线），实际21个目录而非20。
前者修为prepare后保留原生证据并附加预先校验的规格；后者明确将基线包括在绑定检查中。
`mypy --strict src/gpu_agent` 已通过75文件；Ruff首次检查因新增长字符串超长失败，已拆行。
上述均为离线模拟，没有新增GPU/API调用。继续检验规格确实送到模型及工具链。

## 第二轮：修正接线后验收与定向实验准备

同一工作区、解释器和隔离环境变量下执行：

```
env -u PYTHONPATH PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/you/conda_env/agentic-gpu-debugger/bin/python -m pytest tests/unit/test_public_repair_correctness.py tests/unit/test_iterative_repair.py tests/unit/test_agent_loop.py tests/unit/test_provider_output_telemetry.py tests/integration/test_provider_contract.py tests/unit/test_verification_engine.py -q
env -u PYTHONPATH PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/you/conda_env/agentic-gpu-debugger/bin/python -m pytest tests/unit/test_repair_comparison_schedule.py tests/unit/test_diverse_corpus.py -q
/home/you/conda_env/agentic-gpu-debugger/bin/mypy --strict src/gpu_agent
/home/you/conda_env/agentic-gpu-debugger/bin/ruff check src/gpu_agent/public_task.py src/gpu_agent/repair.py src/gpu_agent/service.py src/gpu_agent/evidence/models.py src/gpu_agent/agent/{models,orchestrator,prompts,policy}.py tests/unit/test_public_repair_correctness.py scripts/compare_repair_modes.py
git diff --check
```

结果分别181 passed（49.71秒）、28 passed（0.67秒）、75个源码文件类型检查通过、
静态检查通过、diff检查通过，退出码均0。没有执行全量套件。新增测试包括旋转/分组
算法的安全但错误输出、明确运行失败、超时、正确输出、instrumented错误输出、规格
哈希绑定及完整证据传播。缺规格的旧测试不再声称功能正确，新增有规格的正向测试接替。

task.json为21个公开目录（含case_0000基线）提供公开需求选择和源码绑定，不含修补位置、
clean源码、隐藏输入。纯数值参考函数复用已有实现，但算法选择来自公开task，不来自
verification.truth或隐藏注册信息。打包清单同步加入task文件。候选不能改task或harness。

实验目的：验证本轮公开需求/自检修复能否使之前11个失败单元走通，不测E总体优势。
固定只运行旧失败的case0017/D重复0、1、2，E重复0、1、2；case0020/D重复0、1、2，
E重复0、2。每单元仅发起一次任务，任务内最多3候选、共享40调用、请求120秒、任务600秒。
不设置美元上限、不查余额，不无限重采样。完成11项或意外执行异常即停止此批，保留全部结果。
使用独立 `/home/you/gpu-agent-public-repair-20261001-0nU1Wp/source` 快照和独立env，
不提交原工作区、不改旧结果；启动后追加快照版本、运行命令和实际usage。

### 旧失败记录的离线复核

使用原实验公开store，经tools.dev_failure_inventory._Run核对manifest引用的文件大小与hash，
读取11项self-check.json、正常运行result.json/stdout、public-input.json。
首次临时读取脚本误把run/result.json当ExecutionResult读取runtime_status而报KeyError；
该文件实际为底层ToolResult。改为从self-check读取运行状态，并在ToolResult核对exit_code、
tool_error、timeout/cancel/truncated。没有修改旧文件、执行GPU或调用API。

4项退出1且无基础设施错误，均应为可修订的运行失败；另外7项的真实旧stdout经过新公开
check_public_output比较，全部返回NUMERIC_MISMATCH。此检查证明公开输入足够识别这7项，
不是以隐藏测试作为反馈。11个一基序号为11、12、24、27、34、35、40、42、44、47、48。

### 真实复测启动

独立快照commit：19b53f9d6d312e7d92d33d4e7919abc0de04adaa。
runtime hash：c56bb14a16401eff22ad95e1fc747451bbdbe9df2278de98221fc618ba70af4b。
prompt版本m3-2026-10-01-v12；repair版本public-repair-v2。
env使用原Conda依赖但独立editable安装；检查实际gpu_agent导入该source，Git工作区干净。
加载原provider.env，不打印密钥；unset旧family，timeout=120，knowledge版本
cuda=12.8.1;compute-sanitizer=2025.1.0.0，原20260929-v6索引复制到新结果目录并只读。

实际命令（在快照source目录执行，env -u PYTHONPATH；输出通过tee写experiment.log）：

```
/home/you/gpu-agent-public-repair-20261001-0nU1Wp/env/bin/python /home/you/gpu-agent-public-repair-20261001-0nU1Wp/source/scripts/compare_repair_modes.py --repository /home/you/gpu-agent-public-repair-20261001-0nU1Wp/source --knowledge-index /home/you/.codex/worktrees/a352/agentic-gpu-debugger/.cache/gpu-agent/knowledge/20260929-v6/index.json --output /home/you/gpu-agent-public-repair-20261001-0nU1Wp/results --unit case_0017:D:0 --unit case_0017:D:1 --unit case_0017:D:2 --unit case_0017:E:0 --unit case_0017:E:1 --unit case_0017:E:2 --unit case_0020:D:0 --unit case_0020:D:1 --unit case_0020:D:2 --unit case_0020:E:0 --unit case_0020:E:2
```

工具session=35320，开始时仅确认START 1/11，不将启动说成通过。所有selection、attempt、
public原始记录、最终投影及usage在上述results目录，旧48项目录保持不变。

启动后补跑 `tests/unit/test_evidence.py -q` 得到2 passed（0.12秒），
`tests/unit/test_distribution_resources.py -q` 得到7 passed（1.58秒），均使用上方同一
解释器、清除PYTHONPATH及禁用pytest插件的命令前缀。检查新增字段的持久化边界及已有
分发资源检查未回归。当前工作区与实验快照runtime hash一致，后续仅补充说明文档，
不改变实验正在使用的代码。一次文档apply_patch使用了不存在的定位行被拒绝，未写入；
随后以真实上下文重做文档补充，diff检查通过。

首个case_0017/D/repeat0已VERIFIED_FIXED，2次调用。它首个候选通过，不能当作真实第二轮
修订证据；后续仍按预先固定清单执行，不人为制造模型失败来凑多轮记录。

运行期间只在主工作区补边界/分发测试，不改快照：用同一pytest前缀执行
`tests/unit/test_public_repair_correctness.py tests/unit/test_distribution_resources.py -k 'rejects_symlink or invalid_public_input or built_distributions' -q`
得到3 passed、17 deselected（1.68秒）。验证task符号链接拒绝、不接受模型自定义checker路径、
无效公开输入不得通过、坏输出格式明确失败，以及构建wheel实际包含全部21份task.json。
测试新增字符串首次Ruff报一处行超长，拆行后Ruff与diff检查通过。没有新增生产代码变化。

## 第三轮：2026-10-02核对最终结果

用户询问状态，本轮只读核对experiment.log、results.jsonl、completed.json和两个相关
unit/result.json，未重启实验、未调用API或GPU。completed.json确认11单元完成，快照仍为
19b53f9d6d312e7d92d33d4e7919abc0de04adaa；pgrep未发现compare_repair_modes.py进程，
没有interrupted.json。进程查询返回1表示无匹配进程，不是实验失败。

结果10项VERIFIED_FIXED，1项未生成候选。case_0017的D/E共6项全部通过；case_0020的
D repeat0/2、E repeat0/2通过，D repeat1未进入补丁阶段。
未通过项run=f5a97e3cabda4884a9e1cbc1ea070236，诊断调用记录LLM_CONNECTION_ERROR，
state=UNCERTAIN，elapsed_ms=1269.8527111206204，usage=null。只能确定本次调用连接失败，
不能确定服务器是否收到请求或计费，也不能据此断定是模型推理错误或项目算法缺陷。
没有将它记作GPU验证失败，也没有自动重发不确定请求。

本轮取得真实第二轮修订证据：case_0020/D/repeat0，run=83480ee43d204fbd9ff759f064fb1503。
第一候选build CLEAN、runtime FAILED；新控制流保留FAILED而非覆盖为UNAVAILABLE。
模型收到公开失败反馈后生成第二候选，正常功能检查和四种Sanitizer的功能检查均PASSED，
四工具均CLEAN，最后独立验证VERIFIED_FIXED。该任务3次模型调用；隐藏结果没有回流。
这是修订回路真实工作的证据，不是E优于D的证据（此单元属于D）。

本批合计37次物理调用，已知113065 tokens；1次调用usage未知，美元费用未核算，不能当0。
其余9个通过单元为首候选通过。没有重跑原先成功的37项，没有拼接旧结果。

当前结论：两处目标流程修复已有离线和真实GPU/API证据，10/11定向任务通过；仍保留
1项连接失败，不能宣称11/11或项目所有测试通过。全量新D/E比较未做，自主E优势尚未证明。
原工作区修改未在本轮提交/推送。下一步如继续处理该失败，应先检查公开provider传输记录，
区分请求不确定状态；只处理这一项，不重跑已通过的10项。

## 第四轮：剩余连接失败的原始记录核查

用户要求“检查”，本轮只读，不修改生产代码、不发请求、不重跑实验。通过
tools.dev_failure_inventory._Run逐项验证unit-07公开manifest引用的provider记录，
检查STARTED.json与UNCERTAIN.json；同时定位provider.py的异常分类和SDK重试配置，
对照相邻unit-06、unit-08的公开调用摘要。使用原Conda Python执行只读脚本，以及
rg、sed查代码，没有读取私有验证内容或输出密钥。

已确认：唯一调用属于diagnose，开始于2026-09-30T19:03:11.794306Z，耗时约1.27秒。
无HTTP状态、无usage，没有获得可用于诊断/补丁的模型结果。代码明确将SDK抛出的
openai.APIConnectionError映射为LLM_CONNECTION_ERROR/UNCERTAIN；超时有独立的
LLM_TIMEOUT分支，SDK max_retries=0。因此本次不是记录中的格式拒绝、GPU失败或
120秒请求超时，也没有在SDK内部重复调用。

相同案例及模式的前一单元3次调用全部COMPLETED，后一单元2次调用全部COMPLETED。
这与一次局部连接异常相容，不支持持续的配置/认证失效；但不足以定位网络哪一端故障。
现有遥测没有保留底层异常类型/errno/安全的cause分类，APIConnectionError分支也没有
捕获并分类异常原因，所以无法事后区分DNS、TLS、代理、连接重置或服务端断开。
不能把连接错误笼统说成“DeepSeek模型能力问题”，也不能声称已证明DeepSeek服务器故障。

检查结论：10项真实修复仍有效，余1项是连接层未取得响应；详细网络根因受历史证据
不足限制。今后若补遥测，应仅记录允许的类型/原因码，不保存可能含凭据的完整异常文本。
本轮没有实施遥测变更或新重试；原不确定调用费用仍未知，不能清零或覆盖旧失败。
