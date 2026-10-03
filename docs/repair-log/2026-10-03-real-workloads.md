# 2026-10-03：二维邻域与前缀和案例接入完整修复

## 目标与边界

用户要求扩展真实案例并接入完整修复流程，同时询问自主调查是否必须胜过非自主流程。
D是固定规则选工具、模型诊断与补丁；E由模型在受控动作空间内选择调查步骤。证明E能
真正选择并使用证据是能力验证；证明E超过D是比较结论，不是项目必须满足的产品定义。
本轮不启动新D/E优势对照，不筛选有利E的案例、不修改旧成绩。

工作区保留既有未提交修改。按failure-aware-execution先定义可验证范围：新增case0021
二维五点stencil和case0022分段inclusive scan；两个故障都在真实输出计算路径，不加
旁路探针。前者覆盖二维索引边界，后者覆盖共享内存多轮跨线程依赖。
沿用现有两数组扁平传输ABI，不声称任意shape/多文件CUDA工程支持；公开规格明确逻辑
二维布局及分段规则。扩展可信算法选择、CPU参考、公开task、验证参考源、打包和测试。

## 第一轮：设计与验收

二维网格固定逻辑行宽32，末行允许不足32个元素；越界邻居为零，输出为中心+四邻域+b。
使用真实二维threadIdx/blockIdx，不用额外kernel制造故障。mutant去掉上邻居的边界保护。
scan按连续128元素分段，在每段执行inclusive prefix sum(a+b)，尾段只输出有效元素。
clean用共享内存Hillis–Steele步骤和块屏障；mutant去掉迭代读取前的屏障，racecheck须
在5次独立instrumented执行中均报告登记类别，才能接受故障定义。

验收顺序：手算数值/边界离线测试→可信源码/hash/验证回放接线→只对新增两例跑GPU。
每例clean需通过8个边界输入的功能检查及代表性输入四工具；mutant检出目标故障；删除
核心launch的反例必须不能通过功能验收。之后冻结快照，两个新增case各执行一次E repair
（含公开自检/最多3候选修订/最终独立验证），保留模型失败，不为成功重复抽样。
没有金额上限或余额查询；实际调用和usage照常记录，不读取旧私有holdout。
当前尚未实现或运行；后续逐轮追加实际结果。

## 第二轮：实现和第一次原生验收

已新增case0021/22的clean、mutant、input/task以及diverse registry/provenance；扩充
oracle枚举、独立CPU参考、公开需求、原生验收与最终验证参考源码选择、wheel资源清单。
repair路径无需给模型额外权限，使用同一公开规格/输入检查和隔离执行。默认48项日程
不变，脚本显式--unit可选择已注册的新公开case，并保存独立用途与日程。

离线命令统一前缀 `env -u PYTHONPATH PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/you/conda_env/agentic-gpu-debugger/bin/python -m pytest`：

- `tests/unit/test_diverse_corpus.py tests/unit/test_public_repair_correctness.py tests/unit/test_verification_engine.py -k 'not new_algorithm or case_0021 or case_0022' -q`：102 passed、5 deselected，40.79秒。
- `tests/unit/test_diverse_corpus.py -k '2d_stencil or scan_resets' -q`：2 passed、33 deselected，0.19秒，手算二维边界和跨128段重置。
- `tests/unit/test_repair_comparison_schedule.py tests/unit/test_distribution_resources.py -q`：8 passed，1.71秒。
- Ruff检查/格式化、mypy --strict src/gpu_agent（76文件）、git diff --check均通过。

GPU实际命令（退出1，原始失败保留）：

```
env -u PYTHONPATH /home/you/conda_env/agentic-gpu-debugger/bin/python -I -m gpu_agent benchmark validate-diversity --repository /home/you/.codex/worktrees/a352/agentic-gpu-debugger --output /home/you/gpu-agent-workloads-20261003-1CqAGC/native --case case_0021 --case case_0022
```

报告native/report.json、终端native.log。case0021三个角色均通过；case0022 clean和
ablation通过，mutant报TARGET_FINDING_MISSING。不是没有检出竞争：原始racecheck.log
明确7个ERROR，Read在kernel.cu:11，Write在kernel.cu:9/13，正是tile初始化及每步更新。
原来从既有stencil案例沿用的Write→Read类别与新case实际Read→Write顺序不符。
没有模型参与、没有私有数据、没有放宽数值oracle或Sanitizer解析。

修正的是尚在开发验收的新案例故障描述，不是把错误计算改成正确答案，也不把任意
FINDING算通过。按原始访问位置明确登记Read access and Write access，同时更新mutation
provenance哈希；源码、输入、CPU答案和工具均未变。首次失败报告保持原hash/标签。
下一轮只跑case0022的mutant，要求5次都命中这个精确类别；不重跑已通过5个角色。

## 第三轮：故障定义验收通过，准备真实Agent闭环

实际复测命令为上一命令将output换成native-scan-mutant，并指定
`--case case_0022 --role mutant`。退出0，5次racecheck全部FINDING且精确命中Read→Write。
报告 `/home/you/gpu-agent-workloads-20261003-1CqAGC/native-scan-mutant/report.json`，日志同级
native-scan-mutant.log。本报告明确complete_acceptance_scope=false，不能把单角色报告
说成重新跑了完整验收；与首次已通过角色按相同源码/input/harness/hash核对使用。
首次失败不删除，未改kernel或数值检查标准，没有API调用。

下一步独立source/env快照固定全部实现与元数据；只运行case0021/E和case0022/E各一次，
默认3候选、共享40调用、请求120秒、任务600秒。模型拿公开需求、故障源码、实际工具证据，
不拿clean、mutation元数据、隐藏输入。正常模型失败保留，不无限采样。旧实验仍不变。

## 第四轮：真实E闭环已启动

独立目录source/env位于 `/home/you/gpu-agent-workloads-20261003-1CqAGC`；原Conda依赖未
卸载或替换。实际导入独立source，Git快照干净，commit=0d7db80df1bad79be0c4812ec673ec99c43e51a4，
runtime hash=60cc2c9ab74255edc3e8c9f7677907f46f800d9394fe6969c82e3bf003b96587。
prompt仍m3-2026-10-01-v12，未为新案例增加特殊修补提示；知识库hash仍
15c53142f8949ba1d852a3227fd11460060adec359fda9ee0cf42c52011b8600。

从既有provider.env加载配置但不显示密钥，unset旧family，timeout=120，knowledge版本
cuda=12.8.1;compute-sanitizer=2025.1.0.0。实际命令（在独立source中，输出tee到agent.log）：

```
env -u PYTHONPATH /home/you/gpu-agent-workloads-20261003-1CqAGC/env/bin/python /home/you/gpu-agent-workloads-20261003-1CqAGC/source/scripts/compare_repair_modes.py --repository /home/you/gpu-agent-workloads-20261003-1CqAGC/source --knowledge-index /home/you/.codex/worktrees/a352/agentic-gpu-debugger/.cache/gpu-agent/knowledge/20260929-v6/index.json --output /home/you/gpu-agent-workloads-20261003-1CqAGC/agent --unit case_0021:E:0 --unit case_0022:E:0
```

session=19254，selection.json已冻结两项日程与源码/input/task哈希。只确认已启动，尚未
在此宣称模型修复成功。元数据纠正后复核test_mutation_metadata通过（1 passed、34
deselected，0.17秒）；后续Ruff及diff检查通过。所有未完成检查/真实调用结果继续追加。

## 第五轮：真实结果与失败归因

两任务均已结束，脚本退出0仅表示日程完整，不能解读成2/2修复通过。结果在agent/results.jsonl
及agent/unit-00、unit-01/result.json。case0021 run=44ddb114785b4afc93a5da5b8661f9ad，
memcheck→检索→结束诊断，首候选正常/四instrumented功能检查均通过，四工具CLEAN，
最终独立VERIFIED_FIXED。5次调用、17397 tokens。

case0022 run=eaee4fd7291043f79379fc7941807425，memcheck→racecheck→检索→结束诊断。
8次调用、26629 tokens。第一候选把迭代中读取后、写入前的屏障移到写入后；公开数值检查
NUMERIC_MISMATCH。第二候选另在初始化后加屏障，但仍没有保证每轮所有线程完成读取后才
覆盖tile；普通运行恰好算对，racecheck仍FINDING，synccheck下的程序输出再次数值错误。
第三候选与第二候选hash一致，REPEATED_CANDIDATE停止，没有重复执行相同候选或调用最终验证。
2个真正自检round和3份候选全部保存；这是模型未正确消除循环内读写竞争，不是连接/格式
错误，也不是控制器拒绝合法补丁。没有把它视为已修好，更没有向模型提供clean参考答案。

核查使用tools.dev_failure_inventory._Run读取公开candidate.json/self-check引用，验证hash
及大小。初次打印summary过长，后续记录只保留关键检查和diff结论，不将日志截断误报为
执行截断。实际原始provider usage均存在，共13次调用、44026 tokens、未知usage为0，
美元未计算。两个任务没有改prompt，不因模型失败再采样、不提高预算或放宽racecheck。

验收分层结论：两个新工作负载的定义、CPU答案、正常/故障/删除计算反例GPU验收通过；
两者都已进入真实E调查、候选生成、公开GPU自检路径，scan还实际触发修订与重复候选停止。
最终验证新oracle的离线engine/replay接线测试已通过；真实最终验证只在case0021执行并通过，
case0022因公开失败按设计不执行。不能称新增两例全部被模型修复，也不能将人工clean当Agent产物。

本轮目标是扩展案例和接通流程，不是保证模型修复率100%。对已出现失败已完成归因，
不存在支持继续改同一实现或重采样的证据；因此保留这个真实失败作为能力边界。
如果未来要提高这类推理成功率，需另行设计通用反馈/修复策略并预定评测，不能向case0022
专门提示“把屏障插在这里”。新案例并非采集自第三方生产事故，仍是受控算法工作负载。
