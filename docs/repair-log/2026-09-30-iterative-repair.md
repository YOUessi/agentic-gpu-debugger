# 2026-09-30：公开GPU自检与多轮修复

## 背景与验收

用户已明确要求补上调查→补丁→公开自检→分析失败→修订→独立验证。本轮在
HEAD 87fa492及已有未提交修改之上增量实现，保留旧单候选评测和所有历史结果。
使用failure-aware-execution技能管理多阶段验证：先失败/受影响的离线测试，再
定向真实GPU/API验证；不设置美元上限、不查询余额、不重跑无关成功实验。

## 第一轮：边界核查与设计

service._diagnose当前只生成一个候选；verify_exact调用独立验证器，含未公开
测试，不能把其结果包直接回填模型。原设计限制来自V2规格，不是JD。

实现选择：新增显式repair工作流，默认最多3个候选（可配置），共享原任务的
总调用与时间边界。旧diagnose/evaluate保持单次语义，避免新结果冒充旧消融。
每轮相对原始源码生成完整diff，保存候选、公开自检子run、结果与反馈；最终仍
只把选定候选注册到原diagnosis，确保独立验证比较基准不变。

自检模块只接收public store、公开源码和原公开输入、隔离backend，不接收
evaluator或truth能力。检查编译、运行与四Sanitizer；其通过仅代表公开检查
通过，不代替功能Oracle或隐藏输入验证。最终独立验证只做一次，不反馈模型。
重复候选、基础设施故障、模型失败或迭代耗尽停止并保留记录，不随机重启。

验收：能从一次自检失败进入下一次有证据的修订；干净候选不继续抽样；重复
候选/预算/工具故障正确停止；反馈不含隐藏数据；原单次路径不变；CLI可运行。
本节为执行前计划，尚无通过结论。实际修改、命令、失败和结果随后追加。

## 第二轮：实现与离线验收

新增repair.py，只持有public store和隔离执行工厂，不导入独立验证器或truth。
service.repair显式执行新工作流，repair_candidates保存每轮候选和反馈，
原始source snapshot不变，最终仍注册一个最终候选。provider.revise_patch复用
原patch schema/范围检查/调用账本，feedback独立字段不伪装成原始诊断。
新增CLI repair及规格/模式契约增补；PROMPT_VERSION为v11，新增公开反馈说明，
v10/v11动作回放均使用full-source-v2。旧diagnose/evaluate不启用自检循环。

首次mypy发现Literal默认值类型和局部变量复用类型冲突，已修正；Ruff发现
过长字符串及测试import分组，已修正。新增7项测试首次全过，随后扩充真实
provider端口模拟、共享调用上限、编译失败/超时/全clean、自检及隐藏反馈边界。
12项通过（2.11秒），注意这些是离线模拟，不是GPU实验成绩。

执行目录为当前工作区，解释器/home/you/conda_env/agentic-gpu-debugger/bin/python。
实际相关回归命令：

```bash
env -u PYTHONPATH PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/you/conda_env/agentic-gpu-debugger/bin/python -m pytest tests/unit/test_iterative_repair.py tests/unit/test_agent_loop.py tests/unit/test_provider_output_telemetry.py tests/integration/test_provider_contract.py tests/unit/test_verification_engine.py -q
/home/you/conda_env/agentic-gpu-debugger/bin/mypy --strict src/gpu_agent
/home/you/conda_env/agentic-gpu-debugger/bin/ruff check src/gpu_agent/repair.py src/gpu_agent/service.py src/gpu_agent/agent/provider.py src/gpu_agent/agent/prompts.py src/gpu_agent/agent/policy.py src/gpu_agent/cli.py tests/unit/test_iterative_repair.py
env -u PYTHONPATH /home/you/conda_env/agentic-gpu-debugger/bin/python -m gpu_agent repair --help
git diff --check
```

相关回归166 passed in 50.33s（含新增12项，不能相加）；mypy检查74源文件通过，
Ruff、CLI帮助与diff检查通过。没有重跑全量套件或旧GPU实验。

下一步冻结独立实验快照，仅对前次失败的case_0009/E执行一次真实新工作流，
候选上限3，共享40次调用，无美元限制。若第一候选即通过，不为了证明多轮而
人为制造失败或重跑抽样；多轮失败→修订状态机已有离线验证，真实覆盖应如实标注。

## 第三轮：定向真实验证已启动

目录/home/you/gpu-agent-iterative-repair-20260930-k7N8XJ，source为独立本地快照，
env为继承原依赖但独立editable安装的venv；未卸载原Conda安装。复制明确项目
文件，不复制密钥和旧实验。新快照提交1fb39eb6cc63f12fdde2c882f527f13191a432be，
runtime hash为e87fa6b81682d6bf01918b826cbb4b754d4e7ff976b17f6c2365e5d7fb9feb00。
知识库复制并设只读，corpus hash为
15c53142f8949ba1d852a3227fd11460060adec359fda9ee0cf42c52011b8600。
已确认Python实际导入新快照，Git干净，四工具兼容文档检索非空。

在独立source目录，加载既有provider.env但不显示密钥，unset旧family，设置
GPU_AGENT_RUN_ROOT到该目录public，GPU_AGENT_EVALUATOR_ROOT到verification，
GPU_AGENT_KNOWLEDGE_INDEX到只读knowledge-index.json，单请求timeout120秒。
实际命令：

```bash
/home/you/gpu-agent-iterative-repair-20260930-k7N8XJ/env/bin/python -I -m gpu_agent repair /home/you/gpu-agent-iterative-repair-20260930-k7N8XJ/source/benchmarks/public/case_0009/public_input --allow-paid-calls --max-candidates 3 --max-llm-calls 40
```

stdout/stderr通过tee保存到experiment.log。此时仅标注已启动，结论待实际结果。
原工作区不commit/push，不对旧实验做任何写入。

### 第三轮实际结果

CLI退出0，run为2f9ef93d7af244ec8c794ec00e6a6045，自检子run为
65a8746e1c714465bede611b991e37ef。第一个候选就通过公开自检：编译CLEAN，
运行SUCCESS，memcheck/racecheck/initcheck/synccheck均CLEAN。随后独立strict
验证为VERIFIED_FIXED / ALL_REQUIRED_CHECKS_PASSED，功能Oracle与四工具通过。
最终候选hash为937328807059f6bae50dacd80c2128b9e5c34bfd852df51d7c6b3e6031df8d03。
独立验证审计run为62ffe58bb150f0604dcef6b324db0cc7，其内容不反馈给模型。

真实模型deepseek-v4-pro，共6次物理调用：planner4、诊断1、补丁1；全部COMPLETED，
记录16932 total_tokens，unknown usage为0。费用未核算，不记0美元，没有余额查询。
调查动作依次为memcheck→racecheck→检索→结束，没有inspect_source。
诊断及公开自检阶段UTC为03:41:29.965590到03:42:07.973933；这不包含其后独立
验证耗时，不能冒充整个流程耗时。完整终端输出在experiment.log。

这是一次预先指定的旧失败案例的新版本运行，不覆盖旧NOT_FIXED，不重复启动
任务刷成功。由于第一候选就成功，此次没有真正发出revise_patch请求；真实
“失败后第二轮模型修订”的覆盖尚未取得。不能把该成功归因于多轮机制，
也不能称多轮修复率已提高。新回路的失败→修订、停止和隐藏隔离由离线测试
覆盖，真实单轮成功退出与最终独立验证由本次GPU/API覆盖。

执行结束后重新核对snapshot仍干净、runtime hash不变。没有修改运行中源码。
补充CLI退出码/评测边界测试只在原工作区进行，不改实验snapshot：

```bash
env -u PYTHONPATH PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/you/conda_env/agentic-gpu-debugger/bin/python -m pytest tests/unit/test_iterative_repair.py -k 'cli or refuses' -q
```

结果4 passed、12 deselected（0.63秒）；这4项为前述166项之外新增，不能将
12 deselected再计入通过。覆盖VERIFIED_FIXED退出0、NOT_FIXED或未验证退出1，
拒绝绑定评测和A-C消融使用新回路。补充测试初次Ruff报import空行，已修正；
最终Ruff重新通过。没有因此重跑成功GPU实验。

## 最终交接

已实现多轮公开修复的服务、provider反馈、隔离自检、逐轮产物与CLI，新增规格
增补，保留旧单次评测。默认3候选可配置，40调用共享，不重置总账；重复候选
或不可用工具停止。公开自检没有数值Oracle，最终独立验证仍负责功能判断。
测试范围是166项相关回归及随后4项CLI/边界测试，不是全项目全量测试。
真实一次新入口case_0009/E通过；真实第二轮修订尚未触发，广泛成功率未评测。

使用 `gpu-agent repair <公开源码目录> --allow-paid-calls --max-candidates 3`。
目录须包含kernel.cu及其公开input.json（无input时沿用已有开发默认输入）；
provider与知识库使用既有配置。每个diagnosis的repair/目录保存policy、逐轮
candidate/result/feedback和summary，public下子run保存原始执行证据，provider/
保留实际调用。summary的PUBLIC_CHECKS_PASSED不是最终VERIFIED_FIXED。
原工作区尚未提交/推送。下一轮真实若出现公开自检失败，直接审查本流程留下的
反馈和后续候选；不为了演示第二轮重新抽样已成功案例。
