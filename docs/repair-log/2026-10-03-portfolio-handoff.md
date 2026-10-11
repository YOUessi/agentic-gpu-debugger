# 2026-10-03：投递收尾

## 目标与授权范围

用户确认投递收尾：整理最新代码、统一介绍和结果、选择成功演示。本轮整理并提交本地
项目成果，不新增功能、不重跑付费/GPU实验、不提交申请或擅自推送远端。
按failure-aware-execution先核实文件范围和证据，再做交付检查与提交。
工作区design/v2-operator-workflow，起点HEAD87fa492，既有大量实现修改与新增文件。
保留.claude-deps.tgz、.claude-regress.txt、.cursor和用户AGENTS.md，不放入本轮交付提交。

## 第一轮：材料整理

README原来以旧冻结评测/门禁开头，demo仍只有早期单候选记录，不利于展示当前闭环。
本轮改为突出当前能力及一个成功一个失败样例，保留历史材料但明确版本差异；新增中英文
介绍及面试讲解，避免模型100%修复、E优于D、生产事故来源等不成立的表述。

从0d7db80快照的两个公开result及经_Run核验的候选artifact抽取仓库可读摘要：只包含
固定版本、运行ID、调用/token计数、动作、检查结果、公开候选diff及原artifact/result哈希。
未打包完整实验目录、provider配置、模型请求、原始响应或私有验证内容；摘要明确不是
完整原生证据档案，不能凭这个摘要声称可独立复核所有底层执行。
尚未提交；后续追加检查命令、结果、提交及明确排除项。

## 第二轮：交付检查与最终范围

执行目录均为上述主工作区，解释器/工具来自
`/home/you/conda_env/agentic-gpu-debugger/bin`。本次没有新增GPU或模型调用。
为防止系统ROS路径或外部pytest插件干扰，测试显式清除PYTHONPATH并禁用插件自动加载。

实际执行的针对性测试命令为：

```bash
env -u PYTHONPATH PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/you/conda_env/agentic-gpu-debugger/bin/python -m pytest tests/unit/test_iterative_repair.py tests/unit/test_public_repair_correctness.py tests/unit/test_repair_preflight.py tests/unit/test_transport_diagnostics.py tests/unit/test_diverse_corpus.py tests/unit/test_distribution_resources.py tests/unit/test_retrieval_completion.py tests/unit/test_knowledge_metadata.py tests/unit/test_investigation_witness.py tests/unit/test_repair_comparison_schedule.py tests/integration/test_provider_contract.py -q
/home/you/conda_env/agentic-gpu-debugger/bin/ruff check src scripts tools tests/unit/test_iterative_repair.py tests/unit/test_public_repair_correctness.py tests/unit/test_repair_preflight.py tests/unit/test_transport_diagnostics.py tests/unit/test_diverse_corpus.py tests/unit/test_distribution_resources.py
/home/you/conda_env/agentic-gpu-debugger/bin/mypy --strict src/gpu_agent
env -u PYTHONPATH /home/you/conda_env/agentic-gpu-debugger/bin/python -I -m gpu_agent repair --help
```

结果：pytest为152 passed in 6.06s，退出0；Ruff通过；mypy检查76文件通过；CLI帮助退出0，
示例使用的repair参数存在。这是受影响模块离线回归，不是全量测试、真实GPU或付费API复测。
只读核查README、demo、portfolio、current-status及本文共5份文档的本地链接，全部存在；
重新计算两份原始result.json的SHA256并逐项对照摘要run_id，均一致；两例调用数13、
token数44026核对一致。脱敏摘要不替代完整日志，成功与失败都保留。

检查中保留的失败：首次对demo文件在同一个apply_patch内同时Delete/Add被工具拒绝，
没有产生部分写入；改用Update后成功。首次暂存后git diff --cached --check发现历史提交
目录末尾多一个空行，去掉这一个排版空行后再次检查，不改变历史文字或实验记录。
此前git diff --check只覆盖已跟踪改动，不能替代对新增文件暂存后的检查。

提交范围为项目规格、README、pyproject及docs/knowledge/scripts/src/tests/benchmarks/tools
内的既有实现与投递材料，共128文件。未纳入.claude-deps.tgz、.claude-regress.txt、.cursor/
及AGENTS.md，保留它们原状。对暂存文件扫描私钥头和常见长token格式未发现匹配；
这只是特定模式检查，不等于能识别所有未知密钥格式。没有纳入外部实验目录或私有数据。

交付使用包含本文的本地Git提交作为代码检查点，不推送远端、不提交招聘申请。
具体commit可用git log -1获取。尚未完成的是远端推送（本轮不执行），不是新增工程阻塞。
已知模型边界继续保留：case0022未修好；不能宣称E优于D、任意CUDA通用修复或所有实验通过。
这些不通过结果不通过重复抽样、隐藏记录或放宽验证来消除。本轮收尾不扩展实验范围。
