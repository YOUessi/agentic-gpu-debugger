# 2026-10-03：CI同名测试模块收集失败

## 现象、复现与原因

用户提供Python矩阵CI日志：离线pytest在执行前退出2，gpu和unit目录各有
test_diverse_corpus.py，被pytest默认导入方式识别成同一顶层模块。-m标记过滤在
模块收集之后，不能阻止该冲突。不是CUDA、DeepSeek或缓存损坏。
在最新工作树codex/latest-complete-20261003、起点e9ac346，用如下命令复现：

```bash
env -u PYTHONPATH PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/you/conda_env/agentic-gpu-debugger/bin/python -m pytest --collect-only -q tests/gpu/test_diverse_corpus.py tests/unit/test_diverse_corpus.py
```

结果6个GPU节点已收集、1个import file mismatch、退出2，与用户CI相同。
此前定向回归只选择unit文件，没有同时收集GPU同名文件，所以漏检。152项通过
不能代表完整CI通过。遵照failure-aware-execution先复现再修改，不重跑付费实验。

## 修复与验收计划

把GPU文件改名为test_diverse_corpus_gpu.py，保留所有测试、断言和标记，
不改全项目导入机制、不跳过出错文件、不清缓存掩盖冲突。历史记录保留原命令。
随后执行CI相同的完整离线选择表达式，另做收集检查以及命名冲突回归保护。
使用专用环境绑定新工作树，避免子进程误用旧editable checkout。
测试结果及后续失败若有，追加于本文，不把收集通过当成全量执行通过。

## 已完成的检查及实际命令

用原Conda解释器创建本工作树`.venv`（--system-site-packages），在其中执行
`python -m pip install --no-deps --no-build-isolation -e .`成功。原Conda editable
安装未被卸载或重定向；`.venv/bin/python -I -c 'import gpu_agent; print(gpu_agent.__file__)'`
确认指向最新工作树。以下命令均在最新工作树执行：

```bash
env -u PYTHONPATH PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest --collect-only -q tests/gpu/test_diverse_corpus_gpu.py tests/unit/test_diverse_corpus.py tests/unit/test_test_module_names.py
.venv/bin/python -m ruff check src tests
.venv/bin/python -m ruff format --check src tests
.venv/bin/python -m mypy --strict src/gpu_agent
.venv/bin/python -m build --no-isolation
```

收集42个测试成功，包括原6个GPU参数化节点，没有删掉任何GPU测试；Ruff通过，
170个文件格式通过，mypy全部76源码文件通过。sdist与wheel构建通过。另建
`local-experiments/ci-collection-20261003/wheel-env`安装刚生成的wheel（--no-deps
--force-reinstall），切换/tmp使用隔离模式确认包从该wheel的site-packages导入；
toolchain lock可加载，corpus registry、modes、retrieval-eval、sources资源全部存在。
本机没有python3.12命令，不声称已运行GitHub的两种Python矩阵。

完整离线执行命令如下，使用pipefail保留pytest真实退出码：

```bash
set -o pipefail
env -u PYTHONPATH PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q -m 'not gpu and not container and not live_llm and not release' 2>&1 | tee local-experiments/ci-collection-20261003/offline.log
```

此命令清除宿主ROS路径、关闭外部插件自动加载，测试选择表达式与CI相同；
并非模拟GitHub整台主机。构建原始输出保存于同目录build.log。两个日志属于新一轮
证据，不修改之前归档清单，也不声称它们已包含在旧清单中。

## GitHub状态与同步边界

用户明确报错来自GitHub Actions。本地修复并不自动改变GitHub已失败任务的状态。
尝试只读请求https://api.github.com/repos/YOUessi/agentic-gpu-debugger/actions/runs?per_page=5
以确认失败任务head分支，但curl返回HTTP403；管道下游因此无法解析JSON。
没有据此猜测远端分支、没有请求或输出密钥。已向用户询问失败任务链接。
原CI工作流在pull_request及main push时触发，不能假设推到任意新分支就会执行。
本轮修复保存在最新本地分支，不擅自把本地归档提交合并到main或其他正在开发的分支。

## 第二轮：完整离线运行暴露监听代次轮换误报

完整运行结果为1400 passed、53 deselected、103 errors，1830.94秒，退出1。
所有103个错误来自test_holdout_scoring.py共享scoring_base初始化，堆栈一致：
service.diagnose → backend.run → EvidenceRepository.view → RunStore._load_at，
抛出ValueError("manifest changed while reading")。不是103种独立故障。
原始全量日志offline.log保留，不覆盖。单独执行第一个错误用例：

```bash
env -u PYTHONPATH PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest tests/unit/test_holdout_scoring.py::test_exact_package_zero_mutation_plan -q -x --tb=short
```

200.30秒后同样1 error（holdout-first-error.log）。本轮读取的是合成测试夹具，
没有读取真实私有评测内容，也没有要求用户补充人工评分。

直接原因：FileChanges.version在_versions超过4096时重启监听器并改变epoch。
第一次version可能把监听数从4096增到4097，第二次version就重启；manifest字节
与stat都没变，但_load_at把epoch变化等同于文件被改写。小型回归测试用真实
inotify和预置4096项准确复现：修复前1 failed、1 passed，4.57秒，退出1。
这确认是生产存储读取的边界缺陷，不是模型问题，也不是同名测试改名导致的。

修复方案：stat身份/时间/大小真实变化仍立即拒绝；只有监听代次变化时丢弃这次
读取和缓存，最多重新读取到3次，必须得到新的一对稳定代次才返回。持续不稳定仍
报原错误，不缓存跨代次读取的字节，不关闭监听器、不信任旧缓存、不放宽artifact hash。
实际仅修改store.py的_load_at，另加test_store_watcher_epoch.py覆盖真实轮换、
连续不稳定的次数边界，以及真实文件变化仍被拒绝。

```bash
env -u PYTHONPATH PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest tests/unit/test_store_watcher_epoch.py tests/unit/test_store.py tests/unit/test_test_module_names.py -q
env -u PYTHONPATH PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest tests/unit/test_holdout_scoring.py -q --tb=short
env -u PYTHONPATH PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest tests/unit/test_batch_complexity.py tests/unit/test_dev_failure_inventory.py -q --tb=short
```

第一组29 passed in 0.33s，store-after.log；后二组为受影响回归，分别保存
holdout-after.log、cache-after.log。不为此再次重跑已通过的完整1503项或GPU/API。
这意味着最终结果必须分开报告“初次全量结果”和“修复后受影响结果”，不能说同一次
全量运行全绿。生产读取变更发生在先前构建后，先前wheel检查只证明改名前后包装路径，
不是本次store改动的新wheel验收。

## 后续只读远端核对与已完成受影响检查

API返回403后，按browser技能使用浏览器只读查看Actions列表和详情，确认失败任务是
https://github.com/YOUessi/agentic-gpu-debugger/actions/runs/37050813046 ，PR #1，
head分支design/v2-operator-workflow，Python3.11和3.12均在测试步骤退出2。
任务详情可公开读到分支与状态，但原始日志需要登录；没有登录、提交或重运行远端任务。
已询问用户是否在验收后同步修复到该PR分支，只同步修复，不上传本地原始实验归档。

缓存复杂度和失败清点相关22项于10.68秒通过（cache-after.log）；修改后的Ruff通过、
171文件格式通过、mypy76文件通过。另在store修改后再次执行build --no-isolation，
sdist/wheel均成功，原始输出build-after-store.log；未将此说成新GPU/API验收。

## 最终结果与交接

受影响评分模块最终113 passed in 733.19s（12分13秒），退出0，原始日志
holdout-after.log。原103个错误所在模块全部完成，未跳过、未改期望、未删断言。
另两组29和22项全部通过，共164项受影响检查（不同组分开执行）。首次全量依然如实
记录为1400通过、103错误、53排除；没有声称修改后的1506项在一次全量运行中全部通过，
也没有运行本机不可用的Python3.12。先前53个按CI标记排除项不是离线通过项。

重新生成wheel中的gpu_agent/store.py逐字节对照工作树通过，SHA256为
`cf43a03490cfc74609c74f1a745ec55a91e2d19a0e9661a25473c7a2a9dfec23`。
git diff --check通过。所有运行已结束，无后台付费或GPU任务；另一个工作树的测试
进程不属于本任务，未中断、未纳入本轮结果。

把测试改名、重名回归、监听轮换修复与三条针对性回归以及本文提交到
codex/latest-complete-20261003。本轮未推送GitHub，远端CI仍是旧提交结果。
已经只读定位到PR #1，已询问是否把修复同步推送至design/v2-operator-workflow；
未收到确认前，不擅自推送其他分支、改main或把本地归档提交混入PR。
因此本地修复和受影响验收已完成，远端重新运行与Python3.12结果仍待同步后确认。

## 2026-10-04：用户授权推送后的同步

用户明确回复“推送”。目标仍为PR #1的design/v2-operator-workflow，不合并main。
原PR工作树存在用户未提交文档，故从最新远端2fa987b创建独立临时工作树，
只移植61520e4修复，不带e9ac346本地归档提交。cherry-pick --no-commit出现两处
文档上下文冲突：current-status及repair-log索引引用本地归档阶段新增内容。
保留PR原有路径，仅加入CI修复说明和存在的记录链接；源码及测试未发生冲突，
逐文件与61520e4比较以确认一致。原工作区、原始实验副本和本地新分支均不改写。
再次做完整离线收集和存储/命名针对性检查，不重复已有昂贵实验。
推送不等于GitHub CI已通过，云端任务结果仍需单独确认。

同步检查结果：四份实际修复源码/测试与61520e4逐文件无差异；PR分支不包含本地归档
工具的test_local_archive.py，这是未移植归档提交的预期差别。29项存储与命名测试
0.32秒全部通过；完整离线选择收集1505项、按标记排除53项，2.21秒退出0。
git diff --cached --check通过。上述收集不声称1505项已在本临时工作树执行通过。
