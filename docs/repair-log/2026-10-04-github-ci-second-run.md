# 2026-10-04：GitHub第二次CI失败状态核查

## 用户问题与实际检查

用户要求检查推送修复后是否再次失败。本轮只读核查远端，没有修改生产代码、
推送新提交或重新运行测试。优先查询可用GitHub连接器未发现对应工具；此前API
读取返回403，按browser技能使用现有浏览器访问公开Actions页面。

## 已确认事实

最新任务为https://github.com/YOUessi/agentic-gpu-debugger/actions/runs/37152381894 ，
CI #2，PR #1 synchronize触发，分支design/v2-operator-workflow，状态Failure，
总时长56m35s。Python3.11任务111288735801与Python3.12任务111288735673均报
Process completed with exit code 1，注释指向step7（离线测试步骤）。
这不同于首次任务37050813046的收集阶段exit2，但仅凭退出码无法确定具体新根因，
也不能据此保证旧问题全部消失。页面还存在Node版本弃用及Ubuntu迁移警告；
没有证据将这些警告认定为pytest失败原因。

## 当前缺失证据与下一步

任务摘要和job页面均显示Sign in to view logs。当前浏览器未登录，浏览器发现仅
返回这一实例，没有其他可用已登录实例。没有读取cookie、查找密钥或绕过登录。
所以尚未取得FAILED/ERROR节点名、pytest最终统计和traceback；不凭退出码猜测，
不把前一次本地定向通过当成本次云端通过。应请用户在内置浏览器登录后通知，
或提供两个任务测试步骤末尾的失败摘要及堆栈，然后才能定向定位和决定修复。

本轮检查已确认远端失败；原因诊断受日志访问限制，未执行GPU/API或完整离线回归。
保留上一轮1400通过/103错误及随后113、29、22项通过的历史记录，不改写为新成绩。

## 用户提供日志后的第二轮定位

收到两份完整粘贴日志：附件faa570fe-9bc8-49e2-af5f-ec04757896ea及
8d4f56eb-68a6-4616-ae23-b25d0693fe4d。Python3.11.16为2 failed、1503 passed、
53 deselected（2641.31秒）；Python3.12.14为3 failed、1502 passed、53 deselected
（3353.17秒）。先前重名及轮换错误未出现在这两份最终失败列表。

两版均失败的是CLI帮助字符串断言与依赖闭包检查；3.12额外失败于build --no-isolation。
CI使用无约束pip install -e '.[dev]'，未使用requirements.lock；日志明确显示已安装
依赖要求httpcore2==2.13.1，而锁文件2.13.0，因此属于安装环境和锁文件不一致。
CLI输出有ANSI但当前证据尚不能证明只有ANSI原因；构建stderr被capture_output隐藏，
只能确认子进程失败，不能直接断言缺哪个包。

原本地环境typer0.27.2、rich15.0.0、httpx2/httpcore2均2.13.0、setuptools79.0.1，
build1.6.1、openai3.13.0。FORCE_COLOR=1下单独CLI测试仍通过（1 passed 0.28s），
这条不复现的结果保留。下一步建独立环境模拟CI安装，限于失败3项，记录实际输出；
不修改原环境、不调用GPU/API、不盲目重跑完整套件。拟修复需先验证：让CI确实使用
锁文件；显式安装非隔离构建所需后端；CLI测试按参数语义及可见帮助文字验证，不依赖
ANSI字节和窄终端折行。保持原参数必须存在的断言，不删测试或降低验收。

## 实际复现和修复

从原解释器用venv创建独立/tmp/gpu-agent-ci-env-20261004-EJvyvQ，不继承site-packages。
按旧CI执行pip install -e '.[dev]'，实际装入openai3.24.0、httpx2/httpcore2
2.13.1（而原环境是3.13.0及2.13.0）。对3个失败目标定向运行，结果1 failed、
2 passed，2.35秒（local-experiments/ci-install-20261004/before.log）；锁冲突完全复现，
CLI和打包此时通过。CLI未做到原样复现，不能把推断写成已证实的唯一根因。

在这个临时环境单独卸载setuptools后，build --no-isolation立即报
BackendUnavailable: Cannot import 'setuptools.build_meta'。证明配置确实没有保证
非隔离后端存在；但用户的3.12日志未展示stderr，所以是否同因仍需云端结果验证。
卸载只作用于临时环境，不影响原Conda或项目环境。没有因此添加无关依赖。

实际改动：

1. CI先安装requirements.lock，再以--no-deps --no-build-isolation安装editable项目，
   并执行pip check；pip缓存键显式使用锁文件。不升级模型SDK或重新选择价格。
2. 锁文件补齐本地已验证的build1.6.1、pyproject_hooks1.3.3、setuptools79.0.1；
   dev依赖显式包含setuptools>=77。原模型运行依赖版本不变。
3. 闭包检查扩展到dev依赖并要求实际安装版本等于锁定版本，避免拿未锁定的本机元数据
   假装验证锁文件。保留原来的版本约束和传递依赖检查。
4. CLI测试覆盖color=False/True，终端列宽160，先检查实际注册选项集合，再对
   Text.from_ansi后的帮助文字检查原四个必需参数，帮助命令仍必须退出0。
5. 打包测试继续要求退出0；失败时断言包含完整stdout/stderr，不再只显示CalledProcessError。
6. CI增加依赖/CLI/打包快速预检步骤，放在完整测试之前，尽早暴露这类环境问题。

## 修复后的检查与限制

在临时环境按新CI命令安装锁与editable成功；第一次pip check因宿主PYTHONPATH
带入ROS的launch-ros而报告缺pyyaml，未通过安装pyyaml掩盖，改为env -u PYTHONPATH
运行后No broken requirements found。CI不使用此宿主ROS路径。定向命令：

```bash
env -u PYTHONPATH CI=true GITHUB_ACTIONS=true FORCE_COLOR=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /tmp/gpu-agent-ci-env-20261004-EJvyvQ/bin/python -m pytest tests/unit/test_distribution_resources.py tests/unit/test_retrieval.py::test_lock_covers_declared_dependencies_and_runtime_closure -q --tb=short
```

结果9 passed in 2.53s（after.log），包括实际构建sdist/wheel、包内资源、两种颜色设置
的帮助和完整依赖闭包。Ruff通过、171文件格式通过。未重跑已通过的1500余项，
没有GPU/API调用。Python3.12本机无可用解释器，所以没有声称完成3.12复现或全量通过。
本轮仅本地实现，未推送；旧CI结果不改写，新结果需要同步后由GitHub重新确认。
最终mypy --strict src/gpu_agent检查76文件通过，git diff --check通过。
将本轮变更及此前状态核查一起提交到codex/latest-complete-20261003，保持最新本地
工作区可追溯；没有改变远端PR分支，没有删除旧失败记录或修改实验结果。
