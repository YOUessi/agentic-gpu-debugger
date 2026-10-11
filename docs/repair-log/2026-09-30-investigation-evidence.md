# 2026-09-30：自主调查能力证据与收益对照设计

## 背景、范围、验收

用户同意先证明能力、再评估收益。本轮复用已完成的真实公开D/E记录，增加
只读证据核对及离线分支测试，不重跑成功GPU/API。用failure-aware-execution
技能管理证据边界：动作获准不等于已执行、真实路线变化不自动证明因果收益。
不读取holdout，不调用模型，不修改生产策略，不要求E胜D。

第一层验收：确认E实际走provider.plan而非RuleRouter；所选工具确实执行且
结果进入下一步证据；发现故障后检索并结束；模拟控制变量测试确认改变工具
证据能影响模型接口分支并被执行。最后一项测试只能证明编排机制，不冒充真实
模型因果实验。初始强制memcheck、必需文档等约束必须显式说明。

## 第一轮：已发现的证据与局限

原生case_0003/E记录fa9c05c445994b8795526674db753a29显示：memcheck CLEAN后
选择initcheck，得到FINDING后检索，最后finish。但模型最初已从源码怀疑未
初始化读取，且错误地以为memcheck能确认该问题。因此不能将它描述为完全
不知道问题后被反证启发、也不能将CLEAN解释为证明不存在越界。
接下来核对step引用的原生bundle和实际工具结果，而不是只读模型rationale。

代码审阅定位：agent/orchestrator.py的investigate仅mode D调用rule_router，
E调用provider.plan；通过策略检查后用registry[action_type]执行工具。
本轮不修改该逻辑，只验证并记录。新的核查产物与第二层预先方案随后追加。

## 第二轮：实际证据核对与机制测试

新增tools/investigation_witness.py，输入显式public root和run ID，不扫描私有
目录，拒绝绑定评测/非public/非E。沿每个step的evidence_ref读取原生bundle，
核对模型可见Sanitizer状态、工具result以及stdout/stderr大小与hash，再核对
下一步确实多了模型指定的工具结果，检索的chunk产物存在。成功planner次数
应与动作数相符。工具没有输出源码、模型理由、密钥或隐藏信息。
该工具为已有指定成功终止轨迹的窄核查器，不支持把任意失败轨迹都判为通过；
artifact一致性不是独立硬件证明，也不证明模型真正因果上依赖了某段证据。

首次以python -I执行脚本失败：隔离模式移除了脚本同目录的导入路径，导致
dev_failure_inventory无法导入。没有实验重跑，也未修改生产代码；修正命令为
env -u PYTHONPATH，保留脚本目录，避免ROS环境干扰。随后核查全部通过。

实际命令（cwd为/home/you/.codex/worktrees/a352/agentic-gpu-debugger）：

```bash
env -u PYTHONPATH /home/you/conda_env/agentic-gpu-debugger/bin/python tools/investigation_witness.py /home/you/gpu-agent-investigation-20260930/results/public 1d18fb108cb34c5ea989f69762aeabec b4ca835e6bb54534ab24b178a7510790 fa9c05c445994b8795526674db753a29 cc2232efe86f406598d0232b55872d94 > .gpu-agent/investigation-capability/v10-witnesses.json
env -u PYTHONPATH /home/you/conda_env/agentic-gpu-debugger/bin/python tools/investigation_witness.py /home/you/gpu-agent-iterative-repair-20260930-k7N8XJ/public 2f9ef93d7af244ec8c794ec00e6a6045 > .gpu-agent/investigation-capability/v11-witness.json
env -u PYTHONPATH PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/you/conda_env/agentic-gpu-debugger/bin/python -m pytest tests/unit/test_investigation_witness.py -q
/home/you/conda_env/agentic-gpu-debugger/bin/ruff check tools/investigation_witness.py tests/unit/test_investigation_witness.py
/home/you/conda_env/agentic-gpu-debugger/bin/ruff format --check tools/investigation_witness.py tests/unit/test_investigation_witness.py
git diff --check
```

两份只读核查均退出0。v10四轨迹按上列顺序的完成planner数为3、5、4、4；
v11为4。路径分别是memcheck命中直接检索、racecheck命中后额外synccheck、
initcheck命中检索、synccheck命中检索，以及v11的racecheck命中检索。
没有把v10 case0009的补丁失败改成成功，只确认其调查的执行证据。

新增测试最初8 passed in 1.10s，补充无法判断后终止测试后9 passed in 1.40s。
前8项包含于9项，不相加。模拟GPU边界的两个控制变量分支在相同源码下让
memcheck分别FINDING/CLEAN，模拟planner分别检索/转initcheck，明确禁止调用
RuleRouter，真实编排分派满足预期。其余检查缺失结果、错工具、未完成执行、
历史结果被改、空检索的拒绝；终止测试确认不编造诊断或生成候选。
首次Ruff报测试import排序，已修正；最终Ruff、format及diff检查均退出0。
未运行全量测试；没有生产逻辑改动，无新GPU/API调用或费用。

## 第二层方案与交接边界

新增docs/investigation-capability-and-benefit-CN.md：说明第一层证据及其边界，
预定8案例×D/E×3重复的48单元开发对照，统一模型/知识库/公开输入/修复轮数
与调用、时间边界，固定随机顺序；主指标修复数，次指标调用/工具/阶段耗时。
保留所有失败和未执行，重复不算独立案例，不以E必须胜出为验收。

方案未执行，不新增模型费用，不声称新收益；启动前仍需冻结实际日程/版本
和新增算法入口预检查。当前完成的是第一层基本机制与真实轨迹核验，不是
真实模型反事实测试，更不是广泛优势证明。对模型理由中已存在的错误假设
照实保留，不包装成完美推理。旧数据未写入、未提交或推送。
