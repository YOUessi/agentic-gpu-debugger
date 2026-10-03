# 2026-09-29：自主调查优势的证据排查

## 请求与边界

用户同意先分析现有公开 D/E 调查记录，定位无效动作与真实阻塞，再决定改哪里。
本轮不调用 DeepSeek、不启动 GPU、不重跑旧实验、不读取 evaluator 或私有 holdout。
不以“自主多步”直接推导“比规则好”，不承诺修复率必然提高。工作区 HEAD 87fa492
包含此前未提交的检索与算法扩展，保留这些改动，不把当前代码当成旧实验代码。

使用 failure-aware-execution 流程：先核对版本、逐步证据和终止原因，再提出有
依据的最小修复。当前是活动开发中的定向排查，按项目要求在本文记录，不另建仪表盘。

## 第一轮：核实数据和可回答的问题

公开 store 为 `/home/you/gpu-agent-release-e80ce75/public`，evaluation run
`1bc6af1b1acb4940a2c0829fc5d274aa`。读取该 run 指向的 summary、schedule 和
D/E diagnosis 的 controller action、decision、provider 状态记录。复用已有
dev_failure_inventory 的受限读取器校验大小/hash和路径，不读取候选源码或模型原文。

预期输出：D/E 各单元的最终结果、planner/diagnose/patch 调用数、允许/拒绝动作、
第一次具备 finish 条件后继续做了什么、源码读取是否完全重复。统计分母按单元
与调用分别计数。额外动作不能自动叫“浪费”，源码读取可能改善定位；只能标记
“满足控制器最低结束条件后仍发生”，是否有效须另作因果实验。

当前已核对 docs/evaluation-report.md 记载开发集 D=44/48、E=41/48 修复成功；
此前对话引用的 E/D 相等不能不加核实沿用。下一步由原始记录独立重算。

## 第二轮：原始轨迹统计与确定原因

### 实际实现与校验方式

新增 `tools/dev_investigation_audit.py`，仅接受 e80ce75 的完整 240 单元 development
清单，从中定位 96 个 D/E diagnosis。读取关联的 step、decision、evidence snapshot
和 provider 状态；STARTED 与终态按 invocation ID 合并，不能算两次调用。拒绝
非 development、版本不符、重复单元/步骤、缺失 decision、大小/hash不符和链接文件。
输出只包含控制器字段、计数、hash、run ID 和阶段，不输出源码、查询原文或模型解释。
它没有导入运行中的 gpu_agent，也不加载密钥、调用模型或连接 GPU。

源码查看是否增加事实，通过两种证据交叉核查：一是 step 中已有完整 public source；
二是有下一步时比较前后的 public evidence 是否变化。未保存下一步就标为无法比较，
不补造轨迹。allowed 表示策略允许，不冒充每个动作已执行成功；是否完成诊断/生成
补丁/验证通过分别使用原始记录字段。

### 重算后的真实结果

D：48 单元，48 诊断、48 补丁，44 VERIFIED_FIXED、3 NOT_FIXED、1 INCONCLUSIVE。
模型物理调用 102 次，拆为 diagnose 51、patch 51，没有 planner 调用。
Sanitizer 调用 93 次、检索 48 次。189 个动作提议全部允许。

E：48 单元，44 诊断、44 补丁，41 VERIFIED_FIXED、3 NOT_FIXED、3 LLM_TIMEOUT、
1 LLM_INVALID_OUTPUT。模型物理调用 326 次，其中 plan 224、diagnose 53、patch 49。
Sanitizer 调用 92 次、检索 45 次。220 个提议，219 允许，1 次重复工具提议被拒。

这次总调用差值 326-102=224，恰好等于 E 的 planner 调用数；非 planner 调用
两者均为 102。这只说明调用成本结构，不表示删除 planner 后能保留 E 的决策。
E 的工具总数略少不能说成更高效，因为三项提前超时。仅在 E 已 DIAGNOSED 的
44 个配对单元中，E 使用 87 次 Sanitizer，对应 D 为 84 次，E 反而多 3 次。
这是描述性、带选择条件的比较，不是总体显著性检验。

按 case+repeat 配对：37 对双方修复、4 对只有 E 修复、7 对只有 D 修复。
不能把 48 对当作 48 个独立案例；实际只有 16 个 case，也不能把 E 独赢的四项
自动归因于调查能力，后续补丁生成随机性同样会影响结果。

### 发现一：源码动作是信息获取上的空转，不是重复读取循环

E 共允许 37 次 inspect_source，涉及 35 个单元；每次动作的输入里已含完整源码。
有下一步 snapshot 的 36 次，前后 PublicEvidence 全部相同；另一次无下一步，
不能比较。完全覆盖此前已读范围的次数为 0，因此不是“同一范围无限重读”。
其中 16 次发生在已满足控制器最低结束条件之后，全部为 inspect_source。

代码证据：冻结版与当前版 `public_evidence_from_bundle` 都将整个 kernel 内容
放入 PublicSource；provider.plan 又把该 evidence 完整序列化给模型。
`AgentOrchestrator._source` 只验证行范围并写 source-reads/<action_id>.json，
不向 evidence 增加代码摘录或新事实。冻结版该函数注释也明确写着源码已提供。
因此它可表示模型选择“关注某处”，却不是一次提供新信息的工具调查。将其宣传
为获取新源码证据不准确。一次额外动作又需要后续 planner 调用，存在可改进的
调用开销；但删除后是否影响模型注意力/推理，仍须同版本开发实验，不能预报收益。

示例 `37795d11f3de7e67e751be25ae24150f`（case_0016 repeat1）：memcheck →
synccheck → 检索 → inspect_source → finish。源码查看前 missing_evidence 已空，
该单元最终修复；“已经可结束”不是说后续动作必然有害，只说明最低条件已满足。

### 发现二：没有证据支持继续修“经常缺文档/反复拒绝”

96 个 D/E 单元没有 MANDATORY_EVIDENCE_MISSING 的动作拒绝，也没有调查预算耗尽。
唯一拒绝发生于 E case_0011 repeat2，run `fd7adfa2e6a96296147f46ca524f45e7`：
initcheck 已有结果后又提议一次 initcheck，被 DUPLICATE_NO_BENEFIT 拒绝，随后
一次重新规划选择 finish，最终 VERIFIED_FIXED。因此原有一次重新规划在该真实
例子中发挥了作用，本轮没有理由为提高成绩直接放宽重规划次数或重复工具权限。

更早版本的“缺文档19项、重复动作12项”不能套在 e80ce75 上。当前 prompt 已有
controller_state/missing_evidence 描述，本轮不把已完成的修复再重复一遍。

### 发现三：失败要按阶段区分

E 三个 LLM_TIMEOUT 全部是 planner 调用：case_0008 repeat2 的
`f2619ebd325f2cee55fc4befe395a7df`、case_0014 repeat1 的
`a8494328a75a078f49aeb0400eaa9181`、case_0013 repeat2 的
`c05a3080aa0c1c2f42c15cf30a8f304a`。它们没有走到诊断或补丁完成，不能称为模型
修错补丁；也不能仅凭客户端 timeout 判断服务端是否收到、为何慢或是否收费。

E 终止格式失败是 case_0002 repeat2，run `2eec904c586375b8a8d39ed2629821c8`：
已收集工具、文档并 finish，之后 diagnose 两次 LLM_INVALID_OUTPUT。不是 planner
无法选择工具。E 三个 NOT_FIXED（case_0010 repeat1、case_0006 repeat2、
case_0008 repeat0）已经生成补丁，是验证未通过，不是调查没开始。

特别核对了 e80ce75 到当前源码的差异：当前 policy 已移除隐式60秒硬截断，provider
已能配置/记录 request_timeout_seconds，并有 extra_forbidden 的专用纠正提示。
这些是已有后续改动，本轮没有再改。旧超时/格式失败不能当作当前改动无效的证据，
也不能用当前已有补丁反向改写 e80ce75 的三次超时和一次格式失败。

### 发现四：局部存在自主路线选择，但总体优势仍未证明

case_0009 三次以及 case_0010 三次，D 都先 memcheck、synccheck 再 racecheck；
E 选择 memcheck 后直接 racecheck，少跑一种工具。源码中的 `__syncthreads` 会让
RuleRouter 的 `"__sync" in source` 启发式先选择 synccheck，E 在这些案例里没有
沿用同一路径，这是可引用的不同路线证据，不仅仅是多步调用。

另一方面，E 在 case_0003、0012、0013 的部分重复中，先做了 D 没有做的
racecheck，再到 initcheck；case_0012 repeat1 甚至多做 racecheck 和 synccheck。
不能只挑选省工具的轨迹来宣传，也不能据一条路线决定固定全局工具顺序。

## 第三轮：排查工具自身的测试和证据留档

工作目录 `/home/you/.codex/worktrees/a352/agentic-gpu-debugger`，解释器
`/home/you/conda_env/agentic-gpu-debugger/bin/python`。执行命令：

```bash
mkdir -p .gpu-agent/investigation-review
/home/you/conda_env/agentic-gpu-debugger/bin/python tools/dev_investigation_audit.py \
  /home/you/gpu-agent-release-e80ce75/public 1bc6af1b1acb4940a2c0829fc5d274aa \
  > .gpu-agent/investigation-review/trajectories-final.json
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/you/conda_env/agentic-gpu-debugger/bin/python -I \
  -m pytest tests/unit/test_dev_investigation_audit.py tests/unit/test_dev_failure_inventory.py -q
/home/you/conda_env/agentic-gpu-debugger/bin/ruff check \
  tools/dev_investigation_audit.py tests/unit/test_dev_investigation_audit.py
```

真实公开读取退出0，96 个单元完整统计。新增9项合成测试与原读取器相关12项合计
21 passed，最终0.17s；测试覆盖无原文输出、无输入修改、错误split/commit/重复
单元拒绝、hash篡改和symlink拒绝、读取范围重叠与拒绝动作不计入允许数。
首轮 Ruff 报7处长行，格式化后修正，最终 Ruff 通过；没有失败功能测试被忽略。
原始测试日志为 tests.log、tests-final.log。未跑全量离线/GPU/API，因为本轮
没有改运行时行为；API/GPU 新调用均为0。未提交/推送。

冻结评测 summary artifact SHA256：
`605674879f2a85b230bf4ca07e235452674674bf9ea943794ecb2fc5898b25f2`。
最终脚本 SHA256：
`ab3d3df059edda6cc39e85ab26a9ad90da37848ed9c6b2278d264cf3991d2cb9`。
完整96条轨迹、各run manifest hash和汇总见 trajectories-final.json；较早的
trajectories.json 是增加“下一步 snapshot”比较前版本，保留但不混作最终统计。

## 下一步应该怎么修（本轮未实施，不能写成已修复）

第一优先级是 inspect_source 的信息契约，而不是再补一段“不要重复读”的提示。
短期保持当前“完整源码已提供”的输入合同，应从此情况下提供给 planner 的可选
动作中移除无新增内容的 inspect_source，并使 schema、控制器可用动作和文档一致。
不能只在执行后拒绝它，否则只是把无效动作换成重试/终止；也不能偷偷转发为
finish，因为模型还可能需要检索或别的工具。单元测试要证明输入仍提供相同源码、
诊断/补丁标准没放宽、旧记录可回放，缺少证据仍不能结束。

若以后要让查看源码成为真正调查，采用按需提供带行号片段的另一份明确合同：
初始不提供全文，工具返回之前没有的片段，并完整审计。那会改变 D/E 的可见信息，
是下一版设计升级，不能混进本轮小修或与旧结果直接作因果比较。

之后仅对公开开发案例做定向验证：先选包含源码空转的案例和不同路线案例，保存
新版本结果，观察调用数是否降低、诊断/修复是否退化。旧成功 API 单元本轮未重跑。
若要得出总体优势，需事先确定配对指标、统一模型/知识库/工具权限，再完整评测；
本次统计不能证明“去掉37个动作就能省37次调用且修复率不变”。

本轮完成的是原因定位与可复查证据；“Agent 相对规则的优势”仍未得到证明。
不通过扩大重试、重复采样、篡改旧成绩或挑成功案例来达成该结论。

## 第四轮：授权后修复完整源码合同中的空转动作

用户已明确要求修复，并将整个过程继续归入“Agent 自主调查的优势尚未证明”。
本轮先证明 schema 不再提供 inspect_source、控制器不会执行注入的旧动作、源码
仍完整提供、验证标准不变；这不等于证明调用次数下降或相对 D 的修复率提高。

### 原因与兼容风险

executor 在 D/E 原生记录复核中会重新计算 decide_action。若直接全局改变规则，
历史允许的 inspect_source 会被重新判为非法，所以必须按版本保留旧策略回放。

### 实际修改

- models：旧 domain action 和 LegacyPlannerOutput 保留，新 PlannerOutput 的
  结构化 schema 只提供四 Sanitizer、检索、finish、无法判断，不提供 inspect_source。
- prompts：升至 m3-2026-09-29-v10，明确完整源码已在 evidence.sources 中，直接
  分析；下一种工具与结束仍由模型选择，不自动转发 finish。
- policy：新 diagnosis-full-source-v2 拒绝注入旧动作，返回 SOURCE_ALREADY_AVAILABLE。
  其余证据、预算和重复动作约束不变。
- orchestrator：删除运行时源码查看动作注册入口。
- executor：按受绑定 prompt version 为历史记录选择 v1，新版本使用 v2；不能
  信任模型自行提供的策略版本，不能改旧产物以适配新规则。

没有修改完整源码输入、诊断/补丁共享规则、Oracle 或 Sanitizer 标准，没有费用
控制或私有数据访问。旧检索和案例扩展改动保留。

### 测试开始

更新旧成功查看测试为“注入旧动作被拒、无源码读取、重新规划反馈正确”；保留
旧 wire 约束遥测测试；新增历史策略兼容、新 wire 拒绝与完整源码保留测试。
首次 Ruff 有长行、未用导入/排序告警，已格式化。功能测试进行中，结果待追加。

### 静态检查遇到的问题与修正

strict mypy 报 executor 的 binding.prompt_version 类型为 str|None，不能直接传给
只收 str 的策略选择函数。实际修复是显式支持入参类型并在缺失版本时拒绝回放，
而不是用类型忽略或把缺失版本默认为新规则。新增 None 拒绝测试，以及当前
PROMPT_VERSION 必须映射当前策略的测试，防止以后升级提示版本忘记同步映射。
修正后 73 个 src 文件 strict mypy 通过，Ruff 通过。

新策略/旧策略/旧动作注入/新wire定向检查：4 passed、20 deselected（0.79s）。
另将完整源码断言放入真正 service→provider 输入的端到端模拟测试，逐次核对
provider 输入源码与该run原始snapshot一致，并核对没有 source-reads 产物：
1 passed、23 deselected（0.84s）。模拟测试不冒充真实 DeepSeek 行为。

主相关回归只覆盖 test_agent_loop、test_provider_output_telemetry、test_mode_contract、
test_evaluation_modes 四文件，不是全项目离线套件。因为本轮改变策略版本与原生
记录复核，所以需要覆盖模式执行/绑定/回放，不仅测试一个 schema 字符串。
该组仍在运行，未完成前不报告“所有测试通过”。

本轮没有运行 GPU/API、没有生成新的实验修复率、没有重跑旧成功付费单元。
旧 e80ce75 结果仍是 D44/48、E41/48，不能因源码动作修复将它改成新版本成绩。

### 本轮最终结果与复现命令

相关四文件回归最终 **199 passed，449.71s（7分29秒），退出0**。范围包含 Agent
循环、输出遥测、五模式合同和评测记录复核；这些是模拟 provider/backend 的离线
测试，不是新 DeepSeek 或 GPU 实验，也不是所有项目测试。日志为
`.gpu-agent/investigation-review/full-source-contract-tests.log`。

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/you/conda_env/agentic-gpu-debugger/bin/python -I \
  -m pytest tests/unit/test_agent_loop.py tests/unit/test_provider_output_telemetry.py \
  tests/unit/test_mode_contract.py tests/unit/test_evaluation_modes.py -q --tb=short
/home/you/conda_env/agentic-gpu-debugger/bin/mypy --strict src/gpu_agent
/home/you/conda_env/agentic-gpu-debugger/bin/ruff check src \
  tests/unit/test_agent_loop.py tests/unit/test_provider_output_telemetry.py
git diff --check
```

主回归运行期间增加的当前提示版本映射/缺失版本拒绝与完整源码端到端断言，已
另做上述4项和1项定向检查。它们与199项重合，不相加为204个独立测试；未为这类
断言补强重复整组7分钟回归。mypy全73文件、Ruff、七个受改代码文件格式检查和
git diff --check 均通过。没有重放全部旧240条原生证据；旧策略的兼容性由版本
选择、旧动作解析/判定以及模式复核相关测试覆盖，不冒称旧实验全量重新验收。

### “Agent 自主调查优势尚未证明”的连续状态

已完成：公开旧实验96条D/E轨迹核查；定位源码动作无信息增量；修复模型可见
schema、运行时入口、策略和版本兼容；同步模式文档；相关离线测试通过。

尚未完成：新版本的真实开发集D/E对比及其调用数、成功率分析；因此总体优势
仍标为“尚未证明”，不是因本轮测试通过就关掉整个主题。

下一轮应先固定待测代码/知识库/模型配置，明确这是完整源码合同v10，并预先写下
要比较的公开案例、重复次数与指标。先以出现源码空转及路线分歧的开发案例做
定向检查，确认新记录不再有成功的inspect_source，保留所有超时/拒绝/错误补丁。
对同一批D/E统计真实调用数、工具数、诊断/补丁/VERIFIED_FIXED，不混用旧版本
成功条目补分母。定向测试只能说明修复行为和开发效果；要主张总体优势，还需
更完整的配对评测。不能提前承诺一定省37次调用、一定不退化或一定胜过D。

本轮没有付费/GPU调用、没有新增费用控制、没有提交或推送，旧结果和本轮此前
文档修改均保留。详细记录覆盖排查、修法、兼容风险、检查失败和最终测试结果。
