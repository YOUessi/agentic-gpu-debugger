# 2026-09-30：自主调查动作修复后的真实定向验证

承接 2026-09-29-agent-investigation.md。用户明确要求“验证”，本轮运行真实
DeepSeek 与隔离 GPU，而不是继续用离线测试代替。使用 failure-aware-execution
技能：先固定范围，再运行，保留全部失败；遇到基础设施问题先诊断，不重复派发。

## 预先固定的设计（运行前记录）

选择 case_0001、0003、0009、0016：分别覆盖越界、初始化、race、同步；前次公开
轨迹中有源码空转和 D/E 路线分歧。每个案例 D/E 各一次，共8单元，随机种子
20260930 固定串行顺序。不是按新结果筛选成功例，不因为失败自动追加重复。
这是一轮开发定向验证，仅4个独立案例，不能用其证明总体统计优势。

两模式使用同一新代码、知识库、DeepSeek模型和输入，诊断/补丁生成器和 full
验证相同；不把注册的 target_tool 提供给 E 指定路线。每单元沿用已授权40次
模型调用的执行边界，无美元上限、不查询余额。费用未知时写 null，记录全部
调用与 token，不把未知当0。不查询或利用原私有 holdout 的内容。

观察项：是否出现 inspect_source 及成功执行；D/E实际物理调用数、planner调用数、
工具路线、诊断/候选/VERIFIED_FIXED；超时、格式失败、NOT_FIXED 分开保留。
源码空转消失只是合同验收，修复率和调用改善须看真实结果。知识库已有后续更新，
因此本次与旧 e80ce75 不能视为单因素因果实验。

## 执行与版本隔离方案

现有工作区包含此前用户任务的未提交修改，不直接为实验提交或覆盖它。复制明确
的项目源码/资源到独立实验快照，在快照内建立本地提交，并使用独立Python环境
指向该快照。API密钥仍从既有 provider.env 载入，不复制到快照、不打印内容。
新脚本 compare_investigation_modes.py 不伪装成“旧失败重跑”，不筛掉旧成功案例
来改变比较分母；固定8个新版本单元，结果独立保存，不拼接旧实验。

尚未运行时没有通过结论。以下继续追加预检查、实际命令、版本和结果。

## 第一轮：预检查与实验启动

新增 compare_investigation_modes.py 和固定日程测试。日程8项均匀配对、无重复，
测试通过，Ruff首次报一行输出过长，拆成相邻字符串后修正。

执行前发现脚本初稿把知识库 corpus_hash 当成检索的 version 参数；阅读
KnowledgeIndex.retrieve/parse_version 确认该参数实际要求 CUDA/Sanitizer 兼容
版本组合，而不是 corpus版本或hash。尚未发出任何API请求时即修正，添加四工具
文档兼容性预检查，避免付费后才发现检索不可用。不是模型或API的错误。
实测 selector `cuda=12.8.1;compute-sanitizer=2025.1.0.0`，四工具预检查通过。

独立目录 `/home/you/gpu-agent-investigation-20260930`：source 为本地源码快照，
env 为 system-site-packages venv，继承原环境依赖，但 editable 安装只在新环境
中生效。确认 gpu_agent.__file__ 指向 snapshot/source/src/gpu_agent；原Conda
安装未卸载。只复制明确项目目录，未复制 .claude-deps、.claude-regress、密钥或
旧实验目录。原工作区没有提交或推送。

初次快照 db14fec 包含已完成改动；修复上述预检查后，最终实验从快照本地提交
7bac87c 开始。实际运行中源码不再更改，每单元重核 clean commit、runtime hash
和冻结知识库字节。不是拿原工作区HEAD冒充这份新代码。

GPU availability=READY；模型配置 deepseek-v4-pro，密钥仅验证存在，不输出。
本轮显式沿用此前定向测试的120秒单请求设置（环境原默认60秒），两模式相同；
不是调整美元额度，不为超时自动补跑。Prompt为m3-2026-09-29-v10。
runtime hash为 `77b9f59749e5501db66822b4ec33833192864a03f82720da14f11015b01fceed`，
corpus hash为 `15c53142f8949ba1d852a3227fd11460060adec359fda9ee0cf42c52011b8600`。
更多绑定字段由results/selection.json在调用前写入。

实际启动命令（source既有配置时关闭shell回显，不打印密钥）：

```bash
set -a
source /home/you/.config/agentic-gpu-debugger/provider.env
set +a
unset GPU_AGENT_CORPUS_FAMILY_ROOT
export GPU_AGENT_LLM_TIMEOUT_SECONDS=120
/home/you/gpu-agent-investigation-20260930/env/bin/python \
  /home/you/gpu-agent-investigation-20260930/source/scripts/compare_investigation_modes.py \
  --repository /home/you/gpu-agent-investigation-20260930/source \
  --knowledge-index /home/you/.codex/worktrees/a352/agentic-gpu-debugger/.cache/gpu-agent/knowledge/20260929-v6/index.json \
  --output /home/you/gpu-agent-investigation-20260930/results
```

实验日志为experiment.log；每次attempt在请求前落盘，每单元结果fsync写入
results.jsonl，全部完成才写completed.json。新verification目录是本轮公开案例
候选的独立验证产物，不是读取旧私有holdout。进程已启动，结果待后续追加。

### 补充预检查测试及本机环境干扰

实验启动后仅在原开发工作区加强日程测试，不改正在运行的独立快照。
新增非法知识库版本测试最初错误地期待 ValueError，实际 parse_version 将其
包装为 KnowledgeVersionUnavailableError；修正测试期待为真实的类型化异常，
生产校验逻辑没有改动。首次复测使用登录shell，自动加载到系统ROS的pytest插件，
在收集之前报缺少lark；栈指向 /opt/ros/humble 的Python3.10包，并非项目测试失败。
移除该命令的PYTHONPATH并关闭第三方pytest插件自动加载后，针对性测试通过。
没有为此安装依赖、修改系统ROS或重启正在执行的GPU/API实验。

实际复测命令：

```bash
env -u PYTHONPATH PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/you/conda_env/agentic-gpu-debugger/bin/python -m pytest \
  tests/unit/test_investigation_comparison_schedule.py -q
/home/you/conda_env/agentic-gpu-debugger/bin/ruff check \
  scripts/compare_investigation_modes.py tests/unit/test_investigation_comparison_schedule.py
```

结果：1 passed in 0.54s；Ruff All checks passed。
该测试覆盖8项固定配对日程、非法版本拒绝、四工具兼容检索预检查。
它不是8个真实实验的通过凭据，真实结论以下面的GPU/API记录为准。

## 第二轮：真实失败的定位（不修改本批实验）

运行到case_0009/E时出现NOT_FIXED。7次物理调用全部COMPLETED，无API错误码，
诊断正确指出32线程映射到16个共享槽，后置屏障不能消除先前的写写竞争。
但补丁将 `threadIdx.x & 15U` 改为
`threadIdx.x >= 16U ? threadIdx.x - 16U : threadIdx.x`。
对于实际0到31的线程索引，这两个表达式等价：0和16仍写槽0，1和17仍写槽1。
所以这不是修复，仅是等价改写。真实racecheck仍为FINDING，验证原因
ORIGINAL_FINDING_PRESENT；其他工具CLEAN，普通输出Oracle通过。

诊断run：b4ca835e6bb54534ab24b178a7510790；candidate run：
8cd4da7e21057b7c0589c3eeb65025c9。候选内容通过public manifest中的hash/size检查读取。
问题可确定为本次模型候选的语义错误，不能泛化成所有DeepSeek调用失败，
也不是把格式正常误当修复成功：独立GPU验证正确拒绝了该候选。
当前系统的一次候选设计并不保证模型将正确诊断落实为正确代码。

本轮不加入专门针对case_0009的规则，不修改Oracle，不将失败替换成重跑成功，
继续既定8项验证。若后续要提高此类失败的恢复能力，应另立版本评估通用的
公开工具自检/有限修订回路；它是设计升级，不是本次移除空转源码动作的修补。
目前没有实施此升级，也不能承诺该升级必然改善所有案例。

## 最终验收：8项执行完毕，但自主调查优势没有证明

进程正常退出0，completed.json记录8项；结果行数也是8。实验快照最终仍为干净
工作区，提交为7bac87c47c51e88d39824380e7e741c4b124d014。没有中途换代码，
没有额外补跑。复核public原生step/decision时逐项核对artifact大小和SHA256，
动作类型相符，全部使用diagnosis-full-source-v2。没有读取旧holdout内容。

逐单元结果（调用数为物理请求，tokens为provider返回的total_tokens）：

- case_0001/D：VERIFIED_FIXED，2次、5814 tokens、100.05秒；
  run 2604a74927df4f9584f36993bdcc8116。
- case_0009/D：VERIFIED_FIXED，2次、5850 tokens、100.72秒；
  run c988973e49224830b34617a8ecda2669。
- case_0016/D：VERIFIED_FIXED，2次、6103 tokens、102.12秒；
  run 177effa0171047458cfda0dd20bb21d9。
- case_0003/D：VERIFIED_FIXED，2次、5637 tokens、103.10秒；
  run af662f578da64841a91c32ba7159c172。
- case_0001/E：VERIFIED_FIXED，5次、15184 tokens、114.05秒；
  run 1d18fb108cb34c5ea989f69762aeabec。
- case_0009/E：NOT_FIXED，7次、20714 tokens、40.16秒；
  run b4ca835e6bb54534ab24b178a7510790。原因详见上一节，不隐藏失败。
- case_0003/E：VERIFIED_FIXED，6次、17775 tokens、118.53秒；
  run fa9c05c445994b8795526674db753a29。
- case_0016/E：VERIFIED_FIXED，6次、17233 tokens、116.77秒；
  run cc2232efe86f406598d0232b55872d94。

D为4/4，8次调用（诊断4、补丁4），23404 tokens；E为3/4，24次调用
（planner16、诊断4、补丁4），70906 tokens。总计32次、94310 tokens；
全部调用COMPLETED、error_code为空、usage都有记录，无超时或格式重试。
不以旧价格推算费用，cost_usd保留null，美元未核算并不代表免费。
没有查询账户余额或设置美元上限。

两模式均为16个动作提案，全部获准；E的16个planner调用对应16个动作，
没有inspect_source，也没有重复动作拒绝。相同案例的动作路线如下：

- case_0001：D/E均memcheck → 检索 → 结束诊断。
- case_0003：D/E均memcheck → initcheck → 检索 → 结束诊断。
- case_0016：D/E均memcheck → synccheck → 检索 → 结束诊断。
- case_0009：D为memcheck → synccheck → racecheck → 检索 → 结束；
  E为memcheck → racecheck → 检索 → synccheck → 结束。

因此，本轮确实消除了“源码已完整提供却仍花调用查看源码”的动作，
但E没有减少工具总量：D/E各8次Sanitizer动作、4次检索。
E在case_0009先找到race证据，之后仍补跑synccheck；顺序不同不是效率提升。
仅凭已有最小结束证据，也不能把之后的每次检查都定性为错误；这是可审计的
额外动作，而非证据不足或控制器失效。

成功的7项通过编译、运行、Oracle和四工具严格验证；失败的1项被正确拒绝。
诊断与补丁契约可执行，不等于模型每次能给出正确修复。
D累计405.98秒，E389.51秒，不能据此宣称E更快：E中的NOT_FIXED单元没有与
成功单元相同的验证完成路径，耗时仅40.16秒，失败造成的短路混淆总耗时。
三个双方均成功的案例里，E分别114.05/118.53/116.77秒，D分别
100.05/103.10/102.12秒；本轮也没有呈现时间优势。

### 结论和下一轮边界

已验证完成的是全源码动作合同修复和一次真实端到端比较；尚未证明的是
“Agent比规则路由更有价值”。本次仅4个开发案例、每模式每案例一次，
不能推广到广泛CUDA任务，不将4/4与3/4解释为统计显著差异。
这不是完整240单元重评，也没有拼接到原评测。

结果没有完美，因此失败已继续定位到等价下标改写，而不是仅记录“实验跑完”。
本轮没有发现需立即更改的源码动作合同缺陷；不能为了得到完美数字无限重试
随机补丁或继续添加针对已知case的提示。下一轮若要提升修复能力，应单独设计
通用公开自检/有限修订实验；若要证明调查价值，应预先指定能区分路线的公开
案例和调用/工具指标，同一新版本公平比较。不得改现有结果或反向利用holdout。

### 原始证据位置及校验

- /home/you/gpu-agent-investigation-20260930/results/selection.json：调用前固定配置。
- /home/you/gpu-agent-investigation-20260930/results/results.jsonl：8项完整结果和usage。
- /home/you/gpu-agent-investigation-20260930/results/completed.json：完成标记与提交。
- /home/you/gpu-agent-investigation-20260930/results/public：本轮原生诊断、动作、候选记录。
- /home/you/gpu-agent-investigation-20260930/results/verification：本轮独立验证产物。
- /home/you/gpu-agent-investigation-20260930/experiment.log：原始执行进度。

额外离线检查：新增日程测试1 passed；Ruff check通过，format --check两文件通过，
git diff --check通过。本轮未重跑已经通过的199项相关回归，未重跑旧GPU/API单元。
原工作区未commit或push；只有独立实验快照有前述本地提交。
