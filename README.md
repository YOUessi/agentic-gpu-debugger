# Agentic GPU Debugger

An evidence-grounded CUDA debugging agent: investigate with GPU tools, propose a patch,
self-check and revise using public evidence, then verify independently.

面向 CUDA 故障的调试工程作品。Python 编排模型、CLI、工具与证据；CUDA C++提供真实
工作负载，候选代码在隔离GPU后端执行。最终结果由数值检查和Compute Sanitizer判断，
不是模型自评。

## 核心能力与演示

- 调查：模型在受控动作空间选择四种Sanitizer与官方文档检索。
- 修复：生成受限diff，公开GPU自检失败后修订，拒绝重复候选空转。
- 验证：公开检查通过后执行独立strict验证，隐藏测试结果不反馈给模型。
- 可核查：保存源码、输入、补丁和工具产物哈希，以及实际调用与失败原因。

同一快照 `0d7db80` 的真实E实验：二维stencil经5次模型调用修复并通过独立验证；
分段scan经8次调用仍有竞争，重复候选后停止，未被误报成功。不宣称E已优于固定规则D。

[演示与复现](docs/demo.md) · [公开证据摘要与补丁](docs/evidence/portfolio-summary.json) ·
[项目介绍与讲解](docs/portfolio-CN.md) · [能力边界](docs/limitations.md)

## Full-stack CUDA Test & Repair Console（V3 分支）

`feat/v3-fullstack-ai-test-console` 在现有 Agent 核心上增加 React/TypeScript 操作台与
FastAPI adapter。Web 层直接读取 public RunStore，并通过 `ApplicationService` 发起
repair/strict verification；不会复制 Agent、执行器或 verifier，也不接受任意宿主机路径。

```bash
python -I -m pip install -e '.[web]'
cd dashboard && npm install && npm run build && cd ..
gpu-agent web --host 127.0.0.1 --port 8000
```

Dashboard 展示 RunStore 分页/筛选、Agent investigation trajectory、grounded diagnosis、
多轮 repair candidate/diff、public self-check、strict verification 和注册 artifact/log。
前端使用 Vitest 与 Playwright；API 使用临时 RunStore 测试。Round 2 增加 public case
catalog、运行中自动轮询和可点击 evidence citation；Round 3 将 Repair 改为持久化异步
Web Job，提交后立即返回 job ID，并在 controller 创建 diagnosis run 后自动挂接实时
RunManifest timeline，同时提供 Original Source / Selected Candidate Diff 并排查看。Round 4
新增 Batch & Evaluation Analytics，可对 public seed batch 和 A–E evaluation 做服务端投影、
mode 对比、大规模记录表、failure-family 分布、latency/调用量/token/cost 分析；可单独挂载
历史 public RunStore，且不会读取 evaluator/private evidence。Round 5 又补充了 mode 可视化图表
和 evaluation record → diagnosis run → evidence/candidate/verification 的只读 lineage drill-down；
Round 6 增加 evaluation trend、兼容性门控的 regression comparison，以及 CSV/JSON export。
设计与安全边界见
[Full-stack Console](docs/fullstack-console.md)，开发中遇到的问题与修复过程持续记录在
[Full-stack 开发日志](docs/fullstack-development-log.md)。

## 实验证据与版本边界

当前已实现证据驱动诊断、单候选与多轮补丁修订、Docker GPU 隔离、四种 Compute
Sanitizer、public/private Oracle、strict 验证、mutation 注册门、评测记录和 release
gate。冻结版本 `e80ce75` 已完成 16 个公开案例和 8 个私有案例的原生验证、
240 个开发集单元和 120 个 holdout 单元的真实 DeepSeek 评测，以及 18 项 GPU/隔离
补充验收。当前工作区包含后续工程修复，不能把冻结实验算成这些修改的重新评测。
实际结果、失败和验证边界见 [评测结果](docs/evaluation-report.md)。

后续新增了四种独立算法的公开开发扩展（循环移位、stencil、加权直方图、分组
归约），各自有 CPU Oracle 与核心路径故障。入口、验收范围和未覆盖能力见
[案例多样性说明](docs/case-diversity-CN.md)，逐轮测试见
[详细记录](docs/repair-log/2026-09-29-case-diversity.md)。它们不计入旧实验成绩。

2026-10-03进一步加入二维五点stencil与128元素分段inclusive scan，共6个扩展案例。
新增两例的原生GPU验收已通过；真实E修复中二维stencil通过最终验证，分段scan未修好，
公开自检检出残留竞争，重复候选后停止。这是保留的模型失败，不计作通过。
见[本轮工作负载记录](docs/repair-log/2026-10-03-real-workloads.md)。

## 独立环境

当前实现与各批实验不是同一个版本；以[当前版本状态](docs/current-status-CN.md)为准。
当前代码包含连接原因分类和repair前置规格检查；历史实验证据绑定各自快照，不混用版本。

多轮修复新入口（需既有provider与知识库配置）：

```bash
gpu-agent repair benchmarks/public/case_0009/public_input --allow-paid-calls --max-candidates 3
```

公开GPU自检失败后修订，公开检查通过后独立strict验证；保存每轮候选和反馈。
这不是旧评测的重跑，也不保证每个模型候选都能修好。默认3个候选可配置，
共享总调用边界，无美元限额。`diagnose`仍保留单候选入口。
详见[实现与测试记录](docs/repair-log/2026-09-30-iterative-repair.md)。

repair v2还要求源码目录中的公开 `task.json`：固定算法标识、版本及原始kernel的SHA256。
自带公开案例已经提供该文件。功能需求送入模型，自检使用同目录公开输入检查数值与
Sanitizer；缺少规格不能判自检通过。支持的功能定义在 `src/gpu_agent/public_task.py`，
不是从隐藏验证推导需求。详见[本轮修复记录](docs/repair-log/2026-10-01-public-repair-correctness.md)。

Repair v3提供显式启用的失败候选重新调查；默认仍为v2。它按公开失败信号决定直接修订
还是调查当前候选，并在原有共享预算内使用新诊断生成针对原始源码的完整补丁：

```bash
gpu-agent repair benchmarks/public/case_0009/public_input --allow-paid-calls \
  --reinvestigate --max-reinvestigations 1 --max-candidates 3
```

重新调查不重置预算，公开自检仍需通过后才进入独立strict验证。使用限制、诊断源码
作用域和产物见[Repair v3操作说明](docs/v2-operator-runbook.md#repair-v3)，实际测试与
失败记录见[2026-10-08记录](docs/repair-log/2026-10-08-repair-v3-reinvestigation.md)。
当前不据此宣称修复率或真实模型能力收益已验证。

使用 Conda 同时固定 Python 和原生 CUDA 开发工具，环境内的 Python 依赖用 pip 锁定。不叠加 venv，不复用其他项目的 PyTorch 环境。Conda **不是**安全沙箱；候选代码只能在后续的 Docker 隔离后端执行。

首次创建（已存在环境时不要重复创建或覆盖）：

```bash
conda env create --prefix /home/you/conda_env/agentic-gpu-debugger -f environment.yml
conda activate /home/you/conda_env/agentic-gpu-debugger
python -I -m pip install -r requirements.lock
python -I -m pip install --no-deps -e .
```

`environment.yml` 固定 CUDA 12.8 Update 1 的必需组件，未包含不需要的 Nsight GUI/数学库。实际安装的原生包 URL/SHA256 已保存到 `environment.lock.txt`；精确复现时用 `conda create --prefix /absolute/new/prefix --file environment.lock.txt` 替代上面的 YAML 创建命令，再安装 Python 依赖与本包。该 lock 仅适用于 Linux x86_64。

本机 shell 的 `PYTHONPATH` 包含 ROS Python 3.10 路径，因此使用 `python -I` 排除外部 Python 路径和用户 site-packages，不改动 ROS 或全局配置。依赖安装也使用该模式。

## 环境诊断

```bash
python -I -m gpu_agent env --json
python -I -m gpu_agent env --cuda-root /usr --json
```

也提供 `gpu-agent env` 控制台入口。默认 CUDA root 为当前 Python 环境 prefix；可以用 `--cuda-root`、`--cuda-bin` 或 `GPU_AGENT_CUDA_ROOT` / `GPU_AGENT_CUDA_BIN` 指定绝对路径，CLI 参数优先。不根据 torch/CUDA runtime 版本推断 NVCC。

退出码：0 = 元数据符合基线，1 = 未就绪，2 = 配置无效。输出包含实际路径、工具版本、GPU/Driver/SM、原始探测结果和 reason codes。未知版本保留 null。

`ready=true` 仅表示 `readiness_scope=metadata_only`，`execution_verified` 始终为 false。真实编译运行在 T02 验收；容器与 Sanitizer 执行能力在 T03 验收。版本查询不能证明头文件/链接器/GPU 执行路径已经可用。

当前元数据门禁限定 Linux x86_64、NVCC 12.8.x、Sanitizer 2025.1.x、GCC 6–14 和目标 SM 8.9。Driver 570.124.06 是本项目采用的保守基线，不是 CUDA 12.x minor compatibility 的最低要求。依据：[CUDA 12.8.1 安装指南](https://docs.nvidia.com/cuda/archive/12.8.1/cuda-installation-guide-linux/index.html)、[Release Notes](https://docs.nvidia.com/cuda/archive/12.8.1/cuda-toolkit-release-notes/index.html)。当前版本只执行操作者指定的版本查询程序，不接受模型工具调用。

## 验证

```bash
python -I -m pytest tests/unit -q
python -I -m pytest tests/integration -q
python -I -m ruff check src tests
python -I -m mypy src/gpu_agent
python -I -m pip check
```

已注册 `gpu`、`container`、`live_llm`、`release` 标记。`--require-live` 将带这些标记的 skipped 测试变为失败；收集阶段 skip 也失败。单元测试、真实 GPU 验收与模型实验分别记录，不互相替代。

## 公开 Seed 批量验证

已有 16 个公开 seed（四种 Sanitizer 家族各四个）可以通过同一个原生控制器串行执行、读取报告和导出证据：

```bash
gpu-agent benchmark run-seeds --repository "$PWD" --data-root /可信public_store的父目录 --preflight-only
gpu-agent benchmark run-seeds --repository "$PWD" --data-root /可信public_store的父目录 --case case_0001
gpu-agent benchmark batch-report BATCH_ID --data-root /可信public_store的父目录
gpu-agent benchmark export-batch BATCH_ID --data-root /可信public_store的父目录 --output /tmp/gpu-batch.zip
```

真实执行要求 `GPU_AGENT_CORPUS_FAMILY_ROOT` 指向已经存在的可信 family；命令不会隐式创建 family、ledger 或注册案例。只有显式使用底层 `--register` 才请求注册，且仍需通过原有 `BenchmarkBuilder` 门禁。详细安全边界和操作步骤见 [公开 GPU Seed 批次运行说明](docs/GPU_BATCH_CURRENT_CN.md)。

## T06 诊断与单候选工作流

`gpu-agent diagnose PATH` 接受一个 CUDA 源文件或含 `kernel.cu` 的目录，输出实际
`run_id`。源码被快照为固定的 `kernel.cu`；匹配 vector harness 接口时仅加入控制器
指定的三个公共 harness 文件，否则走单文件隔离编译。模型没有 shell、任意路径、
URL 或验证器控制权。standalone 模式需要重新构建包含 `build_standalone` 操作的
container runner 镜像；现有四文件 vector 协议保持不变。

```bash
gpu-agent diagnose benchmarks/public/case_0001/public_input --allow-paid-calls --max-llm-calls 40
# 将上一条命令输出的实际 run_id 用于下列命令：
gpu-agent verify RUN_ID --generated-candidate --strict
gpu-agent report RUN_ID
# 或注册一个人工 unified diff；与 --generated-candidate 二选一：
gpu-agent verify RUN_ID /absolute/path/to/candidate.diff --strict
```

远程 provider 通过 OpenAI-compatible 适配层调用模型，必须显式配置
`OPENAI_BASE_URL`、`OPENAI_MODEL` 和控制器环境中的 `OPENAI_API_KEY`。本程序不自动读取
`.env`，也不索取或打印密钥。缺少配置时保存 `LLM_UNAVAILABLE` 结果，LLM 调用数为零。
兼容 endpoint（例如 DeepSeek）还须声明 `GPU_AGENT_STORE_FALSE_SUPPORTED=1`，否则 fail closed；所有
Responses 请求发送 `store=false`；兼容适配器按其协议处理，不声称这等于零数据保留。

`GPU_AGENT_KNOWLEDGE_INDEX` 指向 T05 已 ingest 的本地索引，
`GPU_AGENT_KNOWLEDGE_VERSION` 使用 `cuda=VERSION;compute-sanitizer=VERSION`。
版本缺失或不兼容时返回知识证据不可用，不会猜测版本或联网搜索任意 URL。
V2 已提供完全离线的 BM25、确定性向量余弦和 hybrid RRF 三种检索候选；29 条公开
development 标注的 hit@k/延迟比较、默认方法选择边界和复现代码见
[V2 本地检索比较](docs/retrieval-comparison.md)。该比较不读取 private holdout。

每次诊断默认最多 40 次物理 LLM 请求、38 个 Agent 步骤，预留最终诊断和补丁。
plan、diagnose、patch 各有一次格式/内容重试，均计入物理调用次数。
CLI 需显式使用 `--allow-paid-calls`，否则不发送模型请求；只记录费用，没有美元上限。
SDK 自动重试关闭；timeout 记为 `UNCERTAIN`，不盲目重放。
最多注册一个 candidate；验证失败不再次生成补丁。未注册可信 Oracle 的 standalone
程序验证结果为 `INCONCLUSIVE / ORACLE_UNAVAILABLE`。Fake 测试通过不能作为真实模型、
GPU 或容器验收；M1 仍需通过带 `--require-live` 的真实 OOB 闭环。

不要把 API key 写入仓库、命令历史或报告。推荐放在权限为 600 的用户配置文件中，
仅在控制器 shell 内加载；密钥永远不会传入候选容器。

## 四工具、私有 corpus 与评测

`memcheck` 是 race/init/sync 的前置内存安全检查；strict 模式冻结并执行四工具集合。
必需工具 unsupported、timeout、截断或没有完整 summary 都会阻止 `VERIFIED_FIXED`。
私有 corpus 必须位于仓库和 public RunStore 之外，普通用户只能复现 public development
验证。case 只有在 clean Oracle/required checks 与 mutant target finding 均有真实 run ID
时才能注册。

五组评测 A–E 的协议见 [evaluation/protocol.md](evaluation/protocol.md)。完整批次至少为
`24×5×3=360` 个单元；模型调用必须经过显式授权，费用只记录，不设置金额上限。
当前状态与边界见 [验收](docs/acceptance.md)、[评测](docs/evaluation-report.md) 和
[限制](docs/limitations.md)。生产执行必须按
[V2 操作员手册](docs/v2-operator-runbook.md) 依次完成外部 Ed25519 signer、16+8 注册、
模型调用授权、240+120 评测、120 条盲评标签、score/collect/freeze/derive/check
及构建发布。仓库不提供生产 signer，也不包含生产私钥；在真实原生证据通过最终 release
check 之前，V2 仍为关闭状态，不得发布。

## M0：真实 clean kernel 验收

在仓库根目录、专用 Conda 环境中运行：

```bash
python -I -m pytest tests/gpu/test_clean_kernel.py --require-live --gpu-run-root runs/m0 -q -s
```

正常用例会输出一个唯一 run ID，验证长度 1、257、1025 的 vector add，保留源码快照、编译 argv、二进制 hash、输入、输出、日志及验收结果。另两项 GPU 用例注入清理/证据读取故障，预期得到失败 run；它们用于验证不会误记成功，记录留在 pytest 临时目录。

`runs/` 被 Git 忽略。`runs/m0/<run_id>/manifest.json` 保存注册的 artifact refs 和状态审计事件；原始 blobs 不进入仓库。将状态审计与 manifest 一起原子提交，避免独立 audit 文件的双写不一致。

本地后端只接受包内 `trusted_sources.json` 登记的完整源码/harness/parser hash 集合。修改任何文件后不能自动重新批准并运行；必须先审核。编译使用固定参数和干净环境，显式选择 `Settings.host_compiler`，不会继承 `NVCC_PREPEND_FLAGS`、`LD_PRELOAD` 或 API key。输入/二进制引用绑定到具体执行记录。

这不是不可信代码沙箱。T02 不提供任意源码 CLI，不运行模型补丁；LocalBackend 的 Sanitizer 请求明确返回 `UNSUPPORTED`，不是 `CLEAN`。后续 T03 才建立隔离编译/执行和 memcheck OOB 证据。

真实验证与安装调整见 [T01 记录](docs/t01-validation.md) 和 [T02/M0 记录](docs/t02-validation.md)。设计与依赖顺序见 [V2 规格](PROJECT_SPEC_CN_V2.md) 和 [实施计划](IMPLEMENTATION_PLAN_CN_V2.md)。不按开发天数安排任务。


## Agent Planner 自适应实验（2026-11-14，独立研究分支）

为了同时研究严格修复率和 Planning 成本，新增**仅公开开发模式可选**的
\`GPU_AGENT_PLANNER_POLICY=adaptive-v1\`。它只跳过强制首轮 memcheck 的
模型规划，以及规则 Router 与证据门禁都同意时的结束规划；模糊故障仍
由 LLM 自主决定调查动作。遇到已完成但 schema 无效的模型规划，
可经规则路由器做安全降级，**不会**重试超时等不确定请求。

\`\`\`bash
export GPU_AGENT_ORCHESTRATOR=langgraph
export GPU_AGENT_COMPONENT_LAYER=langchain
export GPU_AGENT_PLANNER_POLICY=adaptive-v1  # 默认 baseline
\`\`\`

每步选择来源写入 public RunStore 的 \`agent/adaptive-policy/*.json\`；
实际工具调用仍由 \`decide_action\`、预算、白名单和 GPU 隔离检查。
拒绝在 D / A–C / 正式评测和旧 Orchestrator 上启用，不影响冻结的
历史实验。**目前只是一项需对照检验的研究假设，未证明提高修复率。**

## 集成版：LangGraph + LangChain + Fullstack（2026-11-14）

此集成分支从 PR #8 的后端出发合入 Fullstack Console；修复 Web
`on_run_created` 在 LangGraph public-repair 路径未透传的问题。仅在本分支
完成新 SHA 的 API、前端与 GPU 验收后才能报告产品级交付。

- 开发机 Mac：`cd dashboard && npm ci && npm run test && npm run build && npm run lint`。
- Python 环境：`python -m pip install -r requirements.lock`，
  `python -m pip install --no-deps -e '.[dev,web]'`。
- 启用图与组件：`GPU_AGENT_ORCHESTRATOR=langgraph`，
  `GPU_AGENT_COMPONENT_LAYER=langchain`；与旧流程等预算对照前不改默认组件。
- Web：`gpu-agent web --host 127.0.0.1 --port 8000`，当前仍只接受受控的 public case。
- 严格验证只读取独立 evaluator store，Graph 不能读取隐藏输入或绕过 policy。
- 不宣称生产级恢复/恶意 CUDA 沙箱/新的真实 GPU 成功率。

## LangChain Core 组件层（2026-11-13，开发分支）

在 LangGraph 状态图下可以显式启用 **LangChain Core / LCEL** 的模型和 RAG
适配层；默认保持原生组件，以便做真实的同预算对照：

```bash
export GPU_AGENT_ORCHESTRATOR=langgraph
export GPU_AGENT_COMPONENT_LAYER=langchain  # 默认 native
gpu-agent diagnose benchmarks/public/case_0001/public_input --allow-paid-calls
```

- `RunnableLambda` 连接 Planner、Diagnosis、Patch、Revision 的**既有**
  `LLMProvider`，响应仍由 Pydantic / Policy / Scope Guard 校验；
  不额外创建 `ChatOpenAI` 客户端，不改变请求量、重试与请求审计；
- `BaseRetriever` 包装版本锁定的离线官方 CUDA 文档检索。只把
  `Document` 作为展示/适配格式，回写的原始 `DocumentChunk` 必须与
  内置 `KnowledgeIndex` 的查询结果和内容哈希完全一致；
- 禁止 `LANGSMITH_TRACING`、`LANGCHAIN_TRACING`、
  `LANGCHAIN_TRACING_V2` 等自动外部追踪，防止源码/证据通过回调外传；
- 此路径**仅用于 D/E 公开开发实验**，未批准参与冻结 release/holdout；
  不开启 LangChain Agent 自主工具执行，不开启额外网络检索；
- `GPU_AGENT_COMPONENT_LAYER=native` 可以恢复原生组件。
  本组件层不提供 GPU Checkpoint/Resume，也不能代替独立 Verifier。

因此这里使用了真实 LangChain Runnable/LCEL 与 Retriever 抽象，
但**没有采用 LangChain 高层 `create_agent()` 或 `ChatOpenAI`**
替换已有安全模型传输。其收益必须通过实验比较验证。

## LangGraph 调查编排（2026-11-12 开发分支）

开发模式 D/E 的 Agent Loop 默认执行 LangGraph StateGraph（observe → plan →
authorize → execute_tool → observe，直至 diagnose 或 inconclusive）；A/B/C
固定证据采集的消融协议与严格验证器保持不变。公开 `repair` 顶层流程也由
LangGraph 拆成调查修复、公开检查门禁和独立 strict 验证三个节点。
LangGraph 仅调度节点，
**Policy、GPU Sandbox、预算、EvidenceRepository、RunStore 和私有 Oracle 仍由控制器持有**。

LangGraph 1.2.20 与全部可解析依赖已并入 `requirements.lock`。可设置
`GPU_AGENT_ORCHESTRATOR=legacy` 显式切回原循环以核对行为；错误的配置值直接拒绝。
目前 **没有配置持久化 LangGraph checkpointer**，因为 GPU/付费请求尚无可安全重放的
跨进程幂等事务。RunStore 已持久化执行证据，但不等于可以在任意节点安全 Resume。
历史冻结评测和真实 GPU 实验不因编排切换而自动成为新版本成绩。

## Repair V3：公开多轮 CUDA 修复与历史失败经验

当前正式评测 A–E 与 V2 release evidence 的合同未更改；公开开发模式可显式启用多轮
`gpu-agent repair`，失败候选能够在来源校验后重新调查，按原始源码生成后续补丁，
并通过真实 GPU 功能/四种 Sanitizer 自检。只有公开检查通过，才会进行一次隔离的
独立严格验证。现有 T06 的单候选描述属于较早阶段，不能当成 Repair V3 的候选限制。

```bash
gpu-agent repair benchmarks/public/case_0009/public_input --allow-paid-calls \
  --reinvestigate --unbounded-sanitizer-calls \
  --max-candidates 3 --max-reinvestigations 1 --max-llm-calls 40
```

`--unbounded-sanitizer-calls` 仅取消开发 V3 **独立 Sanitizer 调查次数**限制；
仍保留模型请求、任务截止时间、Agent step 数及补丁候选数限制。
候选自身的四工具自检调用另计，且不能替代私有严格验证。

历史失败经验库为**可选只读索引**。从明确选择的、已完成且有来源绑定的公开失败
RunStore 导出，然后冻结、合并两个索引并显式启用：

```bash
python scripts/export_repair_memory.py --public-store /runs/public-A \
  --run-id <public-run-id> --output /path/public-A.json
python scripts/export_repair_memory.py --public-store /runs/public-B \
  --run-id <public-run-id> --output /path/public-B.json
python scripts/merge_repair_memory.py --index /path/public-A.json \
  --index /path/public-B.json --output /path/frozen-memory.json
export GPU_AGENT_REPAIR_MEMORY_INDEX=/path/frozen-memory.json
# 此后显式运行 Repair V3；不要把经验自动写回当前评测批次。
```

经验项保留算法/故障类别、原公共 Run ID、候选 Hash 和结果证据 Hash，
只作为低信任的修复提示，不能宣称是 NVIDIA 规范、当前观测或官方验证结论。
冻结索引的语料 Hash 会写入 `repair/experience-retrieval.json`；正式对照需明确
禁用或冻结 Memory，避免同批次信息泄漏。历史示例：
[冻结公开失败经验](docs/repair-log/artifacts/2026-10-09-repair-memory/merged_public_failures_20261009.json)。

真实 GPU 工程闭环验证已覆盖“首个候选失败→重新调查→第二候选→独立验证”。
[复杂案例探索对照](docs/repair-log/2026-10-09-v2-v3-complex-pilot.md)
得到 V2 2/2、V3 1/2 的单次结果，不能声称 V3 成功率更高。
[记忆跨算法试用](docs/repair-log/2026-10-09-repair-memory-live-stencil.md)
真实加载并检索 3 条经验后修复 Stencil 且通过严格验证，但尚没有
memory-on/off 的因果增益对照。完整来源边界见
[V3 操作员手册](docs/v2-operator-runbook.md#repair-v3) 与
[按日期的失败与修复记录](docs/repair-log/README.md)。
