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
