# Agentic GPU Debugger

面向 CUDA 故障的证据驱动诊断工具。Python 负责 Agent/CLI/证据与验证编排，CUDA C++ 负责真实 kernel 和可信 host harness。

当前已实现证据驱动诊断、单一模型补丁、Docker GPU 隔离、四种 Compute
Sanitizer、public/private Oracle、strict 验证、mutation 注册门、评测记录和 release
gate。真实 DeepSeek OOB 闭环与四工具验收已通过；完整 16+8 corpus 和五模式付费评测
尚未完成，因此当前不是可发布版本。

## 独立环境

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

已注册 `gpu`、`container`、`live_llm`、`release` 标记。`--require-live` 将带这些标记的 skipped 测试变为失败；收集阶段 skip 也失败，防止缺少必需环境时假通过。release 选择面覆盖隔离、四工具、private Oracle、live LLM、A–E、corpus 和 manifest；计数由保存的 pytest JUnit 结果派生。当前 release manifest 不存在，离线测试只证明缺证据时门禁正确关闭。

## T06 诊断与单候选工作流

`gpu-agent diagnose PATH` 接受一个 CUDA 源文件或含 `kernel.cu` 的目录，输出实际
`run_id`。源码被快照为固定的 `kernel.cu`；匹配 vector harness 接口时仅加入控制器
指定的三个公共 harness 文件，否则走单文件隔离编译。模型没有 shell、任意路径、
URL 或验证器控制权。standalone 模式需要重新构建包含 `build_standalone` 操作的
container runner 镜像；现有四文件 vector 协议保持不变。

```bash
gpu-agent diagnose benchmarks/public/case_0001/public_input
# 将上一条命令输出的实际 run_id 用于下列命令：
gpu-agent verify RUN_ID --generated-candidate --strict
gpu-agent report RUN_ID
# 或注册一个人工 unified diff；与 --generated-candidate 二选一：
gpu-agent verify RUN_ID /absolute/path/to/candidate.diff --strict
```

远程 provider 使用官方 OpenAI Python SDK Responses structured outputs，必须显式配置
`OPENAI_BASE_URL`、`OPENAI_MODEL` 和控制器环境中的 `OPENAI_API_KEY`。本程序不自动读取
`.env`，也不索取或打印密钥。缺少配置时保存 `LLM_UNAVAILABLE` 结果，LLM 调用数为零。
兼容 endpoint（例如 DeepSeek）还须声明 `GPU_AGENT_STORE_FALSE_SUPPORTED=1`，否则 fail closed；所有
实际请求发送 `store=false`，不声称这等于零数据保留。

`GPU_AGENT_KNOWLEDGE_INDEX` 指向 T05 已 ingest 的本地索引，
`GPU_AGENT_KNOWLEDGE_VERSION` 使用 `cuda=VERSION;compute-sanitizer=VERSION`。
版本缺失或不兼容时返回知识证据不可用，不会猜测版本或联网搜索任意 URL。

每次诊断最多 6 次物理 LLM 请求（含整个 run 唯一一次格式重试），预留最终诊断和补丁，
最多 4 次 planner 请求。SDK 自动重试关闭；timeout 记为 `UNCERTAIN`，不重放。
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
`24×5×3=360` 个单元；没有显式 API 费用上限时 runner 在第一次外部调用前停止。
当前状态与边界见 [验收](docs/acceptance.md)、[评测](docs/evaluation-report.md) 和
[限制](docs/limitations.md)。

发布声明不能只填写一个 JSON：`ReleaseEvidenceIndex` 会重新读取 public/evaluator
RunStore，验证终态、artifact hash、当前 commit、toolchain/corpus/model config、完整
schedule/attempt/record 和真实 pytest 结果。private holdout 在 public store 中只能使用
opaque alias；映射保存在 evaluator store。当前 corpus、alias 生产路径、价格证明和付费
批次都未完成，所以项目是可运行原型与评测基础，不是 Portfolio-ready release。

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
