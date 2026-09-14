# Agentic GPU Debugger V2 实施计划

**状态：** 用户已批准执行；T01 已实现并完成环境元数据验证，T02 尚未开始。M0 尚未完成；真实 GPU 编译运行不能由 T01 替代。记录见 [T01 验证记录](docs/t01-validation.md)。

**Goal：** 在真实 RTX 4090 Laptop 上完成可复现的 CUDA 诊断、单次候选补丁、隔离验证和五组评测。

**Architecture：** 先交付一个 global OOB 纵向切片。受控执行器提供事实，Agent 在预算内获取证据，独立 Verifier 使用可信 Oracle 判定结果。后续沿现有切片扩展四种 Sanitizer、mutation 和评测。

**Tech Stack：** Python 3.11、Pydantic 2、Typer、pytest、Ruff、mypy、CUDA 12.8.x、Compute Sanitizer、Docker/NVIDIA Container Toolkit；RAG 首先采用本地词法检索；模型通过 Provider 接口调用。

**Spec：** [PROJECT_SPEC_CN_V2.md](PROJECT_SPEC_CN_V2.md)，基准提交 `33a6f23`。

**执行方式：** 当前对话串行维护主线，使用 `superpowers:executing-plans`，按任务完成测试、检查与本地提交。开发模型采用用户选择的 Astra/high；产品运行模型独立配置。无需为了每个任务创建新对话或并行 Agent。

**推进规则：** 按依赖与验收推进；不设置日程。M0、M1 及后续各里程碑交付可运行结果后评审。实际安装、模型调用和 GPU 实验产生的结果才能标记完成。

## 1. 当前事实与实施约束

计划编写时仓库只有 V2 规格，工作区干净。现场探测得到：

- GPU：RTX 4090 Laptop，SM 8.9；Driver 580.178.04。
- 当前 shell：`/usr/bin/nvcc` 11.5，Compute Sanitizer 2021.3.1。
- Conda：`/home/you/anaconda3/condabin/conda`。
- Docker：29.1.3；GPU 容器可用性仍需在 T03 实测。

继承规格的约束：

1. Python 3.11 或 3.12 专用环境；本计划选择 3.11。
2. CUDA 12.x 支持 `sm_89`；首选同一 12.8.x release 的本机和容器工具链。
3. 普通 Agent 不提供 Shell、任意 URL、宿主路径或 checker 执行入口。
4. 原始源码只读，每个 candidate 和 verification run 都有独立 ID/hash。
5. 每次 diagnosis 最多一个候选补丁；验证失败不会自动再次修补。
6. `RunStore` 保存运行 artifacts；Evaluator 私有数据位于仓库和 Agent workspace 外。
7. 用户源码、模型候选以及自动 mutation 代码必须在隔离后端编译和运行。
8. `VERIFIED_FIXED` 必须具备真实执行、Public/Private Oracle 和必需检查的完整证据。
9. Agent 预算：8 steps、4 sanitizer calls、3 RAG calls、5 source reads、6 LLM calls。
10. 16 个 public cases（每类至少 4 个）与 8 个 private cases 是 M6 发布门槛，不是早期切片门槛。
11. 无 GPU/API key 时可以验证单元测试；不得把跳过的集成测试计为验收成功。
12. FastAPI、Web UI、Multi-Agent、性能优化、CI bot、自动 reducer 不进入本计划。

## 2. 将规格落实为可执行规则

以下是本计划明确记录的实施解释，评审时与规格一起核对。

### 2.1 实际验证结果的含义

`VERIFIED_FIXED` 仅表示指定 candidate 在记录的输入集、Oracle 和检查集合下通过；不表示对所有输入证明正确。报告必须包含 verification scope、case/input-set hash 和工具版本。

退出码 0 不足以判定 clean。Sanitizer 空输出、截断输出、未启动目标程序、格式不支持均不能得到 `CLEAN`。diagnose 对运行成功的程序仍允许调查，因为 race/init/sync 缺陷可能没有普通 runtime 错误。

### 2.2 必需性与支持度

规格的 `CheckApplicability` 保留供展示，内部补充正交字段：

```python
class CheckRequirement(BaseModel):
    tool: SanitizerTool
    required: bool
    support: Literal["SUPPORTED", "UNSUPPORTED", "NOT_APPLICABLE"]
    reason_code: str
```

`required=True` 遇到 unsupported 不会丢失强制性，而是阻断成功判定。Strict 将所有已知适用工具纳入本次 required set；中途工具故障不能自动降级到 Standard。未知支持状态须探测，不能默认跳过。

### 2.3 私有输入、Oracle 与 harness

诊断 Agent 不能读取 holdout。验证时 Verifier 必须将当前输入传给 candidate，否则程序无法计算；因此 candidate 可以看到当前实际输入，但不能看到生成种子、其余 suite、期望结果或 Oracle 实现。通过每个输入新建进程/隔离 workspace 防止跨输入保存状态。

Oracle 在可信控制端比较候选输出。Private input/output、Sanitizer 私有执行日志不进入 Agent 可读 RunStore；普通报告只返回汇总和 opaque input-set ID。Private 细节进入同一 artifact 存储机制的独立 Evaluator root。

第一条 slice 把可修改代码限定为 `kernel.cu`；可信 host harness 独立编译、只读挂载。仅允许 `.cu/.cuh` 扩展名不足以保护 harness，Patch Guard 必须同时检查逐文件 allowlist。用户自带 `main()` 的自由程序可以诊断；没有可信 Oracle 协议时 verify 返回 `INCONCLUSIVE / ORACLE_UNAVAILABLE`。

### 2.4 实际执行隔离

隔离覆盖编译和运行。Docker 根文件系统只读；编译和输出写入限额 tmpfs。Host RunStore 不作为 candidate 的可写挂载。控制端读取产物时拒绝符号链接、特殊文件、路径穿越和过大文件。

CPU/PID/RAM、tmpfs 与日志限制不能形成 GPU 显存硬配额；单卡串行调度、超时、故障后健康检查是本项目能实际提供的约束。禁用自动 GPU reset，不影响同机其他任务。

### 2.5 预算和结束

diagnosis wall-time 默认 600 秒，单次 build 120 秒、run 30 秒、sanitizer 120 秒、LLM 请求 60 秒；实际 timeout 取阶段限额与剩余预算较小值。这些是执行超时参数，不是开发工期。

6 次 LLM 调用中预留 1 次最终诊断和 1 次补丁生成，最多 4 次 Planner 调用。重试也计费计预算；无足够预算时停止。verification 使用单独 600 秒上限和输入/检查矩阵，不消耗 diagnosis 的 4 次 Sanitizer 调用额度。

## 3. 文件布局与依赖

文件按功能切片逐步创建，不在 T01 生成空目录森林。`__init__.py` 随包首次创建。

```text
pyproject.toml                 安装入口、测试标记、静态检查配置
environment.yml                项目 Python 环境
environment.lock.txt           实际解析的 Conda explicit lock
requirements.lock              首次解析后固定 Python 依赖
.gitignore                     排除 env、runs、密钥、private data
.env.example                   只有配置名称，不含凭据
README.md                      当前可用能力和运行方式
src/gpu_agent/
  cli.py                       Typer 薄入口
  config.py                    类型化配置与路径
  environment.py               工具链探测、兼容性门禁
  contracts.py                 通用 ID、状态、ToolResult
  store.py                     原子 artifacts、manifest、审计
  execution/
    models.py                  类型化执行请求和结果
    backend.py                 ExecutionBackend Protocol
    process.py                 有界进程读取、取消、超时
    local.py                   仅审核 hash 的本机执行
    isolated.py                Docker 执行、导入导出和清理
  evidence/
    models.py                  EvidenceBundle、Finding、SourceLocation
    sanitizer.py               四工具解析与结果判定
    repository.py              public 视图和 artifact 能力校验
  knowledge/
    models.py                  DocumentChunk、RetrievalResult
    ingest.py                  官方 allowlist 和版本化章节
    retrieve.py                首个词法检索实现
  agent/
    models.py                  Action、Budget、DiagnosisResult
    policy.py                  合法动作、预算、完成条件
    provider.py                Provider Protocol 和远程适配
    prompts.py                 版本化 Planner/Diagnosis/Patch 提示
    orchestrator.py            observe/act 循环
    rule_router.py             确定性路由 baseline
  patching.py                  scope、diff、candidate provenance
  verification/
    models.py                  OracleResult、CheckRequirement、Verdict
    oracle.py                  数值比较、可信 checker registry
    policy.py                  必需检查、Standard/Strict
    engine.py                  candidate 重建、测试与 verdict
  service.py                   CLI 共用 diagnose/verify/report
  reporting.py                 public Markdown/JSON 输出
  benchmark/
    models.py                  CaseManifest、mutation/split metadata
    builder.py                 clean→mutant→validated corpus
    evaluation.py              五组执行和盲评导出
    metrics.py                 指标、分母、重复实验汇总
containers/Dockerfile          CUDA devel 和可信 harness 环境
containers/toolchain.lock.json 已实测的 tag/digest/flags
benchmarks/harness/             可信 host harness
benchmarks/public/              neutral case ID 的公开输入
benchmarks/development_truth/   已公开开发集的 Oracle/参考答案
knowledge/sources.json          官方来源和版本
tests/unit/                    无 GPU 无 API 的行为测试
tests/integration/             进程/文件/容器/provider 测试
tests/gpu/                     必须真实执行的 CUDA 检查
tests/e2e/                     完整 diagnose/verify 与发布门禁
docs/                          操作、决策、验收和评测报告
```

依赖顺序：

- M0：T01 环境诊断 → T02 clean CUDA 与产物记录。
- M1：T03 隔离 OOB → T04 Oracle/候选验证 → T05 官方知识 → T06 Agent 闭环。
- M2：T07 四类真实 seed。
- M3：T08 动态调查、规则路由和故障恢复。
- M4：T09 Strict、可信 checker 和完整 verdict。
- M5：T10 mutation/corpus。
- M6：T11 五组实验 → T12 发布文档与验收。

## 4. 测试与提交约定

每个任务遵循失败测试→最小实现→相关测试→静态检查→明确文件暂存→本地提交。下方例子定义需要实现的行为，测试辅助工厂由对应任务创建，不代表已经存在。

执行命令均在项目专用 Conda 环境中运行。代码改动后的常规检查：

```bash
python -m pytest tests/unit -q
python -m ruff check src tests
python -m mypy src/gpu_agent
git diff --check
```

T01 在 `pyproject.toml` 注册 `gpu`、`container`、`live_llm`、`release` 标记和 `--require-live` pytest 参数。日常缺能力可 skip；发布运行指定 `--require-live` 后缺能力必须 fail。`release` 检查测试清单的 expected/executed/skipped 数量，零案例和全 skip 不能通过。

每个任务只提交本任务明确路径；禁止 `git add .` 将 logs/private data 混入。任务结束记录 commit、真实验证命令、结果及尚未通过的硬件门禁。

## T01 / M0：项目环境与可解释的环境诊断

**执行记录：** 已创建独立 Python 3.11.16 环境，NVCC 12.8.93 / Compute Sanitizer 2025.1.0.0 实测可查询。完整 Toolkit 下载失败后改为同 release 的必需开发组件，详见验证记录。新增 `__main__.py` 支持隔离 Python 模式，增加门禁/进程行为测试和实际环境 JSON；没有提前实现 T02 执行器。

**交付：** 可安装包和 `gpu-agent env`；当前 CUDA 11.5 被明确识别为不满足项目基线。

**创建：** `pyproject.toml`、`environment.yml`、`environment.lock.txt`、`requirements.lock`、`.gitignore`、`.env.example`、`README.md`、`src/gpu_agent/{__init__,cli,config,environment}.py`、`tests/conftest.py`、`tests/unit/test_environment.py`。两个 lock 文件在实际解析后生成，不预填版本。

**契约：** `probe_environment(settings: Settings) -> EnvironmentReport`；`EnvironmentReport` 包含 `ready: bool`、`reason_codes: list[str]`、`toolchain: ToolchainManifest`。`Settings` 显式保存 CUDA bin/root 路径，不能从 torch 版本推断。

- [x] 先检查独立 prefix `/home/you/conda_env/agentic-gpu-debugger`；存在则探测，禁止覆盖。不存在时创建 Python 3.11/pip 环境，激活后安装测试依赖。初始化包配置：运行依赖 Pydantic 2、Typer，开发依赖 pytest/Ruff/mypy；运行 `python -m pip install -e '.[dev]'`，此时不要求 CUDA 已安装。

```bash
conda create --prefix /home/you/conda_env/agentic-gpu-debugger python=3.11 pip
conda activate /home/you/conda_env/agentic-gpu-debugger
```

- [x] 写环境测试和 Typer CLI 测试；`fake_probe` 在本测试模块使用注入的 command→stdout 映射，不触碰宿主 PATH。

```python
def test_runtime_version_does_not_replace_toolkit(fake_probe):
    report = fake_probe(nvcc="11.5", gpu_arch="8.9", torch_cuda="12.8")
    assert report.ready is False
    assert "TARGET_ARCH_UNSUPPORTED" in report.reason_codes
```

- [x] 运行 `python -m pytest tests/unit/test_environment.py -q`，确认缺实现导致失败。
- [x] 在专用环境查询实际可用的 CUDA Toolkit 版本。

```bash
conda search -c nvidia 'cuda=12.8*'
```

- [x] 根据 search 实际返回选择并锁定 12.8.x Toolkit，不使用仅 runtime 的 `cudatoolkit` 作为替代。依赖解析先 dry-run，完成后保存 `environment.lock.txt` 和 Python 依赖 `requirements.lock`，检查其中无私有索引凭据或不可移植的本机 editable 路径；不得改动其他 Conda 环境、全局 PATH 或 GPU driver。发行包不含 Sanitizer 时显式补齐配套组件。
- [x] 实现绝对路径探测 `nvcc --version`、`nvcc --list-gpu-code`、`compute-sanitizer --version`、Driver/SM/GCC；枚举丢失、损坏与不兼容原因。元数据未知使用 null，不能填猜测值。
- [x] 运行单元检查与 `gpu-agent env --json`。安装前报告正确的 not-ready，安装后必须实测 ready；只打印版本不等于 GPU 工作正常。
- [x] 提交上述文件，消息 `feat: inspect project CUDA toolchain`。

**通过标准：** 稳定 CLI、3.11 环境与 12.8.x 工具路径有真实记录；若安装权限/下载受阻，T01 保留准确未完成状态，可继续无 GPU 单元任务，M0 不标完成。

## T02 / M0：受信任 clean kernel 的真实执行和证据保存

**依赖：** T01。

**创建：** `src/gpu_agent/contracts.py`、`store.py`、`execution/{models,backend,process,local}.py`、`benchmarks/harness/{vector_io.cpp,vector_api.h}`、`benchmarks/public/case_0000/public_input/kernel.cu`、`tests/unit/test_store.py`、`tests/integration/test_process.py`、`tests/gpu/test_clean_kernel.py`。

**类型与契约：**

- `ArtifactRef(id, sha256, visibility, relative_path, byte_count)`；visibility 为 public/evaluator。
- `ToolResult[T](request_id, payload: T, exit_code, timed_out, log_refs, elapsed_ms, truncated)`。
- `WorkspaceRequest(source_manifest, trust_level)`、`WorkspaceHandle(id)`；`BuildRequest(workspace_id, target_arch)`、`ExecutionRequest(workspace_id, stdin_ref, timeout_seconds)`、`SanitizerRequest(workspace_id, tool, stdin_ref, timeout_seconds)`。
- `BuildResult(success, binary_ref)`、`ExecutionResult(output_ref, runtime_status)`、`SanitizerResult` 在 T03 补充。
- `ExecutionBackend.prepare/build/run/run_sanitizer/cleanup` 按规格 §8.1 类型签名；不支持的 sanitizer 请求返回 typed unsupported。
- `RunManifest(id, kind, parent_run_id, status, current_phase, last_completed_phase, artifact_refs)`，不在 manifest 内嵌其自身内容 hash。
- `RunStore.create_run(kind, parent_run_id=None) -> RunManifest`；`load(run_id) -> RunManifest`；`put(run_id, name, content: bytes, visibility) -> ArtifactRef`；`read(ref) -> bytes`；`transition(run_id, status, phase) -> RunManifest`。
- `ProcessExecutor.execute(argv: list[str], cwd: Path, timeout_seconds: float, max_log_bytes: int) -> ProcessCapture` 仅内部可信代码调用，不注册 Agent Tool。
- `ProcessCapture(exit_code: int | None, stdout: bytes, stderr: bytes, timed_out: bool, elapsed_ms=0, truncated=False, tool_error=None)`；生产执行必须填写实际耗时。

- [ ] 先定义 `RunStatus`、`CurrentPhase` 与终态不变量的测试。

```python
def test_terminal_run_has_no_active_phase(store):
    run = store.create_run("diagnosis")
    store.transition(run.id, "RUNNING", "COMPILING")
    done = store.transition(run.id, "FAILED", None)
    assert done.current_phase is None
    assert store.load(run.id).status == "FAILED"
```

- [ ] `store` fixture 在 `tests/conftest.py` 创建 `RunStore(tmp_path / "runs")`。添加穿越、symlink、hash mismatch、写入失败保留旧 manifest 的测试并运行失败版本。
- [ ] ProcessExecutor 用参数数组、`shell=False`、独立进程组；同时增量读取 stdout/stderr、达到 2 MiB 保留截断标记并继续排空/丢弃，防止管道死锁。timeout 终止进程组，收集退出结果；不要先 `capture_output` 无限缓存再截断。
- [ ] 真实 subprocess 测试覆盖大输出、子进程、超时、UTF-8 损坏；不运行 fork bomb，使用有限两个进程的夹具。
- [ ] 定义 host harness 的标准协议：stdin JSON `{n, a, b}`；stdout 恰好一个 JSON `{dtype:"float32",shape:[n],values:[...]}`。使用审核的 JSON parser/header，固定版本和来源；每次 CUDA API、launch、synchronize 都检查错误。候选只提供 `run_vector_add` 符号，可信 harness 的输入验证与输出编码不可改。
- [ ] clean `kernel.cu` 提供长度保护、固定 block 256、输出 a+b。`LocalBackend` 只接受审核登记的整个源码/harness hash，不信任目录名或 `trust_level` 用户参数。

```bash
nvcc -std=c++17 -lineinfo -arch=sm_89 kernel.cu vector_io.cpp -o vector_add
python -m pytest tests/unit/test_store.py tests/integration/test_process.py -q
python -m pytest tests/gpu/test_clean_kernel.py --require-live -q
```

- [ ] 命令在 task workspace 执行，实际路径由 backend 组装；测试 clean 长度 1、257、1025，CPU 计算参考结果。保存 compile argv、二进制 hash、结果和日志，验收脚本不得只检查 stdout `PASS`。
- [ ] 提交任务文件，消息 `feat: execute and record trusted CUDA workload`。

**通过标准：** M0 clean kernel 在 SM 8.9 实际编译和正确计算；日志与 artifacts 可追踪，原始输入不变。

## T03 / M1：隔离编译、memcheck 与一个真实 OOB

**依赖：** T02。

**创建：** `containers/Dockerfile`、`containers/toolchain.lock.json`、`docs/execution-environment.md`、`execution/isolated.py`、`evidence/{models,sanitizer,repository}.py`、`benchmarks/public/case_0001/public_input/kernel.cu`、`tests/integration/test_isolation.py`、`tests/unit/test_sanitizer.py`、`tests/gpu/test_oob.py`。

**修改：** `execution/models.py`，补充 `SanitizerResult(findings, completed, parser_version, check_outcome)`；`Finding(tool, category, kernel, source_location, raw_ref)`；`SourceLocation(path, line, function)`；`EvidenceBundle` 使用规格 §8.3 字段。

**契约：** `parse_sanitizer(tool: SanitizerTool, capture: ProcessCapture) -> SanitizerResult`；`EvidenceRepository.public_view(run_id) -> EvidenceBundle`。

`SanitizerTool` 在本任务定义为 StrEnum，成员为 MEMCHECK/RACECHECK/INITCHECK/SYNCCHECK，值为对应的小写工具名；先实现 memcheck，另外三个在 T07 实现。

- [ ] 首先写解析空日志不可 clean、0 退出码但存在 finding、工具崩溃、log truncation、source line 提取测试；fixture 标记为合成单元日志，不能作为 benchmark 证据。

```python
def test_empty_sanitizer_output_is_not_clean():
    capture = ProcessCapture(exit_code=0, stdout=b"", stderr=b"", timed_out=False)
    result = parse_sanitizer(SanitizerTool.MEMCHECK, capture)
    assert result.check_outcome == "TOOL_ERROR"
```

- [ ] 查实际 Docker GPU runtime 配置与运行容器。安装 NVIDIA Container Toolkit 需要系统权限时，准备具体命令及配置备份；需要重启 daemon 时先列出在运行容器，避免打断现有 workload。不能通过 `--privileged` 回避工具兼容性。
- [ ] 选择实际可拉取的官方 CUDA 12.8.x devel 镜像，拉取/检查后锁定 digest。`toolchain.lock.json` 由真实 inspect 结果产生，不能提前编造 sha256。镜像构建期可联网安装审核依赖；候选 build/run 期网络关闭。
- [ ] Docker 请求固定 non-root、network none、cap-drop ALL、no-new-privileges、read-only root、4 CPU/4 GiB RAM/64 PID、受限 tmpfs 与单 GPU；所有值记录在 policy。编译 tmpfs 默认 1 GiB、运行 tmpfs 256 MiB，日志 driver 关闭或严格限额。源码输入只读，产物经过上限校验导出；Docker socket/API key/主目录/private root 不挂载。
- [ ] 给 container 加唯一 run label，取消只停止匹配当前 ID/label 的 container。超时不等于 kill Docker CLI：显式停止对应容器再清理。删除仅限本次临时 container/workspace。
- [ ] OOB kernel 使用固定 block 256、`n=257` 分配 n 个元素并省略 index guard；不假设普通执行必须非零。实际捕获不到 OOB 时检查样例、工具链和日志，不修改期望让假样例通过。

```bash
compute-sanitizer --tool memcheck --error-exitcode 86 ./vector_add
python -m pytest tests/integration/test_isolation.py --require-live -q
python -m pytest tests/gpu/test_oob.py --require-live -q
```

- [ ] 上述命令由 isolated backend 组装运行。memcheck logs 与 program output 分开保存；finding 存在优先于退出码，完整 clean summary 和正常结束共同决定 clean。
- [ ] 隔离测试使用 benign probe 检查 UID、capabilities、network、只读挂载、外部 canary 不可读、日志上限、timeout container 消失。Agent public view 不能读取 sibling run/evaluator artifacts；不把 `.git` 复制入 workspace。
- [ ] 提交任务文件，消息 `feat: isolate CUDA execution and capture OOB evidence`。

**通过标准：** OOB finding 真正复现，clean 样例仍通过，同一隔离配置兼容 Sanitizer；不支持时 M1 阻断，不能切到宿主执行模型代码。

## T04 / M1：可信 Oracle、补丁边界与可审计验证

**依赖：** T03。

**创建：** `patching.py`、`verification/{models,oracle,policy,engine}.py`、`benchmarks/development_truth/case_0001/{case.json,reference.cu}`、`tests/unit/test_oracle.py`、`tests/unit/test_patch_scope.py`、`tests/unit/test_verdict.py`、`tests/gpu/test_candidate_verification.py`。

**契约：**

- `PatchCandidate` 使用规格 §8.7 字段；`apply_candidate(source_snapshot, diff: str, allowed_paths: list[str]) -> PatchCandidate`。
- `NumericOracle(atol, rtol, allow_nan, allow_inf).check(actual, expected) -> OracleResult`。
- `VerificationObservation`：`build_ok`、`runtime_ok`、`original_finding_present`、`public_oracle_passed`、`private_holdout_passed`（均允许 None 表示未评估）、`required_evidence_missing: bool`、`new_blocking_findings: list[Finding]`。
- `decide_verdict(observation: VerificationObservation) -> VerificationVerdict` 是纯函数。
- `VerificationEngine.verify(original_run_id, candidate_id, mode) -> VerificationResult`；输入是 store 注册 ID，不能让 Agent 传入 verifier 私有路径。

- [ ] 先写浮点 shape/NaN/Inf/atol/rtol、缺输出、多余文本、假的 `PASS` 不能通过的单元测试。

```python
def test_clean_but_wrong_candidate_is_a_regression():
    observation = VerificationObservation(
        build_ok=True, runtime_ok=True, original_finding_present=False,
        public_oracle_passed=True, private_holdout_passed=False,
        required_evidence_missing=False, new_blocking_findings=[],
    )
    assert decide_verdict(observation) == VerificationVerdict.REGRESSION_DETECTED
```

- [ ] `tests/unit/test_verdict.py` 参数化覆盖：编译失败→NOT_FIXED；原 finding 保留→NOT_FIXED；unknown original→INCONCLUSIVE；required tool error→INCONCLUSIVE；所有完成且通过→VERIFIED_FIXED。
- [ ] 细化缺证据：因 candidate 编译失败而不执行后续检查属于 downstream not-run，不算额外基础设施缺证据。工具失败导致无法检查才是 required evidence missing，避免优先级把所有编译失败误判 inconclusive。原 finding 与新 finding 同时存在时顶层 NOT_FIXED，同时保留新问题。
- [ ] Scope Guard 限定 `kernel.cu`，拒绝绝对路径、`..`、symlink、rename、二进制 patch、改变 harness/include 外部路径、添加未列入 allowlist 的文件和 base hash 不匹配。只做支持范围内的 unified diff，失败即拒绝，不 fuzz 应用。验证前重新校验 candidate hash，使用只读 snapshot 编译。
- [ ] host side CPU reference 计算 a+b；public 输入之外，Evaluator 在独立 root 生成 1、31、32、33、255、256、257、1023、1024、1025 和固定私有 seed 的随机输入。传给 child 的只有当前 a/b/n，结果在 host 比较。
- [ ] 首先用审核的人工作者 patch 检验 Verifier（`generated_by=human`），另外运行保留 OOB、输出零、只针对 n=257 正确、无效语法的候选。每个输入启动独立任务；每个 holdout 输入执行 mandatory memcheck，错误后可提前终止并注明未运行项。
- [ ] 判定原 finding 时使用 tool/category/kernel/access signature 与补丁行映射；不能仅比较易漂移的行号。找不到原 finding 只有在工具确实完整检查了相同目标/输入时才能设为 false。
- [ ] 私有日志保留在 Evaluator root，RunStore public verification 只有通过数/检查状态/opaque suite hash。公开导出采用字段 allowlist，不靠关键词替换。

```bash
python -m pytest tests/unit/test_oracle.py tests/unit/test_patch_scope.py tests/unit/test_verdict.py -q
python -m pytest tests/gpu/test_candidate_verification.py --require-live -q
```

- [ ] 提交任务文件，消息 `feat: verify scoped candidates against trusted oracles`。

**通过标准：** sanitizer clean 的错误功能补丁会被拒绝；任何缺 Oracle 的普通用户程序不能报修复成功；验证 hash 与实际二进制来源匹配。

## T05 / M1：版本化官方知识与可引用检索

**依赖：** T03 的 evidence contract；T04 验证不依赖 RAG。

**创建：** `knowledge/sources.json`、`knowledge/{models,ingest,retrieve}.py`（后三者在 `src/gpu_agent/` 下）、`tests/unit/test_retrieval.py`、`tests/integration/test_knowledge_ingest.py`。

**契约：** `DocumentChunk(chunk_id, document_title, document_version, section_title, source_url, retrieved_at, content_hash, text)`；`RetrievalResult(chunks, corpus_hash, query)`；`KnowledgeIndex.retrieve(query: str, version: str, k: int=5) -> RetrievalResult`。

- [ ] 写稳定 chunk hash、版本过滤、CUDA 标识符保留、假引用拒绝的失败测试。

```python
def test_unknown_citation_rejected(index):
    result = index.retrieve("out of bounds memory", version="12.8", k=5)
    with pytest.raises(InvalidCitationError):
        validate_citations(["invented-id"], result)
```

- [ ] `index` fixture 在该测试模块由三条自写文档片段构建；`validate_citations(ids, result)` 与异常由 `knowledge/models.py` 定义。
- [ ] allowlist 从官方 Programming Guide、Runtime API、Sanitizer 文档和 NVIDIA Samples 开始。抓取工具仅准备知识库，Agent 查询本地索引；每次 redirect 重新核验域名/路径，设置响应大小和 timeout。用户提供 URL 不进入源列表。
- [ ] 按 heading 分段，保存 title/version/anchor/hash；chunk 文本上限与后续 embedding 实际 tokenizer 限额分离，不默认把 900 token 段塞进 256 token 模型。
- [ ] 首个词法检索使用 token 计数/倒排 BM25，可用经锁定的 `rank-bm25`；词法 tokenizer 保留 `__syncthreads`、`threadIdx.x`、CUDA API 名。查询由 public evidence 生成，不含 ground truth。
- [ ] 用实际官网内容运行 integration；离线库损坏、版本缺失返回 typed error/limitation，不能伪造 citation。只提交 sources manifest 和少量合规样例，raw corpus/cache 不入 Git。

```bash
python -m pytest tests/unit/test_retrieval.py -q
python -m pytest tests/integration/test_knowledge_ingest.py --require-live -q
```

- [ ] 提交任务文件，消息 `feat: retrieve versioned official CUDA evidence`。

**通过标准：** 一次 OOB 查询可返回人工验证相关的官方段落；citation 存在性和相关性分别检测，存在于检索结果不自动代表支撑结论。

## T06 / M1：Agent、单次模型补丁、CLI 完整闭环

**依赖：** T04、T05。

**创建：** `agent/{models,policy,provider,prompts,orchestrator}.py`、`service.py`、`reporting.py`、`tests/unit/test_agent_loop.py`、`tests/integration/test_provider_contract.py`、`tests/e2e/test_oob_flow.py`；修改 `cli.py`。

**契约：**

- `AgentAction` 为 Pydantic discriminated union，动作及 typed_arguments 对应规格 §11.1。
- `AgentBudget` 为本计划 §2.5 限额；`PolicyDecision(allowed, mandatory_actions, prohibited_actions, reason_codes, policy_version)`。
- `LLMProvider.plan(evidence, budget) -> AgentAction`；`diagnose(evidence) -> DiagnosisResult`；`propose_patch(public_source, diagnosis) -> str`（unified diff）。每次返回同时记录 model/usage/provider_request_id。
- `AgentOrchestrator.investigate(run_id) -> DiagnosisResult`；`ApplicationService.diagnose(source: Path) -> RunManifest`；`verify(run_id, candidate_id=None, strict=False) -> VerificationResult`；`report(run_id) -> str`。
- `FakeProvider(actions, diagnosis, diff)` 只用于测试，按序返回预设结果，记录全部收到的模型输入。

- [ ] 写 FakeProvider 请求 memcheck→检索→finish 的测试；检查非法 action、相同无收益动作重复、预算不足与凭据缺失。

```python
def test_runtime_success_still_allows_memcheck(oob_service):
    run = oob_service.diagnose(oob_service.public_source)
    assert "run_memcheck" in oob_service.recorded_actions
    assert run.status == "COMPLETED"
    assert oob_service.patch_provider_calls == 1
```

- [ ] `oob_service` 在 `tests/conftest.py` 组合 FakeBackend（runtime exit=0）、FakeProvider 和临时 Store/RAG；fixture 源码不含答案注释。增加 FakeProvider 输入快照测试：不含 reference.cu、case GT、private seeds、checker 源码或评测标签。
- [ ] M1 支持 memcheck/source/docs/finish/inconclusive；其他动作如果尚未实现返回 unsupported，不执行无意义空壳。phase、budget、typed registry 最小版本现在就要生效。
- [ ] 首个 remote provider 使用单一官方 SDK，在实现时核对当前支持的工具调用/结构化输出 API；根据用户已配置 endpoint/model 选择实际模型。没有凭据时直接返回 `LLM_UNAVAILABLE`，不展示或索取明文 key。控制进程读取 SecretStr，绝不传给 candidate/container。
- [ ] 固定 prompt 文件版本：源码/log/docs 均是数据，不作为指令；引用 source/artifact/chunk ID；诊断保留 facts/findings/docs/inferences。high confidence 不能绕过 evidence policy。
- [ ] 校验失败最多一次格式重试，计入同一 LLM budget；对可能已产生响应的 provider timeout 保留 uncertain invocation，不能无界重发。一次有效 diff 构成唯一 candidate；验证失败不再调用 propose_patch。
- [ ] 接通 CLI：`diagnose PATH` 自动保存至一个新 run 并尝试生成唯一 candidate；`verify RUN_ID --generated-candidate` 或 `verify RUN_ID CANDIDATE_PATH` 二选一。后者注册新 candidate 后执行同一 guard。`--strict` 同时可用；辅助 `env`、`report` 可工作。
- [ ] 不符合 vector/harness protocol 的自由源程序可隔离 build/run/diagnose；未注册可信 Oracle 时 verify 明确 inconclusive。
- [ ] 报告给出四层证据、single patch hash、verdict 条件、检查覆盖、未完成项和真实用量，不能记录自报 chain-of-thought；action rationale 使用简短决策依据。

```bash
python -m pytest tests/unit/test_agent_loop.py tests/integration/test_provider_contract.py -q
python -m pytest tests/e2e/test_oob_flow.py -m 'gpu and container and live_llm' --require-live -q
gpu-agent diagnose benchmarks/public/case_0001/public_input
```

- [ ] 上一个命令输出 run_id；将实际 ID 传给 `gpu-agent verify RUN_ID --generated-candidate`，保存真实报告。RUN_ID 为上一条结果的绑定变量，不写死或猜测。
- [ ] 提交任务文件，消息 `feat: complete agentic OOB diagnose and repair flow`。

**通过标准：** 一次真实模型生成的有效 patch 经隔离编译/Oracle/memcheck 验证；Fake 测试通过但 live 环节未运行时 M1 仍未完成。真实模型失败如实保存并调查原因，不把手工 patch 冒充模型结果。

## T07 / M2：四类故障 seed 与工具能力

**依赖：** M1。

**创建：** `tests/gpu/test_failure_families.py`、`tests/unit/test_check_requirements.py`、`benchmarks/public/case_0002/public_input/kernel.cu` 至 `case_0004`、对应可信 harness 与 `benchmarks/development_truth/case_0002` 至 `case_0004`；修改 `evidence/sanitizer.py`、`verification/policy.py`。

**契约：** `SanitizerTool = MEMCHECK/RACECHECK/INITCHECK/SYNCCHECK`；`plan_checks(target, mode, capability_report) -> list[CheckRequirement]`；`CheckOutcome` 同规格。

- [ ] 用真实版本日志建立解析 fixture，标记来源 Run/工具版本；先写 required+unsupported 仍阻断成功的测试。

```python
def test_required_tool_cannot_be_skipped(capability_report):
    capability_report.mark_unsupported("synccheck")
    checks = plan_checks("synccheck", "standard", capability_report)
    target = next(check for check in checks if check.tool == "synccheck")
    assert target.required is True
    assert target.support == "UNSUPPORTED"
```

- [ ] capability_report fixture 为受控字典包装，mark 方法仅测试辅助；实际能力由版本探测和工具结果得到，不采信模型判断。
- [ ] 分别实现：shared 地址冲突；global buffer 未初始化读取；在当前 GPU/工具版本下能够可靠报告的 barrier misuse。按 seed 实测行为定 expected evidence，不能假定删除 `__syncthreads()` 一定形成可重现错误。
- [ ] race/init/sync 工具不替代 memcheck。Policy 将内存安全预检作为这些检查的前置证据；发现 memcheck 问题先报告，不把后续无意义结果当 clean。
- [ ] race fixture 多次独立执行，预先选择重复数 5 并记录 detection rate；不能重跑到恰好出现一次后丢弃失败运行。

```bash
python -m pytest tests/unit/test_sanitizer.py tests/unit/test_check_requirements.py -q
python -m pytest tests/gpu/test_failure_families.py --require-live -q
```

- [ ] 提交新增 cases、解析器与测试，消息 `feat: validate four CUDA failure families`。

**通过标准：** 每一类 buggy finding 和 clean reference 都经同一隔离配置验证，不能把某个不稳定或 unsupported 用例计入有效 corpus。

## T08 / M3：动态调查、规则路由与故障恢复

**依赖：** T07。

**创建：** `agent/rule_router.py`、`tests/unit/test_budget.py`、`tests/integration/test_recovery.py`、`tests/e2e/test_ambiguous_diagnosis.py`；修改 `agent/policy.py`、`orchestrator.py`、`store.py`。

**契约：** `RuleRouter.next_action(evidence, budget) -> AgentAction` 与 Planner 相同输入；`BudgetLedger.reserve(action)/settle(result)`；`request_more_evidence` 只返回结构化需要项或已有合法读取动作，无无限等待。

- [ ] 写无模型依赖的 ledger 测试，包含并发预留防超额、retry 计数、剩余 wall time 边界。

```python
def test_final_calls_are_reserved(ledger):
    for _ in range(4):
        ledger.reserve("planner_llm")
    with pytest.raises(BudgetExceeded):
        ledger.reserve("planner_llm")
    ledger.reserve("diagnosis_llm")
    ledger.reserve("patch_llm")
```

- [ ] ledger fixture 使用规格预算；超额在工具执行前拒绝，audit 保存 attempted/rejected/started/completed。不靠采样后再判断费用已超额。
- [ ] RuleRouter 首先利用已有证据；unknown symptom 不直接放弃，执行已公布的 memcheck-first 和适用工具 fallback，使用与 Agent 相同预算，避免刻意弱 baseline。
- [ ] ambiguous case 的 ordinary runtime 正常、memcheck clean 后出现 race/init/sync 线索，Agent 根据实际新证据决定下一步。不要求同一固定调用顺序，只检查合法性、证据依据和终止条件。
- [ ] 注入 provider/RAG 失败、非法 tool、进程 timeout、用户取消、manifest 写失败、重复结果与重启；已有 evidence 必须保留。默认不自动恢复带外 native 进程，通过孤儿 container label 检查报告可清理目标。
- [ ] 允许明确标注 `RULE_FALLBACK` 的开发降级；正式 Agent 实验不静默混入 fallback 成绩，provider failure 作为该 mode 失败记录。

```bash
python -m pytest tests/unit/test_budget.py tests/integration/test_recovery.py -q
python -m pytest tests/e2e/test_ambiguous_diagnosis.py --require-live -q
```

- [ ] 提交任务文件，消息 `feat: bound adaptive investigations and rule routing`。

**通过标准：** Agent 多步路径有可见价值测试、预算用尽可解释结束、错误路径不丢 artifacts。

## T09 / M4：Strict 验证、三类 Oracle 与结果完整性

**依赖：** T08。

**创建：** `tests/unit/test_verification_policy.py`、`tests/unit/test_checker_registry.py`、`tests/integration/test_private_visibility.py`、`tests/e2e/test_verification_matrix.py`；修改 `verification/{models,oracle,policy,engine}.py`、`patching.py`、`reporting.py`。

**契约：** `TrustedCheckerRegistry.resolve(checker_id) -> checker` 只返回预注册函数；`VerificationResult` 包含全部 `CheckRequirement`、outcome、not-run reason 和 candidate hash；`ReportExporter.public(run_id) -> bytes` 只序列化 public 字段。

- [ ] 为 target persists+new finding、oracle wrong+target unknown、optional error、strict additional timeout、missing private suite 和重复 verify 写判定测试。

```python
def test_required_unsupported_prevents_success(verification_fixture):
    observation = verification_fixture(required_supported=False)
    assert decide_verdict(observation) == VerificationVerdict.INCONCLUSIVE
```

- [ ] fixture 在测试模块提供 T04 `VerificationObservation`；optional error 只有本次 policy 不要求时可作为 limitation，Strict 提升为 required 后必须阻断 success。
- [ ] 添加 expected-output、CPU reference、registered checker 三种实现和明确类型约束，不能通过 module path 动态 import 用户 checker。stdout `PASS` 仅限可信输出协议且仍配合隐藏数值/行为检查，不能由可修改程序单独自证。
- [ ] 检查集合在验证开始前冻结并保存版本；Agent 只能增加。根据 kernel 事实约束明确 N/A 原因，不把“不想运行”变成 N/A。
- [ ] private visibility 集成测试在 sibling evaluator root 放 canary，通过 tool read、artifact ID、source traversal、公开 report、provider captured request 五条通道测试不可见。candidate 验证只能看到当前输入，下一次任务没有前次可写状态。
- [ ] 重复 verify 产生新的 child run；同一 source/patch hash 明确关联，不覆盖先前结果；测试修改 candidate 文件后验证拒绝或注册不同 candidate。

```bash
python -m pytest tests/unit/test_verification_policy.py tests/unit/test_checker_registry.py tests/integration/test_private_visibility.py -q
python -m pytest tests/e2e/test_verification_matrix.py --require-live -q
```

- [ ] 提交任务文件，消息 `feat: enforce strict verification and private oracle boundaries`。

**通过标准：** 四种 verdict 的真实及异常场景都可解释，private 数据只在 Evaluator 中存在；没有“必需检查没跑却 VERIFIED”的分支。

## T10 / M5：可复现 mutation 与有效 corpus

**依赖：** T09。

**创建：** `benchmark/{models,builder}.py`、`benchmarks/templates/`、`tests/unit/test_corpus_registration.py`、`tests/gpu/test_mutation_validation.py`、`docs/benchmark-protocol.md`。

**契约：** `CaseManifest(id, source_hash, harness_hash, mutation_id, template_id, split, oracle_id, target_tool, expected_finding, validation_run_ids)`；`BenchmarkBuilder.validate(clean_case, mutant) -> CaseValidation`；`register(validation: CaseValidation) -> CaseManifest`；`CaseValidation` 必须含 clean Oracle/checks 与 mutant finding 结果。

- [ ] 写拒绝未验证 mutant、clean 有错误、只 timeout 无目标 finding、hash 不匹配、跨 split 重复模板的测试。

```python
def test_no_finding_means_no_benchmark_case(builder, validation):
    validation.target_confirmed = False
    with pytest.raises(UnvalidatedCaseError):
        builder.register(validation)
```

- [ ] builder/validation fixture 用注册临时 Store 和真实模型工厂；新增 `UnvalidatedCaseError` 在 builder 定义。
- [ ] 采用模板化删除 guard、移除初始化、受控索引冲突和 barrier mutation，每条 operator 记录具体 diff/hash。变异后不因“自己知道删了什么”就确认根因，仍需工具定位和 Oracle 观察。
- [ ] clean 与 mutant 使用完全相同 toolchain、harness、inputs；clean 必须 Oracle PASS/required clean，mutant 必须 target finding 真正出现。验证来源包含全部重复试验，不隐藏未检出运行。
- [ ] 为 Agent 分配 neutral case ID，复制源码至统一 `src/kernel.cu`；外部目录名 memcheck/racecheck、mutation 注释、预期工具字段、参考实现不能泄漏到 public prompt 或 source 路径。
- [ ] Corpus private root 由 `GPU_AGENT_EVAL_ROOT` 配置，必须在 repo 和 Agent roots 外；真实 holdout suite 使用此处存储。public development truth 可公开但 Agent 进程不可读；最终 holdout 先冻结 template/operator split 再评测，公开 hash 和来源类别摘要。
- [ ] 生成 16+8 的候选矩阵，在全部真实验证后才计数；用 CUDA 确定性缺陷实验诊断失败样本，禁止用 mock 补足数量。通用 CaseReducer 此版本不建立空实现。

```bash
python -m pytest tests/unit/test_corpus_registration.py -q
python -m pytest tests/gpu/test_mutation_validation.py --require-live -q
gpu-agent benchmark validate
```

- [ ] 提交 public templates/测试/protocol，绝不提交 holdout root。消息 `feat: curate validated CUDA mutation corpus`。

**通过标准：** 每个有效 case 都有 clean/mutant 两组可追踪运行，私有集与开发集按 template/operator 分组避免近重复泄漏。

## T11 / M6：五组实验、指标与 RAG 比较

**依赖：** T10。

**创建：** `benchmark/{evaluation,metrics}.py`、`evaluation/{modes.json,rubric.md,protocol.md}`、`tests/unit/test_metrics.py`、`tests/integration/test_evaluation_views.py`、`tests/e2e/test_evaluation_run.py`；检索候选实现放 `knowledge/semantic.py`，只在依赖锁定后创建。

**契约：** `EvaluationRunner.run(mode, split, repeats) -> EvaluationManifest`；`score(record, hidden_truth, rubric) -> Score`；`aggregate(records) -> MetricSummary`。`EvaluationRecord` 含 case/template/mode/repeat、input/evidence hashes、完整 executed checks、status、diagnosis、patch/Oracle/verdict、usage、latency 和 failure reason。

- [ ] 指标失败测试：空集合输出 `value=None,n=0`；超时保留在 end-to-end 分母；class/root-cause/location 的有效样本分母另外报告，不能只保留成功案例。

```python
def test_empty_data_is_not_perfect_accuracy():
    summary = aggregate([])
    assert summary.root_cause_accuracy.value is None
    assert summary.root_cause_accuracy.n == 0
```

- [ ] A/B/C 使用固定证据推理比较：A=source+runtime，B=A+RAG，C=A+预收集工具证据；统一 diagnosis/patch provider 和 schema。预收集证据的实际成本不算零，同时报告采集成本和 inference-only 成本。
- [ ] D/E 比较证据获取：共享 tools、起点、RAG、预算、最终诊断与补丁模型，仅 policy 为 RuleRouter/Planner。D 允许预定义 fallback；E 缺 API 失败不能被 D 结果替代。
- [ ] 默认由用户配置 runtime model，Provider 根据能力决定是否发送 temperature；不支持参数标为 N/A，不能为了“相同 temperature”发送非法请求。推理等级、prompt 版本和最大上下文/截断策略全部固定。
- [ ] 每种配置至少 3 次重复，单 GPU 串行，随机化模式顺序并记录种子；不让并行 Sanitizer 争用 GPU 扭曲延迟。记录 LLM/工具调用次数、实际 token 和已知价格版本，价格未知则 cost=null。
- [ ] 指标覆盖规格 §19.4 全项：family exact match、root cause rubric、source location（文件+预注册行区间）、evidence/citation precision、hit@k、compile/Oracle/verified rate、regression detection、inconclusive precision/recall、budget rate、latency/cost。没有不可判定真值病例时 inconclusive precision/recall 报 N/A，并用单独 fault-injection suite 测该能力。
- [ ] 盲评包移除 mode/model/调用轨迹，仅保留评分所需 diagnosis/evidence，隐藏 mapping 由 Evaluator 保存。诊断 correctness 不能仅靠 LLM 自评；有标注 rubric 和人工复核结果。
- [ ] 预定义正负 patch controls 测 regression：原 bug 保留、错误数值、编译失败、clean 正确、新 finding；repair 每次最多一个候选。私有 holdout 最终评分不反馈给 Planner 继续优化该同一案例。
- [ ] 构建至少 20 条人工相关段落标签的 retrieval queries；比较 lexical、semantic、hybrid 中预注册的候选方法。按 embedding 实际 token 限额切段，依赖/模型版本入 lock。依据开发集 hit@k、citation relevance 和延迟选择默认方案，冻结后不以 holdout 调参。

```bash
python -m pytest tests/unit/test_metrics.py tests/integration/test_evaluation_views.py -q
gpu-agent benchmark evaluate --mode all --split development --repeats 3
gpu-agent benchmark evaluate --mode all --split holdout --repeats 3
python -m pytest tests/e2e/test_evaluation_run.py --require-live -q
```

- [ ] 长批次执行前输出预计 case×mode×repeat 调用规模与 API 成本估算；实际费用超用户配置上限时停止并保留部分数据，不能为凑完整实验无限调用。
- [ ] 聚合按 case/template 分组，重复运行不当独立新样本；报告每类计数、均值/中位数、失败列表、区间或小样本限制，不以 16/8 个样本宣布普适结论。
- [ ] 提交代码、protocol、rubric 与脱敏汇总；原始私有 artifacts 留在外部 root。消息 `feat: evaluate diagnosis and evidence acquisition fairly`。

**通过标准：** 五组实验真实可重复，能回答 Agent 与强规则基线在相同预算下的质量/效率差异；没有伪造提升或择优样本。

## T12 / M6：发布文档与可重跑验收

**依赖：** T11。

**创建：** `docs/{architecture,demo,limitations,evaluation-report,acceptance}.md`、`src/gpu_agent/benchmark/release.py`、`tests/e2e/test_release_gate.py`；修改 `README.md`；固定依赖及 toolchain locks。

**契约：** `ReleaseManifest(commit, toolchain_hash, corpus_hash, model_config_hash, test_counts, evidence_run_ids, unresolved_items)`；由验收 runner 使用实际数据生成。

`benchmark/release.py` 定义以上 schema、`ReleaseGate.check(manifest) -> ReleaseGateResult(passed: bool, reason_codes: list[str])`；`test_counts` 具有 expected、executed、skipped_required 等类型化计数字段。

- [ ] 先写缺 live run、SKIPPED、private Oracle 不可用或 corpus 数量不足都不能通过 release gate 的测试。

```python
def test_skipped_live_case_blocks_release(release_gate, manifest):
    manifest.test_counts.skipped_required = 1
    assert release_gate.check(manifest).passed is False
```

- [ ] 测试 fixture 从实际 ReleaseManifest schema 构造，完整字段来自本任务定义，不通过 hardcoded PASS 文本验收。
- [ ] README 写明当前能力、安装/激活、GPU 容器配置、provider 配置、已知威胁边界、private corpus 配置与复现命令；普通用户无 private data 时只能复现 public development 验证，不暗示可重现未发布 holdout。
- [ ] demo 使用真实 run ID/commit，展示 OOB→证据→模型 patch→隔离 Oracle/memcheck→verdict，再展示一个 clean 但错误补丁被拒绝。
- [ ] 报告将 `VERIFIED_FIXED` 限定为特定 suite/toolchain 的结果，列明 GPU 内存/driver 共享边界和不支持的 CUDA 特性。不能写 full-chip hardware verification 成果。
- [ ] 对原生执行路径、path scope、metadata visibility、verdict/skip 分支进行独立检查；不需要自动新建 Codex 任务。发现问题回到所属任务修复，更新验收证据。

```bash
python -m pytest tests/unit tests/integration -m 'not live_llm and not gpu and not container' -q
python -m pytest tests/gpu tests/e2e -m release --require-live -q
python -m ruff check src tests
python -m mypy src/gpu_agent
git diff --check
```

- [ ] 发布检查需验证 release 标记确实覆盖四工具、隔离、隐藏 Oracle、真实 LLM 及五组评测所引用的运行，而非只执行一个轻量 gate 测试。避免无必要地重跑已冻结昂贵评测；验证现有同一 commit/config/hash 的记录即可，代码改变会使对应证据失效。
- [ ] 显式列举将提交文件，检查 secrets、private corpus、raw logs、machine identifiers 和大文件。只提交批准公开的 artifacts，消息 `docs: document verified GPU debugger release`。推送、创建 tag/Release 在用户决定发布时进行。

**通过标准：** clean checkout 的 public demo 可复现；私有实验有可审计记录；所有未满足项明确列出，不能用 README 宣称取代实际执行。

## 5. 规格覆盖索引

- §1–4 产品/JD/范围：T06 产品闭环、T12 README。
- §5 环境：T01/T02；镜像版本 T03；实验版本 T11。
- §6 信任边界：T03/T04/T09。
- §7–8 架构/接口：T02–T06 按首次消费者创建。
- §9 状态：T02 状态不变量、T08 错误恢复。
- §10–11 Diagnose/Agent：T06/T08。
- §12 RAG：T05 词法引用、T11 检索比较。
- §13 Patch：T04 scope/hash、T06 唯一模型补丁、T09 重复验证。
- §14–16 Oracle/Verifier/Policy：T04/T07/T09。
- §17 RunStore：T02；私有 artifact 可见性 T04/T09；导出 T12。
- §18 Builder/隔离：T10，holdout 运行 T11。
- §19 五组实验/指标：T11。
- §20 CLI：T01 env、T06 diagnose/verify/report、T10–T11 benchmark。
- §21 错误处理：T02/T03/T08/T09。
- §22 M0–M6：依赖图与 T01–T12 的里程碑标签。
- §23–26 发布/演示/冻结/评审：T12 和本计划 §2 实施解释。

## 6. 计划自检与执行交接

- [x] 阅读完整 V2，确认当前仓库和工具链事实。
- [x] M1 具有最小隔离后端、scope/hash、私有 Oracle、唯一候选与 Agent 预算，不依赖 M4 才获得基本安全。
- [x] 有成功普通 runtime 仍调查的用例。
- [x] required+unsupported、missing vs downstream-not-run、原 finding 未知分别建模。
- [x] 单任务文件、接口、示例行为测试和真实验证命令均已指定。
- [x] 16+8 corpus 门槛归 M6，早期仅需一个有效 OOB。
- [x] 无开发天数安排，无重复审批模型/任务组织方式。
- [x] 用户评审并批准开始执行本计划。
- [x] 开始 T01，并只根据真实结果勾选任务。

本计划中的命令与测试代码是执行规范；计划文件存在不表示这些测试已通过。T01 的真实结果见验证记录；下一项执行是 T02，后续未执行任务仍保持未完成。

## 7. 官方实现依据

- [CUDA 12.8 Update 1 Linux 安装说明](https://docs.nvidia.com/cuda/archive/12.8.1/cuda-installation-guide-linux/index.html)：专用 Conda Toolkit 安装、编译器支持；实际发行包可用性仍在 T01 查询。
- [Compute Sanitizer 官方说明](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html)：四工具作用、`--error-exitcode` 与 memcheck 前置检查。该页面会更新，实现时只启用本地工具版本实际支持的选项。
- [NVIDIA Container Toolkit 安装说明](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)：配置 Docker GPU runtime 的依据，实际 flags 与权限由 T03 benign probe 验证。
