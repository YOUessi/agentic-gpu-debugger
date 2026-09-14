# Agentic GPU Debugging & Verification System
# 智能体驱动的 GPU 调试与验证系统

**文档状态：** Active Source of Truth

**版本：** V2

**日期：** 2026-09-15

**仓库名：** `agentic-gpu-debugger`

**CLI 名：** `gpu-agent`

**主要目标：** NVIDIA Compute System Arch AI Infra Intern - 2027（JR2023889）作品项目

**替代文档：** `PROJECT_SPEC_CN.md` 与 `IMPLEMENTATION_PLAN_CN.md` V1

> 本文档只定义产品、架构、边界与验收标准。实施计划必须在本文档评审通过后另行编写，并按里程碑和可验证结果组织，不得按天数安排。

---

## 1. 产品定义

### 1.1 正式定义

Agentic GPU Debugging & Verification System 是一个面向 CUDA 故障、以证据为基础的 AI 工程系统。

系统在受控执行环境中真实编译和运行 CUDA 程序，由 Agent 根据源码、运行结果、Compute Sanitizer 和 NVIDIA 官方文档证据动态选择诊断路径；系统在受限 Patch Scope 内生成一个隔离的候选修复，并通过重新编译、功能正确性 Oracle、目标故障检查以及确定性策略要求的回归检查判断修复是否成立。

独立的 Benchmark Builder 通过参数生成和受控缺陷 Mutation 构建经过真实验证的 CUDA Failure Corpus，并通过公开输入/私有 Ground Truth 隔离以及消融评测，分别衡量 RAG、确定性工具证据和 Agent orchestration 的实际贡献。

### 1.2 一句话介绍

> 输入一个有故障的 CUDA 程序，系统真实执行、收集证据、动态诊断、生成隔离补丁，并通过隐藏正确性测试与 Sanitizer 回归检查验证修复。

### 1.3 北极星流程

```text
CUDA Failure
    ↓
Controlled Build & Execution
    ↓
Evidence Collection
    ↓
Agent-directed Investigation
    ↓
Evidence-grounded Diagnosis
    ↓
One Isolated Patch Candidate
    ↓
Oracle + Sanitizer Verification
    ↓
Auditable Verdict
```

任何功能进入当前范围前，都必须回答：

> 它是否直接提高上述闭环的真实性、安全性、可解释性、可评测性或可复现性？

如果答案是否定的，则不进入当前版本。

---

## 2. 招聘目标与能力证据

目标岗位关心 AI-assisted development、AI-integrated flows、RAG、AI Agent、GPU architecture、testing use cases、verification、documentation 和协作交付。

本项目提供以下可检查证据：

- **AI Infrastructure：** Execution Backend、Tool Policy、RunStore、EvidenceRepository、评测框架。
- **AI-integrated flow：** Build → Run → Investigate → Diagnose → Patch → Verify。
- **AI Agent：** 根据新增证据动态选择合法调查动作，并受预算、状态和策略约束。
- **RAG：** 只检索可追踪、版本化的 NVIDIA 官方 CUDA 文档。
- **GPU Domain：** CUDA 执行模型、内存、线程、同步与 Compute Sanitizer。
- **Testing Use Cases：** 经过真实执行确认的 failure corpus，而非只准备几个演示文件。
- **Verification：** 功能 Oracle、目标 finding、回归 finding 和隐藏输入共同决定 verdict。
- **Evaluation：** 规则系统、静态工具证据与 Agent 的公平消融实验。
- **Documentation：** 每次 Run 的 artifact、报告、限制和最终实验结论均可审计。

本项目不声称复刻 NVIDIA 内部 pre-silicon/post-silicon full-chip verification 平台。它在公开可实现的 CUDA 软件层面复现相同的工程思想：生成测试、观察失败、获取证据、诊断根因、验证修复和沉淀回归用例。

---

## 3. 用户、输入与输出

### 3.1 主要用户

- 学习或开发 CUDA 的工程师。
- 需要定位 CUDA memory、race、initialization 或 synchronization 故障的开发者。
- 需要构建可复现 GPU failure benchmark 的研究者或学生。

### 3.2 主要输入

诊断任务输入：

- 一个仓库内受信任的 benchmark case，或隔离执行环境中的 CUDA 源码。
- 受控 build configuration。
- runtime 参数与 public smoke inputs。
- 可选的期望功能描述；它不能替代受信任 Oracle。

验证任务输入：

- 原始 diagnosis `run_id`。
- 一个 `PatchCandidate` 或允许范围内的候选源码副本。
- Evaluator 私有可见的 Oracle 与 holdout inputs。

### 3.3 主要输出

- 环境和工具链快照。
- Build、runtime 与 sanitizer 的原始日志和结构化结果。
- Agent action/audit trace。
- Evidence-grounded diagnosis。
- 带 provenance 的候选补丁。
- Public 与 private Oracle 结果。
- 必需的 sanitizer/regression check 结果。
- `VerificationVerdict` 与细粒度 reason code。
- 可阅读 Markdown 报告和机器可读 JSON artifact。

---

## 4. 目标与非目标

### 4.1 当前目标

1. 跑通一个真实 global-memory OOB 的完整纵向闭环。
2. 扩展到 memory access、shared-memory race、uninitialized memory、synchronization 四类故障。
3. 让 Agent 的调查路径由证据驱动，而不是由单个关键词固定映射。
4. 生成一次隔离候选补丁并给出可审计验证结论。
5. 构建无答案泄漏的 public/private benchmark 边界。
6. 用公平消融实验判断 RAG、工具证据和 Agent 的实际价值。

### 4.2 当前非目标

- 不声称支持 full-chip、RTL、pre-silicon simulator 或硬件根因定位。
- 不做通用 CUDA 聊天机器人。
- 不训练或微调基础模型。
- 不做无限自动修复循环。
- 不允许 Agent 任意执行 Shell。
- 不覆盖用户原始源码。
- 不以 FastAPI、SaaS 或复杂 Web UI 为当前交付重点。
- 不做 Nsight 性能调优、occupancy 优化或多 GPU 性能分析。
- 不做 Multi-Agent。
- 不做 CI/PR Bot。
- 不承诺安全执行恶意 native CUDA 代码。
- 不在第一阶段实现源码/输入 delta debugging；只保留未来扩展位置。

---

## 5. 运行基线与可复现性

### 5.1 目标环境

- Linux。
- NVIDIA RTX 4090 Laptop GPU，Compute Capability 8.9。
- 支持 `sm_89` 的 CUDA 12.x Toolkit。
- 与 Toolkit 配套的 Compute Sanitizer。
- Python 3.11 或 3.12 专用环境。
- NVIDIA Driver 满足所选 Toolkit/runtime 要求。

PyTorch 的 `torch.version.cuda` 只表示 PyTorch 构建/运行时版本，不能替代 `nvcc` 和 Compute Sanitizer 的 Toolkit 检查。

### 5.2 ToolchainManifest

每次 Run 必须持久化：

```text
gpu_name
gpu_uuid_hash
compute_capability
driver_version
nvcc_path
nvcc_version
cuda_target_arch
compute_sanitizer_path
compute_sanitizer_version
python_version
os_release
execution_backend
container_image_digest
rag_corpus_version
prompt_version
llm_provider
llm_model
```

敏感标识不得直接进入公开报告；例如 GPU UUID 只保存哈希或省略。

### 5.3 环境门禁

运行前必须区分：

- Driver 能力。
- CUDA Toolkit 编译能力。
- PyTorch CUDA runtime（如果使用）。
- Compute Sanitizer 可用性。
- 目标 GPU 架构支持。

工具链不兼容时不得继续生成正式 benchmark 结果，应返回结构化 `INCONCLUSIVE` 或环境错误。

---

## 6. 威胁模型与信任边界

### 6.1 保护目标

系统旨在降低受信任 benchmark、用户源码和半可信 AI-generated patch 对宿主环境造成意外影响的风险，包括：

- 非预期文件读写。
- 非预期网络访问。
- 子进程爆炸。
- CPU、内存、磁盘或 GPU 资源滥用。
- 超时与挂死。
- 修改 Oracle、harness、benchmark metadata 或 private ground truth。

### 6.2 非保护目标

GPU 容器共享 Host Kernel 和 GPU Driver。本项目不宣称容器可以安全执行主动恶意、具备逃逸意图的 native CUDA/C++ 代码。

如需支持 hostile untrusted code，需要 VM/MicroVM、GPU passthrough 或独立机器，超出当前范围。

### 6.3 执行信任级别

#### TrustedLocalBackend

只允许运行仓库中经过人工审核的 benchmark/harness 代码。

- 主要用于最早纵向切片和本地开发。
- 不接受任意用户源码。
- 不执行模型生成的候选补丁，除非用户显式选择不安全开发模式；该模式的结果不得作为正式 Portfolio benchmark。

#### IsolatedGPUBackend

用于用户源码和模型生成补丁。

最低策略：

- 非 root 用户。
- 只挂载任务 workspace。
- 根文件系统只读。
- workspace 之外不可写。
- 默认关闭网络。
- drop Linux capabilities。
- PID、CPU、内存、磁盘和 wall-time 限制。
- 只暴露所需 GPU 设备。
- 不挂载 Docker socket、SSH key、用户主目录或项目 private ground truth。
- 容器镜像通过 digest 固定。

### 6.4 物理可见性原则

禁止依赖 prompt 中的“不要读取答案”。Agent workspace 必须在文件系统层面只包含其角色允许访问的内容。

---

## 7. 总体架构

```text
                            User / CLI
                                │
                         diagnose / verify
                                │
                                ▼
                      Agent Orchestrator
                                │
             ┌──────────────────┼──────────────────┐
             ▼                  ▼                  ▼
      ExecutionBackend   EvidenceRepository    RAG Service
             │                  │                  │
       build / run /            │            NVIDIA Docs
       sanitizers               │
             └──────────────┬───┴──────────────────┘
                            ▼
                    Diagnosis Engine
                            │
                            ▼
                      PatchCandidate
                            │
                       Scope Guard
                            │
                            ▼
                    Verification Engine
                  ┌─────────┼───────────┐
                  ▼         ▼           ▼
             Public      Private     Required
             Oracle      Holdout      Checks
                  └─────────┼───────────┘
                            ▼
                  VerificationVerdict

       Benchmark Builder ──► Validated Failure Corpus
                  │                    │
                  └────────► Evaluator ◄────────┘

横切能力：RunStore / Policy Layer / Audit Trail / Security Boundary
```

### 7.1 核心子系统

1. `ExecutionBackend`
2. `EvidenceRepository`
3. `AgentOrchestrator`
4. `RAGService`
5. `DiagnosisPatchEngine`
6. `VerificationEngine`
7. `BenchmarkBuilderEvaluator`

`RunStore` 是唯一持久化边界。Policy 与 Audit 是横切能力，不建立第二套业务存储。

---

## 8. 最小稳定契约

第一条纵向切片前只锁定以下核心契约，不预先锁死全部文件结构。

### 8.1 ExecutionBackend

职责：准备隔离 workspace，并执行类型化的 build、run 和 sanitizer 请求。

```text
prepare(WorkspaceRequest) -> WorkspaceHandle
build(BuildRequest) -> BuildResult
run(ExecutionRequest) -> ExecutionResult
run_sanitizer(SanitizerRequest) -> SanitizerResult
cleanup(WorkspaceHandle) -> CleanupResult
```

不得暴露通用 `run_shell(command: str)`。

### 8.2 ToolResult

统一 envelope：

```text
tool_name
request_id
started_at
finished_at
elapsed_ms
exit_code
timed_out
stdout_artifact
stderr_artifact
payload_type
typed_payload
tool_error
```

不同工具使用独立 typed payload，不能将所有结果退化为任意 `dict` 或日志字符串。

### 8.3 EvidenceBundle

只包含已经观察到的事实和来源：

```text
environment
source_snapshot
build_result
execution_result
sanitizer_results
source_locations
retrieved_chunks
limitations
```

推理结论不得写入 evidence 字段。

### 8.4 EvidenceRepository

为不同角色提供受控视图：

- Diagnosis Agent：public source、runtime、sanitizer、official RAG。
- Patch Engine：允许修改的 public source 与 diagnosis，不可见 private oracle/ground truth。
- Verification Engine：candidate、public/private oracle、required checks。
- Evaluator：private ground truth、系统输出与评分 rubric。

### 8.5 AgentAction

```text
action_id
action_type
typed_arguments
rationale
expected_information_gain
budget_snapshot
```

### 8.6 DiagnosisResult

```text
diagnostic_outcome
failure_family
root_cause
source_locations
observed_facts
tool_findings
documentation_evidence
model_inferences
recommended_change
confidence_label
limitations
```

`confidence_label` 不是校准概率，除非后续完成专门校准实验。

### 8.7 PatchCandidate

```text
patch_id
parent_run_id
base_source_hash
patched_source_hash
unified_diff
generated_by
provider
model
prompt_version
created_at
allowed_paths
scope_validation
```

### 8.8 PolicyDecision

```text
policy_name
policy_version
decision
mandatory_actions
prohibited_actions
reason_codes
```

### 8.9 OracleResult

```text
oracle_id
oracle_type
input_set_id
passed
expected_summary
actual_summary
atol
rtol
nan_policy
inf_policy
failure_reason
```

公开报告不得泄漏 private expected values 或 private seeds。

### 8.10 VerificationResult

```text
verdict
failure_stage
reason_code
original_finding_present
public_oracle_passed
private_holdout_passed
required_checks
new_findings
candidate_hash
limitations
```

### 8.11 RunStore

负责原子保存和加载所有 run artifacts。RunStore 是持久化 source of truth；EvidenceBundle 是领域模型，不建立第二个 Evidence Store。

---

## 9. Run 生命周期

### 9.1 RunStatus

```text
QUEUED
RUNNING
COMPLETED
FAILED
CANCELLED
```

### 9.2 CurrentPhase

仅当 `RunStatus=RUNNING` 时存在：

```text
PREPARING
COMPILING
EXECUTING
COLLECTING_EVIDENCE
DIAGNOSING
PATCH_GENERATING
VERIFYING
FINALIZING
```

终止时：

```text
current_phase = null
last_completed_phase = <phase>
```

### 9.3 DiagnosticOutcome

```text
CONFIRMED
PROBABLE
INCONCLUSIVE
NOT_APPLICABLE
```

### 9.4 VerificationVerdict

```text
VERIFIED_FIXED
NOT_FIXED
REGRESSION_DETECTED
INCONCLUSIVE
```

流程是否完成与修复是否成功是两个独立维度。例如候选补丁编译失败时，Run 可以正常 `COMPLETED`，但 Verdict 是 `NOT_FIXED`。

---

## 10. Diagnose 工作流

```text
1. Validate request and trust level
2. Create run and workspace
3. Capture ToolchainManifest
4. Build source
5. Execute public smoke input
6. Parse deterministic observations
7. Start Agent investigation loop
8. Collect allowed tool evidence
9. Retrieve version-matched official documentation
10. Produce schema-valid DiagnosisResult
11. Generate at most one PatchCandidate
12. Scope-check and persist candidate
13. End diagnosis run
```

Build、runtime 或工具异常不能导致已有 artifacts 丢失。

---

## 11. Agent 调查设计

### 11.1 Action Space

允许动作：

```text
inspect_source
inspect_environment
run_program
run_memcheck
run_racecheck
run_initcheck
run_synccheck
retrieve_official_docs
request_more_evidence
finish_diagnosis
declare_inconclusive
```

不允许动作：

- 任意 Shell。
- 任意网络请求。
- 读取 private oracle、fixed reference 或 ground truth。
- 修改源码、harness、metadata 或 evaluator。
- 绕过 Policy 强制动作。

### 11.2 Agent Loop

```text
Observation
    ↓
Planner proposes AgentAction
    ↓
State + Policy + Budget validation
    ↓
Typed tool execution
    ↓
EvidenceBundle update
    ↓
New Observation
```

### 11.3 AgentBudget

预算是可配置策略并写入 run manifest。默认上限建议：

```text
max_agent_steps = 8
max_sanitizer_calls = 4
max_rag_calls = 3
max_source_reads = 5
max_llm_calls = 6
max_wall_time_seconds = configurable
```

具体默认值在真实纵向切片运行后校准，但任何正式实验必须固定并记录预算版本。

### 11.4 停止条件

Agent 必须在以下任一条件满足时停止：

1. 证据满足 Diagnosis Policy，执行 `finish_diagnosis`。
2. 没有剩余合法动作可以提高信息量，返回 `INCONCLUSIVE`。
3. 预算耗尽，返回 `INCONCLUSIVE / AGENT_BUDGET_EXHAUSTED`。
4. 执行环境出现阻断性错误，根据 Policy 终止。
5. 用户取消。

模型自报 confidence 不能单独作为停止条件。

### 11.5 Agent 的必要性

规则路由器是正式 baseline。只有当 Agent 在模糊症状、多步调查或工具效率方面表现出可测优势时，才宣称 Agent orchestration 带来价值。

---

## 12. RAG 设计

### 12.1 允许知识源

- CUDA C++ Programming Guide。
- CUDA Runtime API。
- Compute Sanitizer Documentation。
- NVIDIA 官方 CUDA Samples 文档。

所有 chunk 必须保存：

```text
document_title
document_version
section_title
source_url
retrieved_at
content_hash
chunk_id
```

### 12.2 版本一致性

优先检索与当前 ToolchainManifest 相匹配的 CUDA/Compute Sanitizer 文档。无法匹配时必须在 diagnosis limitations 中标明版本差异。

### 12.3 检索策略

第一条纵向切片允许使用可解释的 lexical retrieval。Portfolio 版本必须对 lexical、semantic 或 hybrid 方案做检索评测；只有被数据证明有收益的复杂检索方案才进入默认路径。

### 12.4 Citation Guard

Diagnosis 只能引用当前 `RetrievedChunk` 中存在的 chunk。不存在的 `chunk_id` 必须被拒绝，不能默默生成虚假 citation。

---

## 13. Diagnosis 与 Patch

### 13.1 证据分层

报告必须明确分开：

1. `OBSERVED_FACT`
2. `TOOL_FINDING`
3. `DOCUMENTATION_EVIDENCE`
4. `MODEL_INFERENCE`

不得将模型推理包装为 Sanitizer 或文档事实。

### 13.2 单次候选修复

当前版本每次 diagnosis 最多生成一个 PatchCandidate：

```text
Diagnosis
    ↓
Generate ONE Candidate
    ↓
Scope Guard
    ↓
Persist Candidate
    ↓
Separate Verification Run
```

验证失败后不自动生成第二个补丁。新的补丁必须由新的用户请求或后续版本的受控 repair policy 发起。

### 13.3 Patch Scope Guard

第一版允许修改：

```text
*.cu
*.cuh
```

第一版禁止修改：

- benchmark metadata。
- build/runtime harness。
- Makefile/CMake/configuration。
- Oracle/checker/reference implementation。
- public/private inputs。
- evaluation code。
- ground truth。
- workspace 之外的任何路径。

Patch 必须以 unified diff 表示，在应用前验证路径、base hash 和作用域；应用后验证 patched hash。

Patch Scope Guard 不能替代执行隔离。合法 `.cu` 文件仍然可以包含危险 native code。

---

## 14. Oracle 设计

### 14.1 受信任 Oracle 类型

第一版只支持：

1. `EXPECTED_OUTPUT`：结构化标量、数组或受控 stdout。
2. `CPU_REFERENCE`：仓库内经过审核的 reference implementation。
3. `TRUSTED_CHECKER`：仓库内注册、类型化并经过审核的 checker。

用户随源码上传的可执行 checker 不被信任，不得直接运行。

### 14.2 数值比较

数值 Oracle 必须定义：

```text
expected_shape
absolute_tolerance
relative_tolerance
allow_nan
allow_inf
dtype
seed_or_input_set_id
```

默认比较原则：

```text
abs(actual - expected) <= atol + rtol * abs(expected)
```

必须在比较前检查 shape、长度、dtype、NaN 和 Inf 策略。

### 14.3 Public Smoke Inputs

诊断阶段可见，用于稳定复现原始故障。它们不能单独决定最终修复有效。

### 14.4 Private Holdout Inputs

仅 Evaluator/Verification Engine 可见。至少覆盖：

- warp/block 边界。
- 小尺寸和非整除尺寸。
- 多个可复现随机 seed。
- 与 public smoke 不同的 shape 或数据分布。

Agent、Patch Engine 和 candidate workspace 不得看到 private seed、expected values 或 reference implementation。

---

## 15. Verification 设计

### 15.1 CheckApplicability

```text
REQUIRED
APPLICABLE
NOT_APPLICABLE
UNSUPPORTED
```

### 15.2 CheckOutcome

```text
CLEAN
FINDING
TOOL_ERROR
TIMEOUT
NOT_RUN
```

“是否应运行”与“运行结果是什么”必须分别建模。`UNSUPPORTED` 不能被误判为程序 finding。

### 15.3 Verification Mode

#### Standard

- Build。
- Runtime。
- Public Oracle。
- Private holdout Oracle。
- 原始目标 Sanitizer。
- Deterministic Verification Policy 标记为 REQUIRED 的检查。

#### Strict

- Standard 的全部检查。
- 当前 case/toolchain 下所有 APPLICABLE sanitizer。
- Agent 可建议附加检查，但不能减少 REQUIRED/APPLICABLE 集合。

### 15.4 Verdict 规则

#### VERIFIED_FIXED

必须同时满足：

```text
candidate build PASS
runtime PASS
public smoke oracle PASS
private holdout oracle PASS
original finding absent
all REQUIRED checks CLEAN
all executed blocking regression checks CLEAN
```

#### NOT_FIXED

- Candidate 无法编译；或
- 原始 finding 仍存在；或
- Candidate 无法进入有效验证，且原因可明确归因于 candidate 本身。

#### REGRESSION_DETECTED

原始 finding 已消失，但出现任一情况：

- Public 或 private Oracle 失败。
- 出现新的 blocking sanitizer finding。
- 出现新的 runtime failure。

#### INCONCLUSIVE

- 执行后端失败。
- 必需工具不可用或 unsupported，导致证据不足。
- 超时且无法归因于 candidate 功能错误。
- Oracle 或 private holdout 不可用。
- 证据相互矛盾且无法可靠判定。

### 15.5 判定优先级

```text
Infrastructure/required evidence unavailable -> INCONCLUSIVE
Candidate build invalid -> NOT_FIXED
Original finding persists -> NOT_FIXED
Original finding absent but new functional/tool failure -> REGRESSION_DETECTED
All mandatory conditions pass -> VERIFIED_FIXED
Otherwise -> INCONCLUSIVE
```

---

## 16. Policy Layer

Policy 使用版本化、可测试的确定性规则。LLM 可以增加调查或验证动作，但不能删除强制动作。

### 16.1 ExecutionPolicy

- 根据 trust level 选择 backend。
- 决定网络、资源、文件系统和 GPU 暴露。
- 拒绝不受支持的执行模式。

### 16.2 ToolPolicy

- 限定 typed tool registry。
- 验证 AgentAction 与当前 phase。
- 应用调用预算。

### 16.3 PatchScopePolicy

- 验证路径、文件类型、base hash 和 diff。
- 拒绝对 harness、Oracle、metadata 和 ground truth 的修改。

### 16.4 VerificationPolicy

- 目标 Sanitizer 永远 REQUIRED。
- 基于 case metadata 和环境能力确定其他 REQUIRED/APPLICABLE 检查。
- Agent 只能添加检查。

### 16.5 BenchmarkVisibilityPolicy

- 创建只含 public input 的 Agent workspace。
- private ground truth 永不挂载到 Agent/candidate execution environment。
- 记录每个角色实际获得的 artifact view。

---

## 17. RunStore 与审计

### 17.1 原则

- 每个 Run 使用独立不可复用的 `run_id`。
- 原始输入先做只读快照。
- Artifact 写入必须原子化。
- 已有 deterministic artifacts 不因 LLM/RAG 失败而丢失。
- 所有状态变化和 PolicyDecision 都进入 audit log。
- 公共报告不得包含 API key、私有 seed、私有 expected output 或主机敏感路径。

### 17.2 建议布局

```text
runs/<run_id>/
├── manifest.json
├── toolchain.json
├── input/
│   └── public source snapshot
├── workspace/
├── artifacts/
│   ├── build.json
│   ├── execution.json
│   ├── sanitizers/
│   ├── retrieval.json
│   ├── evidence.json
│   ├── diagnosis.json
│   ├── patch.json
│   └── verification.json
├── logs/
├── audit.jsonl
└── report.md
```

Private holdout 和 ground truth 不复制到该目录；Evaluator 保存在独立、不可见的存储边界。

---

## 18. Benchmark Builder

### 18.1 Pipeline

```text
Trusted Correct Program
    ↓
Baseline Validation
    ├── Oracle PASS
    └── Required Sanitizers CLEAN
    ↓
Controlled Mutation
    ↓
Mutant Build & Execution
    ↓
Target Failure Confirmed?
    ├── No -> Discard
    └── Yes
        ↓
Ground Truth Record
        ↓
Public/Private Split
        ↓
Validated Failure Corpus
```

只有 target finding 经过真实 GPU 执行确认的 mutant 才能进入 corpus。

### 18.2 首批 Failure Families

- Memory access：OOB、illegal、misaligned。
- Shared-memory race。
- Uninitialized memory read。
- Synchronization/barrier misuse。

### 18.3 Mutation 范围

第一版采用人工审核的 template mutation：

- 删除 bounds check。
- 改变索引边界或分配/访问长度关系。
- 删除或移动同步点。
- 将 thread-local/shared 索引改为冲突索引。
- 删除初始化步骤。
- 构造受控 divergent barrier。

不在第一版实现通用源码重写器或自动 delta debugging。

### 18.4 Corpus 隔离

开发集可以在仓库中包含：

```text
case_<id>/
├── public_input/
└── private_ground_truth/
```

但 Agent 只能收到复制后的 `public_input` workspace，不能访问仓库根目录、Git 历史或 private path。

最终评测还必须包含不进入公开仓库的 private holdout corpus。Holdout 应按 kernel template 或 mutation operator family 隔离，而不仅仅更换随机 seed。

### 18.5 防止 Patch 过拟合

Candidate 验证必须使用诊断阶段未公开的多个输入。只对 public smoke case 成功的补丁不能得到 `VERIFIED_FIXED`。

---

## 19. Evaluation 设计

### 19.1 五组系统

#### A. LLM_ONLY

源码 + public runtime/build log，无 RAG、无 Sanitizer。

#### B. LLM_RAG

A + 官方文档检索。

#### C. LLM_STATIC_TOOLS

系统预先提供固定的 sanitizer evidence；LLM 不决定工具。

#### D. RULE_ROUTER_RAG_TOOLS

确定性 parser/规则选择工具，结合 RAG 和相同 diagnosis schema，不使用 Agent planning。

#### E. AGENT_RAG_TOOLS

Agent 在相同 action space、工具集合和预算下动态获取证据。

### 19.2 两类实验问题

#### Diagnosis Reasoning Quality

在控制输入证据的条件下比较不同信息组合对 diagnosis 的影响。

#### Evidence Acquisition Policy

在相同工具集合、上限预算和 case 输入下比较 Rule Router 与 Agent Planner。

### 19.3 实验控制

必须固定或记录：

- 模型/provider/version。
- prompt version。
- temperature 与采样参数。
- 输出 schema。
- public case 输入。
- tool/action budget。
- RAG corpus 和索引版本。
- toolchain/container digest。
- 每个系统实际获得的 evidence。

非确定性配置至少重复运行三次。人工评分必须在不知道系统标签的情况下按预注册 rubric 进行。

### 19.4 指标

#### Diagnosis Quality

- Failure Family Accuracy。
- Root Cause Accuracy。
- Source Location Accuracy。
- Inconclusive Precision/Recall。

#### Evidence Quality

- Evidence Precision。
- Unsupported Claim Rate。
- Citation Precision。
- Retrieval Hit@K。

#### Repair Quality

- Patch Compile Rate。
- Public Oracle Pass Rate。
- Private Holdout Pass Rate。
- Verified Fix Rate。
- Regression Detection Rate。

#### Agent Efficiency

- Tool Calls/Case。
- Sanitizer Calls/Case。
- LLM Calls/Case。
- End-to-End Latency。
- Token/Monetary Cost。
- Budget Exhaustion Rate。

### 19.5 报告原则

- 不预写“提升百分比”。
- 只使用真实 RTX 4090 执行结果。
- 报告均值、中位数、分布和失败案例，不只报告最佳值。
- Agent 与规则系统持平时如实说明。
- 区分准确率收益与证据获取效率收益。
- 公开实验配置、版本和已知限制。

---

## 20. CLI 产品界面

### 20.1 主入口

```bash
gpu-agent diagnose <case-or-source>
gpu-agent verify <run-id> <candidate-or-generated-patch>
```

### 20.2 辅助入口

```bash
gpu-agent env
gpu-agent report <run-id>
gpu-agent benchmark validate
gpu-agent benchmark evaluate --mode <mode>
```

CLI 只是接口层，不包含业务逻辑。当前版本不要求 FastAPI。

---

## 21. 错误处理原则

### 21.1 工具缺失或不兼容

- 保存环境探测结果。
- 不伪造工具输出。
- 返回明确 reason code。
- 需要该工具才能判断时返回 `INCONCLUSIVE`。

### 21.2 Build/Runtime Timeout

- 终止进程组或容器任务。
- 保存截断前的受限日志。
- 标记 timeout 来源和阶段。
- 不将基础设施 timeout 自动解释为 CUDA 根因。

### 21.3 LLM/RAG Failure

- 保留全部确定性 evidence。
- 可以使用明确标注的 rule-based fallback。
- 不将 fallback 输出冒充 Agent 结果。

### 21.4 非法 AgentAction

- Policy 拒绝执行。
- 写入 audit log。
- 允许在预算内重新规划一次；重复非法动作则终止为 `INCONCLUSIVE`。

### 21.5 Artifact 或 Hash 不一致

- 拒绝验证。
- 返回 `INCONCLUSIVE / ARTIFACT_INTEGRITY_ERROR`。
- 不允许自动选择“最新文件”代替声明的 PatchCandidate。

---

## 22. 纵向切片与里程碑

所有里程碑只以可运行成果和验收条件定义，不绑定日期或天数。

### M0：工具链与安全基线

成果：

- 专用 Python 环境。
- 支持 `sm_89` 的 CUDA Toolkit 与配套 Sanitizer。
- ToolchainManifest 探测。
- TrustedLocal 与 Isolated backend 的边界说明。

验收：

- 能真实编译、运行一个 clean CUDA kernel。
- 能记录完整 toolchain artifact。
- 不兼容环境被可靠拒绝。

### M1：Global OOB 纵向闭环

成果：

- 一个经过审核的 OOB benchmark。
- 可执行模型候选补丁的最小 `IsolatedGPUBackend`。
- Build/run/memcheck。
- EvidenceBundle。
- 最小 AgentAction loop。
- 官方文档检索。
- DiagnosisResult。
- 一个 PatchCandidate。
- Oracle + memcheck verification。
- 完整 RunStore artifacts。

验收：

- 原始 case 被真实 memcheck 捕获并定位。
- Diagnosis 引用真实 evidence 和文档。
- Candidate 不覆盖原始文件。
- Candidate workspace 不挂载用户主目录、仓库根目录或 private ground truth。
- 在隔离后端上通过 public/private Oracle 与 memcheck 后得到 `VERIFIED_FIXED`。
- 任一强制条件缺失时不得输出 `VERIFIED_FIXED`。

### M2：四类故障工具层

成果：

- memcheck、racecheck、initcheck、synccheck typed adapters。
- CheckApplicability/CheckOutcome。
- 每类至少一个 validated seed case。

验收：

- 每类 seed 的 target finding 可稳定复现。
- 对应 reference/fixed 版本 Oracle 通过且目标 finding 消失。
- unsupported/tool error 不被误判为 program finding。

### M3：Agent、规则路由与预算

成果：

- Rule Router baseline。
- 动态 Agent Planner。
- Policy validation、budget、stop conditions。

验收：

- Agent 不能执行非法动作或越权读取。
- Budget exhaustion 得到可审计 `INCONCLUSIVE`。
- 至少一个 ambiguous case 需要根据新增 evidence 决定第二步工具。

### M4：Patch 与严格 Verification

成果：

- Patch provenance/hash。
- Scope Guard。
- Public/private Oracle。
- Standard/Strict verification。
- 完整 verdict/reason matrix。

验收：

- 编译失败、原问题保留、新功能回归、工具缺失和完全修复均得到正确 verdict。
- Candidate 看不到 private holdout 或 Oracle。
- 过拟合 public smoke 的错误 patch 被 private holdout 拒绝。

### M5：Benchmark Builder

成果：

- Trusted clean kernels。
- 受控 mutation templates。
- 自动执行、确认和 corpus 注册。
- public/private visibility boundary。

验收：

- 未经 target finding 确认的 mutant 不能进入 corpus。
- Agent workspace 物理上不存在 private ground truth。
- corpus artifact 记录 mutation provenance 和 toolchain。

### M6：评测与 Portfolio Release

成果：

- 五组系统配置。
- 诊断推理与证据获取两类实验。
- 至少 16 个 public validated cases，每类至少 4 个。
- 至少 8 个来自不同 template/operator 的 private holdout cases。
- README、架构说明、示例报告、演示脚本和实验报告。

验收：

- 实验控制和评分 rubric 可审计。
- 结果来自真实 RTX 4090 环境。
- 所有数字可由保存的配置和 artifacts 追溯。
- 报告同时展示成功、失败、inconclusive 和限制。
- 面试演示能够在一个 OOB case 上完成 Diagnose → Patch → Verify。

---

## 23. Portfolio Release 总验收

只有满足以下条件才能称为 Portfolio-ready：

### 产品闭环

- `diagnose` 与 `verify` 主路径真实可运行。
- 至少一个案例完成自动 Candidate Patch 与隔离验证。
- 任何 `VERIFIED_FIXED` 都有完整 Oracle 和 sanitizer evidence。

### 工程可信度

- 不存在 generic shell tool。
- 用户/模型生成 native code 不在默认 trusted-local 模式执行。
- Agent、Patch、Verifier 和 Evaluator 的文件可见性经过测试。
- RunStore artifact、hash 和 provenance 完整。

### Agent 可信度

- Agent 受 action space、phase、policy 和 budget 约束。
- 至少一个 case 展示多步、非固定线性的调查路径。
- Rule Router 是正式 baseline，不通过弱 baseline 人为抬高 Agent 表现。

### Benchmark 可信度

- 所有 cases 经过 clean baseline 与 mutant failure 双重验证。
- Public input 与 private ground truth 物理隔离。
- Private holdout 使用未公开输入和至少部分未公开 operator/template。

### 报告可信度

- 只报告真实实验数据。
- 说明环境、预算、模型、prompt、RAG corpus 和工具版本。
- 不隐瞒 Agent 与规则系统持平或失败的结果。
- 明确项目不是 full-chip verification，也不是恶意代码强安全沙箱。

---

## 24. 最终演示场景

```bash
gpu-agent diagnose benchmarks/public/memcheck/oob_001/public_input
```

演示应展示：

1. ToolchainManifest。
2. 原始 CUDA build/run。
3. Agent 选择并调用 memcheck。
4. Sanitizer finding 与源码位置。
5. NVIDIA 官方文档 citation。
6. 事实、工具、文档与模型推理分层的 DiagnosisResult。
7. 带 hash 和 diff 的 PatchCandidate。

随后：

```bash
gpu-agent verify <run-id> --generated-candidate --strict
```

演示应展示：

1. Candidate 在隔离 backend 编译运行。
2. Public smoke Oracle。
3. Private holdout Oracle，只展示通过情况，不泄漏答案。
4. Required/applicable sanitizer checks。
5. 最终 verdict 和 reason code。
6. 可追溯 report 与 artifacts。

---

## 25. 已冻结的设计决策

以下决策在 V2 评审后不得在实现中静默改变：

1. 产品核心是 CUDA Diagnose → Patch → Verify 闭环。
2. 每次 diagnosis 最多生成一个候选补丁。
3. 原始源码永不被覆盖。
4. 用户/模型 native code 需要隔离执行；容器不被宣称为对抗性强沙箱。
5. Tool Registry 不提供通用 Shell。
6. LLM 不能绕过确定性 Policy。
7. `RunStatus`、执行 phase、diagnosis outcome 与 verification verdict 分离。
8. `VERIFIED_FIXED` 必须通过 public、private Oracle 和全部 REQUIRED checks。
9. Agent/Patch workspace 物理不可见 private ground truth。
10. Benchmark 只接收经过真实执行确认的 mutant。
11. Rule Router 是正式 baseline。
12. Evaluation 同时衡量质量和证据获取效率。
13. FastAPI、复杂 UI、Multi-Agent、Nsight 和 CI 不属于当前版本。
14. 实施计划不得使用日程或天数组织。

---

## 26. V2 评审清单

评审者需要确认：

- 产品定义是否与 NVIDIA JD 对齐且没有夸大。
- Threat Model 是否诚实。
- TrustedLocal/Isolated 边界是否可实施。
- Agent 是否有真实决策空间，同时受到充分约束。
- Oracle 与 holdout 是否能阻止无效/过拟合补丁通过。
- Verification verdict 是否语义互斥且可确定执行。
- Benchmark 是否存在答案泄漏通道。
- 五组 ablation 是否公平。
- 第一条 OOB vertical slice 是否能在不构建全部平台的前提下落地。
- Portfolio acceptance 是否能通过真实 artifacts 证明。
- 文档是否不存在按天数安排。

评审通过后，下一份文档才是基于本规格编写的 `IMPLEMENTATION_PLAN_CN_V2.md`。
