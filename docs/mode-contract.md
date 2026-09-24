# A–E 评测模式契约（V2.1）

本文件是模式语义、证据门禁、补丁权限、记录状态与 Release Gate 的唯一来源。代码、
测试与 `evaluation/protocol.md` 必须与此表一致；修改任一列需同时修改其余各处。
离线端到端检查见 `tests/unit/test_mode_contract.py`。

## 1. 模式定义

| 模式 | 证据获取 | 诊断 | 补丁与验证 |
|---|---|---|---|
| A LLM_ONLY | build + 普通运行（无 Sanitizer、无 RAG） | 统一诊断模型 | 统一补丁模型 → 验证 |
| B LLM_RAG | A + 控制器固定查询的官方文档检索 | 统一诊断模型 | 同上 |
| C LLM_STATIC_TOOLS | A + 控制器预先执行 memcheck 与目标工具 | 统一诊断模型 | 同上 |
| D RULE_ROUTER | RuleRouter 在预算内选择工具与检索 | 统一诊断模型 | 同上 |
| E AGENT | Planner 在同一预算与动作空间内选择 | 统一诊断模型 | 同上 |

- 所有模式共用同一 provider、模型、诊断 prompt、补丁 prompt、schema、provider policy、
  pricing attestation 与验证器；D 与 E 的唯一变量是证据获取策略。
- 不再存在确定性诊断（`_deterministic_diagnosis` 与 `_ControllerOnlyProvider` 已删除）。
- 模型接受的诊断原文存为 `agent/model-diagnosis.json`；终态诊断由
  `derive_final_diagnosis(model, evidence)` 推导，回放校验器逐字节复算。

## 2. 诊断证据门禁（按本次运行实际获得的证据）

`policy.validate_diagnosis` 对每个 `DIAGNOSED` 结果要求：

| 条件 | 规则 |
|---|---|
| observed facts | 至少一条，引用 ID 必须来自本次 observed facts |
| tool finding 引用 | 本次证据含 tool finding 时必需；引用 ID 必须存在 |
| 文档引用 | 本次证据含文档时必需；引用 ID 必须存在 |
| source_locations | 非空；`kernel.cu`；行号在源码范围内 |
| 位置一致 | 任一 finding 带位置时，每个诊断位置必须等于某个 finding 位置 |

- 规则与模式无关、只看证据，因此 A（无工具无文档）也可以合法 DIAGNOSED。
- 模型返回 `INCONCLUSIVE`/`LLM_UNAVAILABLE` → 终态 `MODEL_DECLARED_INCONCLUSIVE`。
- 已知限制：provider 在调用内用同一门禁校验并重试一次；两次都不合规时记录为
  `LLM_INVALID_OUTPUT`（FAILED），`output_diagnostics.failure_class=DOMAIN_REJECTED`
  可与格式错误区分。
- 格式重试额度按调用类别各一次（plan、diagnose、patch 互不占用），仍计入总调用次数边界。
  不可解析的输出只记录形状分类（`<json>: raw_diff` / `code_fence` / `truncated_json` /
  `invalid_escape` 等），不保存原文；重试提示按“格式错误”与“内容被拒”分别给出。

## 3. 补丁与验证

- 每个 `DIAGNOSED` 单元生成一个补丁（provider 内 diff 校验 + 一次重试），执行器随后
  执行一次验证；`INCONCLUSIVE` 单元不生成补丁。补丁 prompt 不含任何 case 特定措辞。
- 验证 truth 按 case 解析（`verification/truth.py`）：用原始 kernel 的 hash 在
  `benchmarks/corpus-registry.json` 中唯一定位 case，取其 `target_tool` 与
  `expected_finding`；数值 oracle 参数来自 `development_truth/case_0001/case.json`，
  每个 case 的私有输入种子由基准种子与 case_id 派生。
- 验证不依赖 agent 证据：公开输入取自诊断 run 的普通执行；原始缺陷是否仍在，按
  「目标工具 + 注册 finding 类别」判断。因此 A/B（未跑 Sanitizer）与 D/E 用同一基准。
- 每个验证输入都运行 memcheck 与该 case 的目标工具；full 模式在 memcheck 干净时再跑
  其余工具。race/sync 类缺陷不改变数值输出，只有目标工具能区分修好与否。
- 私有 holdout 从 evaluator store 的已提交 corpus、按 evaluation unit 的 cutoff 与
  case ID 解析 truth，并核对原始源码及 harness。支持已注册的 vector-add-cpu-v1 oracle；
  不支持的 oracle 保持 `ORACLE_OR_BASELINE_UNAVAILABLE`。私有描述不进入模型输入。

## 3a. 输入、标签与 planner

- 每个公开 case 的注册输入存为 `public_input/input.json`（与 corpus 验证的
  `input_set_hash` 一致）；评测单元必须使用它，执行器回放时核对 `public-input.json`。
- `failure_family` 是冻结词表（`agent/models.py::FailureFamily`），
  `evaluation/development-labels.json` 冻结每个公开 case 的族标签；位置与根因标签未裁定，
  暂不计分。`python -m gpu_agent.benchmark.dev_report RUN_ROOT RUN_ID` 输出按模式的汇总。
- E 的 planner prompt 只给控制器硬约束和目标，不再复述 RuleRouter 的固定流程。
- E 每一步都收到 `controller_state`：结束前仍缺的证据类别（`missing_evidence`，与
  `decide_action` 的结束门禁同一函数 `policy.missing_evidence`）、已执行动作（仅控制器字段）、
  已读源码范围、上一次被拒的原因码。它不给工具顺序，下一步仍由 planner 自己选。
- E 允许一次 replan：第一次被策略拒绝的提议消耗一步，planner 只收到拒绝原因码；
  第二次拒绝为终态（INCONCLUSIVE）。D 不 replan。
- 公开 kernel 已去除描述缺陷的注释和提示性标识符；case_0001 的输入校验改回与 clean
  模板一致（不再硬编码 n=257）。这些改动改变了 mutant hash，corpus 必须重新验证注册。

## 4. 单元记录状态（`evaluation.evaluation_record_status`，唯一实现）

| 情况 | status | failure_reason |
|---|---|---|
| 诊断 DIAGNOSED（无论验证 verdict，或补丁失败） | `COMPLETED` | `None`；verdict 另存，补丁失败原因追加在 `diagnosis.limitations` 末尾 |
| agent 自身结果：`INVALID_DIAGNOSIS_EVIDENCE`、`MODEL_DECLARED_INCONCLUSIVE`、`NO_INFORMATION_GAIN`、`AGENT_BUDGET_EXHAUSTED`、控制器拒绝的 planner 动作（`DUPLICATE_NO_BENEFIT` 等） | `INCONCLUSIVE` | 原因码 |
| provider / 工具 / 知识库 / 构建 / 容器等基础设施故障（`LLM_*`、`SANITIZER_EVIDENCE_UNAVAILABLE`、`KNOWLEDGE_UNAVAILABLE`、`EXECUTION_INFRASTRUCTURE_UNAVAILABLE` …） | `FAILED`（含 `TIMEOUT` 的为 `TIMEOUT`） | 原因码 |

- 单元内任何失败都落为单元记录，批次继续：plan/diagnose/patch 的终态 provider 失败、
  准备阶段（构建、运行、容器 attestation）失败都有回放路径。
- 以下情况停批次：控制器状态不一致（回放不符）、记录无法写入、provider/pricing 配置错误
  （`PRICING_ATTESTATION_REQUIRED`、`MODEL_CONFIG_MISMATCH`、`PAID_CALLS_NOT_ALLOWED`、
  `LLM_UNAVAILABLE`、`LLM_CAPABILITY_UNAVAILABLE`——它们对每个单元相同）、执行代码漂移。
- 费用：0 次调用记 0；有调用但 usage 或价格未知记 `null`。仅记录实际消耗，不检查余额，
  不设单次或总美元上限，也不因金额中断运行或拒绝发布证据。CLI 不再接受美元上限参数。
  旧 schedule 的金额字段仅为兼容读取与签名核对保留，不再控制执行。

## 5. Release Gate

Gate 检查覆盖率与证据链，不检查是否全部成功：

- 每个 (case, mode, repeat) 恰好一条记录；`COMPLETED` 当且仅当 `failure_reason is None`；
- 每条记录的 lineage、hash、cutoff、commit、config 与原生 artifact 一致；
- 评测 binding 必须含 `runtime_code_hash`：`for_release` 要求被 import 的 `gpu_agent`
  就在被捕获仓库的 `src/gpu_agent`，并对其全部 `.py` 求哈希；每个单元重新计算并写入
  `agent/runtime-code.json`，漂移即停批次，回放校验逐条比对。这样 editable install
  指向别处、批次中途改代码的运行不能作为发布证据；
- 失败、超时、INCONCLUSIVE 与各 verdict 按实际分布报告，不设成功率门槛。

## 6. `gpu-agent diagnose` 开发模式

```
gpu-agent diagnose SOURCE --allow-paid-calls --max-llm-calls 40
```

- 不带 `--allow-paid-calls` 时不发送请求，结果为 `PAID_CALLS_NOT_ALLOWED`。
- 每次物理请求（包括格式重试）计入调用次数，到达调用边界即停止；不设置美元上限。
  持久化实际 token usage，STARTED 后无终态的请求按 UNCERTAIN 展示，不能伪造为 0 消耗。
- run 写入 `agent/development-mode.json`，不是评测或发布证据；已绑定评测的 service 拒绝开启。

## 7. 中断恢复与缓存边界

- 已开始但没有完整 completion receipt 的开发集单元保持不确定状态，resume 不重新调用模型。
  只有 receipt、调度与原生证据复算一致时，才恢复公开记录。新执行不得重用已有 execution claim。
- 正常执行只检查新写入的 record；恢复与最终审计仍检查完整证据。Release Gate 精确检查
  attempts、claims、records，以及开发集 completions，不放宽额外 artifact 白名单。
- RunStore 根据文件身份、大小、mtime 和 ctime 缓存 manifest；返回独立的可变字段副本。
  Linux 目录名缓存使用 inotify；队列溢出或监听不可用时重新扫描，不省略 manifest 身份检查。
  fork 后重置继承的锁并创建独立监听，子进程不能消耗父进程的目录事件。
- 已验证的是 validator 调用次数、目录扫描次数及 manifest 解析字节的增长边界。
  子目录逐项 stat、逐渐增长的父 manifest 写入仍存在；不声称整个批次已证明严格 O(n)，
  也不把离线计数测试当成真实 240 单元墙钟性能验收。
