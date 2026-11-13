# Full-stack 开发问题与修复记录

本文是 `feat/v3-fullstack-ai-test-console` 的持续工程日志。以后每轮开发遇到可复现问题、架构冲突、测试失败或兼容性问题时，都在这里留下“现象 → 原因 → 修复 → 验证”记录。它不是提交历史的替代品，也不把未验证猜测写成已确认根因。

## 记录规则

每条记录至少包含：日期/阶段与影响范围、现象、已确认原因、修复、验证和仍存在的边界。

---

## 2026-10-03 / Full-stack Round 1

### FS-001：GitHub App 无法直接创建分支

**现象：** GitHub connector 创建新分支返回 HTTP 403 `Resource not accessible by integration`。

**原因：** 当前 GitHub App 安装权限可读取仓库，但 ref create 权限不足。

**修复：** 不修改旧工作区；在已授权 Tang 设备 fetch `design/v2-operator-workflow@2fa987b`，通过仓库已有 SSH remote 创建并 push 新分支。

**验证：** GitHub 再读取分支确认 HEAD，Round 1 commit `88e1143` 已成功 push。

### FS-002：FastAPI Response union 触发 Pydantic response model 错误

**现象：** `/` 路由标注 `FileResponse | PlainTextResponse` 后，TestClient 创建应用时报 `Invalid args for response field`。

**原因：** FastAPI 尝试从 union response 类型生成 Pydantic response field。

**修复：** 路由设置 `response_model=None`，返回类型统一为 `Response`。

**验证：** `tests/web` 通过，FastAPI 可同时返回 API 与构建后的 React 首页。

### FS-003：历史 RunStore verification schema 与当前模型不完全一致

**现象：** 对 Tang 历史真实 RunStore 调用 `/api/stats` 返回 500；旧 `verification/result.json` 含当前 `VerificationResult` 不再接受的历史字段。

**原因：** Web 展示层错误地把不可变历史 artifact 强制重新解释为当前严格 domain schema。

**修复：** Web projection 对历史 public JSON 使用 schema-tolerant dict 读取；核心 `ApplicationService` / verifier 仍使用当前严格模型，不迁移或改写旧 artifact。

**验证：** 历史 run `ebce9bb...` 可显示 `VERIFIED_FIXED`，旧 artifact 保持原样。

### FS-004：TypeScript verbatimModuleSyntax 要求 type-only import

**现象：** `npm run build` 报 `FormEvent is a type and must be imported using a type-only import`。

**原因：** Vite 模板 TypeScript 配置启用 `verbatimModuleSyntax`。

**修复：** 改为 `import type { FormEvent } from 'react'`。

**验证：** TypeScript + Vite production build 通过。

### FS-005：Playwright strict locator 命中两个同名元素

**现象：** `getByText('Shared Memory Race')` 同时命中表格与侧栏统计。

**原因：** 测试 locator 没有限定语义区域。

**修复：** locator 限定到 `getByRole('table')`。

**验证：** Playwright 主流程测试通过。

### FS-006：旧 Agent 运行只有 decision，没有 step artifact

**现象：** 历史真实 run 有 `actions/*/decision.json`，详情页却显示 0 steps。

**原因：** 新 trajectory 使用 step + decision；早期 RunStore 只有 decision。

**修复：** Web projection 优先读取 step；缺 step 时从已注册 decision 生成只读兼容视图，不伪造 rationale/evidence。

**验证：** 历史 run 可显示 controller-recorded action 类型；新 run 仍使用完整 step 数据。

---

## 2026-10-03 / Full-stack Round 2

### FS-007：Public case metadata 分散在两个 registry

**现象：** `case_0001~0016` 在 `corpus-registry.json`，`case_0017~0022` 在 `diverse-registry.json`；前端原先要求手输 case ID。

**原因：** benchmark 演进后 metadata 分散，不能只读取单一 registry。

**修复：** 新增 Public Case Catalog，合并两个 registry，并用每个 public case 的 `task.json` 与 `kernel.cu` SHA256 再确认 repair-ready 身份；前端改为 catalog selector。

**验证：** `/api/cases` 在真实仓库返回 23 个 public case；`case_0017~0022` 元数据与对应 task 正确。Python targeted regression 44 passed，Playwright catalog selector 流程通过。

### FS-008：Playwright 把原生 select 的 option 判为 hidden

**现象：** case catalog E2E 中 `getByRole('option').toBeVisible()` 失败，但 locator 已正确找到 `case_0021`。

**原因：** 原生 `<select>` 未展开时 option 在浏览器可访问树中存在，但视觉状态不是 visible；测试把“存在且被选中”错误写成“可见”。

**修复：** 改用 `getByLabel('Public case').toHaveValue('case_0021')`，并检查 option 数量；测试用户可观察的选择状态，而不是浏览器内部绘制行为。

**验证：** Playwright 两条 E2E 流程均通过，catalog selector 使用 value/count 断言稳定通过。

### FS-009：Case catalog effect 触发 exhaustive-deps 警告

**现象：** Oxlint 指出加载 case catalog 的 `useEffect` 读取 `form.case_id` 却使用空依赖数组。

**原因：** 初始化逻辑捕获了 render 时的 form 对象。

**修复：** 把 readiness 判断移入 `setForm(current => ...)` functional update，只依赖异步返回的 items，不再捕获 `form`。

**验证：** `npm run lint` 为 0 warnings / 0 errors；TypeScript production build 同时通过。

### FS-010：Diagnosis citation 同时使用 artifact ID 与文档 chunk ID

**现象：** Diagnosis 的 `citation_ids` 不能统一按 RunStore artifact ID 直接打开：运行/工具证据引用 artifact ID，而 RAG 文档引用 `DocumentChunk.chunk_id`。

**原因：** 两类 citation 有不同身份空间；文档 chunk 作为 JSON artifact 持久化，但对模型暴露的是稳定 chunk ID。

**修复：** Web projection 增加 citation resolver：直接 artifact ID 映射到注册 artifact；文档 citation 扫描已注册 `docs/*.json`，把 chunk ID 映射到其 artifact、标题、section、source URL 与文本 preview。前端 claim 旁显示可点击 citation chip。

**验证：** Tang 历史真实 OOB run 解析出 6 个 citation，其中包含 build/runtime/memcheck artifact 与 NVIDIA 官方文档 chunk；Playwright 点击 artifact citation 后成功显示持久化内容。

### Round 2 验证摘要

- Python targeted regression：44 passed（Web + service + iterative repair + public repair correctness）。
- 最终 Web API smoke：4 passed。
- Ruff：通过；`mypy --strict src/gpu_agent/web`：通过。
- Vitest：4 passed；TypeScript/Vite production build：通过。
- Oxlint：0 warnings / 0 errors；Playwright：2 passed。
- Tang 真实历史 RunStore：`/api/cases` 返回 23 个 public case；历史 OOB run 成功解析 6 个 evidence citation。
- 本轮没有发起新的付费模型调用，也没有把前端测试冒充成新的真实 GPU 验收。

---

## 2026-10-03 / Full-stack Round 3

### FS-011：同步 Repair API 无法支持“启动后立即看进度”

**现象：** Round 2 的 `POST /api/repair` 会一直等待 `ApplicationService.repair()` 完成；前端只有请求结束后才拿到 run ID，因此无法从 PREPARING/COMPILING 阶段开始展示实时进度。

**原因：** HTTP 请求生命周期与 GPU/LLM 长任务生命周期绑定在一起。

**修复：** 新增持久化 Web Repair Job：`POST /api/jobs/repair` 立即返回 202 + job ID，由受限线程池执行原 `ApplicationService.repair()`；`GET /api/jobs/{job_id}` 只返回编排状态。原同步接口保留兼容，但前端切换到 async job。

**验证：** FastAPI/Job Manager 测试确认提交响应先返回 QUEUED，随后 job 绑定 diagnosis run 并进入 COMPLETED；Playwright 从 New Repair → Background Job → 自动打开 Run Detail 的流程通过。

### FS-012：异步 Job 元数据不能成为第二套“执行真相”

**现象：** 引入 background job 后，如果把阶段/结果直接记录在 job 对象里，会和 RunStore 形成双事实源；服务重启还可能留下永远 RUNNING 的 UI job。

**原因：** Job 是 Web 编排状态，不是 GPU 执行证据；最终 diagnosis/repair/verification 必须继续由 RunStore 定义。

**修复：** job 只持久化 `QUEUED/RUNNING/COMPLETED/FAILED`、case、run_id、错误码和最终 public verdict 引用；实际 pipeline 仍读取 RunManifest/events/artifacts。job JSON 使用 controller-owned 0700 目录和 0600 文件、原子 replace；启动时把遗留 QUEUED/RUNNING 标记为 `WEB_CONTROLLER_RESTARTED`。

**验证：** job persistence/recovery 单测通过；重启恢复不会改写对应 RunStore run。

### FS-013：异步 Job 需要在外部 GPU/LLM 工作前拿到 diagnosis run_id

**现象：** `ApplicationService.repair()` 原本只在全部流程完成后返回 `RunManifest`，background job 无法可靠知道自己对应哪个 run；通过“扫描新目录猜 run”在并发情况下存在歧义。

**原因：** service 内部创建 run，但没有非权威观察接口。

**修复：** `repair()` 增加可选 `on_run_created` observer，并在 diagnosis run 创建完成、进入 PREPARING/任何 provider/GPU 外部工作之前调用。observer 只接收 manifest copy，不获得 shell、verifier 或 evaluator capability；默认 `None`，旧调用路径不变。

**验证：** 单测确认 observer 看到同一 run ID 且初始状态为 QUEUED，随后真实 RunStore 状态正常完成。

### FS-014：源码对比不能在浏览器重新实现一套 CUDA patch 语义

**现象：** “Original vs Candidate” 需要展示候选修改，但把 unified diff 在 React 里重新应用会复制 Python patcher 的规则，可能与 controller 的合法 candidate 不一致。

**原因：** patch application、路径白名单和 provenance 校验属于核心后端语义。

**修复：** 当前 UI 并排展示 RunStore 注册的原始 `sources/kernel.cu` 与已验证/注册 candidate 的 `unified_diff`，明确标为 `Selected candidate diff`；不在前端推导新的 patched source。后续若需要完整 patched source，应调用核心 `materialize_candidate` 的服务端只读 projection，而不是 JS 重写 patcher。

**验证：** Tang 真实 OOB run 可显示 2012 字符原始 kernel 与 713 字符 candidate diff，无前端 console error。

### FS-015：React polling/source effect 依赖必须保持稳定

**现象：** Round 3 首次 lint 对 source artifact 与 repair job effect 报 `exhaustive-deps` 警告。

**原因：** effect 捕获了对象引用，但依赖数组只列对象字段，静态分析无法证明闭包稳定。

**修复：** 在 render 阶段提取稳定 primitive（`sourceArtifactId`、`repairJobId`、`repairJobStatus`），effect 只依赖这些值。

**验证：** `npm run lint` 恢复 0 warnings / 0 errors。

### Round 3 验证摘要

- Python targeted regression：48 passed（Web API/Job Store + service + iterative/public repair）。
- Ruff：通过；`mypy --strict src/gpu_agent`：82 个 source file 无问题。
- Vitest：4 passed；TypeScript/Vite production build：通过。
- Oxlint：0 warnings / 0 errors；Playwright：3 passed（Run Detail、Case Catalog、Async Repair Job）。
- Tang 真实历史 OOB run：8 个 RunManifest timeline event、2 个 source/diff pane、6 个 citation；原始 kernel 2012 字符，candidate diff 713 字符，浏览器无 console error。
- 本轮没有新增付费模型调用或真实 GPU 实验；async/job/UI 测试不替代原 CUDA/LLM 证据。

---

## 2026-10-03 / Full-stack Round 4

### FS-016：当前 operational RunStore 没有历史 batch/evaluation 数据

**现象：** 当前 `.gpu-agent/runs` 只有 diagnosis/candidate/verification；真实 seed batch 和 A–E evaluation 证据保存在历史 public RunStore。直接把主服务切到历史 store 会让当前 Repair 操作与历史分析数据耦合。

**原因：** Operational workflow 与历史 benchmark/evaluation 本来就是不同生命周期的数据域。

**修复：** 增加可选 `GPU_AGENT_ANALYTICS_RUN_ROOT`。Run/Repair 继续使用当前 public RunStore；AnalyticsCatalog 只读另一个 public RunStore。未配置时才复用当前 store。接口和 UI 明确显示 analytics store 路径。

**验证：** Tang 实际以当前 `.gpu-agent/runs` 作为 operational store，同时挂载 `/home/you/gpu-agent-v21-9d75699/public` 作为 analytics store；两套页面数据可同时读取。

### FS-017：Analytics 不能越过 public/evaluator 边界重新计算正式私有指标

**现象：** 项目已有 `metrics.aggregate()`，但它需要 evaluator/private labels。若 Dashboard 为了“指标更全”直接接 evaluator store，会破坏项目原有隐藏评测边界。

**原因：** 正式 benchmark scoring 与运营可视化不是同一个权限域。

**修复：** AnalyticsCatalog 构造时强制 `visibility=public`；只读取 public `batch/summary.json`、`evaluation/manifest.json`/public records。页面展示的是 descriptive public operational metrics（verified rate、latency、调用量、token、已知 cost），不声称是新的 release/holdout score。

**验证：** API 单测确认空 public analytics store 不会回退读取 evaluator store；真实历史 public evaluation 可以独立投影。

### FS-018：240 个 evaluation unit 不应在每次刷新时逐 artifact 读取

**现象：** 一个真实 development evaluation 含 240 个 record，run 中约 963 个 artifact。逐 `evaluation/records/*.json` 读取会制造大量小文件 I/O。

**原因：** Evaluation 已经持久化了聚合 `evaluation/manifest.json`，其中含 public records；Dashboard 不需要重新走 EvaluationRunner 的验证路径。

**修复：** Analytics 优先一次读取 `evaluation/manifest.json`，按 artifact SHA 缓存在只读 projection 层；只有旧 run 缺 manifest 时才回退读取 individual records。分页/筛选在已验证 public projection 上执行。

**验证：** Tang 的真实 evaluation manifest 约 598 KB，单次投影得到 240 records、5 个 mode 汇总；生产页面可一次打开并服务端分页。

### FS-019：损坏的历史 analytics run 不能被静默忽略

**现象：** 初版 overview 在 JSON/schema 读取失败时直接 `continue`，可能让一个损坏 evaluation 从分母中消失，看起来像“没有这次运行”。

**原因：** 可视化层的容错不应改变统计语义。

**修复：** Overview 新增 `projection_errors`，失败 run 不进入成功统计，同时返回 `<run_id>:ANALYTICS_PROJECTION_INVALID`。前端显示黄色 Projection Warning，不把失败投影当 0 值或成功 run。

**验证：** 注入非法 `evaluation/manifest.json` 的 API 测试确认 evaluation_count 保持 0 且明确返回 projection error。

### FS-020：仅把 RunStore 对象标成 public 不能证明底层目录真的是 public store

**现象：** `GPU_AGENT_ANALYTICS_RUN_ROOT` 是路径配置；如果误指向 evaluator 目录，再用默认 `RunStore(path)` 打开，对象的 `visibility` 字段会是 public，但历史 manifest 中的 ArtifactRef 实际仍标记为 evaluator。

**原因：** Store 对象的运行时标签不能替代持久化 artifact 自身的可见性证据。

**修复：** AnalyticsCatalog 在枚举和读取 batch/evaluation 前检查每个 RunManifest 的所有 ArtifactRef，任何 `visibility != public` 都 fail closed；overview 返回 503 `ANALYTICS_STORE_UNSAFE`，而不是投影或泄露 private 内容。

**验证：** 单测创建真实 evaluator-visibility artifact 后用默认 public label 重新打开同一路径，Analytics overview 被 503 拒绝。

### Round 4 验证摘要

- Python targeted regression：52 passed（全部 Web + service + iterative/public repair 相关测试）。
- Ruff：通过；`mypy --strict src/gpu_agent`：83 个 source file 无问题。
- Vitest：4 passed；TypeScript/Vite production build：通过。
- Oxlint：0 warnings / 0 errors；Playwright：5 passed，其中新增 Evaluation Analytics 与 Seed Batch 两条流程。
- Tang 真实双-store 演示：operational 仍为当前 `.gpu-agent/runs`，analytics 挂载历史 `/home/you/gpu-agent-v21-9d75699/public`。页面读取到 1 个 240-unit public development evaluation、5 个 A–E mode 汇总和 1 个 16-case seed batch；batch 16 行均可展开，浏览器 console error 为 0。
- 历史 public evaluation 页面显示 83/240 `VERIFIED_FIXED`、public-record mean latency 约 39.4s、known public cost 约 $2.6858；这些是挂载历史快照的描述性数据，不作为当前 HEAD 的重新评测结果。
- 本轮没有新增付费模型调用或真实 GPU 实验；Analytics 是既有 public evidence 的只读投影。

---

## 2026-10-03 / Full-stack Round 5

### FS-021：Evaluation record 必须保留 immutable lineage，不能只显示统计字段

**现象：** Round 4 的数据表能看到 case/mode/verdict/latency，但无法从一个异常 unit 回到实际 diagnosis run，因此“统计异常 → 原始证据”链路断开。

**原因：** Evaluation public record 本来就持久化了 `lineage.diagnosis_run_id / candidate_run_id / verification_run_id`，Web projection 当时没有暴露这些 ID。

**修复：** `EvaluationRecordRow` 增加三类 lineage ID；record inspector 只展示这些不可变 run ID，并提供 `Open diagnosis run`。不通过 case_id/template 猜测对应 run。

**验证：** Tang 历史 240-unit evaluation 的首个 VERIFIED_FIXED record 成功映射到 diagnosis `40e7b46a...`、candidate `04c688d...`、verification `a69cda8...`，与原 public record 完全一致。

### FS-022：历史 Analytics diagnosis run 不能复用当前 operational `/api/runs` 写路径

**现象：** Analytics store 与当前 operational RunStore 可不同；如果点击历史 record 后直接调用 `/api/runs/{id}`，要么 404，要么未来可能错误连接当前 Strict Verify 等写操作。

**原因：** 历史 analytics drill-down 是只读证据浏览，当前 run console 是可操作 workflow，两者权限语义不同。

**修复：** 新增独立 `/api/analytics/runs/{run_id}` 和 `/api/analytics/runs/{run_id}/artifacts/{artifact_id}`。后端先再次检查 public ArtifactRef visibility，再用 analytics RunCatalog 读取；前端 drawer 明确标记 `Read only`，不提供 Strict Verify 或 Repair 按钮。

**验证：** 真实历史 diagnosis `40e7b46a...` 可读取 47 个 public artifact、3 个 resolved citation、candidate diff 和 1 个 public verification child；当前 operational store 未被切换或修改。

### FS-023：Mode 图表必须是描述性可视化，不能暗示统计显著性

**现象：** A–E mode 的 verified rate/latency/LLM calls 很适合画图，但这些 unit 是固定 benchmark attempts，不应把简单柱状图解释成独立样本置信区间或模型优劣统计结论。

**原因：** Dashboard 这里只有 public records 的描述统计，没有额外的独立性假设、bootstrap 或 significance test。

**修复：** 图表只显示 verified rate、mean latency、mean LLM calls，并在每个 chart 标记 `descriptive · public records`；不绘制置信区间、不输出 winner/ranking 结论。

**验证：** Tang 真实 evaluation 正确生成 3 个 chart、5 个 mode bar group，数值与 public manifest 投影一致。

### FS-024：跨 store evidence 点击必须使用 Analytics artifact namespace

**现象：** 从历史 diagnosis drawer 点击 citation 时，如果仍调用 operational `getArtifact()`，会从当前 RunStore 查找相同 artifact ID，产生 404 或错误数据域。

**原因：** Artifact ID 只在其 RunStore/Run 上有意义，不能跨 operational/analytics store 混用。

**修复：** Analytics drawer 使用独立 `getAnalyticsArtifact()`，路径固定为 `/api/analytics/runs/{run_id}/artifacts/{artifact_id}`；后端只允许该 historical public run 已注册 artifact。

**验证：** Playwright 可从 evaluation record → historical diagnosis → citation chip 打开 persisted memcheck evidence；真实浏览器 drill-down 中 citation 数 3、artifact 数 47、console error 为 0。

### Round 5 验证摘要

- Python targeted regression：52 passed；Ruff 通过；`mypy --strict src/gpu_agent`：83 个 source file 无问题。
- Vitest：4 passed；TypeScript/Vite production build：通过；Oxlint：0 warnings / 0 errors。
- Playwright：5 passed；Evaluation Analytics 流程新增 charts → record inspector → diagnosis lineage → citation artifact 的完整 E2E。
- Tang 真实历史 evaluation `f7a7b393...`：3 个 mode chart 正常渲染；首个 VERIFIED_FIXED record 的 diagnosis/candidate/verification lineage 与 public manifest 一致。
- 真实 diagnosis `40e7b46a...` drill-down：failure family=`out_of_bounds`、47 个 public artifact、3 个 resolved citation、candidate diff、1 个 verification child；浏览器 console error 为 0。
- 本轮没有新增付费模型调用或真实 GPU 实验；图表与 drill-down 均是既有 public evidence 的只读展示。

---

## 2026-10-03 / Full-stack Round 6

### FS-025：跨 Evaluation 不能只看 verified rate 差值就叫 regression

**现象：** 历史 public store 里同时存在 development、holdout、失败未执行完成的 evaluation。若直接把两个 verified rate 相减，会把不同 split、不同 corpus 或不完整执行误写成回归/提升。

**原因：** 指标只有在同一完整 population 上才具备 unit-level 可比性；历史 run 的 `split/corpus_cutoff/expected_units/modes/repeats` 可能不同。

**修复：** Comparison 先检查 run status、split、corpus cutoff、expected units、repeats、mode 集合与 execution completeness。任何不一致都返回 `comparable=false` 和明确 reason，不计算 unit regression 或 metric delta。

**验证：** Tang 的 release public store 中，completed holdout 对 failed holdout 被阻止，返回 `RUN_NOT_COMPLETED/CANDIDATE_INCOMPLETE/UNIT_KEY_MISMATCH`；development 对 holdout 也因 population mismatch 被阻止。

### FS-026：Evaluation schedule ordinal 不能作为跨 run 对齐主键

**现象：** Evaluation schedule 会随机化 ordinal；同一个 case/mode/repeat 在不同 evaluation 中不保证 ordinal 相同。按表格行号比较会把不同 unit 错配。

**原因：** `ordinal` 是执行顺序，不是 benchmark unit 身份。

**修复：** Regression comparison 用稳定 tuple `(case_id, template_id, mode, repeat)` 建索引；发现 duplicate key 或两边 unit key 集合不一致时直接标记不可比。只有匹配 population 中 `VERIFIED_FIXED → 非 VERIFIED_FIXED` 才记为 regression，反向记为 improvement。

**验证：** Synthetic API 测试构造 3 个相同 unit，确认 1 regression + 1 improvement + 1 unchanged，并保留 baseline/candidate diagnosis lineage。

### FS-027：失败或部分执行 Evaluation 不能贡献“0%”假统计

**现象：** 真实 failed holdout evaluation 持久化了 `expected_units=120` 但 `executed_units=0`。若按空 records 算成 verified rate=0%，页面会把“没有执行”误读成“全部失败”。

**原因：** 未执行与执行后未修复是不同状态。

**修复：** EvaluationCard 对空 records 保持 `verified_rate=null`；comparison 对 incomplete run 返回 `BASELINE_INCOMPLETE/CANDIDATE_INCOMPLETE`，matched/regression 统计清零。Trend 可以显示该 run，但不会把它和完整 run 自动解释为性能下降。

**验证：** 真实 failed holdout 在 overview 中显示 0/120 executed 且 verified rate 为空；与 completed holdout 比较被 compatibility gate 拦截。

### FS-028：CSV 导出需要防 spreadsheet formula injection

**现象：** Evaluation record 中 template/failure reason 等文本来自持久化数据；若单元格以 `= + - @` 等开头，直接下载 CSV 后用 Excel/Sheets 打开可能被解释为公式。

**原因：** CSV 是数据格式，但常见消费者会主动执行公式语法。

**修复：** 导出字符串字段统一经过 `_csv_safe()`，危险前缀加单引号；CSV 带 UTF-8 BOM，并设置 attachment filename。导出限制最多 10,000 records，避免无界内存构造。

**验证：** API 测试注入 template `=SUM(A1:A2)`，导出 CSV 中变为 `'` + `=SUM(A1:A2)`；JSON export 保持结构化数据。

### FS-029：Trend 与 Regression 必须在 UI 上分成两种语义

**现象：** 用户需要“趋势”，但历史 timeline 可能同时包含 development/holdout/cutoff 变化。把折线或柱条连接起来容易暗示这些 run 可直接比较。

**原因：** 时间顺序不等于实验 population 一致。

**修复：** `Evaluation trend` 仅逐 run 展示时间、split、cutoff、units、verified/latency 的描述值，并明确提示 direct comparison 需要 compatibility gate；真正的 Regression Analysis 放在独立 comparison workbench 中。

**验证：** Playwright 覆盖 2-run trend + compatible comparison + regression lineage；真实 3-run release store 能展示 timeline，同时不自动生成跨 population regression。

### Round 6 验证摘要

- Python targeted regression：55 passed；Ruff 通过；`mypy --strict src/gpu_agent`：83 个 source file 无问题。
- Vitest：4 passed；TypeScript/Vite production build：通过；Oxlint：0 warnings / 0 errors；Playwright：6 passed。
- Playwright 新增完整 compatible comparison：2 个同 population run → regression workbench → 1 regression / 1 improvement / 1 unchanged → baseline diagnosis lineage drill-down。
- Tang 真实 release public store 挂载 3 个 evaluation：120-unit completed holdout、240-unit development、0/120 failed holdout。Trend 正常显示 3 个历史点；completed holdout → failed holdout 被 gate 拦截，未产生假 regression 数字。
- 真实 development export：JSON 约 178 KB、CSV 约 60 KB，均带 attachment filename；页面 Export CSV/JSON 链接指向对应 historical public evaluation。
- 本轮没有新增付费模型调用或真实 GPU 实验；comparison/trend/export 均使用既有 public evidence。

---

## 模板：后续问题

### FS-XXX：标题

**现象：**

**原因：**

**修复：**

**验证：**

**后续：**
