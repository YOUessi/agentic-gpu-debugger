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

## 模板：后续问题

### FS-XXX：标题

**现象：**

**原因：**

**修复：**

**验证：**

**后续：**
