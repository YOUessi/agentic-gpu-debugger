# Full-stack CUDA Test & Repair Console

本文件描述 `feat/v3-fullstack-ai-test-console` 分支新增的操作员界面。V2 的 Agent、RunStore、执行隔离、RAG、repair 与 verification 仍是核心实现；Web 层只新增受控 API 和可视化，不复制业务逻辑。

## 架构

```text
React / TypeScript
        |
        | REST
        v
FastAPI Web Adapter
        |
        +---- read-only RunCatalog ----> operational public RunStore
        |
        +---- read-only AnalyticsCatalog ----> public analytics RunStore
        |                                  (same store by default, optional archive)
        |
        +---- persistent Web Job Store (orchestration metadata only)
        |                |
        |                +---- bounded repair worker pool
        |                             |
        +--------------------> ApplicationService
                                      |
                            Agent / Repair / Verification
                                      |
                            Docker GPU / Compute Sanitizer
```

`RunStore` 继续是执行证据与状态的唯一事实源。Dashboard 没有第二套可写数据库，也不会把 UI 状态当作验证结果。列表接口一次构建 run/child inventory，再做服务端过滤与分页，避免对每一行重复扫描全部子 run。

## API

- `GET /api/health`：服务与 public RunStore 状态。
- `GET /api/cases`：合并 public case registry，返回已校验的 task/algorithm/tool metadata。
- `GET /api/stats`：诊断、修复与 verification 汇总。
- `GET /api/analytics/overview`：只读投影 public seed batch / evaluation manifests。
- `GET /api/analytics/evaluations/{run_id}`：A–E mode、failure family、latency、tool/model usage 与分页记录表。
- `GET /api/analytics/evaluations/compare?baseline=...&candidate=...`：仅在 split/corpus/modes/repeats/unit key 与执行完整性一致时计算 unit-level regression/improvement；否则返回明确不可比原因。
- `GET /api/analytics/evaluations/{run_id}/export.csv|json`：导出完整 public Analytics projection；CSV 对 spreadsheet formula 前缀做安全转义，并限制最大导出记录数。
- `GET /api/analytics/batches/{run_id}`：public seed batch 的 clean/mutant、Sanitizer detection 与注册状态。
- `GET /api/analytics/runs/{run_id}`：从 evaluation lineage 回到同一 public analytics store 中的 diagnosis run，只读展示 controller timeline、diagnosis、candidate、verification 与 evidence。
- `GET /api/analytics/runs/{run_id}/artifacts/{artifact_id}`：只读访问该 historical public diagnosis run 已注册 artifact；不会落到当前 operational store。
- `GET /api/runs`：服务端分页、搜索、status/kind 过滤。
- `GET /api/runs/{run_id}`：Diagnosis、Agent trajectory、repair rounds、candidate、verification 与 artifact inventory。
- `GET /api/runs/{run_id}/artifacts/{artifact_id}`：只读取该 public run 已注册的 artifact，并设置返回大小上限。
- `POST /api/jobs/repair`：立即返回 `202 + job_id`，由受限 worker pool 异步执行 public repair。
- `GET /api/jobs/{job_id}`：读取 Web 编排状态与已绑定 run_id；job 不是执行事实源。
- `POST /api/repair`：保留同步兼容接口；只接受 `case_####`，不接受任意宿主机路径。
- `POST /api/runs/{run_id}/verify`：复用 `ApplicationService.verify_exact()`，Web 层不实现另一套 verifier。

模型付费调用仍需要请求中显式 `allow_paid_calls=true`；实际调用边界继续由 `DevelopmentCallPolicy` 与 Agent budget 控制。Candidate 仍在既有隔离 GPU 后端执行。

## 前端

Dashboard 展示工程闭环，而不是聊天 UI：RunStore 运行、failure family、Agent trajectory、evidence-grounded diagnosis、多轮 candidate self-check/diff、四种 Sanitizer、strict verification 与 public artifact/log viewer。

Round 2 增加了 public case catalog selector、运行中自动轮询，以及可点击的 diagnosis evidence citation。citation resolver 同时支持 RunStore artifact ID 与 NVIDIA 文档 `DocumentChunk.chunk_id`，不会把 UI 链接当作新的证据来源。

Round 3 将 Repair 改为异步 Web Job：HTTP 提交立即返回，job 在 `ApplicationService` 创建 diagnosis run 后绑定 run_id，前端随后直接轮询 RunManifest/events。Run Detail 新增 controller timeline 与 Original Source / Selected Candidate Diff 并排视图。Web Job JSON 只用于 UI 编排，重启后遗留 QUEUED/RUNNING 会标记为 `WEB_CONTROLLER_RESTARTED`，不会改写 RunStore。

Round 4 增加独立 Analytics 页：读取 public seed batch summary 与 public evaluation manifest，展示大表格、A–E mode 对比、failure-family 分布、verified rate、latency、LLM/Sanitizer 调用、token 与已知 cost。它不读取 evaluator/private store，也不调用 `metrics.aggregate()` 伪造带私有标签的正式评测；这里只做 public operational projection。可通过 `GPU_AGENT_ANALYTICS_RUN_ROOT` 指向一个历史 public RunStore，同时保持当前 operational RunStore 不变。Analytics 会重新检查持久化 ArtifactRef 的 visibility；若配置目录包含 evaluator artifact，overview fail closed 为 `ANALYTICS_STORE_UNSAFE`。

Round 5 在 Analytics 内加入三组无第三方图表依赖的 mode 可视化（verified rate、mean latency、LLM calls），并把每条 evaluation record 的 immutable lineage 暴露为只读导航：record → diagnosis_run_id → historical public diagnosis run → evidence/candidate/verification。这个 drill-down 使用独立 `/api/analytics/runs/...` 命名空间，不复用当前 operational `/api/runs/...`，因此不会把历史 run 误接到当前 Strict Verify 等写操作。

Round 6 增加 Evaluation History Timeline、跨 evaluation 比较和 CSV/JSON 导出。趋势卡片只展示每个 run 自身的 public 描述统计；真正的 regression comparison 先验证 population compatibility，并按 `(case_id, template_id, mode, repeat)` 对齐 unit，禁止用 ordinal 或不同 split/corpus 的结果直接做“回归”判断。可比时显示 fixed→not-fixed regressions、not-fixed→fixed improvements、mode delta 与对应 diagnosis lineage；不可比时只返回原因，不计算 unit-level regression。

持续遇到的问题与修复过程记录在 [Full-stack 开发问题与修复记录](fullstack-development-log.md)。

历史 RunStore 可能含旧 schema。Web projection 对展示字段采用向后兼容读取，不修改或迁移原 artifact；核心 `ApplicationService` 仍使用当前严格模型。

## 开发运行

后端：

```bash
python -I -m pip install -e '.[web]'
export GPU_AGENT_REPOSITORY_ROOT="$PWD"
# 可选：把只读 Analytics 指向历史 public RunStore；未设置时复用当前 public RunStore。
export GPU_AGENT_ANALYTICS_RUN_ROOT="/path/to/historical/public"
gpu-agent web --host 127.0.0.1 --port 8000
```

前端开发：
```bash
cd dashboard
npm install
npm run dev
```

Vite 将 `/api` 代理到 `127.0.0.1:8000`。本地完整演示先执行 `npm run build`，随后 `gpu-agent web` 会从 `dashboard/dist` 提供前端。

## 验证

```bash
python -m pytest tests/web tests/unit/test_service.py -q
ruff check src/gpu_agent/web tests/web src/gpu_agent/cli.py
mypy --strict src/gpu_agent/web

cd dashboard
npm run test
npm run build
npm run lint
npm run test:e2e
```

Playwright E2E 使用受控 API mock 验证 Dashboard 与 Run Detail 主流程；Python API 测试使用临时 RunStore 验证持久化投影。两类测试均不能替代真实 GPU/LLM 验收。
