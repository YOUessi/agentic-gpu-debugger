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
        +---- read-only RunCatalog ----> public RunStore
        |
        +---- ApplicationService
                    |
          Agent / Repair / Verification
                    |
        Docker GPU / Compute Sanitizer
```

`RunStore` 继续是执行证据与状态的唯一事实源。Dashboard 没有第二套可写数据库，也不会把 UI 状态当作验证结果。列表接口一次构建 run/child inventory，再做服务端过滤与分页，避免对每一行重复扫描全部子 run。

## API

- `GET /api/health`：服务与 public RunStore 状态。
- `GET /api/stats`：诊断、修复与 verification 汇总。
- `GET /api/runs`：服务端分页、搜索、status/kind 过滤。
- `GET /api/runs/{run_id}`：Diagnosis、Agent trajectory、repair rounds、candidate、verification 与 artifact inventory。
- `GET /api/runs/{run_id}/artifacts/{artifact_id}`：只读取该 public run 已注册的 artifact，并设置返回大小上限。
- `POST /api/repair`：只接受 `case_####`，服务端解析仓库内 `benchmarks/public/<case>/public_input`，不接受任意宿主机路径。
- `POST /api/runs/{run_id}/verify`：复用 `ApplicationService.verify_exact()`，Web 层不实现另一套 verifier。

模型付费调用仍需要请求中显式 `allow_paid_calls=true`；实际调用边界继续由 `DevelopmentCallPolicy` 与 Agent budget 控制。Candidate 仍在既有隔离 GPU 后端执行。

## 前端

Dashboard 展示工程闭环，而不是聊天 UI：RunStore 运行、failure family、Agent trajectory、evidence-grounded diagnosis、多轮 candidate self-check/diff、四种 Sanitizer、strict verification 与 public artifact/log viewer。

历史 RunStore 可能含旧 schema。Web projection 对展示字段采用向后兼容读取，不修改或迁移原 artifact；核心 `ApplicationService` 仍使用当前严格模型。

## 开发运行

后端：

```bash
python -I -m pip install -e '.[web]'
export GPU_AGENT_REPOSITORY_ROOT="$PWD"
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
