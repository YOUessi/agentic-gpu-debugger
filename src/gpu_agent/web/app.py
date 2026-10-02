"""FastAPI adapter for the CUDA debugging operator console."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from gpu_agent.agent.provider import DevelopmentCallPolicy
from gpu_agent.public_task import PublicRepairInputError
from gpu_agent.repair import RepairPolicy
from gpu_agent.service import ApplicationService
from gpu_agent.web.cases import PublicCaseCatalog
from gpu_agent.web.catalog import RunCatalog
from gpu_agent.web.models import (
    CaseSummary,
    RepairRequest,
    RepairResponse,
    RunDetail,
    RunListResponse,
    RunStats,
    VerifyRequest,
    VerifyResponse,
)


def _repository_root() -> Path:
    configured = os.environ.get("GPU_AGENT_REPOSITORY_ROOT")
    return Path(configured).absolute() if configured else Path.cwd().absolute()


def _case_path(repository: Path, case_id: str) -> Path:
    root = repository / "benchmarks" / "public"
    selected = (root / case_id / "public_input").absolute()
    if selected.parent.parent != root.absolute() or not selected.is_dir():
        raise ValueError("public case is unavailable")
    return selected


def create_app(
    service: ApplicationService | None = None,
    *,
    repository: Path | None = None,
) -> FastAPI:
    runtime = service or ApplicationService.configured()
    catalog = RunCatalog(runtime.store)
    repo = (repository or _repository_root()).absolute()
    case_catalog = PublicCaseCatalog(repo)

    app = FastAPI(
        title="Agentic GPU Debugger Operator Console",
        version="0.3.0",
        docs_url="/api/docs",
        redoc_url=None,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
        allow_methods=["GET", "POST"],
        allow_headers=["content-type"],
    )

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "store": str(runtime.store.root)}

    @app.get("/api/cases", response_model=list[CaseSummary])
    def cases() -> list[CaseSummary]:
        return case_catalog.list_cases()

    @app.get("/api/stats", response_model=RunStats)
    def stats() -> RunStats:
        return catalog.stats()

    @app.get("/api/runs", response_model=RunListResponse)
    def runs(
        page: Annotated[int, Query(ge=1)] = 1,
        page_size: Annotated[int, Query(ge=1, le=100)] = 25,
        query: str | None = None,
        status: str | None = None,
        kind: str | None = None,
    ) -> RunListResponse:
        items, total = catalog.list_runs(
            page=page,
            page_size=page_size,
            query=query,
            status=status,
            kind=kind,
        )
        return RunListResponse(items=items, total=total, page=page, page_size=page_size)

    @app.get("/api/runs/{run_id}", response_model=RunDetail)
    def run_detail(run_id: str) -> RunDetail:
        try:
            return catalog.detail(run_id)
        except (OSError, ValueError):
            raise HTTPException(status_code=404, detail="RUN_NOT_FOUND") from None

    @app.get("/api/runs/{run_id}/artifacts/{artifact_id}", response_class=PlainTextResponse)
    def artifact(
        run_id: str,
        artifact_id: str,
        max_bytes: Annotated[int, Query(ge=1024, le=1024 * 1024)] = 256 * 1024,
    ) -> str:
        try:
            return catalog.artifact_text(run_id, artifact_id, max_bytes)
        except (OSError, ValueError):
            raise HTTPException(status_code=404, detail="ARTIFACT_NOT_FOUND") from None

    @app.post("/api/repair", response_model=RepairResponse)
    async def repair(request: RepairRequest) -> RepairResponse:
        try:
            source = _case_path(repo, request.case_id)
            worker = ApplicationService.configured() if service is None else runtime
            if request.allow_paid_calls:
                worker.allow_development_paid_calls(
                    DevelopmentCallPolicy(max_llm_calls=request.max_llm_calls)
                )
            run, verified = await run_in_threadpool(
                worker.repair,
                source,
                policy=RepairPolicy(max_candidates=request.max_candidates),
                mode=request.mode,
            )
        except PublicRepairInputError as exc:
            raise HTTPException(status_code=400, detail=exc.code) from None
        except (OSError, ValueError):
            raise HTTPException(status_code=400, detail="REPAIR_INPUT_INVALID") from None
        return RepairResponse(
            run_id=run.id,
            status=run.status.value,
            verification_verdict=verified.verdict.value if verified else None,
        )

    @app.post("/api/runs/{run_id}/verify", response_model=VerifyResponse)
    async def verify(run_id: str, request: VerifyRequest) -> VerifyResponse:
        try:
            result, verification_run = await run_in_threadpool(
                runtime.verify_exact,
                run_id,
                None,
                request.strict,
            )
        except (OSError, ValueError):
            raise HTTPException(status_code=400, detail="VERIFICATION_UNAVAILABLE") from None
        return VerifyResponse(
            verification_run_id=verification_run,
            result=result.model_dump(mode="json"),
        )

    dashboard = repo / "dashboard" / "dist"
    assets = dashboard / "assets"
    if assets.is_dir():
        app.mount("/assets", StaticFiles(directory=assets), name="dashboard-assets")

    @app.get("/", include_in_schema=False, response_model=None)
    def index() -> Response:
        page = dashboard / "index.html"
        if page.is_file():
            return FileResponse(page)
        return PlainTextResponse(
            "Dashboard is not built. Run npm install && npm run build in dashboard/.",
            status_code=200,
        )

    return app
