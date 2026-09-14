"""CLI workflows delegate to the same controller service and verification guard."""

from pathlib import Path
from typing import Annotated

import typer
from pydantic import ValidationError

from gpu_agent.config import Settings
from gpu_agent.environment import probe_environment

app = typer.Typer(no_args_is_help=True, help="Evidence-driven CUDA debugger.")


@app.callback()
def main() -> None:
    """Inspect the environment before running any workload."""


@app.command("env")
def environment_command(
    json_output: Annotated[bool, typer.Option("--json", help="Emit structured JSON.")] = False,
    cuda_root: Annotated[Path | None, typer.Option(envvar="GPU_AGENT_CUDA_ROOT")] = None,
    cuda_bin: Annotated[Path | None, typer.Option(envvar="GPU_AGENT_CUDA_BIN")] = None,
    host_compiler: Annotated[Path, typer.Option()] = Path("/usr/bin/g++"),
    nvidia_smi: Annotated[Path, typer.Option()] = Path("/usr/bin/nvidia-smi"),
) -> None:
    """Check toolchain metadata; exit 1 if not ready, 2 for invalid configuration."""
    try:
        settings = Settings(
            cuda_root=cuda_root if cuda_root is not None else Settings().cuda_root,
            cuda_bin=cuda_bin,
            host_compiler=host_compiler,
            nvidia_smi=nvidia_smi,
        )
    except ValidationError as exc:
        raise typer.BadParameter(str(exc)) from exc
    report = probe_environment(settings)
    if json_output:
        typer.echo(report.model_dump_json(indent=2))
    else:
        typer.echo(f"Toolchain metadata: {'READY' if report.ready else 'NOT READY'}")
        typer.echo(f"CUDA bin: {report.toolchain.cuda_bin}")
        typer.echo(f"NVCC: {report.toolchain.nvcc_version or 'unknown'}")
        typer.echo(f"Compute Sanitizer: {report.toolchain.sanitizer_version or 'unknown'}")
        for reason in report.reason_codes:
            typer.echo(f"- {reason}")
        typer.echo("GPU execution not verified; clean-kernel acceptance is a separate step.")
    raise typer.Exit(0 if report.ready else 1)


@app.command("diagnose")
def diagnose_command(source: Path) -> None:
    """Snapshot a source file/directory, investigate and attempt one model patch."""
    from gpu_agent.service import ApplicationService

    try:
        service = ApplicationService.configured()
        run = service.diagnose(source)
    except (OSError, ValueError):
        raise typer.BadParameter(
            "Source or controller configuration is unavailable or invalid."
        ) from None
    typer.echo(f"run_id {run.id}")
    result = service.diagnosis(run.id)
    typer.echo(result.diagnostic_outcome)
    for limitation in result.limitations:
        typer.echo(limitation)


@app.command("verify")
def verify_command(
    run_id: str,
    candidate_path: Annotated[Path | None, typer.Argument()] = None,
    generated_candidate: Annotated[bool, typer.Option("--generated-candidate")] = False,
    strict: Annotated[bool, typer.Option("--strict")] = False,
) -> None:
    """Verify exactly one generated candidate or controller-supplied unified diff."""
    from gpu_agent.service import ApplicationService

    if generated_candidate == (candidate_path is not None):
        raise typer.BadParameter("Select exactly one: CANDIDATE_PATH or --generated-candidate.")
    try:
        service = ApplicationService.configured()
        candidate_id = service.register_patch(run_id, candidate_path) if candidate_path else None
        result = service.verify(run_id, candidate_id, strict)
    except (OSError, ValueError):
        raise typer.BadParameter(
            "Run/candidate is unavailable or failed the registration guard."
        ) from None
    typer.echo(result.model_dump_json(indent=2))


@app.command("report")
def report_command(run_id: str) -> None:
    """Render public evidence, candidate, coverage and provider usage."""
    from gpu_agent.service import ApplicationService

    try:
        report = ApplicationService.configured().report(run_id)
    except (OSError, ValueError):
        raise typer.BadParameter("Run/report is unavailable or invalid.") from None
    typer.echo(report)
