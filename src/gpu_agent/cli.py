"""CLI workflows delegate to the same controller service and verification guard."""

from pathlib import Path
from typing import Annotated

import typer
from pydantic import ValidationError

from gpu_agent.benchmark.evaluation import (
    EvaluationBindings,
    EvaluationRunner,
    EvaluationSelection,
    EvaluationSplit,
)
from gpu_agent.benchmark.executor import (
    CaseExecutionAttestationUnavailable,
    EvaluationExecutor,
    registered_cases,
)
from gpu_agent.config import Settings
from gpu_agent.environment import probe_environment

app = typer.Typer(no_args_is_help=True, help="Evidence-driven CUDA debugger.")
benchmark_app = typer.Typer(no_args_is_help=True)
app.add_typer(benchmark_app, name="benchmark")


@benchmark_app.command("validate")
def benchmark_validate(
    clean_execution: Path,
    mutant_execution: Path,
    corpus_root: Annotated[Path, typer.Option("--corpus-root")],
) -> None:
    """Register attested clean/mutant executions (currently fails closed).

    Existing runs cannot bind all CaseExecution input/toolchain/oracle assertions.
    No serialized booleans or run IDs may substitute for those missing attestations.
    """
    try:
        raise CaseExecutionAttestationUnavailable("CASE_EXECUTION_ATTESTATION_UNAVAILABLE")
    except CaseExecutionAttestationUnavailable:
        raise typer.BadParameter(
            "CASE_EXECUTION_ATTESTATION_UNAVAILABLE: current run artifacts cannot attest "
            "serialized case validation claims; corpus was not modified."
        ) from None


@benchmark_app.command("evaluate")
def benchmark_evaluate(
    mode: Annotated[str, typer.Option("--mode")],
    split: Annotated[str, typer.Option("--split")],
    repeats: Annotated[int, typer.Option("--repeats", min=3)],
    max_cost_usd: Annotated[float | None, typer.Option("--max-cost-usd")] = None,
    max_unit_cost_usd: Annotated[float | None, typer.Option("--max-unit-cost-usd")] = None,
    corpus_root: Annotated[Path | None, typer.Option("--corpus-root")] = None,
    case_root: Annotated[Path | None, typer.Option("--case-root")] = None,
    commit: Annotated[str | None, typer.Option("--commit")] = None,
    toolchain_hash: Annotated[str | None, typer.Option("--toolchain-hash")] = None,
    model_config_hash: Annotated[str | None, typer.Option("--model-config-hash")] = None,
) -> None:
    """Run registered cases with explicit caps and controller-owned provenance bindings."""
    from typing import cast

    from gpu_agent.agent.prompts import PROMPT_VERSION
    from gpu_agent.service import ApplicationService
    from gpu_agent.store import RunStore, reject_symlinks

    # This check precedes any configured service, corpus, or provider construction.
    if max_cost_usd is None or max_unit_cost_usd is None:
        raise typer.BadParameter("COST_CAP_REQUIRED: both total and unit caps must be explicit.")
    if mode not in {"A", "B", "C", "D", "E", "all"} or split not in {"development", "holdout"}:
        raise typer.BadParameter("EVALUATION_SELECTION_INVALID")
    if corpus_root is None or case_root is None:
        raise typer.BadParameter("EVALUATION_CONTROLLER_ROOTS_REQUIRED")
    try:
        bindings = EvaluationBindings.model_validate(
            {
                "commit": commit,
                "prompt_version": PROMPT_VERSION,
                "toolchain_hash": toolchain_hash,
                "model_config_hash": model_config_hash,
                "max_cost_usd": max_cost_usd,
                "max_unit_cost_usd": max_unit_cost_usd,
            }
        )
        corpus_root, case_root = corpus_root.absolute(), case_root.absolute()
        reject_symlinks(corpus_root)
        reject_symlinks(case_root)
        if not corpus_root.is_dir():
            raise ValueError("corpus unavailable")
        corpus = RunStore(corpus_root, visibility="evaluator" if split == "holdout" else "public")
        cases = {
            key: case
            for key, case in registered_cases(corpus).items()
            if case.split == ("private" if split == "holdout" else "public")
        }
        if not cases or any(
            case.toolchain_hash != bindings.toolchain_hash for case in cases.values()
        ):
            raise ValueError("registered case bindings unavailable")
        sources = {key: case_root / key / "public_input" / "kernel.cu" for key in cases}
        mode_count = 5 if mode == "all" else 1
        typer.echo(
            f"{len(cases)} case × {mode_count} mode × {repeats} repeats = "
            f"{len(cases) * mode_count * repeats} units"
        )
        typer.echo(
            f"Hard maximum cost: ${max_cost_usd:.2f}; unit reservation: ${max_unit_cost_usd:.2f}"
        )
        service = ApplicationService.configured()
        executor = EvaluationExecutor(service, corpus, sources)
        runner = EvaluationRunner(
            service.store,
            {key: case.template_id for key, case in cases.items()},
            executor.execute,
            commit=bindings.commit,
            prompt_version=bindings.prompt_version,
            toolchain_hash=bindings.toolchain_hash,
            model_config_hash=bindings.model_config_hash,
            max_cost_usd=max_cost_usd,
            max_unit_cost_usd=max_unit_cost_usd,
        )
        result = runner.run(cast(EvaluationSelection, mode), cast(EvaluationSplit, split), repeats)
    except (OSError, ValueError):
        raise typer.BadParameter("EVALUATION_CONTROLLER_INPUT_INVALID") from None
    typer.echo(f"run_id {result.run_id}")
    typer.echo(f"Executed {result.executed_units}/{result.expected_units}")
    if result.stopped_reason:
        typer.echo(result.stopped_reason)
        raise typer.Exit(1)


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
