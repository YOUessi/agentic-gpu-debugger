"""CLI workflows delegate to the same controller service and verification guard."""

import os
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Literal

import typer
from pydantic import ValidationError

from gpu_agent.benchmark.builder import BenchmarkBuilder, UnvalidatedCaseError
from gpu_agent.benchmark.evaluation import (
    EvaluationRunner,
    EvaluationSelection,
    EvaluationSplit,
)
from gpu_agent.benchmark.validation import CaseExecutionAttestationUnavailable
from gpu_agent.config import Settings
from gpu_agent.environment import probe_environment
from gpu_agent.store import RunStore

if TYPE_CHECKING:
    from gpu_agent.benchmark.ledger import CorpusFamily
    from gpu_agent.benchmark.release import ReleaseEvidenceIndex

app = typer.Typer(no_args_is_help=True, help="Evidence-driven CUDA debugger.")
benchmark_app = typer.Typer(no_args_is_help=True)
release_app = typer.Typer(no_args_is_help=True)
app.add_typer(benchmark_app, name="benchmark")
app.add_typer(release_app, name="release")


def _release_artifact_context() -> tuple["CorpusFamily", tuple[Path, Path]]:
    from gpu_agent.benchmark.ledger import CorpusFamily

    family_root = os.environ.get("GPU_AGENT_CORPUS_FAMILY_ROOT")
    if not family_root:
        raise ValueError("trusted corpus family configuration is required")
    family = CorpusFamily.open(Path(family_root))
    forbidden_roots = (
        family.corpus_store("public").root,
        family.corpus_store("evaluator").root,
    )
    return family, forbidden_roots


def _derive_release_evidence(
    selection_path: Path,
    repository: Path,
    *,
    family: "CorpusFamily",
    forbidden_roots: tuple[Path, Path],
) -> "ReleaseEvidenceIndex":
    from gpu_agent.benchmark.controller_artifacts import read_private_external
    from gpu_agent.benchmark.release import ReleaseEvidenceIndex, ReleaseEvidenceSelection
    from gpu_agent.provenance import capture_repository_snapshot

    selection = ReleaseEvidenceSelection.model_validate_json(
        read_private_external(
            selection_path,
            repository=repository,
            forbidden_roots=forbidden_roots,
            limit=1024 * 1024,
        )
    )
    actual = capture_repository_snapshot(
        repository.absolute(), expected_commit=selection.repository.commit
    )
    return ReleaseEvidenceIndex.derive(
        selection,
        family.corpus_store("public"),
        family.corpus_store("evaluator"),
        family,
        actual,
    )


@release_app.command("derive-manifest")
def release_derive_manifest(
    selection: Annotated[Path, typer.Option("--selection")],
    repository: Annotated[Path, typer.Option("--repository")] = Path("."),
) -> None:
    """Print release claims derived from a frozen native evidence selection."""
    from gpu_agent.benchmark.release import (
        ReleaseManifest,
        validate_external_release_artifact_path,
    )

    try:
        selection = validate_external_release_artifact_path(selection, repository)
    except ValueError:
        raise typer.BadParameter("RELEASE_ARTIFACT_PATH_INVALID") from None
    try:
        family, forbidden_roots = _release_artifact_context()
    except ValueError:
        raise typer.BadParameter("RELEASE_EVIDENCE_INCOMPLETE") from None
    try:
        selection = validate_external_release_artifact_path(
            selection,
            repository,
            forbidden_roots=forbidden_roots,
        )
    except ValueError:
        raise typer.BadParameter("RELEASE_ARTIFACT_PATH_INVALID") from None
    try:
        evidence = _derive_release_evidence(
            selection,
            repository,
            family=family,
            forbidden_roots=forbidden_roots,
        )
        manifest = ReleaseManifest.from_evidence(evidence)
    except (OSError, ValueError):
        raise typer.BadParameter("RELEASE_EVIDENCE_INCOMPLETE") from None
    typer.echo(manifest.model_dump_json(indent=2))


@release_app.command("check")
def release_check(
    manifest: Annotated[Path, typer.Option("--manifest")],
    selection: Annotated[Path, typer.Option("--selection")],
    repository: Annotated[Path, typer.Option("--repository")] = Path("."),
) -> None:
    """Validate declarative release claims against native, same-commit evidence."""
    from gpu_agent.benchmark.controller_artifacts import read_private_external
    from gpu_agent.benchmark.release import (
        ReleaseGate,
        ReleaseManifest,
        validate_external_release_artifact_path,
    )

    try:
        manifest = validate_external_release_artifact_path(manifest, repository)
        selection = validate_external_release_artifact_path(selection, repository)
    except ValueError:
        raise typer.BadParameter("RELEASE_ARTIFACT_PATH_INVALID") from None
    try:
        family, forbidden_roots = _release_artifact_context()
    except ValueError:
        raise typer.BadParameter("RELEASE_EVIDENCE_INVALID") from None
    try:
        manifest = validate_external_release_artifact_path(
            manifest,
            repository,
            forbidden_roots=forbidden_roots,
        )
        selection = validate_external_release_artifact_path(
            selection,
            repository,
            forbidden_roots=forbidden_roots,
        )
    except ValueError:
        raise typer.BadParameter("RELEASE_ARTIFACT_PATH_INVALID") from None
    try:
        claims = ReleaseManifest.model_validate_json(
            read_private_external(
                manifest,
                repository=repository,
                forbidden_roots=forbidden_roots,
                limit=1024 * 1024,
            )
        )
        evidence = _derive_release_evidence(
            selection,
            repository,
            family=family,
            forbidden_roots=forbidden_roots,
        )
        result = ReleaseGate().check(claims, evidence)
    except (OSError, ValueError):
        raise typer.BadParameter("RELEASE_EVIDENCE_INVALID") from None
    typer.echo(result.model_dump_json(indent=2))
    if not result.passed:
        raise typer.Exit(1)


@release_app.command("collect-evidence")
def release_collect_evidence(
    repository: Annotated[Path, typer.Option("--repository")],
    development_evaluation_run_id: Annotated[str, typer.Option("--development-evaluation-run-id")],
) -> None:
    """Bind the fixed live suite to an already signed development evaluation."""
    from gpu_agent.benchmark.ledger import CorpusFamily
    from gpu_agent.benchmark.schedule_authority import EvaluationScheduleVerifier
    from gpu_agent.contracts import RunStatus
    from gpu_agent.release_controller import ReleaseEvidenceController

    try:
        family_root = os.environ.get("GPU_AGENT_CORPUS_FAMILY_ROOT")
        if family_root is None:
            raise ValueError("trusted corpus family configuration is required")
        family = CorpusFamily.open(Path(family_root))
        store = family.corpus_store("public")
        evaluation = store.load(development_evaluation_run_id)
        if (
            evaluation.kind != "evaluation"
            or evaluation.status != RunStatus.COMPLETED
            or evaluation.binding is None
            or evaluation.binding.purpose != "evaluation"
        ):
            raise ValueError("development evaluation is not complete and bound")
        receipt = EvaluationScheduleVerifier.for_family(family, store).verify(
            development_evaluation_run_id
        )
        if receipt.request.split != "development":
            raise ValueError("release tests require the development evaluation binding")
        binding = evaluation.binding.model_copy(update={"purpose": "release_acceptance"})
        run_id = ReleaseEvidenceController(
            store,
            repository,
            binding,
        ).collect(receipt.request.corpus_cutoff)
    except (OSError, ValueError):
        raise typer.BadParameter("RELEASE_TEST_EVIDENCE_FAILED") from None
    typer.echo(f"release_test_run_id {run_id}")


@benchmark_app.command("provision-family")
def benchmark_provision_family(
    controller_root: Annotated[Path, typer.Option("--controller-root")],
    public_store: Annotated[Path, typer.Option("--public-store")],
    evaluator_store: Annotated[Path, typer.Option("--evaluator-store")],
    repository: Annotated[Path, typer.Option("--repository")],
    schedule_public_key: Annotated[Path, typer.Option("--schedule-public-key")],
) -> None:
    """Provision a corpus family with an external production schedule public key."""
    from gpu_agent.benchmark.controller_config import provision_production_family

    try:
        family = provision_production_family(
            controller_root=controller_root,
            public_store=public_store,
            evaluator_store=evaluator_store,
            repository=repository,
            schedule_public_key=schedule_public_key,
        )
    except (OSError, ValueError):
        raise typer.BadParameter("PRODUCTION_FAMILY_PROVISION_FAILED") from None
    typer.echo(f"controller_root {family.root}")
    typer.echo(f"namespace_hash {family.namespace_hash}")


@benchmark_app.command("attest-pricing")
def benchmark_attest_pricing(
    repository: Annotated[Path, typer.Option("--repository")],
    commit: Annotated[str, typer.Option("--commit")],
    input_usd_per_million: Annotated[float, typer.Option("--input-usd-per-million", min=0)],
    output_usd_per_million: Annotated[float, typer.Option("--output-usd-per-million", min=0)],
    source_uri: Annotated[str, typer.Option("--source-uri")],
    reviewed_at: Annotated[datetime, typer.Option("--reviewed-at")],
    source_content_hash: Annotated[str, typer.Option("--source-content-hash")],
    output: Annotated[Path, typer.Option("--output")],
) -> None:
    """Write an owner-only reviewed provider rate card bound to the clean commit."""
    from gpu_agent.benchmark.controller_config import (
        reviewed_pricing_attestation,
        write_private_new,
    )

    try:
        attestation = reviewed_pricing_attestation(
            repository=repository,
            expected_commit=commit,
            input_usd_per_million=input_usd_per_million,
            output_usd_per_million=output_usd_per_million,
            source_uri=source_uri,
            reviewed_at=reviewed_at,
            source_content_hash=source_content_hash,
        )
        write_private_new(output, attestation.model_dump_json().encode())
    except (OSError, ValueError):
        raise typer.BadParameter("PRICING_ATTESTATION_FAILED") from None
    typer.echo(f"model_config_hash {attestation.model_config_hash}")
    typer.echo(str(output.absolute()))


def _configured_evaluation_runner(
    *,
    repository: Path,
    case_root: Path,
    corpus_root: Path,
    split: str,
    commit: str,
    toolchain_hash: str,
    model_config_hash: str,
    max_cost_usd: float,
    max_unit_cost_usd: float,
) -> EvaluationRunner:
    """Build the paid runner only from controller-owned, pre-attested configuration."""
    from urllib.parse import urlsplit

    from gpu_agent.agent.policy import LLMCallGate
    from gpu_agent.agent.prompts import PROMPT_VERSION
    from gpu_agent.agent.provider import OpenAIProviderSettings, OpenAIResponsesProvider
    from gpu_agent.benchmark.evaluation import EvaluationProviderPolicy, PricingAttestation
    from gpu_agent.benchmark.executor import EvaluationExecutor, registered_cases
    from gpu_agent.benchmark.holdout import HoldoutController
    from gpu_agent.benchmark.ledger import CorpusFamily
    from gpu_agent.benchmark.schedule_authority import (
        EvaluationScheduleVerifier,
        ExternalCommandScheduleCommitClient,
    )
    from gpu_agent.service import ApplicationService
    from gpu_agent.store import read_regular

    family_root = os.environ.get("GPU_AGENT_CORPUS_FAMILY_ROOT")
    authority_command = os.environ.get("GPU_AGENT_SCHEDULE_AUTHORITY_COMMAND")
    pricing_path = os.environ.get("GPU_AGENT_PRICING_ATTESTATION")
    if not family_root or not authority_command or not pricing_path:
        raise ValueError("production evaluation controller is incomplete")
    family = CorpusFamily.open(Path(family_root))
    if family.schedule_authority_profile != "PRODUCTION":
        raise ValueError("production schedule authority is not configured")
    visibility: Literal["public", "evaluator"] = "public" if split == "development" else "evaluator"
    corpus = family.corpus_store(visibility)
    if corpus.root != corpus_root.absolute():
        raise ValueError("requested corpus root differs from trusted family")
    service = ApplicationService.for_release(
        repository,
        purpose="evaluation",
        expected_commit=commit,
        prompt_version=PROMPT_VERSION,
        model_config_hash=model_config_hash,
        require_corpus_family=True,
    )
    binding = service.binding
    if (
        binding is None
        or binding.toolchain_lock_hash != toolchain_hash
        or binding.model_config_hash != model_config_hash
    ):
        raise ValueError("evaluation binding differs from requested configuration")
    pricing = PricingAttestation.model_validate_json(
        read_regular(Path(pricing_path).absolute(), 256 * 1024)
    )
    if pricing.source != "REVIEWED":
        raise ValueError("reviewed pricing attestation is required")
    settings = OpenAIProviderSettings.from_environment()
    probe = OpenAIResponsesProvider(settings, LLMCallGate(), service.store, "0" * 32)
    probe.ensure_available()
    endpoint_host = urlsplit(settings.endpoint or "").hostname or ""
    policy = EvaluationProviderPolicy(
        provider=probe.provider_name,
        endpoint_host=endpoint_host,
        configured_model=probe.model_name or "",
        allowed_response_models=[probe.model_name or ""],
        prompt_version=PROMPT_VERSION,
        pricing_hash=pricing.rate_card_hash,
    )
    if (
        policy.sha256 != model_config_hash
        or pricing.provider != policy.provider
        or pricing.model != policy.configured_model
    ):
        raise ValueError("provider, pricing, and model binding differ")
    service._bind_pricing_attestation(pricing)
    verifier = EvaluationScheduleVerifier.for_family(family, service.store)
    cases = registered_cases(corpus, binding, family)
    source_root = case_root.absolute()
    sources = {case_id: source_root / case_id / "public_input" for case_id in cases}
    if not sources or any(not path.is_dir() for path in sources.values()):
        raise ValueError("registered evaluation source is unavailable")
    holdout_controller = None
    holdout_batch = None
    if split == "holdout":
        holdout_controller = HoldoutController(
            service.store,
            corpus,
            binding=binding,
            _schedule_verifier=verifier,
        )
        holdout_batch = holdout_controller.prepare()
    executor = EvaluationExecutor(
        service,
        corpus,
        sources,
        holdout_controller=holdout_controller,
        holdout_batch=holdout_batch,
        _corpus_family=family,
        _schedule_verifier=verifier,
    )
    return EvaluationRunner(
        service.store,
        executor,
        commit=commit,
        prompt_version=PROMPT_VERSION,
        toolchain_hash=toolchain_hash,
        model_config_hash=model_config_hash,
        binding=binding,
        max_cost_usd=max_cost_usd,
        max_unit_cost_usd=max_unit_cost_usd,
        holdout_controller=holdout_controller,
        holdout_batch=holdout_batch,
        schedule_client=ExternalCommandScheduleCommitClient(Path(authority_command)),
    )


@benchmark_app.command("validate")
def benchmark_validate(
    clean_execution: str,
    mutant_execution: str,
    corpus_root: Annotated[Path, typer.Option("--corpus-root")],
    visibility: Annotated[Literal["public", "evaluator"], typer.Option("--visibility")],
) -> None:
    """Register two exact native clean/mutant execution run IDs."""
    try:
        builder = BenchmarkBuilder(RunStore(corpus_root, visibility=visibility))
        manifest = builder.register(builder.validate(clean_execution, mutant_execution))
    except (OSError, ValueError, CaseExecutionAttestationUnavailable, UnvalidatedCaseError):
        raise typer.BadParameter(
            "CASE_EXECUTION_ATTESTATION_UNAVAILABLE: exact native run artifacts are "
            "missing, unbound, incomplete, or inconsistent; corpus was not modified."
        ) from None
    # Evaluator identities are deliberately never echoed by this public CLI surface.
    typer.echo(f"registered corpus evidence ({manifest.split})")


@benchmark_app.command("run-seeds")
def benchmark_run_seeds(
    data_root: Annotated[Path, typer.Option("--data-root")],
    repository: Annotated[Path, typer.Option("--repository")] = Path("."),
    case: Annotated[list[str] | None, typer.Option("--case")] = None,
    preflight_only: Annotated[bool, typer.Option("--preflight-only")] = False,
    register: Annotated[bool, typer.Option("--register")] = False,
) -> None:
    """Run public clean/mutant seeds serially; no LLM calls or paid evaluation."""
    from gpu_agent.benchmark.batch import prepare_seed_batch, run_public_seeds
    from gpu_agent.benchmark.batch_security import BatchInputError

    try:
        prepared = prepare_seed_batch(repository, data_root, tuple(case or ()))
        if preflight_only:
            typer.echo(prepared.report.model_dump_json(indent=2))
            return
        result = run_public_seeds(prepared, register=register, progress=typer.echo)
    except BatchInputError as exc:
        raise typer.BadParameter(str(exc)) from None
    except (OSError, ValueError) as exc:
        raise typer.BadParameter(
            f"BATCH_SETUP_FAILED ({type(exc).__name__}): check the data root and corpus family."
        ) from None
    typer.echo(f"Batch {result.batch_run_id}: {result.status}")
    typer.echo("Use benchmark batch-report or export-batch to inspect retained evidence.")
    if result.status == "CANCELLED":
        raise typer.Exit(130)
    if not result.all_passed:
        raise typer.Exit(1)


@benchmark_app.command("batch-report")
def benchmark_batch_report(
    batch_run_id: str,
    data_root: Annotated[Path, typer.Option("--data-root")],
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Read a saved batch or its last progress snapshot without rerunning any GPU work."""
    from gpu_agent.benchmark.batch_report import load_batch_summary, render_batch
    from gpu_agent.benchmark.batch_security import public_store_path

    try:
        public_root = public_store_path(data_root)
        if not public_root.is_dir():
            raise ValueError("RunStore does not exist")
        summary = load_batch_summary(RunStore(public_root), batch_run_id)
    except (OSError, ValueError):
        raise typer.BadParameter("BATCH_REPORT_UNAVAILABLE") from None
    typer.echo(summary.model_dump_json(indent=2) if json_output else render_batch(summary))


@benchmark_app.command("export-batch")
def benchmark_export_batch(
    batch_run_id: str,
    data_root: Annotated[Path, typer.Option("--data-root")],
    output: Annotated[Path, typer.Option("--output")],
) -> None:
    """Export only this public batch's registered artifacts; no evaluator state or keys."""
    from gpu_agent.benchmark.batch_report import export_batch
    from gpu_agent.benchmark.batch_security import public_store_path

    try:
        public_root = public_store_path(data_root)
        if not public_root.is_dir():
            raise ValueError("RunStore does not exist")
        path = export_batch(RunStore(public_root), batch_run_id, output)
    except (OSError, ValueError) as exc:
        raise typer.BadParameter(f"BATCH_EXPORT_FAILED ({type(exc).__name__})") from None
    typer.echo(str(path))


@benchmark_app.command("evaluate")
def benchmark_evaluate(
    ctx: typer.Context,
    mode: Annotated[str, typer.Option("--mode")],
    split: Annotated[str, typer.Option("--split")],
    repeats: Annotated[int, typer.Option("--repeats", min=3)],
    max_cost_usd: Annotated[float | None, typer.Option("--max-cost-usd")] = None,
    max_unit_cost_usd: Annotated[float | None, typer.Option("--max-unit-cost-usd")] = None,
    corpus_root: Annotated[Path | None, typer.Option("--corpus-root")] = None,
    case_root: Annotated[Path | None, typer.Option("--case-root")] = None,
    repository: Annotated[Path, typer.Option("--repository")] = Path("."),
    commit: Annotated[str | None, typer.Option("--commit")] = None,
    toolchain_hash: Annotated[str | None, typer.Option("--toolchain-hash")] = None,
    model_config_hash: Annotated[str | None, typer.Option("--model-config-hash")] = None,
) -> None:
    """Paid batches require cost attestation; injected controller runners support offline tests."""
    from typing import cast

    from gpu_agent.benchmark.executor import CostBoundUnavailable

    # This check precedes any configured service, corpus, or provider construction.
    if max_cost_usd is None or max_unit_cost_usd is None:
        raise typer.BadParameter("COST_CAP_REQUIRED: both total and unit caps must be explicit.")
    try:
        if isinstance(ctx.obj, EvaluationRunner):
            runner = ctx.obj
        else:
            if None in {corpus_root, case_root, commit, toolchain_hash, model_config_hash}:
                raise CostBoundUnavailable("COST_BOUND_UNAVAILABLE")
            assert corpus_root is not None and case_root is not None
            assert commit is not None and toolchain_hash is not None
            assert model_config_hash is not None
            runner = _configured_evaluation_runner(
                repository=repository,
                case_root=case_root,
                corpus_root=corpus_root,
                split=split,
                commit=commit,
                toolchain_hash=toolchain_hash,
                model_config_hash=model_config_hash,
                max_cost_usd=max_cost_usd,
                max_unit_cost_usd=max_unit_cost_usd,
            )
    except (CostBoundUnavailable, OSError, ValueError):
        raise typer.BadParameter(
            "COST_BOUND_UNAVAILABLE: paid evaluation requires reviewed pricing attestation "
            "before provider execution."
        ) from None
    if mode not in {"A", "B", "C", "D", "E", "all"} or split not in {"development", "holdout"}:
        raise typer.BadParameter("EVALUATION_SELECTION_INVALID")
    try:
        if (
            runner.bindings.max_cost_usd != max_cost_usd
            or runner.bindings.max_unit_cost_usd != max_unit_cost_usd
        ):
            raise ValueError("injected runner caps differ from requested caps")
        if runner.schedule_client is None:
            raise typer.BadParameter(
                "SCHEDULE_ATTESTATION_REQUIRED: external schedule authority is unavailable."
            )
        schedule = EvaluationRunner._schedule(
            runner, cast(EvaluationSelection, mode), cast(EvaluationSplit, split), repeats
        )
        cases = {item.case_id for item in schedule.items}
        mode_count = 5 if mode == "all" else 1
        typer.echo(
            f"{len(cases)} case × {mode_count} mode × {repeats} repeats = "
            f"{len(cases) * mode_count * repeats} units"
        )
        typer.echo(
            f"Cost reservation: ${max_cost_usd:.2f}; unit reservation: ${max_unit_cost_usd:.2f}"
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
