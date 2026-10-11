"""CLI workflows delegate to the same controller service and verification guard."""

import hashlib
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
knowledge_app = typer.Typer(no_args_is_help=True, help="Offline knowledge inspection.")
app.add_typer(knowledge_app, name="knowledge")


@benchmark_app.command("validate-diversity")
def validate_diversity(
    output: Annotated[Path, typer.Option()],
    repository: Annotated[Path, typer.Option()] = Path("."),
    case: Annotated[list[str] | None, typer.Option("--case")] = None,
    role: Annotated[list[str] | None, typer.Option("--role")] = None,
) -> None:
    """Run native acceptance for new public algorithms; no model or registration."""
    from gpu_agent.benchmark.diversity import run_diversity

    try:
        report = run_diversity(
            repository,
            output,
            tuple(case or ()),
            tuple(role) if role is not None else ("clean", "mutant", "ablation"),
        )
    except (OSError, ValueError) as exc:
        typer.echo(f"DIVERSITY_INPUT_INVALID: {exc}", err=True)
        raise typer.Exit(2) from None
    typer.echo(f"Report: {output / 'report.json'}")
    if not report["passed"]:
        raise typer.Exit(1)


@knowledge_app.command("metadata")
def knowledge_metadata(
    manifest: Annotated[Path, typer.Option()],
    receipts: Annotated[Path, typer.Option()],
    expected_receipts_sha256: Annotated[str, typer.Option()],
    source_id: Annotated[str | None, typer.Option()] = None,
) -> None:
    """Read recorded source versions and provenance; never infer release-note contents."""
    import json

    from gpu_agent.knowledge.metadata import lookup_metadata

    try:
        result = lookup_metadata(manifest, receipts, expected_receipts_sha256, source_id)
    except (ValueError, OSError):
        typer.echo("KNOWLEDGE_METADATA_INVALID", err=True)
        raise typer.Exit(2) from None
    typer.echo(json.dumps(result, ensure_ascii=False, indent=2))


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


def _holdout_scoring_error_code(exc: OSError | ValueError) -> str:
    if isinstance(exc, OSError):
        return "HOLDOUT_SCORING_FAILED"
    message = str(exc)
    if any(token in message for token in ("conflict", "differs", "ambiguous", "claim is unsafe")):
        return "HOLDOUT_SCORING_CONFLICT"
    if any(
        token in message
        for token in (
            "label package",
            "private external artifact",
            "external artifact path",
            "scoring roots are invalid",
        )
    ):
        return "HOLDOUT_LABEL_PACKAGE_INVALID"
    if "external artifact output is unsafe" in message:
        return "HOLDOUT_SCORING_FAILED"
    return "HOLDOUT_SCORING_EVIDENCE_MISMATCH"


def _release_freeze_error_code(exc: OSError | ValueError) -> str:
    message = str(exc)
    if "output conflicts" in message:
        return "RELEASE_SELECTION_OUTPUT_CONFLICT"
    if "output is unsafe" in message or "artifact path is unsafe" in message:
        return "RELEASE_SELECTION_OUTPUT_UNSAFE"
    if "repository changed" in message:
        return "RELEASE_REPOSITORY_CHANGED"
    if "evidence is incomplete" in message:
        return "RELEASE_EVIDENCE_INCOMPLETE"
    return "RELEASE_ROOTS_INVALID"


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
        repository,
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
    except (OSError, ValueError):
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
    except (OSError, ValueError):
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


@release_app.command("freeze-selection")
def release_freeze_selection(
    development_evaluation_run_id: Annotated[str, typer.Option("--development-evaluation-run-id")],
    holdout_evaluation_run_id: Annotated[str, typer.Option("--holdout-evaluation-run-id")],
    private_binding_run_id: Annotated[str, typer.Option("--private-binding-run-id")],
    release_test_run_id: Annotated[str, typer.Option("--release-test-run-id")],
    output: Annotated[Path, typer.Option("--output")],
    repository: Annotated[Path, typer.Option("--repository")],
) -> None:
    """Gate and atomically publish a canonical four-root release selection."""
    from gpu_agent.benchmark.controller_artifacts import validate_external_artifact_path
    from gpu_agent.benchmark.release import ReleaseEvidenceFreezer, ReleaseEvidenceRoots

    try:
        if not repository.is_absolute():
            raise ValueError("release roots are invalid")
        family, forbidden_roots = _release_artifact_context()
        output = validate_external_artifact_path(
            output,
            repository=repository,
            forbidden_roots=forbidden_roots,
        )
        roots = ReleaseEvidenceRoots(
            development_evaluation_run_id=development_evaluation_run_id,
            holdout_evaluation_run_id=holdout_evaluation_run_id,
            private_binding_run_id=private_binding_run_id,
            release_test_run_id=release_test_run_id,
        )
        frozen = ReleaseEvidenceFreezer().freeze(
            roots,
            family.corpus_store("public"),
            family.corpus_store("evaluator"),
            family,
            repository,
            output,
        )
    except (OSError, ValueError) as exc:
        raise typer.BadParameter(_release_freeze_error_code(exc)) from None
    typer.echo(f"selection_path {frozen.output}")
    typer.echo(f"selection_sha256 {frozen.selection_sha256}")
    typer.echo(f"corpus_cutoff {frozen.corpus_cutoff}")
    typer.echo(f"public_case_count {frozen.public_case_count}")
    typer.echo(f"private_case_count {frozen.private_case_count}")
    typer.echo(f"acceptance_run_count {frozen.acceptance_run_count}")


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


@benchmark_app.command("score-holdout")
def benchmark_score_holdout(
    evaluation_run_id: Annotated[str, typer.Option("--evaluation-run-id")],
    private_binding_run_id: Annotated[str, typer.Option("--private-binding-run-id")],
    labels: Annotated[Path, typer.Option("--labels")],
    repository: Annotated[Path, typer.Option("--repository")],
    metrics_output: Annotated[Path | None, typer.Option("--metrics-output")] = None,
) -> None:
    """Bind one complete private adjudication package to a holdout evaluation."""
    from gpu_agent.benchmark.controller_artifacts import (
        read_private_external,
        write_private_atomic_new,
    )
    from gpu_agent.benchmark.holdout_scoring import HoldoutScoringController
    from gpu_agent.benchmark.ledger import CorpusFamily
    from gpu_agent.benchmark.schedule_authority import EvaluationScheduleVerifier
    from gpu_agent.contracts import RunStatus

    try:
        family_root = os.environ.get("GPU_AGENT_CORPUS_FAMILY_ROOT")
        if family_root is None or not repository.is_absolute():
            raise ValueError("holdout scoring controller input is invalid")
        family = CorpusFamily.open(Path(family_root))
        public = family.corpus_store("public")
        evaluator = family.corpus_store("evaluator")
        forbidden_roots = (public.root, evaluator.root)
        # Fail closed on path and private-file metadata before any scoring mutation.
        read_private_external(
            labels,
            repository=repository,
            forbidden_roots=forbidden_roots,
            limit=16 * 1024 * 1024,
        )
        evaluation = public.load(evaluation_run_id)
        if evaluation.binding is None or evaluation.binding.purpose != "evaluation":
            raise ValueError("holdout evaluation binding is invalid")
        controller = HoldoutScoringController(
            public,
            evaluator,
            binding=evaluation.binding,
            _schedule_verifier=EvaluationScheduleVerifier.for_family(family, public),
        )
        result = controller.score(
            evaluation_run_id,
            private_binding_run_id,
            labels,
            repository,
        )
        session = evaluator.load(result.scoring_run_id)
        metrics_refs = [
            ref for ref in session.artifact_refs if ref.name == "holdout-scoring/metrics.json"
        ]
        if (
            session.kind != "holdout_scoring"
            or session.status != RunStatus.COMPLETED
            or session.binding != evaluation.binding
            or len(metrics_refs) != 1
        ):
            raise ValueError("holdout scoring evidence is incomplete")
        metrics = evaluator.read(metrics_refs[0])
        if metrics_refs[0].sha256 != result.metrics_hash:
            raise ValueError("holdout scoring metrics hash is invalid")
        if metrics_output is not None:
            write_private_atomic_new(
                metrics_output,
                metrics,
                repository=repository,
                forbidden_roots=forbidden_roots,
            )
    except (OSError, ValueError) as exc:
        raise typer.BadParameter(_holdout_scoring_error_code(exc)) from None
    typer.echo(f"scoring_run_id {result.scoring_run_id}")
    typer.echo(f"scored {result.scored_count}/{result.expected_record_count}")
    typer.echo(f"metrics_sha256 {result.metrics_hash}")


def _configured_evaluation_runner(
    *,
    repository: Path,
    case_root: Path,
    corpus_root: Path,
    split: str,
    commit: str,
    toolchain_hash: str,
    model_config_hash: str,
) -> EvaluationRunner:
    """Build the paid runner only from controller-owned, pre-attested configuration."""
    from urllib.parse import urlsplit

    from gpu_agent.agent.policy import LLMCallGate
    from gpu_agent.agent.prompts import PROMPT_VERSION
    from gpu_agent.agent.provider import OpenAIProviderSettings, OpenAIResponsesProvider
    from gpu_agent.benchmark.controller_config import validate_production_store_configuration
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
    public_store, evaluator_store = validate_production_store_configuration(family, repository)
    visibility: Literal["public", "evaluator"] = "public" if split == "development" else "evaluator"
    corpus = public_store if visibility == "public" else evaluator_store
    if corpus.root != corpus_root.absolute():
        raise ValueError("requested corpus root differs from trusted family")
    service = ApplicationService.for_release(
        repository,
        purpose="evaluation",
        expected_commit=commit,
        prompt_version=PROMPT_VERSION,
        model_config_hash=model_config_hash,
        require_corpus_family=True,
        workflow_visibility="public",
        cost_policy="record_only",
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
    holdout_service = None
    if split == "holdout":
        holdout_service = ApplicationService.for_release(
            repository,
            purpose="evaluation",
            expected_commit=commit,
            prompt_version=PROMPT_VERSION,
            model_config_hash=model_config_hash,
            require_corpus_family=True,
            workflow_visibility="evaluator",
            cost_policy="record_only",
        )
        if holdout_service.binding != binding:
            raise ValueError("paired evaluation bindings differ")
        holdout_service._bind_pricing_attestation(pricing)
    verifier = EvaluationScheduleVerifier.for_family(family, service.store)
    cases = registered_cases(corpus, binding, family)
    source_root = case_root.absolute()
    sources = {case_id: source_root / case_id / "public_input" for case_id in cases}
    if not sources or any(not path.is_dir() for path in sources.values()):
        raise ValueError("registered evaluation source is unavailable")
    # Check the entire selected corpus before reserving any aliases, unit or
    # signed schedule. A valid registration can use a different source layout
    # from the runtime snapshot (notably shared private harness/input files).
    for case_id, source in sources.items():
        case = cases[case_id]
        if (
            hashlib.sha256(read_regular(source / "kernel.cu", 4 * 1024 * 1024)).hexdigest()
            != case.source_hash
        ):
            raise ValueError("registered evaluation source hash mismatch")
        ApplicationService._public_input(source / "kernel.cu", case.input_set_hash, None)
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
        holdout_service=holdout_service,
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
    corpus_root: Annotated[Path | None, typer.Option("--corpus-root")] = None,
    case_root: Annotated[Path | None, typer.Option("--case-root")] = None,
    repository: Annotated[Path, typer.Option("--repository")] = Path("."),
    commit: Annotated[str | None, typer.Option("--commit")] = None,
    toolchain_hash: Annotated[str | None, typer.Option("--toolchain-hash")] = None,
    model_config_hash: Annotated[str | None, typer.Option("--model-config-hash")] = None,
) -> None:
    """Run a signed evaluation; record usage and costs without dollar limits."""
    from typing import cast

    from gpu_agent.benchmark.executor import CostBoundUnavailable

    if mode not in {"A", "B", "C", "D", "E", "all"} or split not in {
        "development",
        "holdout",
    }:
        raise typer.BadParameter("EVALUATION_SELECTION_INVALID")
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
            )
    except (CostBoundUnavailable, OSError, ValueError):
        raise typer.BadParameter(
            "COST_BOUND_UNAVAILABLE: paid evaluation requires reviewed pricing attestation "
            "before provider execution."
        ) from None
    try:
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
        typer.echo("Usage and cost are recorded only; no dollar ceiling.")
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
def diagnose_command(
    source: Path,
    allow_paid_calls: Annotated[
        bool,
        typer.Option(
            "--allow-paid-calls",
            help="Development only: send real provider requests. Runs are never evaluation "
            "or release evidence.",
        ),
    ] = False,
    max_llm_calls: Annotated[int, typer.Option("--max-llm-calls", min=1, max=40)] = 40,
) -> None:
    """Snapshot a source file/directory, investigate and attempt one model patch.

    Without --allow-paid-calls no provider request is sent and the run records
    PAID_CALLS_NOT_ALLOWED. With it, physical requests (including retries) are bounded
    by --max-llm-calls. Token usage is recorded without a dollar ceiling.
    """
    from gpu_agent.agent.provider import DevelopmentCallPolicy
    from gpu_agent.service import ApplicationService

    policy = DevelopmentCallPolicy(max_llm_calls=max_llm_calls) if allow_paid_calls else None
    try:
        service = ApplicationService.configured()
        if policy is not None:
            service.allow_development_paid_calls(policy)
            typer.echo("DEVELOPMENT RUN: paid provider calls enabled; not evaluation evidence.")
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


@app.command("repair")
def repair_command(
    source: Path,
    allow_paid_calls: Annotated[bool, typer.Option("--allow-paid-calls")] = False,
    max_candidates: Annotated[int, typer.Option("--max-candidates", min=1, max=20)] = 3,
    max_llm_calls: Annotated[int, typer.Option("--max-llm-calls", min=1, max=40)] = 40,
    reinvestigate: Annotated[
        bool, typer.Option("--reinvestigate", help="Enable v3 investigation of failed candidates.")
    ] = False,
    max_reinvestigations: Annotated[int, typer.Option("--max-reinvestigations", min=0, max=3)] = 1,
    unbounded_sanitizer_calls: Annotated[
        bool,
        typer.Option(
            "--unbounded-sanitizer-calls",
            help="Development Repair v3 only: remove the separate sanitizer call count cap. "
            "Agent steps, total time, and paid LLM calls remain bounded.",
        ),
    ] = False,
) -> None:
    """Investigate, self-check/revise using public inputs, then independently verify."""
    from gpu_agent.agent.provider import DevelopmentCallPolicy
    from gpu_agent.public_task import PublicRepairInputError
    from gpu_agent.repair import RepairPolicy
    from gpu_agent.service import ApplicationService

    if unbounded_sanitizer_calls and not reinvestigate:
        raise typer.BadParameter("--unbounded-sanitizer-calls requires --reinvestigate")
    try:
        service = ApplicationService.configured()
        if allow_paid_calls:
            service.allow_development_paid_calls(DevelopmentCallPolicy(max_llm_calls=max_llm_calls))
        run, verified = service.repair(
            source,
            policy=RepairPolicy(
                version="public-repair-v3" if reinvestigate else "public-repair-v2",
                max_candidates=max_candidates,
                max_reinvestigations=max_reinvestigations,
                unbounded_sanitizer_calls=unbounded_sanitizer_calls,
            ),
        )
        typer.echo(f"run_id {run.id}")
        for ref in run.artifact_refs:
            if ref.name == "repair/summary.json":
                typer.echo(service.store.read(ref).decode())
        if verified is not None:
            typer.echo(verified.model_dump_json(indent=2))
        else:
            typer.echo("Independent verification not run: public repair did not pass.")
            for limitation in service.diagnosis(run.id).limitations:
                typer.echo(limitation)
    except PublicRepairInputError as exc:
        raise typer.BadParameter(
            f"{exc.code}: repair requires a supported, source-bound task.json "
            "and valid public input."
        ) from None
    except (OSError, ValueError):
        raise typer.BadParameter("Repair input or controller configuration is invalid.") from None
    if verified is None or verified.verdict != "VERIFIED_FIXED":
        raise typer.Exit(1)


@app.command("report")
def report_command(run_id: str) -> None:
    """Render public evidence, candidate, coverage and provider usage."""
    from gpu_agent.service import ApplicationService

    try:
        report = ApplicationService.configured().report(run_id)
    except (OSError, ValueError):
        raise typer.BadParameter("Run/report is unavailable or invalid.") from None
    typer.echo(report)
