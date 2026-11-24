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
harness_app = typer.Typer(no_args_is_help=True, help="Model-directed GPU coding-agent harness.")
app.add_typer(harness_app, name="harness")
knowledge_app = typer.Typer(no_args_is_help=True, help="Offline knowledge inspection.")
app.add_typer(knowledge_app, name="knowledge")


@harness_app.command("sweep-repo")
def harness_sweep_repo(
    repository: Annotated[Path, typer.Option("--repository")],
    output: Annotated[Path, typer.Option("--output")],
    cases: Annotated[str, typer.Option("--cases")] = "h001",
    repetitions: Annotated[int, typer.Option("--repetitions", min=1, max=3)] = 1,
    max_steps: Annotated[int, typer.Option("--max-steps", min=4, max=24)] = 12,
    max_model_calls: Annotated[
        int,
        typer.Option(
            "--max-model-calls",
            min=6,
            max=32,
            help="Shared executor ceiling; hierarchical arms reserve four auxiliary calls.",
        ),
    ] = 14,
    max_public_checks: Annotated[int, typer.Option("--max-public-checks", min=1, max=3)] = 2,
    include_workflow_d: Annotated[bool, typer.Option("--include-workflow-d")] = False,
    architecture_study: Annotated[bool, typer.Option("--architecture-study")] = False,
    planner_only_study: Annotated[bool, typer.Option("--planner-only-study")] = False,
    evaluator_secret_file: Annotated[Path | None, typer.Option("--evaluator-secret-file")] = None,
    execute: Annotated[bool, typer.Option("--execute")] = False,
    allow_paid_calls: Annotated[bool, typer.Option("--allow-paid-calls")] = False,
    allow_real_gpu: Annotated[bool, typer.Option("--allow-real-gpu")] = False,
) -> None:
    """Freeze and optionally execute a 2x2 or three-arm coding Agent study.

    The default preflight makes ZERO GPU or paid model calls and creates no
    experiment output. --execute requires explicit paid/API, GPU and private
    evaluator authorization; failed/blocked units are never re-sampled.
    """
    from gpu_agent.harness.ablation import (
        preflight_paired_experiment,
        run_paired_experiment,
    )

    selected = tuple(part.strip() for part in cases.split(","))
    if len(set(selected)) != len(selected):
        raise typer.BadParameter("duplicate HardSet case in --cases")
    try:
        if not execute:
            report = preflight_paired_experiment(
                repository,
                evaluator_secret_file=evaluator_secret_file,
                output=output,
                cases=selected,
                repetitions=repetitions,
                max_steps=max_steps,
                max_model_calls=max_model_calls,
                max_public_checks=max_public_checks,
                include_workflow_D=include_workflow_d,
                architecture_study=architecture_study,
                planner_only_study=planner_only_study,
            )
            typer.echo(report.model_dump_json(indent=2))
            return
        if evaluator_secret_file is None:
            raise typer.BadParameter("--evaluator-secret-file required to execute")
        result = run_paired_experiment(
            repository,
            output,
            evaluator_secret_file=evaluator_secret_file,
            cases=selected,
            repetitions=repetitions,
            max_steps=max_steps,
            max_model_calls=max_model_calls,
            max_public_checks=max_public_checks,
            include_workflow_D=include_workflow_d,
            architecture_study=architecture_study,
            planner_only_study=planner_only_study,
            allow_paid_calls=allow_paid_calls,
            allow_real_gpu=allow_real_gpu,
        )
        typer.echo(result.model_dump_json(indent=2))
        if not getattr(result, "ready", True):
            raise typer.Exit(3)
    except (OSError, ValueError) as error:
        typer.echo(f"PAIRED_BENCHMARK_REJECTED: {type(error).__name__}", err=True)
        raise typer.Exit(2) from None


@harness_app.command("verify-repo")
def harness_verify_repo(
    repository: Annotated[Path, typer.Option("--repository")],
    session: Annotated[Path, typer.Option("--session")],
    output: Annotated[Path, typer.Option("--output")],
    evaluator_secret_file: Annotated[Path, typer.Option("--evaluator-secret-file")],
) -> None:
    """Operator-only strict HardSet check; hidden data never becomes an Agent tool."""
    import stat

    from gpu_agent.harness.repo_verification import verify_repository_hardset_candidate
    from gpu_agent.store import read_regular

    secret = evaluator_secret_file.absolute()
    if secret.is_relative_to(repository.resolve()) or secret.is_relative_to(session.resolve()):
        raise typer.BadParameter("evaluator secret must be outside code and Agent session")
    try:
        info = secret.lstat()
        if not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o600:
            raise ValueError("private evaluator secret must be a regular 0600 file")
        projection = verify_repository_hardset_candidate(
            repository=repository,
            session_output=session,
            output=output,
            evaluator_secret=read_regular(secret, 4096),
        )
        typer.echo(projection.model_dump_json(indent=2))
        if projection.verdict != "VERIFIED_FIXED":
            raise typer.Exit(1)
    except (OSError, ValueError):
        typer.echo("OPERATOR_REPOSITORY_VERIFICATION_REJECTED", err=True)
        raise typer.Exit(2) from None


@harness_app.command("summarize-repo")
def harness_summarize_repo(
    output: Annotated[Path, typer.Option("--output")],
) -> None:
    """Summarize trusted PUBLIC tool events; never invoke models or CUDA."""
    from gpu_agent.harness.metrics import summarize_repository_trajectory

    try:
        summary = summarize_repository_trajectory(output)
        typer.echo(summary.model_dump_json(indent=2))
    except (OSError, ValueError):
        typer.echo("INVALID_OR_UNTRUSTED_REPOSITORY_TRAJECTORY", err=True)
        raise typer.Exit(2) from None


@harness_app.command("inspect-session")
def harness_inspect_session(
    journal: Annotated[Path, typer.Option("--journal")],
) -> None:
    """Validate the durable event chain; never replay uncertain GPU/model calls."""
    import json

    from gpu_agent.harness.journal import (
        read_verified_journal,
        recovery_disposition,
    )

    try:
        events = read_verified_journal(journal)
        typer.echo(
            json.dumps(
                {
                    "events": len(events),
                    "recovery_disposition": recovery_disposition(events),
                    "last_event": events[-1]["kind"],
                    "journal_sha256": hashlib.sha256(journal.read_bytes()).hexdigest(),
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
    except (OSError, ValueError):
        typer.echo("INVALID_OR_UNTRUSTED_SESSION_JOURNAL", err=True)
        raise typer.Exit(2) from None


@harness_app.command("repo-run")
def harness_repo_run(
    output: Annotated[Path, typer.Option()],
    goal: Annotated[str, typer.Option("--goal")],
    base_sha: Annotated[str, typer.Option("--base-sha")],
    repository: Annotated[Path, typer.Option()] = Path("."),
    max_steps: Annotated[int, typer.Option("--max-steps", min=1, max=38)] = 18,
    max_model_calls: Annotated[int, typer.Option("--max-model-calls", min=4, max=40)] = 24,
    allow_paid_calls: Annotated[bool, typer.Option("--allow-paid-calls")] = False,
    public_case: Annotated[str | None, typer.Option("--public-case")] = None,
    context_mode: Annotated[
        Literal["legacy", "evidence_first"], typer.Option("--context-mode")
    ] = "evidence_first",
    disable_batch_edits: Annotated[bool, typer.Option("--disable-batch-edits")] = False,
    hierarchical: Annotated[bool, typer.Option("--hierarchical")] = False,
    syntax_editing: Annotated[bool, typer.Option("--syntax-editing")] = False,
    compile_gate: Annotated[bool, typer.Option("--compile-gate")] = False,
    allow_revert: Annotated[bool, typer.Option("--allow-revert")] = False,
    max_compiles: Annotated[int, typer.Option("--max-compiles", min=1, max=8)] = 6,
    max_public_checks: Annotated[int, typer.Option("--max-public-checks", min=1, max=3)] = 2,
    resume: Annotated[bool, typer.Option("--resume")] = False,
    pause_after_turns: Annotated[
        int | None, typer.Option("--pause-after-turns", min=1, max=37)
    ] = None,
) -> None:
    """Model-directed, multi-file editing in an isolated Git worktree.

    Only --public-case h001/h002/h003 enables registered Docker GPU public checks.
    Output remains UNVERIFIED without the separate independent evaluator.
    """
    from gpu_agent.agent.models import ProviderError
    from gpu_agent.harness.repo_runner import run_repository_coding_session

    if not allow_paid_calls:
        raise typer.BadParameter("real LLM use requires --allow-paid-calls")
    try:
        result = run_repository_coding_session(
            repository=repository,
            output=output,
            base_sha=base_sha,
            goal=goal,
            max_steps=max_steps,
            max_model_calls=max_model_calls,
            allow_paid_calls=True,
            public_case_id=public_case,
            max_public_checks=max_public_checks,
            context_mode=context_mode,
            allow_batch_edits=not disable_batch_edits,
            hierarchical_mode=hierarchical,
            syntax_mode=syntax_editing,
            compile_gate=compile_gate, allow_revert=allow_revert, max_compiles=max_compiles,
            resume=resume,
            pause_after_turns=pause_after_turns,
        )
        typer.echo(result.model_dump_json())
        if result.status not in {"DIFF_STAGED_UNVERIFIED", "PAUSED_RESUMABLE"}:
            raise typer.Exit(1)
    except ProviderError as exc:
        typer.echo(f"REPOSITORY_AGENT_MODEL_FAILED: {exc.code}", err=True)
        raise typer.Exit(1) from None
    except (OSError, ValueError) as exc:
        typer.echo(
            f"REPOSITORY_AGENT_REJECTED: {type(exc).__name__}",
            err=True,
        )
        raise typer.Exit(2) from None


@harness_app.command("run")
def harness_run(
    output: Annotated[Path, typer.Option()],
    case: Annotated[str, typer.Option("--case")],
    goal: Annotated[str, typer.Option("--goal")],
    evaluator_secret_file: Annotated[Path, typer.Option("--evaluator-secret-file")],
    max_steps: Annotated[int, typer.Option("--max-steps", min=1, max=40)] = 24,
    max_model_calls: Annotated[int, typer.Option("--max-model-calls", min=4, max=40)] = 32,
    allow_paid_calls: Annotated[bool, typer.Option("--allow-paid-calls")] = False,
    repository: Annotated[Path, typer.Option()] = Path("."),
) -> None:
    """Real model-driven tool loop on a pinned CUDA workspace."""
    import stat

    from gpu_agent.agent.models import ProviderError
    from gpu_agent.harness.runner import run_gpu_coding_harness
    from gpu_agent.store import read_regular

    if not allow_paid_calls:
        raise typer.BadParameter("real model requires explicit --allow-paid-calls")
    secret = evaluator_secret_file.absolute()
    if secret.is_relative_to(repository.resolve()):
        raise typer.BadParameter("private evaluator key must be outside the source")
    try:
        mode = secret.lstat().st_mode
        if not stat.S_ISREG(mode) or stat.S_IMODE(mode) & 0o077:
            raise ValueError("evaluator key must be a regular 0600 file")
        outcome = run_gpu_coding_harness(
            repository,
            output,
            case_id=case,
            goal=goal,
            evaluator_secret=read_regular(secret, 4096),
            max_steps=max_steps,
            max_model_calls=max_model_calls,
            allow_paid_calls=True,
        )
        typer.echo(outcome.model_dump_json())
        if outcome.status != "VERIFIED_FIXED":
            raise typer.Exit(1)
    except ProviderError as exc:
        typer.echo(f"HARNESS_MODEL_FAILURE: {exc.code}", err=True)
        raise typer.Exit(1) from None
    except (OSError, ValueError) as exc:
        typer.echo(f"HARNESS_REJECTED: {type(exc).__name__}", err=True)
        raise typer.Exit(2) from None


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


@benchmark_app.command("validate-hardset")
def validate_hardset(
    output: Annotated[Path, typer.Option()],
    repository: Annotated[Path, typer.Option()] = Path("."),
    case: Annotated[list[str] | None, typer.Option("--case")] = None,
) -> None:
    """Native GPU acceptance of self-authored numerical and multi-kernel faults."""
    from gpu_agent.benchmark.hardset_v2 import run_hardset

    try:
        report = run_hardset(repository, output, tuple(case or ()))
    except (OSError, ValueError) as exc:
        typer.echo(f"HARDSET_INPUT_INVALID: {exc}", err=True)
        raise typer.Exit(2) from None
    typer.echo(f"Native report: {output / 'report.json'}")
    if not report["passed"]:
        raise typer.Exit(1)


@benchmark_app.command("probe-hardset")
def probe_hardset(
    output: Annotated[Path, typer.Option()],
    case: Annotated[str, typer.Option("--case")],
    size: Annotated[list[int], typer.Option("--size")],
    repository: Annotated[Path, typer.Option()] = Path("."),
) -> None:
    """Operator-only bounded public functional probes (never hidden evaluation)."""
    import json

    from gpu_agent.benchmark.hardset_v2_probe import FunctionalProbeAction, HardsetFunctionalProbe

    if not size or len(size) > 15 or len(size) != len(set(size)) or type(case) is not str:
        raise typer.BadParameter("provide 1-15 unique probe sizes and one registered case")
    try:
        with HardsetFunctionalProbe(repository, output, case, max_probes=len(size)) as session:
            observations = [
                session.invoke(FunctionalProbeAction(size=n)).model_dump(mode="json") for n in size
            ]
        (output / "public-probe-results.json").write_text(
            json.dumps(
                {
                    "schema_version": "hardset-public-probe-cli-v2",
                    "case_id": case,
                    "observations": observations,
                    "type": "PUBLIC_EVIDENCE_NOT_STRICT_VERIFICATION",
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        typer.echo(f"Public probe report: {output / 'public-probe-results.json'}")
    except (ValueError, OSError) as exc:
        typer.echo(f"FUNCTIONAL_PROBE_REJECTED: {exc}", err=True)
        raise typer.Exit(2) from None


@benchmark_app.command("investigate-hardset")
def investigate_hardset(
    output: Annotated[Path, typer.Option()],
    case: Annotated[str, typer.Option("--case")],
    mode: Annotated[Literal["D", "E"], typer.Option("--mode")] = "D",
    max_probes: Annotated[int, typer.Option("--max-probes", min=1, max=15)] = 3,
    max_llm_calls: Annotated[int, typer.Option("--max-llm-calls", min=4, max=40)] = 8,
    allow_paid_calls: Annotated[bool, typer.Option("--allow-paid-calls")] = False,
    repository: Annotated[Path, typer.Option()] = Path("."),
) -> None:
    """D fixed-rule vs E audited-model probes; this does not generate a patch."""
    import json

    from gpu_agent.agent.models import AgentBudget
    from gpu_agent.agent.policy import LLMCallGate
    from gpu_agent.agent.provider import (
        DevelopmentCallPolicy,
        OpenAIProviderSettings,
        OpenAIResponsesProvider,
    )
    from gpu_agent.benchmark.hardset_v2 import PUBLIC_REQUIREMENTS
    from gpu_agent.benchmark.hardset_v2_graph import run_hardset_probe_graph
    from gpu_agent.benchmark.hardset_v2_probe import HardsetFunctionalProbe

    if case not in PUBLIC_REQUIREMENTS:
        raise typer.BadParameter("unregistered public HardSet case")
    if mode == "E" and not allow_paid_calls:
        raise typer.BadParameter("E requires explicit --allow-paid-calls")
    if mode == "D" and allow_paid_calls:
        raise typer.BadParameter("D does not make paid model calls")
    if mode == "E" and max_llm_calls < max_probes + 3:
        raise typer.BadParameter("E must reserve enough calls for bounded planning")
    settings = OpenAIProviderSettings.from_environment() if mode == "E" else None
    if mode == "E" and (
        settings is None or not settings.endpoint or not settings.model or settings.api_key is None
    ):
        raise typer.BadParameter("explicit configured model and API key required")
    try:
        with HardsetFunctionalProbe(repository, output, case, max_probes=max_probes) as probe:
            planner = None
            provider = None
            if mode == "E":
                assert settings is not None and probe.run_id is not None
                provider = OpenAIResponsesProvider(
                    settings,
                    LLMCallGate(
                        AgentBudget(
                            max_llm_calls=max_llm_calls,
                            max_wall_time_seconds=600,
                        )
                    ),
                    probe.public_store,
                    probe.run_id,
                    call_policy=DevelopmentCallPolicy(max_llm_calls=max_llm_calls),
                )
                provider.ensure_available()
                planner = provider.plan_hardset_v2
            state = run_hardset_probe_graph(
                probe,
                case_id=case,
                mode=mode,
                max_probes=max_probes,
                planner=planner,
                public_source=probe.public_source,
                functional_requirement=PUBLIC_REQUIREMENTS[case],
            )
            physical_calls = len(provider.invocations()) if provider is not None else 0
        result = {
            "schema_version": "hardset-v2-research-probe-1",
            "type": "PUBLIC_INVESTIGATION_NOT_AGENT_REPAIR",
            "case_id": case,
            "mode": mode,
            "max_probes": max_probes,
            "observations": state["observations"],
            "model_physical_calls": physical_calls,
            "found_public_failure": any(obs["outcome"] != "PASS" for obs in state["observations"]),
            "completion_reason": state["completion_reason"],
        }
        (output / "investigation.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        typer.echo(f"Investigation: {output / 'investigation.json'}")
    except (ValueError, OSError) as exc:
        typer.echo(f"HARDSET_INVESTIGATION_REJECTED: {type(exc).__name__}", err=True)
        raise typer.Exit(2) from None


@benchmark_app.command("run-hardset-agent")
def run_hardset_agent(
    output: Annotated[Path, typer.Option()],
    case: Annotated[str, typer.Option("--case")],
    evaluator_secret_file: Annotated[Path, typer.Option("--evaluator-secret-file")],
    mode: Annotated[Literal["D", "E"], typer.Option("--mode")] = "E",
    max_probes: Annotated[int, typer.Option("--max-probes", min=1, max=15)] = 3,
    max_candidates: Annotated[int, typer.Option("--max-candidates", min=1, max=3)] = 3,
    max_llm_calls: Annotated[int, typer.Option("--max-llm-calls", min=5, max=40)] = 20,
    allow_paid_calls: Annotated[bool, typer.Option("--allow-paid-calls")] = False,
    repository: Annotated[Path, typer.Option()] = Path("."),
) -> None:
    """Real LangGraph CUDA Agent: probe, diagnose, patch, revise, verify."""
    import stat

    from gpu_agent.agent.models import ProviderError
    from gpu_agent.benchmark.hardset_v2_agent import run_autonomous_hardset
    from gpu_agent.store import read_regular

    if not allow_paid_calls:
        raise typer.BadParameter("actual model diagnosis and patch require --allow-paid-calls")
    secret_path = evaluator_secret_file.absolute()
    repository_path = repository.resolve()
    if secret_path.is_relative_to(repository_path):
        raise typer.BadParameter("private evaluator key cannot be in source repository")
    try:
        info = secret_path.lstat()
        if not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) & 0o077:
            raise ValueError("evaluator key must be a regular 0600 private file")
        key = read_regular(secret_path, 4096)
        result = run_autonomous_hardset(
            repository,
            output,
            case,
            mode=mode,
            evaluator_secret=key,
            max_probes=max_probes,
            max_candidates=max_candidates,
            max_llm_calls=max_llm_calls,
            allow_paid_calls=True,
        )
        typer.echo(result.model_dump_json())
        if not result.verified:
            raise typer.Exit(1)
    except ProviderError as exc:
        typer.echo(f"HARDSET_MODEL_FAILED: {exc.code}", err=True)
        raise typer.Exit(1) from None
    except (OSError, ValueError) as exc:
        typer.echo(f"HARDSET_AGENT_REJECTED: {type(exc).__name__}", err=True)
        raise typer.Exit(2) from None


@benchmark_app.command("verify-hardset")
def verify_hardset(
    output: Annotated[Path, typer.Option()],
    case: Annotated[str, typer.Option("--case")],
    patch: Annotated[Path, typer.Option("--patch")],
    evaluator_secret_file: Annotated[Path, typer.Option("--evaluator-secret-file")],
    repository: Annotated[Path, typer.Option()] = Path("."),
) -> None:
    """Operator-only experimental strict evaluator; never exposed to an Agent."""
    import stat

    from gpu_agent.benchmark.hardset_v2_verifier import run_strict_hardset_verification
    from gpu_agent.store import read_regular

    try:
        # Secret must be stored in an operator-controlled, private regular file,
        # outside the repository and outside any public RunStore.
        secret = evaluator_secret_file.absolute()
        if secret.is_symlink() or stat.S_IMODE(secret.stat().st_mode) & 0o077:
            raise ValueError("secret must be a private regular 0600 file")
        secret_bytes = read_regular(secret, 4096)
        diff = read_regular(patch.absolute(), 4 * 1024 * 1024).decode("utf-8")
        result = run_strict_hardset_verification(
            repository,
            output,
            case_id=case,
            unified_diff=diff,
            evaluator_secret=secret_bytes,
        )
        # Print only the bounded public projection, never private run IDs, raw
        # inputs or reference values.
        typer.echo(result.model_dump_json())
        if result.verdict != "VERIFIED_FIXED":
            raise typer.Exit(1)
    except (ValueError, OSError, UnicodeError) as exc:
        typer.echo(f"HARDSET_VERIFIER_REJECTED: {type(exc).__name__}", err=True)
        raise typer.Exit(2) from None


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


@app.command("web")
def web_command(
    host: Annotated[str, typer.Option("--host")] = "127.0.0.1",
    port: Annotated[int, typer.Option("--port", min=1, max=65535)] = 8000,
    reload: Annotated[bool, typer.Option("--reload")] = False,
) -> None:
    """Serve the optional FastAPI + React operator console."""
    try:
        import uvicorn
    except ImportError:
        message = "Install the optional web dependencies with pip install -e '.[web]'."
        raise typer.BadParameter(message) from None
    uvicorn.run(
        "gpu_agent.web.app:create_app",
        host=host,
        port=port,
        reload=reload,
        factory=True,
    )


@app.command("report")
def report_command(run_id: str) -> None:
    """Render public evidence, candidate, coverage and provider usage."""
    from gpu_agent.service import ApplicationService

    try:
        report = ApplicationService.configured().report(run_id)
    except (OSError, ValueError):
        raise typer.BadParameter("Run/report is unavailable or invalid.") from None
    typer.echo(report)
