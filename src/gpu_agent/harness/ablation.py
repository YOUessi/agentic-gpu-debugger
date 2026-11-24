"""Fail-closed paired REAL-model benchmark of repository Coding Agent policies.

Every CUDA run and paid model invocation must originate in the existing
operator-audited runners. This orchestration deliberately does NOT expose any
hidden evaluator input, evaluator secret, or shell command to model tools.
"""

from __future__ import annotations

import hashlib
import os
import re
import stat
import subprocess
import time
from pathlib import Path
from typing import Literal, TypedDict
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field

from gpu_agent.agent.models import ProviderError
from gpu_agent.agent.provider import OpenAIProviderSettings, provider_invocations
from gpu_agent.benchmark.hardset_v2 import HARDSET_CASES, load_registry
from gpu_agent.benchmark.hardset_v2_agent import run_autonomous_hardset
from gpu_agent.harness.check_metrics import public_check_lifecycle
from gpu_agent.harness.journal import read_verified_journal
from gpu_agent.harness.metrics import summarize_repository_trajectory
from gpu_agent.harness.repo_audit import read_repository_tool_events
from gpu_agent.harness.repo_runner import run_repository_coding_session
from gpu_agent.harness.repo_verification import verify_repository_hardset_candidate
from gpu_agent.store import RunStore, read_regular, reject_symlinks, sync_directory

Arm = Literal[
    "legacy_single",
    "legacy_batch",
    "evidence_single",
    "evidence_batch",
    "react_syntax",
    "hierarchical_syntax",
    "workflow_D",
]
Disposition = Literal[
    "EXPERIMENTAL_STRICT_FIXED",
    "STRICT_NOT_FIXED",
    "PUBLIC_NOT_ACCEPTED",
    "MODEL_OR_RUNTIME_ERROR",
    "INCONCLUSIVE",
]
_PLANNER_ONLY_ARMS: tuple[Arm, ...] = ("react_syntax", "hierarchical_syntax")
_ARCHITECTURE_ARMS: tuple[Arm, ...] = (
    "evidence_single",
    "react_syntax",
    "hierarchical_syntax",
)
_ARMS: tuple[Arm, ...] = (
    "legacy_single",
    "legacy_batch",
    "evidence_single",
    "evidence_batch",
)
_GIT_SHA = re.compile(r"^[a-f0-9]{40}$")


class BenchmarkModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class TrialSpec(BenchmarkModel):
    index: int = Field(ge=1, le=100)
    case_id: Literal["h001", "h002", "h003"]
    repetition: int = Field(ge=1, le=3)
    arm: Arm
    physical_model_call_limit: int | None = Field(default=None, ge=1, le=40)
    executor_format_retry_limit: int | None = Field(default=None, ge=0)
    auxiliary_model_call_limit: int | None = Field(default=None, ge=0)


class BenchmarkPlan(BenchmarkModel):
    schema_version: Literal["repository-agent-paired-pilot-v1"] = "repository-agent-paired-pilot-v1"
    base_sha: str = Field(pattern=r"^[a-f0-9]{40}$")
    registry_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    cases: tuple[Literal["h001", "h002", "h003"], ...]
    repetitions: int = Field(ge=1, le=3)
    model_name: str | None
    endpoint_host: str | None
    max_steps: int = Field(ge=4, le=24)
    max_model_calls: int = Field(ge=6, le=32)
    # For new plans max_model_calls is the shared executor ceiling; auxiliary
    # Plan/Reflect and their retries are separately reserved per trial.
    model_budget_scheme: Literal["legacy_shared", "equal_executor_v1"] = "legacy_shared"
    max_public_checks: int = Field(ge=1, le=3)
    include_workflow_D: bool
    architecture_study: bool = False
    planner_only_study: bool = False
    public_probe_budget: Literal[4] = 4
    units: tuple[TrialSpec, ...]


class Preflight(BenchmarkModel):
    plan: BenchmarkPlan
    ready: bool
    blockers: tuple[str, ...]
    no_paid_calls_made: Literal[True] = True


class TrialRecord(BenchmarkModel):
    spec: TrialSpec
    disposition: Disposition
    public_accepted: bool
    strict_verdict: Literal["VERIFIED_FIXED", "NOT_VERIFIED", "INCONCLUSIVE"] | None
    source_candidate_sha256: str | None
    public_run_id: str | None
    model_calls: int | None = Field(default=None, ge=0)
    model_tokens: int | None = Field(default=None, ge=0)
    tool_turns: int | None = Field(default=None, ge=0)
    first_edit_turn: int | None = Field(default=None, ge=1)
    # Legacy name is retained as an admitted request, not proof of GPU execution.
    first_public_check_turn: int | None = Field(default=None, ge=1)
    first_public_check_request_turn: int | None = Field(default=None, ge=1)
    first_public_check_admitted_turn: int | None = Field(default=None, ge=1)
    first_public_check_executed_turn: int | None = Field(default=None, ge=1)
    first_public_check_completed_turn: int | None = Field(default=None, ge=1)
    first_public_check_pass_turn: int | None = Field(default=None, ge=1)
    public_check_dispatches: int | None = Field(default=None, ge=0)
    public_check_completions: int | None = Field(default=None, ge=0)
    public_audit_available: bool = True
    redundant_reads: int | None = Field(default=None, ge=0)
    public_gpu_checks: int | None = Field(default=None, ge=0)
    public_probes: int | None = Field(default=None, ge=0)
    effective_patch: bool = False
    public_check_reached: bool = False
    public_check_executed: bool | None = None
    public_check_completed: bool | None = None
    elapsed_seconds: float = Field(ge=0)
    failure_type: str | None = None
    evaluation_kind: Literal["experimental-HardSet-V2"] = "experimental-HardSet-V2"


class SweepReport(BenchmarkModel):
    schema_version: Literal["repository-agent-paired-report-v1"] = (
        "repository-agent-paired-report-v1"
    )
    plan_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    expected_units: int
    completed_units: int
    missing_units: int
    all_units_accounted_for: bool
    records: tuple[TrialRecord, ...]
    by_arm: dict[str, dict[str, int | float | None]]


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _no_overwrite_write(path: Path, value: BaseModel) -> None:
    """Durably persist immutable evidence; refuse an existing destination."""
    reject_symlinks(path)
    if path.exists() or path.is_symlink():
        raise ValueError("benchmark evidence cannot be overwritten")
    path.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
    with os.fdopen(os.open(path, flags, 0o600), "wb") as out:
        out.write(value.model_dump_json(indent=2).encode() + b"\n")
        out.flush()
        os.fsync(out.fileno())
    sync_directory(path.parent)


def _head(repository: Path) -> str:
    proc = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "--verify", "HEAD"],
        capture_output=True,
        text=True,
        timeout=20,
        check=True,
    )
    sha = proc.stdout.strip()
    if not _GIT_SHA.fullmatch(sha):
        raise ValueError("benchmark requires a full frozen SHA-1 Git commit")
    status = subprocess.run(
        ["git", "-C", str(repository), "status", "--porcelain=v1", "--untracked-files=all", "-z"],
        capture_output=True,
        timeout=20,
        check=True,
    )
    if status.stdout:
        raise ValueError("benchmark requires a clean, committed Git source snapshot")
    return sha


def create_benchmark_plan(
    repository: Path,
    *,
    cases: tuple[str, ...] = ("h001",),
    repetitions: int = 1,
    max_steps: int = 12,
    max_model_calls: int = 14,
    max_public_checks: int = 2,
    include_workflow_D: bool = False,
    architecture_study: bool = False,
    planner_only_study: bool = False,
) -> BenchmarkPlan:
    """Freeze case/arm order before seeing any model answer or GPU feedback."""
    reject_symlinks(repository)
    if (
        not 1 <= len(cases) <= 3
        or len(set(cases)) != len(cases)
        or any(case not in HARDSET_CASES for case in cases)
        or type(repetitions) is not int
        or not 1 <= repetitions <= 3
        or type(max_steps) is not int
        or not 4 <= max_steps <= 24
        or type(max_model_calls) is not int
        or not max_steps + 2 <= max_model_calls <= 32
        or type(max_public_checks) is not int
        or not 1 <= max_public_checks <= 3
        or type(include_workflow_D) is not bool
        or type(architecture_study) is not bool
        or type(planner_only_study) is not bool
        or (planner_only_study and (architecture_study or include_workflow_D))
    ):
        raise ValueError("invalid registered paired experiment contract")
    root = repository.resolve(strict=True)
    _, registry_hash = load_registry(root / "benchmarks")
    settings = OpenAIProviderSettings.from_environment()
    host = urlsplit(settings.endpoint).hostname if settings.endpoint else None
    units: list[TrialSpec] = []
    for repetition in range(1, repetitions + 1):
        for case_index, case_id in enumerate(cases):
            # Rotate precommitted order across cases and repeated trials to
            # avoid systematically giving the new arm the freshest GPU cache.
            arms = (
                _PLANNER_ONLY_ARMS
                if planner_only_study
                else _ARCHITECTURE_ARMS
                if architecture_study
                else _ARMS
            )
            shift = (case_index + repetition - 1) % len(arms)
            sequence: tuple[Arm, ...] = arms[shift:] + arms[:shift]
            for arm in (*sequence, *((("workflow_D",)) if include_workflow_D else ())):
                units.append(
                    TrialSpec(
                        index=len(units) + 1,
                        case_id=case_id,  # type: ignore[arg-type]
                        repetition=repetition,
                        arm=arm,
                        physical_model_call_limit=(
                            max_model_calls + 4 if arm == "hierarchical_syntax" else max_model_calls
                        ),
                        executor_format_retry_limit=(
                            None if arm == "workflow_D" else max_model_calls - max_steps
                        ),
                        auxiliary_model_call_limit=(
                            None
                            if arm == "workflow_D"
                            else 4
                            if arm == "hierarchical_syntax"
                            else 0
                        ),
                    )
                )
    return BenchmarkPlan(
        base_sha=_head(root),
        registry_sha256=registry_hash,
        cases=cases,  # type: ignore[arg-type]
        repetitions=repetitions,
        model_name=settings.model,
        endpoint_host=host,
        max_steps=max_steps,
        max_model_calls=max_model_calls,
        model_budget_scheme="equal_executor_v1",
        max_public_checks=max_public_checks,
        include_workflow_D=include_workflow_D,
        architecture_study=architecture_study,
        planner_only_study=planner_only_study,
        units=tuple(units),
    )


def preflight_paired_experiment(
    repository: Path,
    *,
    evaluator_secret_file: Path | None = None,
    output: Path | None = None,
    cases: tuple[str, ...] = ("h001",),
    repetitions: int = 1,
    max_steps: int = 12,
    max_model_calls: int = 14,
    max_public_checks: int = 2,
    include_workflow_D: bool = False,
    architecture_study: bool = False,
    planner_only_study: bool = False,
) -> Preflight:
    """NO model, GPU or Docker side effects; expose presence not secret contents."""
    plan = create_benchmark_plan(
        repository,
        cases=cases,
        repetitions=repetitions,
        max_steps=max_steps,
        max_model_calls=max_model_calls,
        max_public_checks=max_public_checks,
        include_workflow_D=include_workflow_D,
        architecture_study=architecture_study,
        planner_only_study=planner_only_study,
    )
    errors: list[str] = []
    settings = OpenAIProviderSettings.from_environment()
    if not settings.api_key or not settings.endpoint or not settings.model:
        errors.append("MODEL_NOT_CONFIGURED")
    elif (
        urlsplit(settings.endpoint).hostname != "api.openai.com"
        and not settings.supports_store_false
    ):
        errors.append("MODEL_CAPABILITY_NOT_VERIFIED")
    if evaluator_secret_file is None:
        errors.append("EVALUATOR_SECRET_NOT_CONFIGURED")
    else:
        reject_symlinks(evaluator_secret_file)
        try:
            info = evaluator_secret_file.stat(follow_symlinks=False)
            if (
                not stat.S_ISREG(info.st_mode)
                or stat.S_IMODE(info.st_mode) != 0o600
                or not 32 <= info.st_size <= 4096
                or evaluator_secret_file.resolve().is_relative_to(repository.resolve())
                or (
                    output is not None
                    and evaluator_secret_file.resolve().is_relative_to(output.absolute())
                )
            ):
                errors.append("EVALUATOR_SECRET_NOT_PRIVATE")
        except OSError:
            errors.append("EVALUATOR_SECRET_UNAVAILABLE")
    if output is not None:
        reject_symlinks(output)
        root = repository.resolve()
        target = output.absolute()
        if (
            target.exists()
            or target.is_symlink()
            or target.is_relative_to(root)
            or root.is_relative_to(target)
        ):
            errors.append("INVALID_OUTPUT_ROOT")
    return Preflight(plan=plan, ready=not errors, blockers=tuple(errors))


def _arm_configuration(
    arm: Arm,
) -> tuple[Literal["legacy", "evidence_first"], bool, bool, bool]:
    # context, batch edits, Planner+Reflect, AST editing
    configs: dict[str, tuple[Literal["legacy", "evidence_first"], bool, bool, bool]] = {
        "legacy_single": ("legacy", False, False, False),
        "legacy_batch": ("legacy", True, False, False),
        "evidence_single": ("evidence_first", False, False, False),
        "evidence_batch": ("evidence_first", True, False, False),
        "react_syntax": ("evidence_first", False, False, True),
        "hierarchical_syntax": ("evidence_first", False, True, True),
    }
    if arm not in configs:
        raise ValueError("fixed workflow is a separate control, not a repository ablation arm")
    return configs[arm]


class PartialAudit(TypedDict, total=False):
    run_id: str
    model_calls: int
    model_tokens: int | None
    tool_turns: int
    first_edit_turn: int | None
    first_public_check_turn: int | None
    first_public_check_request_turn: int | None
    first_public_check_executed_turn: int | None
    first_public_check_completed_turn: int | None
    first_public_check_pass_turn: int | None
    public_checks_attempted: int
    public_check_dispatches: int
    public_check_completions: int


def _failed_repository_public_audit(unit_directory: Path) -> PartialAudit:
    """Recover verified *partial* model/tool counts after a failed trial.

    A terminal result.json does not exist after most model exceptions, but the
    provider still holds immutable STARTED/COMPLETED invocation artifacts.
    Never silently turn a failure into zero-cost/no-model evidence.
    """
    source = unit_directory / "agent"
    try:
        journal = read_verified_journal(source / "session-events.jsonl")
        first = journal[0]
        if first["kind"] != "session_started":
            raise ValueError("invalid incomplete session creation")
        run_id = first["payload"]["public_run_id"]
        store = RunStore(source / "public")
        parent = store.load(run_id)
        if parent.kind != "gpu_repository_agent":
            raise ValueError("partial evidence is not a repository run")
        invoked = provider_invocations(store, run_id)
        calls = len(invoked)
        complete_usage = [
            item.usage.total_tokens
            for item in invoked
            if item.usage is not None and item.usage.total_tokens is not None
        ]
        tokens = sum(complete_usage) if len(complete_usage) == calls else None
        count = sum(
            ref.name.startswith("repo-harness/v1/tool-") and ref.name.endswith(".json")
            for ref in parent.artifact_refs
        )
        events = read_repository_tool_events(store, run_id, expected_count=count)
        started = {
            row["payload"]["turn"]: row["payload"]["tool"]
            for row in journal
            if row["kind"] == "tool_started"
        }
        if any(started.get(turn) != row["tool"] for turn, row in enumerate(events, 1)):
            raise ValueError("partial public tools do not match started journal requests")
        first_edit = next(
            (
                index
                for index, row in enumerate(events, 1)
                if row["tool"]
                in {"repo_replace_snippet", "repo_replace_many", "repo_replace_syntax"}
                and row["status"] == "OK"
            ),
            None,
        )
        lifecycle = public_check_lifecycle(store, run_id, journal, events)
        return {
            "run_id": run_id,
            "model_calls": calls,
            "model_tokens": tokens,
            "tool_turns": count,
            "first_edit_turn": first_edit,
            "first_public_check_turn": min(lifecycle.admitted, default=None),
            "first_public_check_request_turn": min(lifecycle.requested, default=None),
            "first_public_check_executed_turn": min(lifecycle.executed, default=None),
            "first_public_check_completed_turn": min(lifecycle.completed, default=None),
            "first_public_check_pass_turn": min(lifecycle.passed, default=None),
            "public_checks_attempted": len(lifecycle.admitted),
            "public_check_dispatches": len(lifecycle.executed),
            "public_check_completions": len(lifecycle.completed),
        }
    except (OSError, ValueError, TypeError, KeyError, IndexError):
        # Missing or corrupt partial audit is a real evidence gap, not zero cost.
        return {}


def _run_one(
    repository: Path,
    root: Path,
    plan: BenchmarkPlan,
    spec: TrialSpec,
    secret: bytes,
) -> TrialRecord:
    """No retries; record failures without selecting only successful trials."""
    folder = root / f"unit-{spec.index:03d}-{spec.case_id}-{spec.arm}"
    if folder.exists():
        raise ValueError("unit output is immutable and already exists")
    begin = time.monotonic()
    metrics = None
    result = None
    projection = None
    mode = spec.arm
    failure: str | None = None
    try:
        if mode == "workflow_D":
            report = run_autonomous_hardset(
                repository,
                folder / "fixed-workflow",
                spec.case_id,
                mode="D",
                evaluator_secret=secret,
                max_probes=3,
                max_candidates=plan.max_public_checks,
                max_llm_calls=plan.max_model_calls,
                allow_paid_calls=True,
            )
            return TrialRecord(
                spec=spec,
                disposition=(
                    "EXPERIMENTAL_STRICT_FIXED"
                    if report.verdict == "VERIFIED_FIXED"
                    else "STRICT_NOT_FIXED"
                    if report.verdict == "NOT_VERIFIED"
                    else "INCONCLUSIVE"
                ),
                public_accepted=report.candidate_sha256 is not None,
                strict_verdict=report.verdict,
                source_candidate_sha256=report.candidate_sha256,
                public_run_id=report.investigation_run_id,
                model_calls=report.model_physical_calls,
                model_tokens=report.model_total_tokens,
                tool_turns=None,  # Its fixed graph is NOT comparable to repo tool turns.
                public_gpu_checks=report.candidates,
                effective_patch=report.candidate_sha256 is not None,
                public_check_reached=report.candidates > 0,
                elapsed_seconds=round(time.monotonic() - begin, 3),
            )
        context, batch, hierarchical, syntax = _arm_configuration(mode)
        result = run_repository_coding_session(
            repository=repository,
            output=folder / "agent",
            base_sha=plan.base_sha,
            goal=(
                "Repair the registered CUDA kernel to satisfy its functional requirement. "
                "Use approved public GPU checks before finish; do not inspect hidden tests."
            ),
            public_case_id=spec.case_id,
            context_mode=context,
            allow_batch_edits=batch,
            hierarchical_mode=hierarchical,
            syntax_mode=syntax,
            max_steps=plan.max_steps,
            max_model_calls=spec.physical_model_call_limit or plan.max_model_calls,
            max_public_checks=plan.max_public_checks,
            allow_paid_calls=True,
        )
        metrics = summarize_repository_trajectory(folder / "agent")
        if result.status == "DIFF_STAGED_UNVERIFIED" and result.public_checks_passed:
            projection = verify_repository_hardset_candidate(
                repository=repository,
                session_output=folder / "agent",
                output=folder / "strict",
                evaluator_secret=secret,
            )
    except (OSError, ValueError, ProviderError, RuntimeError) as exc:
        # Do not publish raw exception messages: SDK errors may contain request
        # contents, tokens, endpoint query strings or private evaluation details.
        failure = type(exc).__name__
    recovered = (
        _failed_repository_public_audit(folder)
        if failure is not None and mode != "workflow_D"
        else {}
    )
    status: Disposition = (
        "MODEL_OR_RUNTIME_ERROR"
        if failure is not None
        else "EXPERIMENTAL_STRICT_FIXED"
        if projection is not None and projection.verdict == "VERIFIED_FIXED"
        else "STRICT_NOT_FIXED"
        if projection is not None and projection.verdict == "NOT_VERIFIED"
        else "INCONCLUSIVE"
        if projection is not None and projection.verdict == "INCONCLUSIVE"
        else "PUBLIC_NOT_ACCEPTED"
    )
    return TrialRecord(
        spec=spec,
        disposition=status,
        public_accepted=bool(result and result.public_checks_passed),
        strict_verdict=projection.verdict if projection else None,
        source_candidate_sha256=result.public_checked_candidate_sha256 if result else None,
        public_run_id=result.public_run_id if result else recovered.get("run_id"),
        model_calls=result.model_calls if result else recovered.get("model_calls"),
        model_tokens=result.model_tokens if result else recovered.get("model_tokens"),
        tool_turns=result.tool_calls if result else recovered.get("tool_turns"),
        first_edit_turn=(
            metrics.first_accepted_edit_turn if metrics else recovered.get("first_edit_turn")
        ),
        first_public_check_turn=(
            metrics.first_public_check_admitted_turn
            if metrics
            else recovered.get("first_public_check_turn")
        ),
        first_public_check_request_turn=(
            metrics.first_public_check_request_turn
            if metrics
            else recovered.get("first_public_check_request_turn")
        ),
        first_public_check_admitted_turn=(
            metrics.first_public_check_admitted_turn
            if metrics
            else recovered.get("first_public_check_turn")
        ),
        first_public_check_executed_turn=(
            metrics.first_public_check_executed_turn
            if metrics
            else recovered.get("first_public_check_executed_turn")
        ),
        first_public_check_completed_turn=(
            metrics.first_public_check_completed_turn
            if metrics
            else recovered.get("first_public_check_completed_turn")
        ),
        first_public_check_pass_turn=(
            metrics.first_public_check_pass_turn
            if metrics
            else recovered.get("first_public_check_pass_turn")
        ),
        public_check_dispatches=(
            metrics.public_check_dispatches if metrics else recovered.get("public_check_dispatches")
        ),
        public_check_completions=(
            metrics.public_check_completions
            if metrics
            else recovered.get("public_check_completions")
        ),
        public_audit_available=metrics is not None or bool(recovered),
        redundant_reads=metrics.redundant_inspection_rejections if metrics else None,
        public_gpu_checks=(
            result.public_checks_attempted if result else recovered.get("public_checks_attempted")
        ),
        public_probes=result.public_probes_attempted if result else None,
        effective_patch=bool(
            metrics.first_accepted_edit_turn if metrics else recovered.get("first_edit_turn")
        ),
        public_check_reached=bool(
            result.public_checks_attempted if result else recovered.get("public_checks_attempted")
        ),
        public_check_executed=(
            bool(metrics.public_check_dispatches)
            if metrics
            else bool(recovered["public_check_dispatches"])
            if recovered
            else None
        ),
        public_check_completed=(
            bool(metrics.public_check_completions)
            if metrics
            else bool(recovered["public_check_completions"])
            if recovered
            else None
        ),
        failure_type=failure,
        elapsed_seconds=round(time.monotonic() - begin, 3),
    )


def summarize_sweep(plan: BenchmarkPlan, records: tuple[TrialRecord, ...]) -> SweepReport:
    """Do not drop failures or impute unobserved token costs as zero."""
    indices = [item.spec.index for item in records]
    if len(set(indices)) != len(indices) or any(
        item.spec != plan.units[item.spec.index - 1] for item in records
    ):
        raise ValueError("paired sweep result does not match frozen unit identities")
    by_arm: dict[str, dict[str, int | float | None]] = {}
    arms = (
        _PLANNER_ONLY_ARMS
        if plan.planner_only_study
        else _ARCHITECTURE_ARMS
        if plan.architecture_study
        else _ARMS
    )
    for arm in (*arms, *(("workflow_D",) if plan.include_workflow_D else ())):
        group = [item for item in records if item.spec.arm == arm]
        reported = [item.model_tokens for item in group]
        edits = [item.first_edit_turn for item in group if item.first_edit_turn is not None]
        by_arm[arm] = {
            "effective_patch_trials": sum(item.effective_patch for item in group),
            "effective_patch_rate": (
                sum(item.effective_patch for item in group) / len(group) if group else None
            ),
            "public_check_reached_trials": sum(item.public_check_reached for item in group),
            "public_check_reached_rate": (
                sum(item.public_check_reached for item in group) / len(group) if group else None
            ),
            "public_check_executed_trials": sum(
                item.public_check_executed is True for item in group
            ),
            "public_check_completed_trials": sum(
                item.public_check_completed is True for item in group
            ),
            "missing_public_audits": sum(not item.public_audit_available for item in group),
            "first_effective_edit_turn_mean": sum(edits) / len(edits) if edits else None,
            "planned": sum(item.arm == arm for item in plan.units),
            "observed": len(group),
            "strict_fixed": sum(item.disposition == "EXPERIMENTAL_STRICT_FIXED" for item in group),
            "failures": sum(item.disposition != "EXPERIMENTAL_STRICT_FIXED" for item in group),
            "measured_model_tokens": (
                sum(value for value in reported if value is not None)
                if len(group) > 0 and all(value is not None for value in reported)
                else None
            ),
        }
    return SweepReport(
        plan_sha256=_sha(plan.model_dump_json().encode()),
        expected_units=len(plan.units),
        completed_units=len(records),
        missing_units=len(plan.units) - len(records),
        all_units_accounted_for=len(plan.units) == len(records),
        records=records,
        by_arm=by_arm,
    )


def run_paired_experiment(
    repository: Path,
    output: Path,
    *,
    evaluator_secret_file: Path,
    cases: tuple[str, ...] = ("h001",),
    repetitions: int = 1,
    max_steps: int = 12,
    max_model_calls: int = 14,
    max_public_checks: int = 2,
    include_workflow_D: bool = False,
    architecture_study: bool = False,
    planner_only_study: bool = False,
    allow_paid_calls: bool = False,
    allow_real_gpu: bool = False,
) -> SweepReport | Preflight:
    """Explicit opt-in and preflight BEFORE creating any paid/side-effecting trial.

    A blocked run records a durable preflight report, never a fabricated trial.
    Even if a unit fails, later frozen units continue; no successful-run picking.
    """
    if not allow_paid_calls or not allow_real_gpu:
        raise ValueError("real paired sweep requires explicit paid model and GPU authorization")
    preflight = preflight_paired_experiment(
        repository,
        evaluator_secret_file=evaluator_secret_file,
        output=output,
        cases=cases,
        repetitions=repetitions,
        max_steps=max_steps,
        max_model_calls=max_model_calls,
        max_public_checks=max_public_checks,
        include_workflow_D=include_workflow_D,
        architecture_study=architecture_study,
        planner_only_study=planner_only_study,
    )
    if output.exists() or output.is_symlink():
        raise ValueError("experiment output already exists; evidence is immutable")
    output.mkdir(parents=True, mode=0o700, exist_ok=False)
    _no_overwrite_write(output / "preflight.json", preflight)
    _no_overwrite_write(output / "frozen-plan.json", preflight.plan)
    if not preflight.ready:
        return preflight
    secret = read_regular(evaluator_secret_file, 4096)
    records: list[TrialRecord] = []
    for spec in preflight.plan.units:
        # Require source registry and current HEAD never drift between arms.
        snapshot = create_benchmark_plan(
            repository,
            cases=cases,
            repetitions=repetitions,
            max_steps=max_steps,
            max_model_calls=max_model_calls,
            max_public_checks=max_public_checks,
            include_workflow_D=include_workflow_D,
            architecture_study=architecture_study,
            planner_only_study=planner_only_study,
        )
        if (
            snapshot.base_sha != preflight.plan.base_sha
            or snapshot.registry_sha256 != preflight.plan.registry_sha256
            or snapshot.model_name != preflight.plan.model_name
            or snapshot.endpoint_host != preflight.plan.endpoint_host
            or snapshot.architecture_study != preflight.plan.architecture_study
            or snapshot.planner_only_study != preflight.plan.planner_only_study
        ):
            raise ValueError("frozen benchmark source changed during comparison")
        record = _run_one(repository, output, preflight.plan, spec, secret)
        _no_overwrite_write(output / f"unit-{spec.index:03d}.json", record)
        records.append(record)
    report = summarize_sweep(preflight.plan, tuple(records))
    _no_overwrite_write(output / "paired-report.json", report)
    return report
