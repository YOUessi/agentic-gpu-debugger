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
from typing import Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field

from gpu_agent.agent.models import ProviderError
from gpu_agent.agent.provider import OpenAIProviderSettings
from gpu_agent.benchmark.hardset_v2 import HARDSET_CASES, load_registry
from gpu_agent.benchmark.hardset_v2_agent import run_autonomous_hardset
from gpu_agent.harness.metrics import summarize_repository_trajectory
from gpu_agent.harness.repo_runner import run_repository_coding_session
from gpu_agent.harness.repo_verification import verify_repository_hardset_candidate
from gpu_agent.store import read_regular, reject_symlinks, sync_directory

Arm = Literal["legacy_single", "legacy_batch", "evidence_single", "evidence_batch", "workflow_D"]
Disposition = Literal[
    "EXPERIMENTAL_STRICT_FIXED",
    "STRICT_NOT_FIXED",
    "PUBLIC_NOT_ACCEPTED",
    "MODEL_OR_RUNTIME_ERROR",
    "INCONCLUSIVE",
]
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
    max_public_checks: int = Field(ge=1, le=3)
    include_workflow_D: bool
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
    first_public_check_turn: int | None = Field(default=None, ge=1)
    redundant_reads: int | None = Field(default=None, ge=0)
    public_gpu_checks: int | None = Field(default=None, ge=0)
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
    by_arm: dict[str, dict[str, int | None]]


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
            shift = (case_index + repetition - 1) % len(_ARMS)
            sequence: tuple[Arm, ...] = _ARMS[shift:] + _ARMS[:shift]
            for arm in (*sequence, *((("workflow_D",)) if include_workflow_D else ())):
                units.append(
                    TrialSpec(
                        index=len(units) + 1,
                        case_id=case_id,  # type: ignore[arg-type]
                        repetition=repetition,
                        arm=arm,
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
        max_public_checks=max_public_checks,
        include_workflow_D=include_workflow_D,
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


def _arm_configuration(arm: Arm) -> tuple[Literal["legacy", "evidence_first"], bool]:
    configs: dict[str, tuple[Literal["legacy", "evidence_first"], bool]] = {
        "legacy_single": ("legacy", False),
        "legacy_batch": ("legacy", True),
        "evidence_single": ("evidence_first", False),
        "evidence_batch": ("evidence_first", True),
    }
    if arm not in configs:
        raise ValueError("fixed workflow is a separate control, not a repository ablation arm")
    return configs[arm]


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
                elapsed_seconds=round(time.monotonic() - begin, 3),
            )
        context, batch = _arm_configuration(mode)
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
            max_steps=plan.max_steps,
            max_model_calls=plan.max_model_calls,
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
        public_run_id=result.public_run_id if result else None,
        model_calls=result.model_calls if result else None,
        model_tokens=result.model_tokens if result else None,
        tool_turns=result.tool_calls if result else None,
        first_edit_turn=metrics.first_accepted_edit_turn if metrics else None,
        first_public_check_turn=metrics.first_public_check_request_turn if metrics else None,
        redundant_reads=metrics.redundant_inspection_rejections if metrics else None,
        public_gpu_checks=result.public_checks_attempted if result else None,
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
    by_arm: dict[str, dict[str, int | None]] = {}
    for arm in (*_ARMS, *(("workflow_D",) if plan.include_workflow_D else ())):
        group = [item for item in records if item.spec.arm == arm]
        reported = [item.model_tokens for item in group]
        by_arm[arm] = {
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
        )
        if (
            snapshot.base_sha != preflight.plan.base_sha
            or snapshot.registry_sha256 != preflight.plan.registry_sha256
            or snapshot.model_name != preflight.plan.model_name
            or snapshot.endpoint_host != preflight.plan.endpoint_host
        ):
            raise ValueError("frozen benchmark source changed during comparison")
        record = _run_one(repository, output, preflight.plan, spec, secret)
        _no_overwrite_write(output / f"unit-{spec.index:03d}.json", record)
        records.append(record)
    report = summarize_sweep(preflight.plan, tuple(records))
    _no_overwrite_write(output / "paired-report.json", report)
    return report
