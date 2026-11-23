"""Operator-only experimental strict validation of a FINISHED repository Agent.

Never install this as a model tool. It verifies the exact registered public
candidate after the session closes, then executes the existing independent
HardSet V2 evaluator under a separate private store. No hidden inputs or
feedback enter the model's worktree, journal or parent public RunStore.
"""

from __future__ import annotations

import difflib
import hashlib
import tempfile
from pathlib import Path

from gpu_agent.benchmark.hardset_v2 import load_registry
from gpu_agent.benchmark.hardset_v2_agent_models import HardsetPublicCheck
from gpu_agent.benchmark.hardset_v2_verifier import (
    HardsetPrivateProjection,
    run_strict_hardset_verification,
)
from gpu_agent.harness.journal import read_verified_journal, recovery_disposition
from gpu_agent.harness.models import RepoHarnessOutcome
from gpu_agent.harness.recovery import read_repository_checkpoint
from gpu_agent.harness.worktree import PinnedGitWorkspace
from gpu_agent.patching import SourceSnapshot, apply_generated_candidate, materialize_candidate
from gpu_agent.store import RunStore, read_regular, reject_symlinks


def verify_repository_hardset_candidate(
    *,
    repository: Path,
    session_output: Path,
    output: Path,
    evaluator_secret: bytes,
) -> HardsetPrivateProjection:
    """Build the exact single-kernel diff and invoke the strict evaluator once.

    Preconditions are checked before the operator's separate evaluation output
    is created. This cannot verify arbitrary multi-file repository changes.
    """
    reject_symlinks(repository)
    reject_symlinks(session_output)
    reject_symlinks(output)
    if (
        output.exists()
        or output.is_symlink()
        or not isinstance(evaluator_secret, bytes)
        or len(evaluator_secret) < 32
    ):
        raise ValueError("new evaluator output and valid private secret are required")
    origin = repository.resolve(strict=True)
    session = session_output.resolve(strict=True)
    target = output.absolute()
    if (
        target.is_relative_to(origin)
        or target.is_relative_to(session)
        or origin.is_relative_to(target)
        or session.is_relative_to(target)
    ):
        raise ValueError("private evaluation output must be outside repository and session")

    report = RepoHarnessOutcome.model_validate_json(read_regular(session / "result.json", 32_000))
    events = read_verified_journal(session / "session-events.jsonl")
    if (
        recovery_disposition(events) != "FINISHED"
        or events[0]["kind"] != "session_started"
        or events[-1]["payload"].get("status") != report.status
        or events[0]["payload"].get("public_run_id") != report.public_run_id
        or events[0]["payload"].get("base_git_commit") != report.base_sha
    ):
        raise ValueError("repository session journal is incomplete or untrusted")
    checkpoint = read_repository_checkpoint(session / "session-checkpoint.json")
    case_id = checkpoint.public_case_id
    if (
        report.status != "DIFF_STAGED_UNVERIFIED"
        or not report.public_checks_passed
        or report.public_checks_attempted < 1
        or case_id not in {"h001", "h002", "h003"}
        or events[0]["payload"].get("public_case_id") != case_id
        or checkpoint.public_run_id != report.public_run_id
        or checkpoint.base_sha != report.base_sha
        or checkpoint.diff_sha256 != report.diff_sha256
        or not checkpoint.last_public_passed
        or checkpoint.last_checked_diff_sha != report.diff_sha256
        or checkpoint.last_checked_candidate_sha != report.public_checked_candidate_sha256
        or checkpoint.public_checks_attempted != report.public_checks_attempted
        or checkpoint.state.get("reason") != "WORKSPACE_DIFF_STAGED_UNVERIFIED"
        or checkpoint.journal_head_sha256 != events[-2]["sha256"]
        or events[-2]["kind"] != "tool_completed"
    ):
        raise ValueError("no independently verifiable registered public candidate")

    store = RunStore(session / "public")
    parent = store.load(report.public_run_id)
    if parent.kind != "gpu_repository_agent" or parent.status.value != "COMPLETED":
        raise ValueError("public session is not terminal")
    if checkpoint.last_public_run_id is None:
        raise ValueError("missing public candidate evidence")
    child = store.load(checkpoint.last_public_run_id)
    if (
        child.kind != "hardset_v2_public_candidate"
        or child.parent_run_id != report.public_run_id
        or child.status.value != "COMPLETED"
    ):
        raise ValueError("public check does not belong to this completed agent session")
    references = [ref for ref in child.artifact_refs if ref.name == "hardset/public-selfcheck.json"]
    if len(references) != 1:
        raise ValueError("public GPU pass has no unique immutable evidence")
    check = HardsetPublicCheck.model_validate_json(store.read(references[0]))
    if (
        not check.passed
        or check.failure_code != "PUBLIC_CHECKS_PASSED"
        or check.public_run_id != child.id
        or check.candidate_sha256 != report.public_checked_candidate_sha256
    ):
        raise ValueError("final public CUDA result is not a passing check of this candidate")

    workspace = PinnedGitWorkspace.attach_existing(
        origin=origin,
        root=session / "worktree",
        sha=report.base_sha,
        accepted_hashes=checkpoint.accepted_hashes,
        max_edited_files=checkpoint.max_edited_files,
    )
    assert case_id is not None
    kernel_path = f"benchmarks/hardset_v2/{case_id}/mutant/kernel.cu"
    if workspace.edited_files != (kernel_path,):
        raise ValueError("independent HardSet checker cannot verify unrelated source edits")
    if workspace.integrity()["diff_sha256"] != report.diff_sha256:
        raise ValueError("final repository source diff differs from the public check")
    candidate = read_regular(workspace._checked(kernel_path, editable=True), 1024 * 1024)
    if hashlib.sha256(candidate).hexdigest() != report.public_checked_candidate_sha256:
        raise ValueError("candidate hash differs from final public GPU result")

    root = origin / "benchmarks"
    registry, _ = load_registry(root)
    case = next(entry for entry in registry.cases if entry.case_id == case_id)
    original = read_regular(root / "hardset_v2" / case_id / "mutant" / "kernel.cu", 1024 * 1024)
    if hashlib.sha256(original).hexdigest() != case.mutant_source_sha256:
        raise ValueError("strict evaluator's original source does not match registry")
    source_lines = original.decode("utf-8").splitlines(keepends=True)
    candidate_lines = candidate.decode("utf-8").splitlines(keepends=True)
    if (
        not source_lines
        or not candidate_lines
        or not all(line.endswith("\n") for line in source_lines + candidate_lines)
    ):
        raise ValueError("the independent patch requires complete LF source lines")
    patch = "".join(
        difflib.unified_diff(
            source_lines,
            candidate_lines,
            fromfile="a/kernel.cu",
            tofile="b/kernel.cu",
            n=3,
        )
    )
    if not patch:
        raise ValueError("the registered candidate does not modify its original")
    with tempfile.TemporaryDirectory(prefix="gpu-repo-strict-preflight-") as temporary:
        folder = Path(temporary)
        (folder / "kernel.cu").write_bytes(original)
        snapshot = SourceSnapshot(
            parent_run_id="0" * 32,
            root=folder,
            hashes={"kernel.cu": case.mutant_source_sha256},
        )
        scope_candidate = apply_generated_candidate(snapshot, patch)
        if materialize_candidate(snapshot, scope_candidate)["kernel.cu"] != candidate:
            raise ValueError("private verifier preflight did not materialize exact candidate")

    # This call is deliberately AFTER the model RunStore and journal are final.
    # Hidden inputs remain in output/evaluator/runs, never session_output/public.
    result = run_strict_hardset_verification(
        origin,
        output,
        case_id=case_id,
        unified_diff=patch,
        evaluator_secret=evaluator_secret,
    )
    if result.candidate_sha256 != hashlib.sha256(candidate).hexdigest():
        raise ValueError("independent strict verifier checked the wrong candidate")
    return result
