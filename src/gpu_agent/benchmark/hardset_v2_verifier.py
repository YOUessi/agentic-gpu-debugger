"""Evaluator-only strict HardSet V2 pilot using unseen deterministic secret inputs.

This engine is NOT the v1 signed verifier. It accepts a scoped unified diff
against an exact pinned mutant, validates source and patch provenance, then
runs under separate evaluator visibility. Public results reveal only a bounded
verdict; hidden inputs, reference outputs and GPU traces never enter the Agent
RunStore. Never promote this experimental verdict to a signed holdout result.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import random
import tempfile
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from gpu_agent.benchmark.hardset_v2 import (
    HARDSET_CASES,
    HardsetCase,
    f32,
    load_registry,
    verify_numeric,
)
from gpu_agent.execution.isolated import IsolatedGPUBackend
from gpu_agent.execution.models import (
    BuildRequest,
    ExecutionRequest,
    SanitizerRequest,
    SanitizerTool,
    WorkspaceRequest,
)
from gpu_agent.patching import (
    SourceSnapshot,
    apply_generated_candidate,
    materialize_candidate,
)
from gpu_agent.store import RunStore, read_regular, reject_symlinks

# Domain-separated and NOT in the public size list. The evaluator secret must
# originate in an operator-controlled file, not in repo, prompt or API payload.
HIDDEN_SIZES = (2, 7, 30, 34, 66, 79, 130, 258, 514)
REQUIRED_SANITIZERS = tuple(SanitizerTool)
_HARNESS = ("harness/vector_io.cpp", "harness/vector_api.h", "harness/vendor/json.hpp")


class HardsetPrivateProjection(BaseModel):
    """The entire public projection. No private artifacts, inputs or scores."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal["hardset-experimental-strict-v2"] = "hardset-experimental-strict-v2"
    case_id: str
    candidate_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    verdict: Literal["VERIFIED_FIXED", "NOT_VERIFIED", "INCONCLUSIVE"]
    reason_code: Literal[
        "ALL_REQUIRED_CHECKS_PASSED",
        "BUILD_FAILED",
        "HIDDEN_NUMERIC_FAILED",
        "HIDDEN_SANITIZER_FAILED",
        "EVALUATOR_UNAVAILABLE",
    ]


def secret_seed(secret: bytes, *, case_id: str, n: int) -> int:
    if not isinstance(secret, bytes) or len(secret) < 32:
        raise ValueError("evaluator requires at least 32 secret bytes")
    if case_id not in HARDSET_CASES or n not in HIDDEN_SIZES:
        raise ValueError("invalid hidden evaluation domain")
    digest = hmac.digest(secret, f"hardset-v2-private:{case_id}:{n}".encode(), "sha256")
    return int.from_bytes(digest[:16], "big")


def hidden_input(case_id: str, n: int, secret: bytes) -> bytes:
    """Construct private domain-separated values; never expose to the planner."""
    rng = random.Random(secret_seed(secret, case_id=case_id, n=n))
    # Extra large logits intentionally differ from public stress shapes.
    extreme = case_id == "h001" and n in {34, 66, 258, 514}
    center = 96.0 if extreme else 0.5
    a = [f32(center + rng.randrange(-24, 24) / 13.0) for _ in range(n)]
    b = [f32(rng.randrange(-19, 19) / 17.0) for _ in range(n)]
    return json.dumps({"n": n, "a": a, "b": b}, separators=(",", ":")).encode()


def _checked_sources(root: Path, case: HardsetCase) -> tuple[bytes, dict[str, bytes]]:
    kernel = read_regular(root / "hardset_v2" / case.case_id / "mutant" / "kernel.cu", 4 << 20)
    if hashlib.sha256(kernel).hexdigest() != case.mutant_source_sha256:
        raise ValueError("source no longer matches mutant registry")
    harness = {Path(name).name: read_regular(root / name, 4 << 20) for name in _HARNESS}
    return kernel, harness


def run_strict_hardset_verification(
    repository: Path,
    output: Path,
    *,
    case_id: str,
    unified_diff: str,
    evaluator_secret: bytes,
) -> HardsetPrivateProjection:
    """Verify one scoped candidate, preserve private evidence and fail closed.

    This is an operator-only controller API. Agent tools MUST NOT receive the
    evaluator_secret, private store, backend or this callable. A test runner may
    invoke it only after the model proposal has been scope-checked.
    """
    reject_symlinks(repository)
    reject_symlinks(output)
    if output.exists():
        raise ValueError("verification output already exists")
    # Refuse two paths before touching CUDA or creating any persistent state.
    if not isinstance(evaluator_secret, bytes) or len(evaluator_secret) < 32:
        raise ValueError("evaluator secret missing or too short")
    root = repository.resolve() / "benchmarks"
    registry, registry_sha = load_registry(root)
    case = next((value for value in registry.cases if value.case_id == case_id), None)
    if case is None:
        raise ValueError("unregistered hardset case")
    kernel, harness = _checked_sources(root, case)
    # Reject malformed/out-of-scope patches BEFORE creating any run or output
    # directory, so a model cannot turn invalid text into evaluator activity.
    with tempfile.TemporaryDirectory(prefix="hardset-v2-patch-preflight-") as preflight:
        source = Path(preflight)
        IsolatedGPUBackend._write_snapshot(source / "kernel.cu", kernel)
        early = SourceSnapshot(
            parent_run_id="0" * 32,
            root=source,
            hashes={"kernel.cu": case.mutant_source_sha256},
        )
        apply_generated_candidate(early, unified_diff)
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    # Public and evaluator stores must never be the same path, including
    # artifact publication. Their default visibility contracts are distinct.
    public_store = RunStore(output / "public")
    private_store = RunStore(output / "evaluator" / "runs", visibility="evaluator")
    public_run = public_store.create_run("hardset_v2_strict_projection")
    public_store.transition(public_run.id, "RUNNING", "FINALIZING")

    with tempfile.TemporaryDirectory(prefix="hardset-v2-public-source-") as temporary:
        source_root = Path(temporary)
        original = source_root / "kernel.cu"
        IsolatedGPUBackend._write_snapshot(original, kernel)
        snapshot = SourceSnapshot(
            parent_run_id=public_run.id,
            root=source_root,
            hashes={"kernel.cu": case.mutant_source_sha256},
        )
        # Hard scope checking happens before any evaluator GPU execution.
        candidate = apply_generated_candidate(snapshot, unified_diff)
        candidate_bytes = materialize_candidate(snapshot, candidate)["kernel.cu"]

    candidate_hash = hashlib.sha256(candidate_bytes).hexdigest()
    public_store.put(
        public_run.id,
        "hardset/candidate-provenance.json",
        json.dumps(
            {
                "case_id": case_id,
                "source_sha256": case.mutant_source_sha256,
                "candidate_sha256": candidate_hash,
                "registry_sha256": registry_sha,
                "patch_id": candidate.patch_id,
            },
            sort_keys=True,
        ).encode(),
        "public",
    )

    private_run = private_store.create_run("hardset_v2_private_verification")
    private_store.transition(private_run.id, "RUNNING", "PREPARING")
    private_store.put(
        private_run.id,
        "hardset/private-verification-contract.json",
        json.dumps(
            {
                "schema_version": "hardset-v2-private-1",
                "case_id": case_id,
                "candidate_sha256": candidate_hash,
                "private_sizes": list(HIDDEN_SIZES),
                "required_sanitizers": [v.value for v in REQUIRED_SANITIZERS],
            }
        ).encode(),
        "evaluator",
    )
    outcome: HardsetPrivateProjection | None = None
    reason: str = "EVALUATOR_UNAVAILABLE"
    verdict: str = "INCONCLUSIVE"
    with tempfile.TemporaryDirectory(prefix="hardset-v2-candidate-") as temporary:
        candidate_root = Path(temporary)
        for name, data in {"kernel.cu": candidate_bytes, **harness}.items():
            IsolatedGPUBackend._write_snapshot(candidate_root / name, data)
        manifests = {
            name: hashlib.sha256(data).hexdigest()
            for name, data in {"kernel.cu": candidate_bytes, **harness}.items()
        }
        backend = IsolatedGPUBackend(private_store, candidate_root, output / "private-workspaces")
        handle = None
        try:
            if not backend.availability().ready:
                reason = "EVALUATOR_UNAVAILABLE"
            else:
                handle = backend.prepare(
                    WorkspaceRequest(
                        run_id=private_run.id,
                        source_manifest=manifests,
                        trust_level="UNTRUSTED",
                    )
                )
                built = backend.build(BuildRequest(workspace_id=handle.id))
                if not built.success:
                    reason, verdict = "BUILD_FAILED", "NOT_VERIFIED"
                else:
                    reason, verdict = "ALL_REQUIRED_CHECKS_PASSED", "VERIFIED_FIXED"
                    for n in HIDDEN_SIZES:
                        private_input = hidden_input(case_id, n, evaluator_secret)
                        ref = private_store.put(
                            private_run.id,
                            f"hardset/hidden-input-{n}.json",
                            private_input,
                            "evaluator",
                        )
                        run = backend.run(
                            ExecutionRequest(
                                workspace_id=handle.id,
                                stdin_ref=ref,
                                timeout_seconds=120,
                            )
                        )
                        complete = not (
                            run.tool_result.tool_error
                            or run.tool_result.timed_out
                            or run.tool_result.cancelled
                            or run.tool_result.truncated
                        )
                        passed = (
                            complete
                            and run.runtime_status == "SUCCESS"
                            and verify_numeric(
                                case_id, private_input, private_store.read(run.output_ref)
                            )
                        )
                        private_store.put(
                            private_run.id,
                            f"hardset/hidden-result-{n}.json",
                            json.dumps({"input_sha256": ref.sha256, "passed": passed}).encode(),
                            "evaluator",
                        )
                        if not complete:
                            reason, verdict = "EVALUATOR_UNAVAILABLE", "INCONCLUSIVE"
                            break
                        if not passed:
                            reason, verdict = "HIDDEN_NUMERIC_FAILED", "NOT_VERIFIED"
                            break
                    if verdict == "VERIFIED_FIXED":
                        size = 66
                        private_input = hidden_input(case_id, size, evaluator_secret)
                        ref = private_store.put(
                            private_run.id,
                            "hardset/sanitizer-hidden-input.json",
                            private_input,
                            "evaluator",
                        )
                        for tool in REQUIRED_SANITIZERS:
                            result = backend.run_sanitizer(
                                SanitizerRequest(
                                    workspace_id=handle.id,
                                    stdin_ref=ref,
                                    tool=tool.value,
                                    timeout_seconds=120,
                                )
                            )
                            payload = (
                                result.tool_result.typed_payload if result.tool_result else None
                            )
                            raw_output = payload.program_output_ref if payload else None
                            passed = bool(
                                result.completed
                                and result.check_outcome == "CLEAN"
                                and raw_output is not None
                                and verify_numeric(
                                    case_id,
                                    private_input,
                                    private_store.read(raw_output),
                                )
                            )
                            private_store.put(
                                private_run.id,
                                f"hardset/hidden-sanitizer-{tool.value}.json",
                                json.dumps(
                                    {"outcome": result.check_outcome, "passed": passed}
                                ).encode(),
                                "evaluator",
                            )
                            if not passed:
                                reason, verdict = "HIDDEN_SANITIZER_FAILED", "NOT_VERIFIED"
                                break
        finally:
            if handle is not None:
                try:
                    backend.cleanup(handle)
                except Exception:
                    reason, verdict = "EVALUATOR_UNAVAILABLE", "INCONCLUSIVE"
    outcome = HardsetPrivateProjection.model_validate(
        {
            "case_id": case_id,
            "candidate_sha256": candidate_hash,
            "verdict": verdict,
            "reason_code": reason,
        }
    )
    private_store.put(
        private_run.id,
        "hardset/private-verdict.json",
        outcome.model_dump_json().encode(),
        "evaluator",
    )
    private_store.transition(
        private_run.id,
        "COMPLETED" if verdict != "INCONCLUSIVE" else "FAILED",
        None,
    )
    # Only the bounded projection crosses into the public store.
    public_store.put(
        public_run.id,
        "hardset/strict-verdict.json",
        outcome.model_dump_json().encode(),
        "public",
    )
    public_store.transition(public_run.id, "COMPLETED", None)
    return outcome
