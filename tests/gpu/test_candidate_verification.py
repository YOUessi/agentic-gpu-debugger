"""Live acceptance: reviewed human repair first, then four negative candidates.

There is no mock or local execution fallback in this module. Ordinary runs skip
only the confirmed missing NVIDIA runtime; --require-live converts that to failure.
"""

import difflib
import hashlib
import json
from pathlib import Path

import pytest

pytestmark = [pytest.mark.gpu, pytest.mark.container]


@pytest.mark.parametrize(
    "variant,want",
    [
        ("human", "VERIFIED_FIXED"),
        ("oob", "NOT_FIXED"),
        ("zero", "REGRESSION_DETECTED"),
        ("257", "REGRESSION_DETECTED"),
        ("syntax", "NOT_FIXED"),
    ],
)
def test_live_candidate_verification(tmp_path, request, variant, want):
    from gpu_agent.execution.isolated import IsolatedGPUBackend
    from gpu_agent.execution.models import BuildRequest, SanitizerRequest, WorkspaceRequest
    from gpu_agent.patching import SourceSnapshot, apply_candidate
    from gpu_agent.store import RunStore
    from gpu_agent.verification.engine import VerificationEngine, register_candidate

    benchmark_root = Path(__file__).resolve().parents[2] / "benchmarks"
    root = request.config.getoption("--gpu-run-root")
    store = RunStore(Path(root) if root else tmp_path / "public-runs")
    backend = IsolatedGPUBackend(store, benchmark_root, tmp_path / "baseline-tasks")
    availability = backend.availability()
    if not availability.ready:
        confirmed = availability.reason.startswith("CONTAINER_UNAVAILABLE:") and any(
            evidence in availability.reason
            for evidence in ("could not select device driver", "nvidia-container-cli")
        )
        if confirmed:
            pytest.skip(availability.reason)
        pytest.fail("Unexpected isolated runtime failure: " + availability.reason)
    paths = [
        "public/case_0001/public_input/kernel.cu",
        "harness/vector_io.cpp",
        "harness/vector_api.h",
        "harness/vendor/json.hpp",
    ]
    manifest = {
        name: hashlib.sha256((benchmark_root / name).read_bytes()).hexdigest() for name in paths
    }
    original = store.create_run("candidate_verification_baseline")
    handle = backend.prepare(
        WorkspaceRequest(run_id=original.id, source_manifest=manifest, trust_level="UNTRUSTED")
    )
    try:
        build = backend.build(BuildRequest(workspace_id=handle.id))
        assert build.success, build.model_dump_json()
        stdin = store.put(
            original.id,
            "public-input.json",
            json.dumps({"n": 257, "a": [1] * 257, "b": [2] * 257}).encode(),
            "public",
        )
        baseline = backend.run_sanitizer(
            SanitizerRequest(workspace_id=handle.id, stdin_ref=stdin, timeout_seconds=120)
        )
        assert baseline.completed and baseline.check_outcome == "FINDING", (
            baseline.model_dump_json()
        )
        assert baseline.findings
        snapshot = SourceSnapshot(
            parent_run_id=original.id,
            root=handle.path,
            hashes={Path(k).name: v for k, v in manifest.items()},
        )
        before = (handle.path / "kernel.cu").read_text()
        # Human-reviewed repair: allow supported positive n and guard final partial blocks.
        after = before.replace("n != 257", "n == 0")
        if variant != "oob":
            after = after.replace("out[i] = a[i] + b[i];", "if (i < n) out[i] = a[i] + b[i];")
        if variant == "zero":
            after = after.replace("a[i] + b[i]", "0.0f")
        elif variant == "257":
            after = after.replace("a[i] + b[i]", "n == 257 ? a[i] + b[i] : 0.0f")
        elif variant == "syntax":
            after += "INVALID CUDA SYNTAX\n"
        diff = "".join(
            difflib.unified_diff(
                before.splitlines(True),
                after.splitlines(True),
                fromfile="a/kernel.cu",
                tofile="b/kernel.cu",
            )
        )
        candidate = apply_candidate(snapshot, diff, ["kernel.cu"])
        assert candidate.generated_by == "human"
        candidate_id = register_candidate(store, candidate)
    finally:
        backend.cleanup(handle)
    store.transition(original.id, "RUNNING", "FINALIZING")
    store.transition(original.id, "COMPLETED", None)
    # Evaluator root is independent even if a persistent public run root was requested.
    private_root = tmp_path / "evaluator"
    result = VerificationEngine(store, private_root).verify(original.id, candidate_id, "full")
    assert result.verdict.value == want, result.model_dump_json(indent=2)
    assert result.candidate_hash == candidate.patched_source_hash
    if variant == "human":
        from gpu_agent.verification.models import VerificationAuditResult

        evaluator = RunStore(private_root / "runs", visibility="evaluator")
        audit = evaluator.load(result.evaluator_audit_run_id)
        audit_ref = next(
            ref for ref in audit.artifact_refs if ref.name == "verification/audit-result.json"
        )
        private_result = VerificationAuditResult.model_validate_json(evaluator.read(audit_ref))
        assert result.public_passed_count == 1 and private_result.private_passed_count == 13
        assert private_result.not_run_count == 0 and result.binary_hashes
        assert all(state == "CLEAN" for state in result.required_checks.values())
    if variant == "syntax":
        assert result.reason_code == "CANDIDATE_BUILD_FAILED"
    assert backend.active_containers() == []
    print(
        f"candidate {variant}: original={original.id}, candidate={candidate_id}, "
        f"public={store.root}, evaluator={private_root}"
    )
