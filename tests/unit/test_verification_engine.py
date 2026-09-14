"""CPU contract tests replace only the unavailable container process boundary."""

import difflib
import hashlib
import json
from pathlib import Path

import pytest


@pytest.fixture
def original(store, tmp_path):
    from gpu_agent.contracts import ToolResult, now
    from gpu_agent.evidence.models import EvidenceBundle
    from gpu_agent.evidence.repository import EvidenceRepository
    from gpu_agent.execution.models import (
        BuildPayload,
        BuildResult,
        Finding,
        SanitizerPayload,
        SanitizerResult,
        SourceLocation,
    )
    from gpu_agent.patching import SourceSnapshot

    repo = Path(__file__).resolve().parents[2] / "benchmarks"
    source_paths = [
        repo / "public/case_0001/public_input/kernel.cu",
        repo / "harness/vector_io.cpp",
        repo / "harness/vector_api.h",
        repo / "harness/vendor/json.hpp",
    ]
    root = tmp_path / "snapshot"
    root.mkdir()
    run = store.create_run("diagnosis")
    refs, hashes = [], {}
    for path in source_paths:
        data = path.read_bytes()
        (root / path.name).write_bytes(data)
        hashes[path.name] = hashlib.sha256(data).hexdigest()
        refs.append(store.put(run.id, "sources/original/" + path.name, data, "public"))
    stdin = store.put(
        run.id,
        "public-input.json",
        json.dumps({"n": 257, "a": [1] * 257, "b": [2] * 257}).encode(),
        "public",
    )
    raw = store.put(run.id, "memcheck.log", b"synthetic unit evidence", "public")
    binary = store.put(
        run.id, "build/original-build/binary", b"synthetic baseline binary", "public"
    )
    build_tool = ToolResult(
        tool_name="build",
        request_id="original-build",
        started_at=now(),
        finished_at=now(),
        elapsed_ms=1,
        exit_code=0,
        timed_out=False,
        stdout_artifact=raw,
        stderr_artifact=raw,
        typed_payload=BuildPayload(argv=["nvcc"], binary_ref=binary).model_copy(
            update={"source_manifest": hashes}
        ),
    )
    build = BuildResult(success=True, binary_ref=binary, tool_result=build_tool)
    store.put(
        run.id, "build/original-build/result.json", build_tool.model_dump_json().encode(), "public"
    )
    finding = Finding(
        tool="memcheck",
        category="Invalid __global__ write",
        kernel="vector_add(float const *, float const *, float *, unsigned long)",
        source_location=SourceLocation(path="/input/kernel.cu", line=9),
        raw_ref=raw,
    )
    tool = ToolResult(
        tool_name="sanitizer",
        request_id="original",
        started_at=now(),
        finished_at=now(),
        elapsed_ms=1,
        exit_code=86,
        timed_out=False,
        stdout_artifact=raw,
        stderr_artifact=raw,
        typed_payload=SanitizerPayload(
            tool="memcheck",
            findings=[finding],
            completed=True,
            check_outcome="FINDING",
            stdin_ref=stdin,
            binary_ref=binary,
        ),
    )
    result = SanitizerResult(
        completed=True, check_outcome="FINDING", findings=[finding], tool_result=tool
    )
    store.put(run.id, "sanitizer/original/result.json", tool.model_dump_json().encode(), "public")
    EvidenceRepository(store).save(
        run.id, EvidenceBundle(source_snapshot=refs, build_result=build, sanitizer_results=[result])
    )
    return run.id, SourceSnapshot(parent_run_id=run.id, root=root, hashes=hashes)


@pytest.mark.parametrize(
    "fault",
    [
        "missing_build",
        "missing_binary",
        "unrelated_binary",
        "incomplete_payload",
        "timed_out",
        "transport_error",
        "wrong_tool",
        "source_manifest",
        "missing_source_manifest",
        "unrelated_input",
        "missing_input",
        "build_timed_out",
        "build_binary_mismatch",
        "outer_findings_mismatch",
        "missing_source_snapshot",
        "mismatched_source_snapshot",
    ],
)
def test_invalid_original_provenance_is_inconclusive(
    store, tmp_path, original, container_boundary, fault
):
    from gpu_agent.evidence.repository import EvidenceRepository
    from gpu_agent.verification.engine import VerificationEngine

    run_id, _ = original
    repo = EvidenceRepository(store)
    bundle = repo.public_view(run_id)
    baseline = bundle.sanitizer_results[0]
    tool = baseline.tool_result
    payload = tool.typed_payload
    build = bundle.build_result
    if fault == "missing_build":
        build = None
    elif fault == "missing_binary":
        payload = payload.model_copy(update={"binary_ref": None})
    elif fault == "unrelated_binary":
        ref = store.put(run_id, "unrelated-binary", b"another executable", "public")
        payload = payload.model_copy(update={"binary_ref": ref})
    elif fault == "incomplete_payload":
        payload = payload.model_copy(update={"completed": False})
    elif fault == "timed_out":
        tool = tool.model_copy(update={"timed_out": True})
    elif fault == "transport_error":
        tool = tool.model_copy(update={"tool_error": "CONTAINER_ERROR"})
    elif fault == "wrong_tool":
        tool = tool.model_copy(update={"tool_name": "run"})
    elif fault in {"source_manifest", "missing_source_manifest"}:
        manifest = {} if fault == "missing_source_manifest" else {"kernel.cu": "0" * 64}
        build = build.model_copy(
            update={
                "tool_result": build.tool_result.model_copy(
                    update={
                        "typed_payload": build.tool_result.typed_payload.model_copy(
                            update={"source_manifest": manifest}
                        )
                    }
                )
            }
        )
    elif fault == "unrelated_input":
        ref = store.put(
            run_id,
            "other-input",
            json.dumps({"n": 257, "a": [3] * 257, "b": [4] * 257}).encode(),
            "public",
        )
        payload = payload.model_copy(update={"stdin_ref": ref})
    elif fault == "missing_input":
        payload = payload.model_copy(update={"stdin_ref": None})
    elif fault == "build_timed_out":
        build = build.model_copy(
            update={"tool_result": build.tool_result.model_copy(update={"timed_out": True})}
        )
    elif fault == "build_binary_mismatch":
        build = build.model_copy(update={"binary_ref": tool.stdout_artifact})
    elif fault == "outer_findings_mismatch":
        payload = payload.model_copy(update={"findings": []})
    elif fault == "missing_source_snapshot":
        bundle = bundle.model_copy(update={"source_snapshot": bundle.source_snapshot[:-1]})
    elif fault == "mismatched_source_snapshot":
        ref = store.put(run_id, "sources/changed/kernel.cu", b"unrelated source\n", "public")
        refs = [ref if Path(r.name).name == "kernel.cu" else r for r in bundle.source_snapshot]
        bundle = bundle.model_copy(update={"source_snapshot": refs})
    tool = tool.model_copy(update={"typed_payload": payload})
    if fault != "unrelated_input":
        # Malformed observations must fail even if the bundle exactly matches the
        # immutable execution record, rather than only through record comparison.
        tool = tool.model_copy(update={"request_id": "invalid-record"})
        store.put(
            run_id,
            f"{tool.tool_name}/invalid-record/result.json",
            tool.model_dump_json().encode(),
            "public",
        )
        if build is not None:
            built = build.tool_result.model_copy(update={"request_id": "invalid-build"})
            store.put(
                run_id,
                "build/invalid-build/result.json",
                built.model_dump_json().encode(),
                "public",
            )
            build = build.model_copy(update={"tool_result": built})
    baseline = baseline.model_copy(update={"tool_result": tool})
    repo.save(
        run_id, bundle.model_copy(update={"build_result": build, "sanitizer_results": [baseline]})
    )
    candidate_id, _ = register_variant(store, original, "human")
    result = VerificationEngine(store, tmp_path / "evaluator").verify(run_id, candidate_id)
    assert result.verdict.value == "INCONCLUSIVE"
    assert result.original_finding_present is None
    assert not container_boundary


def register_variant(store, original, variant):
    from gpu_agent.patching import apply_candidate
    from gpu_agent.verification.engine import register_candidate

    _, snapshot = original
    before = (snapshot.root / "kernel.cu").read_text()
    after = before.replace("n != 257", "n == 0")
    if variant != "oob":
        after = after.replace("out[i] = a[i] + b[i];", "if (i < n) out[i] = a[i] + b[i];")
    if variant == "zero":
        after = after.replace("a[i] + b[i]", "0.0f")
    if variant == "257":
        after = after.replace("a[i] + b[i]", "n == 257 ? a[i] + b[i] : 0.0f")
    if variant == "syntax":
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
    return register_candidate(store, candidate), candidate


@pytest.fixture
def container_boundary(monkeypatch):
    from gpu_agent.execution.isolated import IsolatedGPUBackend
    from gpu_agent.execution.process import ProcessCapture

    calls = []

    def container(self, path, operation, timeout, *, stdin=b"", cancel=None):
        source = (path / "kernel.cu").read_text()
        assert set(p.name for p in path.iterdir()) <= {
            "kernel.cu",
            "vector_io.cpp",
            "vector_api.h",
            "json.hpp",
            "vector_add",
        }
        if operation == "build":
            assert not ((path / "kernel.cu").stat().st_mode & 0o222)
            if "INVALID CUDA SYNTAX" in source:
                return ProcessCapture(1, b"", b"compiler error", False), b"", b""
            return ProcessCapture(0, b"", b"", False), hashlib.sha256(source.encode()).digest(), b""
        data = json.loads(stdin)
        assert set(data) == {"a", "b", "n"}
        calls.append((path, operation, data))
        values = [a + b for a, b in zip(data["a"], data["b"], strict=True)]
        if "0.0f" in source and ("n == 257 ?" not in source or data["n"] != 257):
            values = [0] * data["n"]
        output = json.dumps({"dtype": "float32", "shape": [data["n"]], "values": values}).encode()
        if operation == "memcheck" and "if (i < n)" not in source:
            log = (
                b"========= Invalid __global__ write of size 4 bytes\n"
                b"=========     at vector_add(float const *, float const *, float *, unsigned long)"
                b" in /input/kernel.cu:10\n========= ERROR SUMMARY: 1 error\n"
            )
            return ProcessCapture(86, output, b"", False), b"", log
        return ProcessCapture(0, output, b"", False), b"", b"========= ERROR SUMMARY: 0 errors\n"

    monkeypatch.setattr(IsolatedGPUBackend, "_container", container)
    return calls


@pytest.mark.parametrize(
    "variant,want",
    [
        ("human", "VERIFIED_FIXED"),
        ("zero", "REGRESSION_DETECTED"),
        ("257", "REGRESSION_DETECTED"),
        ("oob", "NOT_FIXED"),
        ("syntax", "NOT_FIXED"),
    ],
)
def test_evaluator_rejects_semantic_failures(
    store, tmp_path, original, container_boundary, variant, want
):
    from gpu_agent.verification.engine import VerificationEngine

    candidate_id, candidate = register_variant(store, original, variant)
    engine = VerificationEngine(store, tmp_path / "evaluator")
    result = engine.verify(original[0], candidate_id, "full")
    assert result.verdict.value == want
    assert result.candidate_hash == candidate.patched_source_hash
    if variant == "human":
        assert result.private_passed_count >= 10
        runs = [x for x in container_boundary if x[1] == "run"]
        checks = [x for x in container_boundary if x[1] == "memcheck"]
        assert len({x[0] for x in runs}) == len(runs)
        assert len(runs) == len(checks) == 1 + result.private_passed_count
        assert {1, 31, 32, 33, 255, 256, 257, 1023, 1024, 1025} <= {x[2]["n"] for x in runs}
        assert result.binary_hashes
        from gpu_agent.store import RunStore

        private = RunStore(tmp_path / "evaluator/runs", visibility="evaluator")
        audit = [
            private.load(path.name)
            for path in private.root.iterdir()
            if private.load(path.name).kind == "verification_audit"
        ][0]
        assert {
            "case.json",
            "reference.cu",
            "oracle-implementation.py",
            "private-suite.json",
            "observation.json",
        } <= {ref.name for ref in audit.artifact_refs}
        assert all(ref.visibility == "evaluator" for ref in audit.artifact_refs)
    if variant == "syntax":
        assert result.required_checks["build"] == "FAILED"
        assert result.required_checks["memcheck"] == "NOT_RUN"
    public = json.dumps(result.model_dump(mode="json"))
    assert all(x not in public for x in ("raw_ref", "expected", "seed", "input.json", "kernel.cu"))
    assert result.not_run_count > 0 if variant != "human" else result.not_run_count == 0
    # Inspect every public artifact, not just the returned model.
    for run_dir in store.root.iterdir():
        manifest = store.load(run_dir.name)
        for ref in manifest.artifact_refs:
            if manifest.kind == "verification":
                assert ref.name == "verification/result.json"
                assert set(json.loads(store.read(ref))) == set(result.model_dump())


def test_candidate_registration_is_one_per_diagnosis(store, original):
    from gpu_agent.verification.engine import register_candidate

    _, candidate = register_variant(store, original, "human")
    with pytest.raises(ValueError, match="one candidate"):
        register_candidate(store, candidate)


def test_private_backend_rejects_public_refs_and_public_view(store, tmp_path):
    from gpu_agent.evidence.repository import EvidenceRepository
    from gpu_agent.execution.isolated import IsolatedGPUBackend
    from gpu_agent.execution.local import _Workspace
    from gpu_agent.execution.models import WorkspaceHandle
    from gpu_agent.store import RunStore

    private = RunStore(tmp_path / "evaluator", visibility="evaluator")
    backend = IsolatedGPUBackend(private, tmp_path / "sources", tmp_path / "tasks")
    private_run = private.create_run("private")
    public_run = store.create_run("public")
    ref = store.put(public_run.id, "input", b"secret", "public")
    state = _Workspace(WorkspaceHandle(id="test", run_id=private_run.id, path=tmp_path))
    with pytest.raises(ValueError):
        backend._input(state, ref)
    with pytest.raises(ValueError):
        backend.evidence.public_view(private_run.id)
    with pytest.raises(ValueError):
        EvidenceRepository(private)


def test_evaluator_root_must_be_independent(store):
    from gpu_agent.verification.engine import VerificationEngine

    with pytest.raises(ValueError):
        VerificationEngine(store, store.root / "private")
    with pytest.raises(ValueError):
        VerificationEngine(store, store.root.parent)


def test_original_signature_survives_patch_line_drift(store, original):
    from gpu_agent.evidence.repository import EvidenceRepository
    from gpu_agent.execution.models import Finding, SanitizerResult, SourceLocation
    from gpu_agent.verification.policy import original_presence

    old = Finding(
        tool="memcheck",
        category="Invalid __global__ write",
        kernel="vector_add()",
        source_location=SourceLocation(path="/input/kernel.cu", line=9),
    )
    moved = old.model_copy(update={"source_location": SourceLocation(path="kernel.cu", line=20)})
    result = SanitizerResult(completed=True, check_outcome="FINDING", findings=[moved])
    assert original_presence([old], result, {9: 20}, same_input=True) is True
    clean = SanitizerResult(completed=True, check_outcome="CLEAN")
    assert original_presence([old], clean, {9: 20}, same_input=True) is None
    tool = EvidenceRepository(store).public_view(original[0]).sanitizer_results[0].tool_result
    payload = tool.typed_payload.model_copy(
        update={"check_outcome": "CLEAN", "findings": [], "binary_ref": tool.stdout_artifact}
    )
    clean = clean.model_copy(
        update={"tool_result": tool.model_copy(update={"exit_code": 0, "typed_payload": payload})}
    )
    assert original_presence([old], clean, {9: 20}, same_input=True) is False
    wrong_tool = clean.model_copy(
        update={
            "tool_result": tool.model_copy(
                update={"typed_payload": payload.model_copy(update={"tool": "racecheck"})}
            )
        }
    )
    assert original_presence([old], wrong_tool, {9: 20}, same_input=True) is None
    assert original_presence([old], clean, {9: 20}, same_input=False) is None
    assert original_presence([old], SanitizerResult(), {9: 20}, same_input=True) is None


def test_holdout_only_memcheck_finding_blocks_success(
    store, tmp_path, original, container_boundary, monkeypatch
):
    from gpu_agent.execution.isolated import IsolatedGPUBackend
    from gpu_agent.execution.process import ProcessCapture
    from gpu_agent.verification.engine import VerificationEngine

    normal = IsolatedGPUBackend._container

    def holdout_finding(self, path, operation, timeout, *, stdin=b"", cancel=None):
        result = normal(self, path, operation, timeout, stdin=stdin, cancel=cancel)
        if operation == "memcheck" and json.loads(stdin)["n"] == 1:
            log = (
                b"========= Invalid __global__ write of size 4 bytes\n"
                b"========= at vector_add(float const *, float const *, float *, unsigned long)"
                b" in /input/kernel.cu:10\n========= ERROR SUMMARY: 1 error\n"
            )
            return ProcessCapture(86, result[0].stdout, b"", False), b"", log
        return result

    monkeypatch.setattr(IsolatedGPUBackend, "_container", holdout_finding)
    candidate_id, _ = register_variant(store, original, "human")
    result = VerificationEngine(store, tmp_path / "evaluator").verify(original[0], candidate_id)
    assert result.verdict.value == "REGRESSION_DETECTED"
    assert result.new_findings == 1
    assert result.private_holdout_passed is None
    assert result.required_checks["private_oracle"] == "INCOMPLETE"


def test_unregistered_program_never_acquires_benchmark_oracle(
    store, tmp_path, original, container_boundary
):
    from gpu_agent.evidence.repository import EvidenceRepository
    from gpu_agent.verification.engine import VerificationEngine

    run_id, snapshot = original
    source = (snapshot.root / "kernel.cu").read_bytes() + b"// ordinary user source\n"
    (snapshot.root / "kernel.cu").write_bytes(source)
    snapshot.hashes["kernel.cu"] = hashlib.sha256(source).hexdigest()
    repository = EvidenceRepository(store)
    bundle = repository.public_view(run_id)
    ref = store.put(run_id, "sources/ordinary/kernel.cu", source, "public")
    refs = [ref if Path(r.name).name == "kernel.cu" else r for r in bundle.source_snapshot]
    repository.save(run_id, bundle.model_copy(update={"source_snapshot": refs}))
    candidate_id, _ = register_variant(store, original, "human")
    result = VerificationEngine(store, tmp_path / "evaluator").verify(run_id, candidate_id)
    assert result.verdict.value == "INCONCLUSIVE"
    assert result.reason_code == "ORACLE_OR_BASELINE_UNAVAILABLE"
    assert not container_boundary


@pytest.mark.parametrize("operation", ["build", "memcheck"])
def test_required_tool_failure_is_inconclusive(
    store, tmp_path, original, container_boundary, monkeypatch, operation
):
    from gpu_agent.execution.isolated import IsolatedGPUBackend
    from gpu_agent.execution.process import ProcessCapture
    from gpu_agent.verification.engine import VerificationEngine

    normal = IsolatedGPUBackend._container

    def failing(self, path, op, timeout, *, stdin=b"", cancel=None):
        if op == operation:
            return ProcessCapture(None, b"", b"", False, tool_error="CONTAINER_ERROR"), b"", b""
        return normal(self, path, op, timeout, stdin=stdin, cancel=cancel)

    monkeypatch.setattr(IsolatedGPUBackend, "_container", failing)
    candidate_id, _ = register_variant(store, original, "human")
    result = VerificationEngine(store, tmp_path / "evaluator").verify(original[0], candidate_id)
    assert result.verdict.value == "INCONCLUSIVE"
    assert result.not_run_count > 0


def test_revalidation_rejects_forged_candidate_hash(store, tmp_path, original, container_boundary):
    from gpu_agent.verification.engine import VerificationEngine

    candidate_id, _ = register_variant(store, original, "human")
    ref = store.load(candidate_id).artifact_refs[0]
    path = store.root / ref.relative_path
    path.chmod(0o600)
    path.write_bytes(
        path.read_bytes().replace(b'"scope_validation":"VALID"', b'"scope_validation":"OTHER"')
    )
    with pytest.raises(ValueError, match="hash mismatch"):
        VerificationEngine(store, tmp_path / "evaluator").verify(original[0], candidate_id)
    assert not container_boundary


def test_standard_mode_runs_all_holdouts(store, tmp_path, original, container_boundary):
    from gpu_agent.verification.engine import VerificationEngine

    candidate_id, _ = register_variant(store, original, "human")
    engine = VerificationEngine(store, tmp_path / "evaluator")
    result = engine.verify(original[0], candidate_id, "standard")
    assert result.private_passed_count == 13
    with pytest.raises(ValueError):
        engine.verify(original[0], candidate_id, "strict")


def test_changed_binary_is_rejected_before_execution(
    store, tmp_path, original, container_boundary, monkeypatch
):
    from gpu_agent.execution.isolated import IsolatedGPUBackend
    from gpu_agent.verification.engine import VerificationEngine

    normal_build = IsolatedGPUBackend.build

    def tamper(self, request):
        result = normal_build(self, request)
        binary = self._workspace(request.workspace_id).handle.path / "vector_add"
        binary.chmod(0o700)
        binary.write_bytes(b"changed after trusted build")
        return result

    monkeypatch.setattr(IsolatedGPUBackend, "build", tamper)
    candidate_id, _ = register_variant(store, original, "human")
    with pytest.raises(ValueError, match="binary hash mismatch"):
        VerificationEngine(store, tmp_path / "evaluator").verify(original[0], candidate_id)
    assert not container_boundary


def test_original_provenance_from_isolated_backend_is_accepted(
    store, tmp_path, original, container_boundary
):
    from gpu_agent.execution.isolated import IsolatedGPUBackend
    from gpu_agent.execution.models import BuildRequest, SanitizerRequest, WorkspaceRequest
    from gpu_agent.verification.engine import VerificationEngine

    _, snapshot = original
    backend = IsolatedGPUBackend(store, snapshot.root, tmp_path / "baseline-tasks")
    run = store.create_run("isolated-baseline")
    handle = backend.prepare(WorkspaceRequest(run_id=run.id, source_manifest=snapshot.hashes))
    try:
        build = backend.build(BuildRequest(workspace_id=handle.id))
        assert build.success
        stdin = store.put(
            run.id,
            "baseline-input.json",
            json.dumps({"n": 257, "a": [1] * 257, "b": [2] * 257}).encode(),
            "public",
        )
        result = backend.run_sanitizer(SanitizerRequest(workspace_id=handle.id, stdin_ref=stdin))
        assert result.completed and result.check_outcome == "FINDING"
    finally:
        backend.cleanup(handle)
    registered = (run.id, snapshot.model_copy(update={"parent_run_id": run.id}))
    candidate_id, _ = register_variant(store, registered, "human")
    result = VerificationEngine(store, tmp_path / "evaluator").verify(run.id, candidate_id)
    assert result.verdict.value == "VERIFIED_FIXED"
