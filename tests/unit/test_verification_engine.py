"""CPU contract tests replace only the unavailable container process boundary."""

import difflib
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest


def _evaluator_store(tmp_path, name="evaluator"):
    from gpu_agent.store import RunStore

    return RunStore(tmp_path / name / "runs", visibility="evaluator")


def _engine(store, tmp_path, name="evaluator"):
    from gpu_agent.verification.engine import VerificationEngine

    return VerificationEngine(store, _evaluator_store(tmp_path, name))


def _public_tree(root):
    return {
        path.relative_to(root): (path.stat().st_mode, path.read_bytes())
        for path in root.rglob("*")
        if path.is_file()
    }


def _audit_result(tmp_path, result):
    from gpu_agent.store import RunStore
    from gpu_agent.verification.models import VerificationAuditResult

    private = RunStore(tmp_path / "evaluator/runs", visibility="evaluator")
    run = private.load(result.evaluator_audit_run_id)
    ref = next(ref for ref in run.artifact_refs if ref.name == "verification/audit-result.json")
    return VerificationAuditResult.model_validate_json(private.read(ref))


def _rewrite_child_index(evaluator, audit_id, child_run_ids, *, schema_version=1):
    audit = evaluator.load(audit_id)
    ref = next(
        (ref for ref in audit.artifact_refs if ref.name == "verification/child-index.json"),
        None,
    )
    assert ref is not None
    content = json.dumps(
        {"schema_version": schema_version, "child_run_ids": child_run_ids},
        separators=(",", ":"),
    ).encode()
    artifact_path = evaluator.root / ref.relative_path
    artifact_path.chmod(0o600)
    artifact_path.write_bytes(content)
    artifact_path.chmod(0o400)
    updated_ref = ref.model_copy(
        update={"sha256": hashlib.sha256(content).hexdigest(), "byte_count": len(content)}
    )
    updated = audit.model_copy(
        update={
            "artifact_refs": [
                updated_ref if item.id == ref.id else item for item in audit.artifact_refs
            ]
        }
    )
    (evaluator.root / audit_id / "manifest.json").write_text(updated.model_dump_json(indent=2))


def _original(store, tmp_path, *, holdout_origin=None):
    from gpu_agent.contracts import RepositorySnapshot, RunBinding, ToolResult, now
    from gpu_agent.environment import load_toolchain_lock
    from gpu_agent.evidence.models import EvidenceBundle
    from gpu_agent.evidence.repository import _evidence
    from gpu_agent.execution.isolated import LOCK_PATH
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
    binding = RunBinding(
        repository=RepositorySnapshot(commit="a" * 40, tracked_tree_hash="b" * 64, clean=True),
        purpose="evaluation",
        toolchain_lock_hash=load_toolchain_lock(LOCK_PATH).lock_hash,
        prompt_version="diagnosis-v1",
        model_config_hash="c" * 64,
    )
    parent = (
        store.create_run("holdout_execution", binding=binding, external_origin=holdout_origin)
        if holdout_origin is not None
        else None
    )
    run = store.create_run("diagnosis", parent.id if parent is not None else None, binding=binding)
    refs, hashes = [], {}
    for path in source_paths:
        data = path.read_bytes()
        (root / path.name).write_bytes(data)
        hashes[path.name] = hashlib.sha256(data).hexdigest()
        refs.append(store.put(run.id, "sources/original/" + path.name, data, store.visibility))
    stdin = store.put(
        run.id,
        "public-input.json",
        json.dumps({"n": 257, "a": [1] * 257, "b": [2] * 257}).encode(),
        store.visibility,
    )
    stdout = store.put(
        run.id,
        "memcheck.stdout",
        json.dumps({"dtype": "float32", "shape": [257], "values": [3.0] * 257}).encode(),
        store.visibility,
    )
    raw = store.put(
        run.id,
        "memcheck.log",
        (
            b"========= Invalid __global__ write of size 4 bytes\n"
            b"=========     at vector_add(float const *, float const *, float *, unsigned long) "
            b"in /input/kernel.cu:9\n"
            b"========= ERROR SUMMARY: 1 error\n"
        ),
        store.visibility,
    )
    binary = store.put(
        run.id,
        "build/original-build/binary",
        b"synthetic baseline binary",
        store.visibility,
    )
    build_tool = ToolResult(
        tool_name="build",
        request_id="original-build",
        started_at=now(),
        finished_at=now(),
        elapsed_ms=1,
        exit_code=0,
        timed_out=False,
        stdout_artifact=stdout,
        stderr_artifact=raw,
        typed_payload=BuildPayload(argv=["nvcc"], binary_ref=binary).model_copy(
            update={"source_manifest": hashes}
        ),
    )
    build = BuildResult(success=True, binary_ref=binary, tool_result=build_tool)
    store.put(
        run.id,
        "build/original-build/result.json",
        build_tool.model_dump_json().encode(),
        store.visibility,
    )
    finding = Finding(
        tool="memcheck",
        category="Invalid __global__ write",
        kernel="vector_add(float const *, float const *, float *, unsigned long)",
        source_location=SourceLocation(
            path="/input/kernel.cu",
            line=9,
            function="vector_add(float const *, float const *, float *, unsigned long)",
        ),
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
        stdout_artifact=stdout,
        stderr_artifact=raw,
        typed_payload=SanitizerPayload(
            status="COMPLETED",
            tool="memcheck",
            findings=[finding],
            completed=True,
            parser_version="compute-sanitizer-2",
            check_outcome="FINDING",
            stdin_ref=stdin,
            binary_ref=binary,
            program_output_ref=stdout,
        ),
    )
    result = SanitizerResult(
        status="COMPLETED",
        completed=True,
        parser_version="compute-sanitizer-2",
        check_outcome="FINDING",
        findings=[finding],
        program_output_ref=stdout,
        tool_result=tool,
    )
    store.put(
        run.id,
        "sanitizer/original/result.json",
        tool.model_dump_json().encode(),
        store.visibility,
    )
    _evidence(store).save(
        run.id, EvidenceBundle(source_snapshot=refs, build_result=build, sanitizer_results=[result])
    )
    return run.id, SourceSnapshot(parent_run_id=run.id, root=root, hashes=hashes)


@pytest.fixture
def original(store, tmp_path):
    return _original(store, tmp_path)


@pytest.fixture
def evaluator_original(tmp_path):
    from gpu_agent.contracts import ExternalRunOrigin

    evaluator = _evaluator_store(tmp_path)
    origin = ExternalRunOrigin(run_id="d" * 32, visibility="public")
    return evaluator, _original(evaluator, tmp_path, holdout_origin=origin), origin


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
    result = _engine(store, tmp_path).verify(run_id, candidate_id)
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
    from gpu_agent.environment import RuntimeToolchainAttestation
    from gpu_agent.execution.isolated import IsolatedGPUBackend
    from gpu_agent.execution.process import ProcessCapture

    calls = []

    def attest(self):
        assert self._expected_toolchain is not None
        return RuntimeToolchainAttestation(
            runtime_session_id=self._runtime_session_id,
            lock_hash=self._expected_toolchain.lock_hash,
            image_id=self._expected_toolchain.image_id,
            cuda_nvcc=self._expected_toolchain.cuda_nvcc,
            compute_sanitizer=self._expected_toolchain.compute_sanitizer,
            compute_capability="8.9",
            target_arch=self._expected_toolchain.target_arch,
            policy_hash=hashlib.sha256(self.policy.model_dump_json().encode()).hexdigest(),
        )

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
        clean = (
            b"========= RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)\n"
            if operation == "racecheck"
            else b"========= ERROR SUMMARY: 0 errors\n"
        )
        return ProcessCapture(0, output, b"", False), b"", clean

    monkeypatch.setattr(IsolatedGPUBackend, "_container", container)
    monkeypatch.setattr(IsolatedGPUBackend, "_attest_runtime", attest)
    return calls


@pytest.mark.parametrize(
    "variant,want",
    [
        ("human", "VERIFIED_FIXED"),
        ("zero", "REGRESSION_DETECTED"),
        # This defect is holdout-only.  The public projection is intentionally
        # identical to a fully passing run; the evaluator audit carries the
        # regression verdict.
        ("257", "VERIFIED_FIXED"),
        ("oob", "NOT_FIXED"),
        ("syntax", "NOT_FIXED"),
    ],
)
def test_evaluator_rejects_semantic_failures(
    store, tmp_path, original, container_boundary, variant, want
):
    candidate_id, candidate = register_variant(store, original, variant)
    engine = _engine(store, tmp_path)
    result = engine.verify(original[0], candidate_id, "full")
    assert result.verdict.value == want
    assert result.candidate_hash == candidate.patched_source_hash
    audit_result = _audit_result(tmp_path, result)
    if variant == "257":
        assert audit_result.verdict.value == "REGRESSION_DETECTED"
        assert audit_result.observation.private_holdout_passed is False
    if variant == "human":
        assert audit_result.private_passed_count >= 10
        runs = [x for x in container_boundary if x[1] == "run"]
        checks = [x for x in container_boundary if x[1] == "memcheck"]
        assert len({x[0] for x in runs}) == len(runs)
        assert len(runs) == len(checks) == 1 + audit_result.private_passed_count
        assert {1, 31, 32, 33, 255, 256, 257, 1023, 1024, 1025} <= {x[2]["n"] for x in runs}
        assert result.binary_hashes
        from gpu_agent.store import RunStore

        private = RunStore(tmp_path / "evaluator/runs", visibility="evaluator")
        audit = [
            private.load(path.name)
            for path in private.root.iterdir()
            if private.load(path.name).kind == "verification_audit"
        ][0]
        origin_binding = store.load(original[0]).binding
        assert audit.binding == origin_binding
        assert audit.external_origin.run_id == original[0]
        inputs = [
            private.load(path.name)
            for path in private.root.iterdir()
            if private.load(path.name).kind == "verification_input"
        ]
        assert inputs and all(item.binding == origin_binding for item in inputs)
        assert all(item.external_origin.run_id == original[0] for item in inputs)
        assert {
            "case.json",
            "reference.cu",
            "oracle-implementation.py",
            "private-suite.json",
            "observation.json",
            "verification/audit-result.json",
        } <= {ref.name for ref in audit.artifact_refs}
        assert all(ref.visibility == "evaluator" for ref in audit.artifact_refs)
    if variant == "syntax":
        assert result.required_checks["build"] == "FAILED"
        assert result.required_checks["memcheck"] == "NOT_RUN"
    public = json.dumps(result.model_dump(mode="json"))
    assert all(x not in public for x in ("raw_ref", "expected", "seed", "input.json", "kernel.cu"))
    assert audit_result.not_run_count > 0 if variant != "human" else audit_result.not_run_count == 0
    # Inspect every public artifact, not just the returned model.
    for run_dir in store.root.iterdir():
        manifest = store.load(run_dir.name)
        for ref in manifest.artifact_refs:
            if manifest.kind == "verification":
                assert ref.name == "verification/result.json"
                assert set(json.loads(store.read(ref))) == set(result.model_dump())


def test_unrelated_malformed_evaluator_run_does_not_affect_derivation(
    store, tmp_path, original, container_boundary
):
    from gpu_agent.verification.engine import VerificationEngine

    evaluator = _evaluator_store(tmp_path)
    unrelated = evaluator.create_run("private_label")
    (evaluator.root / unrelated.id / "manifest.json").write_text("not-json")
    candidate_id, _ = register_variant(store, original, "human")

    result = VerificationEngine(store, evaluator).verify(original[0], candidate_id)

    assert result.verdict.value == "VERIFIED_FIXED"
    assert container_boundary


@pytest.mark.parametrize("fault", ["tampered", "duplicate", "foreign"])
def test_derivation_rejects_invalid_verification_child_inventory(
    store, tmp_path, original, container_boundary, fault
):
    from gpu_agent.verification.derivation import validate_persisted_derivation
    from gpu_agent.verification.engine import VerificationEngine

    evaluator = _evaluator_store(tmp_path)
    candidate_id, _ = register_variant(store, original, "human")
    result = VerificationEngine(store, evaluator).verify(original[0], candidate_id)
    audit_id = result.evaluator_audit_run_id
    assert audit_id is not None
    audit = evaluator.load(audit_id)
    index_ref = next(
        (ref for ref in audit.artifact_refs if ref.name == "verification/child-index.json"),
        None,
    )
    assert index_ref is not None
    child_run_ids = json.loads(evaluator.read(index_ref))["child_run_ids"]
    if fault == "tampered":
        child_run_ids[0] = "f" * 32
    elif fault == "duplicate":
        child_run_ids[1] = child_run_ids[0]
    else:
        child_run_ids[0] = evaluator.create_run("private_label").id
    _rewrite_child_index(evaluator, audit_id, child_run_ids)

    with pytest.raises(ValueError):
        validate_persisted_derivation(
            store,
            evaluator,
            original[0],
            result,
            store.load(original[0]).binding,
        )
    assert container_boundary


def test_derivation_rejects_boolean_child_inventory_schema_version(
    store, tmp_path, original, container_boundary
):
    from gpu_agent.verification.derivation import validate_persisted_derivation
    from gpu_agent.verification.engine import VerificationEngine

    evaluator = _evaluator_store(tmp_path)
    candidate_id, _ = register_variant(store, original, "human")
    result = VerificationEngine(store, evaluator).verify(original[0], candidate_id)
    audit_id = result.evaluator_audit_run_id
    assert audit_id is not None
    audit = evaluator.load(audit_id)
    index_ref = next(
        ref for ref in audit.artifact_refs if ref.name == "verification/child-index.json"
    )
    child_run_ids = json.loads(evaluator.read(index_ref))["child_run_ids"]
    _rewrite_child_index(
        evaluator,
        audit_id,
        child_run_ids,
        schema_version=True,
    )

    with pytest.raises(ValueError, match="child inventory is invalid"):
        validate_persisted_derivation(
            store,
            evaluator,
            original[0],
            result,
            store.load(original[0]).binding,
        )
    assert container_boundary


def test_evaluator_local_verification_inherits_holdout_execution_origin(
    evaluator_original, container_boundary
):
    from gpu_agent.verification.engine import VerificationEngine

    evaluator, original, origin = evaluator_original
    diagnosis_id = original[0]
    candidate_id, _ = register_variant(evaluator, original, "human")

    result = VerificationEngine(evaluator, evaluator).verify(diagnosis_id, candidate_id)

    diagnosis = evaluator.load(diagnosis_id)
    holdout = evaluator.load(diagnosis.parent_run_id)
    audit = evaluator.load(result.evaluator_audit_run_id)
    assert holdout.kind == "holdout_execution"
    assert diagnosis.external_origin == audit.external_origin == origin
    assert audit.parent_run_id == diagnosis_id
    assert all(child.external_origin == origin for child in evaluator.children(audit.id))
    assert result.verdict.value == "VERIFIED_FIXED"
    assert container_boundary


@pytest.mark.parametrize("fault", ["audit_origin_removed", "input_origin_altered"])
def test_evaluator_local_derivation_rejects_changed_holdout_origin(
    evaluator_original, container_boundary, fault
):
    from gpu_agent.contracts import ExternalRunOrigin
    from gpu_agent.verification.derivation import validate_persisted_derivation
    from gpu_agent.verification.engine import VerificationEngine

    evaluator, original, _ = evaluator_original
    diagnosis_id = original[0]
    candidate_id, _ = register_variant(evaluator, original, "human")
    result = VerificationEngine(evaluator, evaluator).verify(diagnosis_id, candidate_id)
    audit_id = result.evaluator_audit_run_id
    target_id = audit_id if fault == "audit_origin_removed" else evaluator.children(audit_id)[0].id
    manifest_path = evaluator.root / target_id / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["external_origin"] = (
        None
        if fault == "audit_origin_removed"
        else ExternalRunOrigin(run_id="e" * 32, visibility="public").model_dump(mode="json")
    )
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="verification (audit|input lineage) is invalid"):
        validate_persisted_derivation(
            evaluator,
            evaluator,
            diagnosis_id,
            result,
            evaluator.load(diagnosis_id).binding,
        )
    assert container_boundary


def test_evaluator_local_derivation_rejects_consistent_evaluator_origin_tamper(
    evaluator_original, container_boundary
):
    from gpu_agent.contracts import ExternalRunOrigin
    from gpu_agent.verification.derivation import validate_persisted_derivation
    from gpu_agent.verification.engine import VerificationEngine

    evaluator, original, _ = evaluator_original
    diagnosis_id = original[0]
    candidate_id, _ = register_variant(evaluator, original, "human")
    result = VerificationEngine(evaluator, evaluator).verify(diagnosis_id, candidate_id)
    audit_id = result.evaluator_audit_run_id
    assert audit_id is not None
    audit = evaluator.load(audit_id)
    index_ref = next(
        ref for ref in audit.artifact_refs if ref.name == "verification/child-index.json"
    )
    child_run_ids = json.loads(evaluator.read(index_ref))["child_run_ids"]
    impossible_origin = ExternalRunOrigin(
        run_id="f" * 32,
        visibility="evaluator",
    ).model_dump(mode="json")
    for run_id in (diagnosis_id, audit_id, *child_run_ids):
        manifest_path = evaluator.root / run_id / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["external_origin"] = impossible_origin
        manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="evaluation diagnosis is invalid"):
        validate_persisted_derivation(
            evaluator,
            evaluator,
            diagnosis_id,
            result,
            evaluator.load(diagnosis_id).binding,
        )
    assert container_boundary


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
    from gpu_agent.store import RunStore
    from gpu_agent.verification.engine import VerificationEngine

    with pytest.raises(ValueError):
        VerificationEngine(store, RunStore(store.root / "private/runs", visibility="evaluator"))
    with pytest.raises(ValueError):
        VerificationEngine(store, RunStore(store.root.parent, visibility="evaluator"))


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
    result = _engine(store, tmp_path).verify(original[0], candidate_id)
    assert result.verdict.value == "VERIFIED_FIXED"
    assert result.new_findings == 0
    audit = _audit_result(tmp_path, result)
    assert audit.verdict.value == "REGRESSION_DETECTED"
    assert audit.observation.private_holdout_passed is False
    assert len(audit.observation.new_blocking_findings) == 1
    assert "private_oracle" not in result.required_checks


@pytest.mark.parametrize(
    "fault,audit_reason",
    [
        ("runtime_tool_error", "RUNTIME_TOOL_ERROR"),
        ("sanitizer_tool_error", "SANITIZER_TOOL_ERROR"),
        ("oracle_failure", "ORACLE_FAILED"),
    ],
)
def test_private_outcomes_cannot_change_public_projection(
    store, tmp_path, original, container_boundary, monkeypatch, fault, audit_reason
):
    from gpu_agent.execution.isolated import IsolatedGPUBackend
    from gpu_agent.execution.process import ProcessCapture
    from gpu_agent.store import RunStore

    candidate_id, _ = register_variant(store, original, "human")
    passing = _engine(store, tmp_path, "passing-evaluator").verify(
        original[0], candidate_id, "standard"
    )
    native = IsolatedGPUBackend._container

    def private_failure(self, path, operation, timeout, *, stdin=b"", cancel=None):
        if operation != "build" and json.loads(stdin)["n"] == 1:
            if fault == "runtime_tool_error" and operation == "run":
                return (
                    ProcessCapture(None, b"", b"", False, tool_error="CONTAINER_ERROR"),
                    b"",
                    b"",
                )
            if fault == "sanitizer_tool_error" and operation == "memcheck":
                return (
                    ProcessCapture(None, b"", b"", False, tool_error="CONTAINER_ERROR"),
                    b"",
                    b"",
                )
            if fault == "oracle_failure":
                bad = json.dumps({"dtype": "float32", "shape": [1], "values": [999.0]}).encode()
                if operation == "run":
                    return ProcessCapture(0, bad, b"", False), b"", b""
                if operation == "memcheck":
                    return (
                        ProcessCapture(0, bad, b"", False),
                        b"",
                        b"========= ERROR SUMMARY: 0 errors\n",
                    )
        return native(self, path, operation, timeout, stdin=stdin, cancel=cancel)

    monkeypatch.setattr(IsolatedGPUBackend, "_container", private_failure)
    failing = _engine(store, tmp_path, "failing-evaluator").verify(
        original[0], candidate_id, "standard"
    )
    assert passing == failing
    passing_audit = RunStore(tmp_path / "passing-evaluator/runs", visibility="evaluator")
    failing_audit = RunStore(tmp_path / "failing-evaluator/runs", visibility="evaluator")
    passing_manifest = passing_audit.load(passing.evaluator_audit_run_id)
    failing_manifest = failing_audit.load(failing.evaluator_audit_run_id)
    passing_result = next(
        ref
        for ref in passing_manifest.artifact_refs
        if ref.name == "verification/audit-result.json"
    )
    failing_result = next(
        ref
        for ref in failing_manifest.artifact_refs
        if ref.name == "verification/audit-result.json"
    )
    from gpu_agent.verification.models import VerificationAuditResult

    assert (
        VerificationAuditResult.model_validate_json(passing_audit.read(passing_result)).reason_code
        == "ALL_REQUIRED_CHECKS_PASSED"
    )
    private_result = VerificationAuditResult.model_validate_json(failing_audit.read(failing_result))
    assert private_result.reason_code == audit_reason
    assert private_result.verdict.value in {"INCONCLUSIVE", "REGRESSION_DETECTED"}


def test_unregistered_program_never_acquires_benchmark_oracle(
    store, tmp_path, original, container_boundary
):
    from gpu_agent.evidence.repository import EvidenceRepository

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
    result = _engine(store, tmp_path).verify(run_id, candidate_id)
    assert result.verdict.value == "INCONCLUSIVE"
    assert result.reason_code == "ORACLE_OR_BASELINE_UNAVAILABLE"
    assert not container_boundary


@pytest.mark.parametrize("operation", ["build", "run", "memcheck"])
def test_required_tool_failure_is_inconclusive(
    store, tmp_path, original, container_boundary, monkeypatch, operation
):
    from gpu_agent.execution.isolated import IsolatedGPUBackend
    from gpu_agent.execution.process import ProcessCapture

    normal = IsolatedGPUBackend._container

    def failing(self, path, op, timeout, *, stdin=b"", cancel=None):
        if op == operation:
            return ProcessCapture(None, b"", b"", False, tool_error="CONTAINER_ERROR"), b"", b""
        return normal(self, path, op, timeout, stdin=stdin, cancel=cancel)

    monkeypatch.setattr(IsolatedGPUBackend, "_container", failing)
    candidate_id, _ = register_variant(store, original, "human")
    result = _engine(store, tmp_path).verify(original[0], candidate_id)
    assert result.verdict.value == "INCONCLUSIVE"
    audit = _audit_result(tmp_path, result)
    assert audit.not_run_count > 0
    assert result.public_passed_count == 0
    assert result.failure_stage == "verification"
    if operation == "build":
        assert result.required_checks["build"] == "TOOL_ERROR"
        assert result.required_checks["runtime"] == "NOT_RUN"
        assert result.reason_code == "BUILD_TOOL_ERROR"
    elif operation == "run":
        assert result.required_checks["runtime"] == "TOOL_ERROR"
        assert result.required_checks["public_oracle"] == "NOT_RUN"
        assert result.reason_code == "RUNTIME_TOOL_ERROR"
    else:
        assert result.required_checks["memcheck"] == "TOOL_ERROR"
        assert result.required_checks["public_oracle"] == "NOT_RUN"
        assert result.reason_code == "SANITIZER_TOOL_ERROR"


def test_revalidation_rejects_forged_candidate_hash(store, tmp_path, original, container_boundary):
    candidate_id, _ = register_variant(store, original, "human")
    ref = store.load(candidate_id).artifact_refs[0]
    path = store.root / ref.relative_path
    path.chmod(0o600)
    path.write_bytes(
        path.read_bytes().replace(b'"scope_validation":"VALID"', b'"scope_validation":"OTHER"')
    )
    with pytest.raises(ValueError, match="hash mismatch"):
        _engine(store, tmp_path).verify(original[0], candidate_id)
    assert not container_boundary


def test_standard_mode_runs_all_holdouts(store, tmp_path, original, container_boundary):
    from gpu_agent.verification.engine import verification_run_id

    candidate_id, _ = register_variant(store, original, "human")
    engine = _engine(store, tmp_path)
    result = engine.verify(original[0], candidate_id, "standard")
    assert _audit_result(tmp_path, result).private_passed_count == 13
    strict = engine.verify(original[0], candidate_id, "full")
    assert strict.verdict.value == "VERIFIED_FIXED"
    assert {tool.value for tool in strict.check_outcomes} == {
        "memcheck",
        "racecheck",
        "initcheck",
        "synccheck",
    }
    assert all(outcome == "CLEAN" for outcome in strict.check_outcomes.values())
    verifications = [
        store.load(path.name)
        for path in store.root.iterdir()
        if store.load(path.name).kind == "verification"
    ]
    assert len(verifications) == 2
    candidate_hash = result.candidate_hash
    assert verification_run_id(original[0], candidate_hash) == verification_run_id(
        original[0], candidate_hash, "standard"
    )
    assert {run.id for run in verifications} == {
        verification_run_id(original[0], candidate_hash, "standard"),
        verification_run_id(original[0], candidate_hash, "full"),
    }


def test_conflicting_exact_public_verification_replay_has_zero_mutation(
    store, tmp_path, original, container_boundary
):
    candidate_id, _ = register_variant(store, original, "human")
    engine = _engine(store, tmp_path)
    result = engine.verify(original[0], candidate_id, "standard")
    before = _public_tree(store.root)
    forged = result.model_copy(update={"reason_code": "FORGED_CONFLICT"})
    with pytest.raises(ValueError, match="existing public verification conflicts"):
        engine._persist_public(original[0], forged, "standard")
    assert _public_tree(store.root) == before


def test_incomplete_public_verification_replay_fails_closed_without_mutation(
    store, tmp_path, original, container_boundary
):
    from gpu_agent.verification.engine import verification_run_id

    candidate_id, candidate = register_variant(store, original, "human")
    store.create_run(
        "verification",
        original[0],
        _run_id=verification_run_id(original[0], candidate.patched_source_hash),
    )
    before = _public_tree(store.root)
    with pytest.raises(ValueError, match="existing public verification conflicts"):
        _engine(store, tmp_path).verify(original[0], candidate_id, "standard")
    assert _public_tree(store.root) == before


@pytest.mark.parametrize("tamper", ["lifecycle", "artifact_owner"])
def test_noncanonical_public_verification_replay_fails_closed_without_mutation(
    store, tmp_path, original, container_boundary, tamper
):
    from gpu_agent.verification.engine import verification_run_id

    candidate_id, _ = register_variant(store, original, "human")
    engine = _engine(store, tmp_path)
    result = engine.verify(original[0], candidate_id, "standard")
    run_id = verification_run_id(original[0], result.candidate_hash, "standard")
    manifest_path = store.root / run_id / "manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    if tamper == "lifecycle":
        manifest["last_completed_phase"] = "PREPARING"
    else:
        forged_owner = "f" * 32
        manifest["artifact_refs"][0]["run_id"] = forged_owner
        artifact_id = manifest["artifact_refs"][0]["id"]
        manifest["artifact_refs"][0]["relative_path"] = f"{forged_owner}/artifacts/{artifact_id}"
    manifest_path.write_text(json.dumps(manifest))
    private = _evaluator_store(tmp_path)
    before_public = _public_tree(store.root)
    before_private = _public_tree(private.root)
    before_calls = list(container_boundary)
    with pytest.raises(ValueError, match="existing public verification conflicts"):
        engine.verify(original[0], candidate_id, "standard")
    assert _public_tree(store.root) == before_public
    assert _public_tree(private.root) == before_private
    assert container_boundary == before_calls


def test_concurrent_exact_public_verification_replay_is_idempotent(
    store, tmp_path, original, container_boundary
):
    candidate_id, _ = register_variant(store, original, "human")
    engine = _engine(store, tmp_path, "shared-concurrent-evaluator")
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(
                engine.verify,
                original[0],
                candidate_id,
                "standard",
            )
            for _ in range(2)
        ]
    results = [future.result() for future in futures]
    assert results[0] == results[1]
    private = _evaluator_store(tmp_path, "shared-concurrent-evaluator")
    audit = private.load(results[0].evaluator_audit_run_id)
    child_index_ref = next(
        ref for ref in audit.artifact_refs if ref.name == "verification/child-index.json"
    )
    child_count = len(json.loads(private.read(child_index_ref))["child_run_ids"])
    # The fixture records only ordinary execution and sanitizer execution;
    # compilation is handled separately by its synthetic container stub.
    assert len(container_boundary) == child_count * 2
    before_calls = list(container_boundary)
    assert engine.verify(original[0], candidate_id, "standard") == results[0]
    assert container_boundary == before_calls
    verifications = [
        store.load(path.name)
        for path in store.root.iterdir()
        if path.is_dir() and len(path.name) == 32 and store.load(path.name).kind == "verification"
    ]
    assert len(verifications) == 1


@pytest.mark.parametrize("state", ["incomplete", "conflicting"])
def test_existing_evaluator_audit_fails_closed_without_mutation(
    store, tmp_path, original, container_boundary, state
):
    from gpu_agent.contracts import ExternalRunOrigin

    candidate_id, candidate = register_variant(store, original, "human")
    private = _evaluator_store(tmp_path)
    audit_id = hashlib.sha256(
        (f"verification-audit-v1:{original[0]}:{candidate.patched_source_hash}:standard").encode()
    ).hexdigest()[:32]
    audit = private.create_run(
        "verification_audit" if state == "incomplete" else "conflicting_audit",
        binding=store.load(original[0]).binding,
        external_origin=ExternalRunOrigin(run_id=original[0], visibility="public"),
        _run_id=audit_id,
    )
    if state == "conflicting":
        private.transition(audit.id, "RUNNING", "FINALIZING")
        private.transition(audit.id, "COMPLETED", None)
    before_public = _public_tree(store.root)
    before_private = _public_tree(private.root)
    with pytest.raises(ValueError, match="existing evaluator verification audit conflicts"):
        _engine(store, tmp_path).verify(original[0], candidate_id, "standard")
    assert _public_tree(store.root) == before_public
    assert _public_tree(private.root) == before_private
    assert container_boundary == []


def test_changed_binary_is_rejected_before_execution(
    store, tmp_path, original, container_boundary, monkeypatch
):
    from gpu_agent.execution.isolated import IsolatedGPUBackend

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
        _engine(store, tmp_path).verify(original[0], candidate_id)
    assert not container_boundary


def test_original_provenance_from_isolated_backend_is_accepted(
    store, tmp_path, original, container_boundary
):
    from gpu_agent.execution.isolated import IsolatedGPUBackend
    from gpu_agent.execution.models import BuildRequest, SanitizerRequest, WorkspaceRequest

    _, snapshot = original
    backend = IsolatedGPUBackend(store, snapshot.root, tmp_path / "baseline-tasks")
    run = store.create_run("isolated-baseline", binding=store.load(original[0]).binding)
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
    result = _engine(store, tmp_path).verify(run.id, candidate_id)
    assert result.verdict.value == "VERIFIED_FIXED"
