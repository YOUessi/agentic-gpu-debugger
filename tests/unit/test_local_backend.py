"""Trust boundary checks use a fake compiler; no GPU is launched here."""

import hashlib
import json
from pathlib import Path

import pytest

from gpu_agent.config import Settings
from gpu_agent.contracts import now
from gpu_agent.execution.process import ProcessCapture
from gpu_agent.store import RunStore


@pytest.fixture
def local(tmp_path, monkeypatch):
    from gpu_agent.execution import local as module

    repo = tmp_path / "repo"
    manifest = {}
    for name in ["kernel.cu", "vector_io.cpp", "vector_api.h", "json.hpp"]:
        path = repo / "reviewed" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"reviewed " + name.encode())
        manifest[f"reviewed/{name}"] = hashlib.sha256(path.read_bytes()).hexdigest()
    monkeypatch.setattr(module, "_load_trusted_sources", lambda: {"case_0000": manifest.copy()})
    store = RunStore(tmp_path / "runs")
    run = store.create_run("test")
    backend = module.LocalBackend(
        Settings(cuda_root=Path("/trusted/cuda")), store, repo, tmp_path / "workspaces"
    )
    return backend, store, run.id, manifest, repo


def prepare(local):
    from gpu_agent.execution.models import WorkspaceRequest

    backend, _, run_id, manifest, _ = local
    return backend.prepare(WorkspaceRequest(run_id=run_id, source_manifest=manifest))


def fake_process(monkeypatch, *, exit_code=0, timed_out=False, truncated=False):
    from gpu_agent.execution import local as module

    calls = []

    def execute(self, argv, cwd, timeout_seconds, max_log_bytes, *, stdin=b"", env=None):
        calls.append((argv, stdin, env))
        if argv[0].endswith("nvcc") and exit_code == 0:
            Path(argv[-1]).write_bytes(b"trusted binary")
        return ProcessCapture(
            exit_code, b'{"values":[3]}', b"log", timed_out, 1.0, now(), now(), truncated=truncated
        )

    monkeypatch.setattr(module.ProcessExecutor, "execute", execute)
    return calls


def test_snapshots_read_bytes_and_preserves_evidence_on_cleanup(local):
    backend, store, run_id, _, repo = local
    handle = prepare(local)
    assert handle.path.parent == backend.workspace_root
    assert handle.path.stat().st_mode & 0o777 == 0o700
    assert (handle.path / "kernel.cu").stat().st_mode & 0o222 == 0
    (repo / "reviewed/kernel.cu").write_bytes(b"changed later")
    assert (handle.path / "kernel.cu").read_bytes() == b"reviewed kernel.cu"
    refs = store.load(run_id).artifact_refs
    assert len(refs) == 4
    cleaned = backend.cleanup(handle)
    assert cleaned.workspace_id == handle.id and cleaned.removed
    assert not handle.path.exists()
    assert {store.read(ref) for ref in refs} == {
        b"reviewed kernel.cu",
        b"reviewed vector_io.cpp",
        b"reviewed vector_api.h",
        b"reviewed json.hpp",
    }


@pytest.mark.parametrize("change", ["claim", "extra", "missing", "hash", "source", "symlink"])
def test_untrusted_or_changed_sources_rejected_before_launch(local, monkeypatch, change):
    from gpu_agent.execution.models import WorkspaceRequest

    backend, _, run_id, manifest, repo = local
    request_manifest = manifest.copy()
    trust_level = "TRUSTED_LOCAL"
    if change == "claim":
        trust_level = "UNTRUSTED"
    elif change == "extra":
        request_manifest["extra.cu"] = "0" * 64
    elif change == "missing":
        del request_manifest["reviewed/json.hpp"]
    elif change == "hash":
        request_manifest["reviewed/kernel.cu"] = "0" * 64
    elif change == "source":
        (repo / "reviewed/kernel.cu").write_bytes(b"unreviewed")
    else:
        (repo / "reviewed/kernel.cu").unlink()
        (repo / "reviewed/kernel.cu").symlink_to(repo / "reviewed/vector_io.cpp")
    calls = fake_process(monkeypatch)
    with pytest.raises(ValueError):
        backend.prepare(
            WorkspaceRequest(
                run_id=run_id, source_manifest=request_manifest, trust_level=trust_level
            )
        )
    assert calls == []


def test_claimed_trust_is_insufficient_without_registry(local, monkeypatch):
    from gpu_agent.execution import local as module

    monkeypatch.setattr(module, "_load_trusted_sources", lambda: {})
    with pytest.raises(ValueError):
        prepare(local)


def test_compile_fixed_argv_clean_environment_and_evidence(local, monkeypatch):
    from gpu_agent.execution.models import BuildRequest

    monkeypatch.setenv("NVCC_PREPEND_FLAGS", "--malicious")
    monkeypatch.setenv("LD_PRELOAD", "/malicious.so")
    monkeypatch.setenv("SECRET", "not inherited")
    calls = fake_process(monkeypatch)
    backend, store, run_id, _, _ = local
    handle = prepare(local)
    result = backend.build(BuildRequest(workspace_id=handle.id))
    assert result.success and result.binary_ref is not None
    argv, _, env = calls[0]
    assert argv == [
        "/trusted/cuda/bin/nvcc",
        "-std=c++17",
        "-lineinfo",
        "-arch=sm_89",
        "-ccbin",
        "/usr/bin/g++",
        "kernel.cu",
        "vector_io.cpp",
        "-I",
        str(handle.path),
        "-o",
        str(handle.path / "vector_add"),
    ]
    assert env == {"PATH": "/trusted/cuda/bin:/usr/bin:/bin", "LC_ALL": "C"}
    assert store.read(result.binary_ref) == b"trusted binary"
    assert store.read(result.tool_result.stderr_artifact) == b"log"
    artifacts = store.load(run_id).artifact_refs
    assert any(
        json.loads(store.read(ref)) == argv for ref in artifacts if ref.name.endswith("argv.json")
    )
    assert any(ref.name.endswith("result.json") for ref in artifacts)
    assert store.load(run_id).status == "QUEUED"


def test_snapshot_tampering_rejected_before_build(local, monkeypatch):
    from gpu_agent.execution.models import BuildRequest

    calls = fake_process(monkeypatch)
    handle = prepare(local)
    (handle.path / "kernel.cu").chmod(0o600)
    (handle.path / "kernel.cu").write_bytes(b"tampered")
    with pytest.raises(ValueError):
        local[0].build(BuildRequest(workspace_id=handle.id))
    assert calls == []


@pytest.mark.parametrize("change", ["binary", "other_run", "forged_ref", "workspace"])
def test_execution_scope_and_binary_binding(local, monkeypatch, change):
    from gpu_agent.execution.models import BuildRequest, ExecutionRequest

    calls = fake_process(monkeypatch)
    backend, store, run_id, _, _ = local
    handle = prepare(local)
    backend.build(BuildRequest(workspace_id=handle.id))
    ref = store.put(run_id, "input.json", b"{}", "public")
    workspace_id = handle.id
    if change == "binary":
        (handle.path / "vector_add").chmod(0o700)
        (handle.path / "vector_add").write_bytes(b"tampered")
    elif change == "other_run":
        other_run = store.create_run("other")
        ref = store.put(other_run.id, "input.json", b"{}", "public")
    elif change == "forged_ref":
        ref = ref.model_copy(update={"sha256": "0" * 64})
    else:
        workspace_id = "unknown"
    with pytest.raises(ValueError):
        backend.run(ExecutionRequest(workspace_id=workspace_id, stdin_ref=ref))
    assert len(calls) == 1


@pytest.mark.parametrize(
    "exit_code,timed_out,truncated,want",
    [
        (0, False, False, "SUCCESS"),
        (1, False, False, "FAILED"),
        (-9, True, False, "TIMEOUT"),
        (0, False, True, "TRUNCATED"),
    ],
)
def test_execution_records_typed_status(local, monkeypatch, exit_code, timed_out, truncated, want):
    from gpu_agent.execution.models import BuildRequest, ExecutionRequest

    fake_process(monkeypatch)
    backend, store, run_id, _, _ = local
    handle = prepare(local)
    backend.build(BuildRequest(workspace_id=handle.id))
    ref = store.put(run_id, "input.json", b'{"n":1}', "public")
    calls = fake_process(monkeypatch, exit_code=exit_code, timed_out=timed_out, truncated=truncated)
    result = backend.run(ExecutionRequest(workspace_id=handle.id, stdin_ref=ref))
    assert result.runtime_status == want
    assert store.read(result.output_ref) == b'{"values":[3]}'
    assert calls[0][1] == b'{"n":1}'


def test_failed_rebuild_revokes_previous_binary(local, monkeypatch):
    from gpu_agent.execution.models import BuildRequest, ExecutionRequest

    fake_process(monkeypatch)
    backend, store, run_id, _, _ = local
    handle = prepare(local)
    backend.build(BuildRequest(workspace_id=handle.id))
    fake_process(monkeypatch, exit_code=1)
    result = backend.build(BuildRequest(workspace_id=handle.id))
    assert not result.success and result.binary_ref is None
    ref = store.put(run_id, "input", b"{}", "public")
    with pytest.raises(ValueError):
        backend.run(ExecutionRequest(workspace_id=handle.id, stdin_ref=ref))


def test_sanitizer_is_typed_unsupported_and_forged_cleanup_rejected(local):
    from gpu_agent.execution.models import SanitizerRequest

    backend = local[0]
    handle = prepare(local)
    result = backend.run_sanitizer(SanitizerRequest(workspace_id=handle.id, tool="memcheck"))
    assert result.status == "UNSUPPORTED"
    assert result.tool_result.tool_error == "UNSUPPORTED"
    with pytest.raises(ValueError):
        backend.cleanup(handle.model_copy(update={"path": handle.path.parent}))
    assert handle.path.exists()


@pytest.mark.parametrize("timeout", [float("nan"), float("inf"), 0, -1])
def test_execution_rejects_invalid_timeout(local, timeout):
    from gpu_agent.execution.models import ExecutionRequest

    backend, store, run_id, _, _ = local
    handle = prepare(local)
    ref = store.put(run_id, "input", b"{}", "public")
    with pytest.raises(ValueError):
        backend.run(
            ExecutionRequest(workspace_id=handle.id, stdin_ref=ref, timeout_seconds=timeout)
        )


def test_execution_evidence_binds_exact_input_and_binary(local, monkeypatch):
    from gpu_agent.execution.models import BuildRequest, ExecutionRequest

    fake_process(monkeypatch)
    backend, store, run_id, _, _ = local
    handle = prepare(local)
    build = backend.build(BuildRequest(workspace_id=handle.id))
    ref = store.put(run_id, "input.json", b'{"n":1}', "public")
    result = backend.run(ExecutionRequest(workspace_id=handle.id, stdin_ref=ref))
    payload = result.tool_result.typed_payload
    assert payload.stdin_ref == ref
    assert payload.binary_ref == build.binary_ref
    request_id = result.tool_result.request_id
    request_ref = next(
        ref
        for ref in store.load(run_id).artifact_refs
        if ref.name == f"run/{request_id}/request.json"
    )
    request = json.loads(store.read(request_ref))
    assert request["stdin_ref"]["id"] == ref.id
    assert request["binary_ref"]["id"] == build.binary_ref.id


def test_unexpected_workspace_file_rejected_before_compile(local, monkeypatch):
    from gpu_agent.execution.models import BuildRequest

    calls = fake_process(monkeypatch)
    handle = prepare(local)
    (handle.path / "stdio.h").write_bytes(b"unreviewed include")
    with pytest.raises(ValueError):
        local[0].build(BuildRequest(workspace_id=handle.id))
    assert not calls


def test_different_architecture_rejected_before_compile(local, monkeypatch):
    from gpu_agent.execution.models import BuildRequest

    calls = fake_process(monkeypatch)
    handle = prepare(local)
    with pytest.raises(ValueError):
        local[0].build(BuildRequest(workspace_id=handle.id, target_arch="sm_90"))
    assert not calls


def test_evaluator_input_and_store_are_rejected(local, monkeypatch, tmp_path):
    from gpu_agent.execution.local import LocalBackend
    from gpu_agent.execution.models import BuildRequest, ExecutionRequest

    backend, _, _, _, repo = local
    evaluator = RunStore(tmp_path / "evaluator", visibility="evaluator")
    with pytest.raises(ValueError):
        LocalBackend(Settings(), evaluator, repo, tmp_path / "private-workspaces")
    fake_process(monkeypatch)
    handle = prepare(local)
    backend.build(BuildRequest(workspace_id=handle.id))
    run = evaluator.create_run("private")
    ref = evaluator.put(run.id, "secret", b"private", "evaluator")
    with pytest.raises(ValueError):
        backend.run(ExecutionRequest(workspace_id=handle.id, stdin_ref=ref))


def test_workspace_symlink_rejected_before_build_or_cleanup(local, monkeypatch, tmp_path):
    from gpu_agent.execution.models import BuildRequest

    calls = fake_process(monkeypatch)
    backend = local[0]
    handle = prepare(local)
    moved = tmp_path / "moved-workspace"
    handle.path.rename(moved)
    handle.path.symlink_to(moved, target_is_directory=True)
    with pytest.raises(ValueError):
        backend.build(BuildRequest(workspace_id=handle.id))
    with pytest.raises(ValueError):
        backend.cleanup(handle)
    assert moved.is_dir() and not calls


def test_missing_compiler_output_cannot_be_success(local, monkeypatch):
    from gpu_agent.execution import local as module
    from gpu_agent.execution.models import BuildRequest

    def missing_output(self, *args, **kwargs):
        return ProcessCapture(0, b"", b"", False, 1.0, now(), now())

    monkeypatch.setattr(module.ProcessExecutor, "execute", missing_output)
    handle = prepare(local)
    result = local[0].build(BuildRequest(workspace_id=handle.id))
    assert not result.success and result.binary_ref is None
    assert result.tool_result.tool_error == "BINARY_UNAVAILABLE"
