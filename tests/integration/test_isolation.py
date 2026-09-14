"""Policy tests are CPU-only; marked live probes require Docker GPU support."""

import base64
import hashlib
import json
from threading import Event

import pytest

from gpu_agent.execution.models import BuildRequest, WorkspaceRequest
from gpu_agent.execution.process import ProcessCapture


@pytest.fixture
def isolated(tmp_path):
    from gpu_agent.execution.isolated import IsolatedGPUBackend
    from gpu_agent.store import RunStore

    public = tmp_path / "public"
    public.mkdir()
    sources = {}
    for name in ["kernel.cu", "vector_io.cpp", "vector_api.h", "json.hpp"]:
        data = b"candidate " + name.encode()
        (public / name).write_bytes(data)
        sources[name] = hashlib.sha256(data).hexdigest()
    (public / ".git").mkdir()
    (public / ".git/config").write_text("secret")
    store = RunStore(tmp_path / "runs")
    run = store.create_run("unit_isolation")
    backend = IsolatedGPUBackend(store, public, tmp_path / "workspaces")
    # Synthetic image identity is consumed only by mocked Docker calls, never live evidence.
    backend._image = "sha256:" + "0" * 64
    handle = backend.prepare(
        WorkspaceRequest(run_id=run.id, source_manifest=sources, trust_level="UNTRUSTED")
    )
    return backend, store, handle


def test_candidate_snapshot_is_bounded_and_omits_git(isolated):
    backend, store, handle = isolated
    assert {p.name for p in handle.path.iterdir()} == {
        "kernel.cu",
        "vector_io.cpp",
        "vector_api.h",
        "json.hpp",
    }
    assert all(p.stat().st_mode & 0o222 == 0 for p in handle.path.iterdir())
    view = backend.evidence.public_view(handle.run_id)
    assert len(view.source_snapshot) == 4
    assert all(ref.run_id == handle.run_id for ref in view.source_snapshot)
    backend.cleanup(handle)
    assert not handle.path.exists()
    assert all(store.read(ref) for ref in view.source_snapshot)


@pytest.mark.parametrize("path", ["../kernel.cu", ".git/kernel.cu", "private/kernel.cu"])
def test_source_boundary_rejects_unsafe_paths(isolated, path):
    backend, _, handle = isolated
    with pytest.raises(ValueError):
        backend.prepare(WorkspaceRequest(run_id=handle.run_id, source_manifest={path: "0" * 64}))


def test_docker_policy_and_targeted_timeout_cleanup(isolated, monkeypatch):
    backend, _, handle = isolated
    calls = []
    name = None

    def execute(argv, cwd, timeout_seconds, max_log_bytes=2097152, **kwargs):
        nonlocal name
        calls.append(argv)
        if argv[1] == "create":
            name = argv[argv.index("--name") + 1]
            assert argv[argv.index("--user") + 1] == "65532:65532"
            assert argv[argv.index("--network") + 1] == "none"
            assert argv[argv.index("--cap-drop") + 1] == "ALL"
            assert argv[argv.index("--security-opt") + 1] == "no-new-privileges"
            assert "--read-only" in argv
            assert argv[argv.index("--cpus") + 1] == "4"
            assert argv[argv.index("--memory") + 1] == "4g"
            assert argv[argv.index("--pids-limit") + 1] == "64"
            assert argv[argv.index("--gpus") + 1] == "device=0"
            assert argv[argv.index("--log-driver") + 1] == "none"
            mounts = [argv[i + 1] for i, arg in enumerate(argv) if arg == "--mount"]
            assert mounts == [f"type=bind,src={handle.path},dst=/input,readonly"]
            assert "size=1g" in argv[argv.index("--tmpfs") + 1]
            return ProcessCapture(0, b"container-id", b"", False)
        if argv[1] == "start":
            return ProcessCapture(-9, b"", b"", True)
        if argv[1] == "inspect":
            return ProcessCapture(0, name.removeprefix("gpu-agent-").encode(), b"", False)
        return ProcessCapture(0, b"", b"", False)

    monkeypatch.setattr(backend._executor, "execute", execute)
    result = backend.build(BuildRequest(workspace_id=handle.id))
    assert not result.success and result.tool_result.timed_out
    assert [c[1] for c in calls] == ["create", "start", "inspect", "stop", "rm"]
    assert calls[-1] == ["docker", "rm", "-f", name]
    assert calls[-2] == ["docker", "stop", "--time", "0", name]


def test_runtime_failure_is_typed_and_never_executes_host_code(isolated, monkeypatch):
    backend, _, handle = isolated
    calls = []

    def execute(argv, *args, **kwargs):
        calls.append(argv)
        if argv[1] == "inspect":
            return ProcessCapture(1, b"", b"Error: No such object", False)
        return ProcessCapture(
            125, b"", b"could not select device driver with capabilities: [[gpu]]", False
        )

    monkeypatch.setattr(backend._executor, "execute", execute)
    result = backend.build(BuildRequest(workspace_id=handle.id))
    assert not result.success
    assert result.tool_result.tool_error == "CONTAINER_UNAVAILABLE"
    assert all(c[0] == "docker" for c in calls)


def test_gpu_runtime_missing_at_start_is_unavailable(isolated, monkeypatch):
    backend, _, handle = isolated
    operation_id = None

    def execute(argv, *args, **kwargs):
        nonlocal operation_id
        if argv[1] == "create":
            operation_id = argv[argv.index("--name") + 1].removeprefix("gpu-agent-")
        if argv[1] == "inspect":
            return ProcessCapture(0, operation_id.encode(), b"", False)
        if argv[1] == "start":
            return ProcessCapture(
                1, b"", b'could not select device driver "" with capabilities: [[gpu]]', False
            )
        return ProcessCapture(0, b"", b"", False)

    monkeypatch.setattr(backend._executor, "execute", execute)
    result = backend.build(BuildRequest(workspace_id=handle.id))
    assert result.tool_result.tool_error == "CONTAINER_UNAVAILABLE"


def test_build_run_and_memcheck_bind_artifacts_and_separate_logs(isolated, monkeypatch):
    from gpu_agent.execution.models import ExecutionRequest, SanitizerRequest

    backend, store, handle = isolated
    operation = operation_id = None
    log = b"========= COMPUTE-SANITIZER\n========= ERROR SUMMARY: 0 errors\n"

    def encoded(data):
        return base64.b64encode(data).decode()

    def execute(argv, *args, **kwargs):
        nonlocal operation, operation_id
        if argv[1] == "create":
            operation, operation_id = (
                argv[-1],
                argv[argv.index("--name") + 1].removeprefix("gpu-agent-"),
            )
        if argv[1] == "inspect":
            return ProcessCapture(0, operation_id.encode(), b"", False)
        if argv[1] == "start":
            envelope = {
                "exit_code": 0,
                "truncated": False,
                "stdout": encoded(b"program"),
                "stderr": encoded(b"program diagnostic"),
                "binary": encoded(b"binary") if operation == "build" else "",
                "sanitizer": encoded(log) if operation == "memcheck" else "",
            }
            return ProcessCapture(0, json.dumps(envelope).encode(), b"", False)
        return ProcessCapture(0, b"", b"", False)

    monkeypatch.setattr(backend._executor, "execute", execute)
    built = backend.build(BuildRequest(workspace_id=handle.id))
    assert built.success
    assert store.read(built.binary_ref) == b"binary"
    stdin = store.put(handle.run_id, "input", b"{}", "public")
    run = backend.run(ExecutionRequest(workspace_id=handle.id, stdin_ref=stdin))
    assert run.runtime_status == "SUCCESS"
    assert run.tool_result.typed_payload.binary_ref == built.binary_ref
    result = backend.run_sanitizer(SanitizerRequest(workspace_id=handle.id, stdin_ref=stdin))
    assert result.completed and result.check_outcome == "CLEAN"
    assert store.read(result.program_output_ref) == b"program"
    payload = result.tool_result.typed_payload
    assert store.read(payload.program_stderr_ref) == b"program diagnostic"
    assert store.read(result.tool_result.stderr_artifact) == log
    assert payload.binary_ref == built.binary_ref and payload.stdin_ref == stdin
    other = store.create_run("sibling")
    sibling_input = store.put(other.id, "input", b"secret", "public")
    with pytest.raises(ValueError, match="another run"):
        backend.run(ExecutionRequest(workspace_id=handle.id, stdin_ref=sibling_input))
    with pytest.raises(ValueError, match="another run"):
        backend.run_sanitizer(SanitizerRequest(workspace_id=handle.id, stdin_ref=sibling_input))
    assert backend.evidence.public_view(handle.run_id).sanitizer_results == [result]


@pytest.mark.parametrize("fault", ["bad_base64", "oversized_log", "cleanup_label", "cancelled"])
def test_export_and_cleanup_fail_closed(isolated, monkeypatch, fault):
    backend, store, handle = isolated
    operation_id = None
    calls = []
    envelope = {
        "exit_code": 0,
        "truncated": False,
        "stdout": "",
        "stderr": "",
        "sanitizer": "",
        "binary": base64.b64encode(b"binary").decode(),
    }
    if fault == "bad_base64":
        envelope["binary"] = "%%%"
    if fault == "oversized_log":
        envelope["stdout"] = base64.b64encode(b"x" * 2097153).decode()

    def execute(argv, *args, **kwargs):
        nonlocal operation_id
        calls.append(argv)
        if argv[1] == "create":
            operation_id = argv[argv.index("--name") + 1].removeprefix("gpu-agent-")
        if argv[1] == "inspect":
            return ProcessCapture(
                0, ("sibling" if fault == "cleanup_label" else operation_id).encode(), b"", False
            )
        if argv[1] == "start":
            if fault == "cancelled":
                return ProcessCapture(-9, b"", b"", False, cancelled=True)
            return ProcessCapture(0, json.dumps(envelope).encode(), b"", False)
        return ProcessCapture(0, b"", b"", False)

    monkeypatch.setattr(backend._executor, "execute", execute)
    result = backend.build(BuildRequest(workspace_id=handle.id))
    assert not result.success and result.binary_ref is None
    if fault == "cleanup_label":
        assert not any(c[1] in {"stop", "rm"} for c in calls)
        assert result.tool_result.tool_error == "CONTAINER_CLEANUP_FAILED"
    elif fault == "cancelled":
        assert result.tool_result.cancelled
        assert calls[-1][1] == "rm"
    else:
        assert result.tool_result.tool_error == "INVALID_EXPORT"
    assert store.read(result.tool_result.stdout_artifact) == b""


def test_timeout_discards_incomplete_transport_envelope(isolated, monkeypatch):
    backend, store, handle = isolated
    operation_id = None

    def execute(argv, *args, **kwargs):
        nonlocal operation_id
        if argv[1] == "create":
            operation_id = argv[argv.index("--name") + 1].removeprefix("gpu-agent-")
        if argv[1] == "inspect":
            return ProcessCapture(0, operation_id.encode(), b"", False)
        if argv[1] == "start":
            return ProcessCapture(-9, b"x" * 3000000, b"diagnostic", True)
        return ProcessCapture(0, b"", b"", False)

    monkeypatch.setattr(backend._executor, "execute", execute)
    result = backend.build(BuildRequest(workspace_id=handle.id))
    assert result.tool_result.timed_out
    assert len(store.read(result.tool_result.stdout_artifact)) <= 2097152


def test_cancellation_reaches_container_start_and_cleans_only_owned_name(isolated, monkeypatch):
    backend, _, handle = isolated
    cancel = Event()
    cancel.set()
    operation_id = None

    def execute(argv, *args, **kwargs):
        nonlocal operation_id
        if argv[1] == "create":
            operation_id = argv[argv.index("--name") + 1].removeprefix("gpu-agent-")
        if argv[1] == "inspect":
            return ProcessCapture(0, operation_id.encode(), b"", False)
        if argv[1] == "start":
            assert kwargs["cancel"] is cancel
            return ProcessCapture(-9, b"", b"", False, cancelled=True)
        return ProcessCapture(0, b"", b"", False)

    monkeypatch.setattr(backend._executor, "execute", execute)
    result, _, _ = backend._container(handle.path, "run", 1, cancel=cancel)
    assert result.cancelled


def test_prepare_rejects_symlink_and_snapshot_tampering(isolated):
    backend, _, handle = isolated
    source = handle.path / "kernel.cu"
    source.chmod(0o644)
    source.write_text("changed candidate")
    with pytest.raises(ValueError, match="snapshot hash"):
        backend.build(BuildRequest(workspace_id=handle.id))
    source.unlink()
    source.symlink_to(handle.path / "json.hpp")
    with pytest.raises(ValueError, match="symlink"):
        backend.build(BuildRequest(workspace_id=handle.id))


@pytest.mark.container
def test_live_isolation_probe_and_timeout_cleanup(tmp_path):
    from gpu_agent.execution.isolated import IsolatedGPUBackend, ProbeKind
    from gpu_agent.store import RunStore

    store = RunStore(tmp_path / "runs")
    backend = IsolatedGPUBackend(store, tmp_path, tmp_path / "workspaces")
    canary = tmp_path / "outside-canary"
    canary.write_text("must not be visible")
    available = backend.availability()
    if not available.ready:
        pytest.skip(available.reason)
    result = backend.probe(ProbeKind.ISOLATION, canary_path=canary)
    assert result.exit_code == 0, result.stderr
    observed = json.loads(result.stdout)
    assert observed["uid"] == 65532
    assert observed["cap_eff"] == "0000000000000000"
    assert observed["interfaces"] == ["lo"]
    assert observed["root_readonly"] and observed["input_readonly"]
    assert observed["canary_visible"] is False
    assert observed["no_new_privs"] == "1"
    flooded = backend.probe(ProbeKind.LOG_LIMIT)
    assert flooded.truncated and len(flooded.stdout) + len(flooded.stderr) <= 2097152
    timed = backend.probe(ProbeKind.TIMEOUT, timeout_seconds=1)
    assert timed.timed_out
    assert backend.active_containers() == []
