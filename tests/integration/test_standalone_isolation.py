"""Single-source programs use the same sandbox and fixed executable protocol."""

import hashlib

import pytest

from gpu_agent.execution.models import BuildRequest, WorkspaceRequest


def test_standalone_compiles_in_container_with_fixed_arguments(store, tmp_path, monkeypatch):
    from gpu_agent.execution.isolated import IsolatedGPUBackend
    from gpu_agent.execution.process import ProcessCapture

    root = tmp_path / "sources"
    root.mkdir()
    data = b"int main() { return 0; }\n"
    (root / "kernel.cu").write_bytes(data)
    run = store.create_run("standalone")
    backend = IsolatedGPUBackend(store, root, tmp_path / "tasks")
    handle = backend.prepare(
        WorkspaceRequest(
            run_id=run.id,
            source_manifest={"kernel.cu": hashlib.sha256(data).hexdigest()},
            trust_level="UNTRUSTED",
        )
    )
    calls = []

    def execute(path, operation, timeout, **kwargs):
        calls.append(operation)
        return ProcessCapture(0, b"", b"", False), b"fake binary", b""

    monkeypatch.setattr(backend, "_container", execute)
    result = backend.build(BuildRequest(workspace_id=handle.id, timeout_seconds=7))
    assert result.success and calls == ["build_standalone"]
    assert "/input/vector_io.cpp" not in result.tool_result.typed_payload.argv
    assert result.tool_result.typed_payload.argv[-2:] == ["-o", "/tmp/vector_add"]
    assert {p.name for p in handle.path.iterdir()} == {"kernel.cu", "vector_add"}
    backend.cleanup(handle)


@pytest.mark.parametrize("name", ["../kernel.cu", "-o.cu", "kernel.cu;id", "/tmp/kernel.cu"])
def test_standalone_source_cannot_be_a_command_or_path(store, tmp_path, name):
    from gpu_agent.execution.isolated import IsolatedGPUBackend

    run = store.create_run("standalone")
    backend = IsolatedGPUBackend(store, tmp_path, tmp_path / "tasks")
    with pytest.raises(ValueError):
        backend.prepare(WorkspaceRequest(run_id=run.id, source_manifest={name: "0" * 64}))


def test_standalone_container_uses_build_resources_and_no_provider_environment(
    store, tmp_path, monkeypatch
):
    from gpu_agent.execution.isolated import IsolatedGPUBackend
    from gpu_agent.execution.process import ProcessCapture

    root = tmp_path / "source"
    root.mkdir()
    data = b"int main() { return 0; }\n"
    (root / "kernel.cu").write_bytes(data)
    backend = IsolatedGPUBackend(store, root, tmp_path / "tasks")
    backend._image = "sha256:" + "0" * 64
    run = store.create_run("standalone")
    handle = backend.prepare(
        WorkspaceRequest(
            run_id=run.id, source_manifest={"kernel.cu": hashlib.sha256(data).hexdigest()}
        )
    )
    calls = []
    monkeypatch.setenv("OPENAI_API_KEY", "secret-canary")
    monkeypatch.setenv("OPENAI_MODEL", "private-model")

    def execute(argv, *args, **kwargs):
        calls.append(argv)
        if argv[1] == "inspect":
            return ProcessCapture(1, b"", b"Error: No such object", False)
        return ProcessCapture(
            125, b"", b"could not select device driver with capabilities: [[gpu]]", False
        )

    monkeypatch.setattr(backend._executor, "execute", execute)
    result = backend.build(BuildRequest(workspace_id=handle.id))
    create = calls[0]
    assert create[create.index("--tmpfs") + 1] == "/tmp:rw,nosuid,nodev,size=1g,mode=1777"
    assert create[create.index("--network") + 1] == "none"
    assert not any("OPENAI" in part or "secret-canary" in part for argv in calls for part in argv)
    assert not result.success and result.tool_result.tool_error == "CONTAINER_UNAVAILABLE"
    assert all(argv[0] == "docker" for argv in calls)


def test_standalone_service_keeps_oracle_unavailable(oob_service, tmp_path):
    from gpu_agent.agent.models import InconclusiveAction

    service, provider, _ = oob_service
    provider.actions = [InconclusiveAction()]
    source = tmp_path / "standalone.cu"
    source.write_text("int main() { return 0; }\n")
    run = service.diagnose(source)
    diff = tmp_path / "human.diff"
    diff.write_text(
        "--- a/kernel.cu\n+++ b/kernel.cu\n@@ -1 +1 @@\n"
        "-int main() { return 0; }\n+int main() { return 1; }\n"
    )
    candidate = service.register_patch(run.id, diff)
    result = service.verify(run.id, candidate, strict=True)
    assert result.verdict.value == "INCONCLUSIVE" and result.reason_code == "ORACLE_UNAVAILABLE"
    assert "ORACLE_UNAVAILABLE" in service.report(run.id)
    # The implicit/generated selector must not silently choose a human-supplied candidate.
    assert service.verify(run.id).reason_code == "CANDIDATE_UNAVAILABLE"


def test_runner_lock_matches_build_inputs_and_pins_immutable_images():
    import json
    import re
    from pathlib import Path

    root = Path(__file__).resolve().parents[2] / "containers"
    lock = json.loads((root / "toolchain.lock.json").read_text())
    assert lock["runner_sha256"] == hashlib.sha256((root / "runner.py").read_bytes()).hexdigest()
    assert (
        lock["dockerfile_sha256"] == hashlib.sha256((root / "Dockerfile").read_bytes()).hexdigest()
    )
    assert re.fullmatch(r"sha256:[a-f0-9]{64}", lock["image_id"])
    assert lock["base_repo_digest"] == (
        "nvidia/cuda@sha256:a99a1860ba8e2916e5c3e73b72ec4c4301653a84586e05bfc9a2aa2d58027e97"
    )


def test_stale_runner_lock_fails_closed(store, tmp_path, monkeypatch):
    import json
    from pathlib import Path

    import gpu_agent.execution.isolated as isolated

    original = Path(__file__).resolve().parents[2] / "containers"
    lock = json.loads((original / "toolchain.lock.json").read_text())
    lock_path = tmp_path / "toolchain.lock.json"
    lock_path.write_text(json.dumps(lock))
    (tmp_path / "runner.py").write_text("modified unpinned runner")
    (tmp_path / "Dockerfile").write_bytes((original / "Dockerfile").read_bytes())
    monkeypatch.setattr(isolated, "LOCK_PATH", lock_path)
    backend = isolated.IsolatedGPUBackend(store, tmp_path, tmp_path / "tasks")
    assert backend._image is None
    assert not backend.availability().ready
