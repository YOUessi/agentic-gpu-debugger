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
    environment = backend.evidence.public_view(run.id).environment
    assert environment["cuda_nvcc"] == "12.8.93"
    assert environment["target_arch"] == "sm_89"
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


def test_bound_execution_requires_matching_lock_and_records_runtime_evidence(
    store, tmp_path, monkeypatch
):
    import json

    from gpu_agent.contracts import RepositorySnapshot, RunBinding
    from gpu_agent.environment import load_toolchain_lock
    from gpu_agent.execution import isolated

    source = tmp_path / "source"
    source.mkdir()
    data = b"int main() { return 0; }\n"
    (source / "kernel.cu").write_bytes(data)
    lock_hash = load_toolchain_lock(isolated.LOCK_PATH).lock_hash
    binding = RunBinding(
        repository=RepositorySnapshot(commit="1" * 40, tracked_tree_hash="2" * 64, clean=True),
        purpose="corpus_validation",
        toolchain_lock_hash=lock_hash,
        prompt_version=None,
        model_config_hash=None,
    )
    run = store.create_run("case_execution", binding=binding)
    backend = isolated.IsolatedGPUBackend(store, source, tmp_path / "tasks")
    from gpu_agent.execution.process import ProcessCapture

    entrypoints = {}
    calls = []
    observed_image = [backend._image]

    def execute(argv, *args, **kwargs):
        calls.append(argv)
        command = argv[1]
        if command == "create":
            name = argv[argv.index("--name") + 1]
            entrypoints[name] = argv[argv.index("--entrypoint") + 1]
            return ProcessCapture(0, b"created", b"", False)
        if command == "inspect" and "--format" not in argv:
            name = argv[-1]
            document = [
                {
                    "Image": observed_image[0],
                    "Config": {
                        "User": "65532:65532",
                        "Labels": {isolated.LABEL: name.removeprefix("gpu-agent-")},
                    },
                    "HostConfig": {
                        "NetworkMode": "none",
                        "CapDrop": ["ALL"],
                        "SecurityOpt": ["no-new-privileges"],
                        "ReadonlyRootfs": True,
                        "NanoCpus": 4_000_000_000,
                        "Memory": 4 * 1024**3,
                        "MemorySwap": 4 * 1024**3,
                        "PidsLimit": 64,
                        "LogConfig": {"Type": "none"},
                        "Tmpfs": {"/tmp": "rw,nosuid,nodev,size=256m,mode=1777"},
                        "DeviceRequests": [{"DeviceIDs": ["0"], "Capabilities": [["gpu"]]}],
                    },
                    "Mounts": [],
                }
            ]
            return ProcessCapture(0, json.dumps(document).encode(), b"", False)
        if command == "inspect":
            name = argv[-1]
            return ProcessCapture(0, name.removeprefix("gpu-agent-").encode(), b"", False)
        if command == "start":
            executable = entrypoints[argv[-1]]
            output = {
                "/usr/local/cuda/bin/nvcc": b"Cuda compilation tools, V12.8.93\n",
                "/usr/local/cuda/bin/compute-sanitizer": (
                    b"Compute Sanitizer version 2025.1.0.0 (build 35583870)\n"
                ),
                "/usr/bin/nvidia-smi": b"8.9\n",
            }[executable]
            return ProcessCapture(0, output, b"", False)
        return ProcessCapture(0, b"", b"", False)

    monkeypatch.setattr(backend._executor, "execute", execute)
    backend.prepare(
        WorkspaceRequest(
            run_id=run.id,
            source_manifest={"kernel.cu": hashlib.sha256(data).hexdigest()},
        )
    )
    environment = backend.evidence.public_view(run.id).environment
    assert environment["toolchain_lock_hash"] == lock_hash
    assert json.loads(environment["policy"])["network"] == "none"
    attestation_ref = next(
        ref
        for ref in store.load(run.id).artifact_refs
        if ref.name == "environment/runtime-attestation.json"
    )
    attestation = json.loads(store.read(attestation_ref))
    assert attestation["compute_capability"] == "8.9"
    assert len(attestation["runtime_session_id"]) == 32
    assert environment["runtime_attestation_sha256"] == attestation_ref.sha256
    assert {argv[argv.index("--entrypoint") + 1] for argv in calls if argv[1] == "create"} == {
        "/usr/local/cuda/bin/nvcc",
        "/usr/local/cuda/bin/compute-sanitizer",
        "/usr/bin/nvidia-smi",
    }

    reused = store.create_run("case_execution", binding=binding)
    reused_handle = backend.prepare(
        WorkspaceRequest(
            run_id=reused.id,
            source_manifest={"kernel.cu": hashlib.sha256(data).hexdigest()},
        )
    )
    reused_ref = next(
        ref
        for ref in store.load(reused.id).artifact_refs
        if ref.name == "environment/runtime-attestation.json"
    )
    assert len([argv for argv in calls if argv[1] == "create"]) == 3
    assert reused_ref.run_id != attestation_ref.run_id
    assert reused_ref.sha256 == attestation_ref.sha256
    assert store.read(reused_ref) == store.read(attestation_ref)
    backend.cleanup(reused_handle)

    observed_image[0] = "sha256:" + "f" * 64
    forged = store.create_run("case_execution", binding=binding)
    forged_backend = isolated.IsolatedGPUBackend(store, source, tmp_path / "forged-tasks")
    monkeypatch.setattr(forged_backend._executor, "execute", execute)
    with pytest.raises(ValueError, match="image or policy mismatch"):
        forged_backend.prepare(
            WorkspaceRequest(
                run_id=forged.id,
                source_manifest={"kernel.cu": hashlib.sha256(data).hexdigest()},
            )
        )

    wrong = binding.model_copy(update={"toolchain_lock_hash": "f" * 64})
    other = store.create_run("case_execution", binding=wrong)
    with pytest.raises(ValueError, match="toolchain binding"):
        backend.prepare(
            WorkspaceRequest(
                run_id=other.id,
                source_manifest={"kernel.cu": hashlib.sha256(data).hexdigest()},
            )
        )


def test_ambiguous_probe_create_always_attempts_label_checked_cleanup(store, tmp_path, monkeypatch):
    from gpu_agent.execution.isolated import IsolatedGPUBackend
    from gpu_agent.execution.process import ProcessCapture

    backend = IsolatedGPUBackend(store, tmp_path, tmp_path / "tasks")
    backend._image = "sha256:" + "0" * 64
    calls = []

    def execute(argv, *_args, **_kwargs):
        calls.append(argv)
        if argv[1] == "create":
            return ProcessCapture(None, b"", b"timed out", True)
        if argv[1:3] == ["inspect", "--format"]:
            return ProcessCapture(0, argv[-1].removeprefix("gpu-agent-").encode(), b"", False)
        return ProcessCapture(0, b"", b"", False)

    monkeypatch.setattr(backend._executor, "execute", execute)
    with pytest.raises(ValueError, match="unavailable"):
        backend._fixed_container_probe("/usr/local/cuda/bin/nvcc", ("--version",))
    assert [argv[1] for argv in calls] == ["create", "inspect", "stop", "rm"]


def test_bound_execution_rejects_observed_runtime_version_mismatch(store, tmp_path, monkeypatch):
    from gpu_agent.contracts import RepositorySnapshot, RunBinding
    from gpu_agent.environment import load_toolchain_lock
    from gpu_agent.execution import isolated

    source = tmp_path / "source"
    source.mkdir()
    data = b"int main() { return 0; }\n"
    (source / "kernel.cu").write_bytes(data)
    lock = load_toolchain_lock(isolated.LOCK_PATH)
    run = store.create_run(
        "case_execution",
        binding=RunBinding(
            repository=RepositorySnapshot(commit="1" * 40, tracked_tree_hash="2" * 64, clean=True),
            purpose="corpus_validation",
            toolchain_lock_hash=lock.lock_hash,
            prompt_version=None,
            model_config_hash=None,
        ),
    )
    backend = isolated.IsolatedGPUBackend(store, source, tmp_path / "tasks")

    def mismatch(*_args, **_kwargs):
        return (
            b"Cuda compilation tools, V0.0.0\n",
            lock.image_id,
            hashlib.sha256(backend.policy.model_dump_json().encode()).hexdigest(),
        )

    monkeypatch.setattr(backend, "_fixed_container_probe", mismatch)
    with pytest.raises(ValueError, match="runtime"):
        backend.prepare(
            WorkspaceRequest(
                run_id=run.id,
                source_manifest={"kernel.cu": hashlib.sha256(data).hexdigest()},
            )
        )


@pytest.mark.parametrize(
    "operation,accepted", [("build_standalone", True), ("run", False), ("memcheck", False)]
)
def test_standalone_binary_passes_real_host_export_decoder(
    store, tmp_path, monkeypatch, operation, accepted
):
    import base64
    import json

    from gpu_agent.execution.isolated import IsolatedGPUBackend
    from gpu_agent.execution.process import ProcessCapture

    root = tmp_path / "source"
    root.mkdir()
    data = b"int main() { return 0; }\n"
    (root / "kernel.cu").write_bytes(data)
    backend = IsolatedGPUBackend(store, root, tmp_path / "tasks")
    run = store.create_run("standalone-export")
    handle = backend.prepare(
        WorkspaceRequest(
            run_id=run.id, source_manifest={"kernel.cu": hashlib.sha256(data).hexdigest()}
        )
    )
    operation_id = None

    def execute(argv, *args, **kwargs):
        nonlocal operation_id
        if argv[1] == "create":
            operation_id = argv[argv.index("--name") + 1].removeprefix("gpu-agent-")
        if argv[1] == "inspect":
            return ProcessCapture(0, operation_id.encode(), b"", False)
        if argv[1] == "start":
            envelope = dict(
                exit_code=0,
                stdout="",
                stderr="",
                sanitizer="",
                truncated=False,
                binary=base64.b64encode(b"compiled binary").decode(),
            )
            return ProcessCapture(0, json.dumps(envelope).encode(), b"", False)
        return ProcessCapture(0, b"", b"", False)

    monkeypatch.setattr(backend._executor, "execute", execute)
    if accepted:
        result = backend.build(BuildRequest(workspace_id=handle.id))
        assert result.success and store.read(result.binary_ref) == b"compiled binary"
    else:
        capture, binary, _ = backend._container(handle.path, operation, 1)
        assert capture.tool_error == "INVALID_EXPORT" and binary == b""
