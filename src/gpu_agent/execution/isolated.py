"""Typed candidate execution in resource-limited Docker GPU containers.

The inherited helpers only manage host-owned snapshots and RunStore metadata.
All prepare/build/run/sanitizer operations are replaced; no local execution path
is reachable. Containers share the host driver/kernel, not a hostile-code VM.
"""

import base64
import binascii
import hashlib
import json
import os
import re
import shutil
import tempfile
from dataclasses import replace
from enum import StrEnum
from pathlib import Path, PurePosixPath
from threading import Event

from pydantic import BaseModel, ConfigDict

from gpu_agent.config import Settings
from gpu_agent.contracts import ArtifactRef, new_id
from gpu_agent.environment import (
    ExpectedToolchain,
    RuntimeToolchainAttestation,
    load_toolchain_lock,
    validate_runtime_toolchain,
)
from gpu_agent.evidence.models import EvidenceBundle
from gpu_agent.evidence.repository import EvidenceRepository
from gpu_agent.evidence.sanitizer import parse_sanitizer
from gpu_agent.execution.local import (
    BINARY_LIMIT,
    LOG_LIMIT,
    SOURCE_LIMIT,
    SOURCE_NAMES,
    LocalBackend,
    _Workspace,
)
from gpu_agent.execution.models import (
    BuildPayload,
    BuildRequest,
    BuildResult,
    ExecutionPayload,
    ExecutionRequest,
    ExecutionResult,
    SanitizerPayload,
    SanitizerRequest,
    SanitizerResult,
    SanitizerTool,
    WorkspaceHandle,
    WorkspaceRequest,
)
from gpu_agent.execution.process import ProcessCapture, ProcessExecutor
from gpu_agent.store import RunStore, read_regular, reject_symlinks

LOCK_PATH = Path(__file__).resolve().parents[3] / "containers/toolchain.lock.json"
LABEL = "io.gpu-agent.operation"


class ProbeKind(StrEnum):
    ISOLATION = "isolation"
    LOG_LIMIT = "log_limit"
    TIMEOUT = "timeout"


class Availability(BaseModel):
    ready: bool
    reason: str


class IsolationPolicy(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    user: str = "65532:65532"
    network: str = "none"
    capabilities: str = "ALL"
    no_new_privileges: bool = True
    read_only_root: bool = True
    cpus: int = 4
    memory: str = "4g"
    pids: int = 64
    build_tmpfs: str = "1g"
    run_tmpfs: str = "256m"
    gpu: str = "device=0"
    log_driver: str = "none"
    log_bytes: int = LOG_LIMIT
    binary_bytes: int = BINARY_LIMIT
    source_bytes: int = SOURCE_LIMIT
    build_timeout_seconds: int = 120
    max_run_timeout_seconds: int = 300


class _Envelope(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    exit_code: int
    truncated: bool
    stdout: str
    stderr: str
    sanitizer: str
    binary: str


class IsolatedGPUBackend(LocalBackend):
    def __init__(self, store: RunStore, public_source_root: Path, workspace_root: Path) -> None:
        # Deliberately do not relax LocalBackend's public-only exact-hash trust gate.
        self.settings, self.store = Settings(), store
        self.repo_root = public_source_root.absolute()
        self.workspace_root = workspace_root.absolute()
        reject_symlinks(self.repo_root)
        reject_symlinks(self.workspace_root)
        self.workspace_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self._workspaces: dict[str, _Workspace] = {}
        self._executor = ProcessExecutor()
        self.policy = IsolationPolicy()
        self.evidence = EvidenceRepository(store, evaluator=store.visibility == "evaluator")
        self._owner = new_id()
        self._image: str | None = None
        self._base: str | None = None
        self._versions: dict[str, str] = {}
        self._expected_toolchain: ExpectedToolchain | None = None
        if LOCK_PATH.exists():
            try:
                self._expected_toolchain = load_toolchain_lock(LOCK_PATH)
                self._image = self._expected_toolchain.image_id
                self._base = self._expected_toolchain.base_repo_digest
                self._versions = {
                    "cuda_nvcc": self._expected_toolchain.cuda_nvcc,
                    "compute_sanitizer": self._expected_toolchain.compute_sanitizer,
                    "target_arch": self._expected_toolchain.target_arch,
                }
            except (OSError, ValueError):
                pass  # A missing/stale lock is typed CONTAINER_UNAVAILABLE at execution time.

    def prepare(self, request: WorkspaceRequest) -> WorkspaceHandle:
        self._active_run(request.run_id)
        run = self.store.load(request.run_id)
        if run.binding is not None:
            if (
                self._expected_toolchain is None
                or run.binding.toolchain_lock_hash is None
                or run.binding.toolchain_lock_hash != self._expected_toolchain.lock_hash
            ):
                raise ValueError("release toolchain binding does not match backend lock")
        snapshots: dict[str, bytes] = {}
        standalone = set(request.source_manifest) == {"kernel.cu"}
        if not standalone and len(request.source_manifest) != 4:
            raise ValueError("expected four public vector sources or one kernel.cu")
        for relative, expected in request.source_manifest.items():
            path = PurePosixPath(relative)
            if (
                path.is_absolute()
                or "\\" in relative
                or "," in relative
                or any(p in {"..", ".git", "private", "evaluator", ".ssh"} for p in path.parts)
                or path.name not in SOURCE_NAMES
                or path.name in snapshots
            ):
                raise ValueError("invalid public source path")
            data = read_regular(self.repo_root / relative, SOURCE_LIMIT)
            if hashlib.sha256(data).hexdigest() != expected:
                raise ValueError("source hash mismatch")
            snapshots[path.name] = data
        reject_symlinks(self.workspace_root)
        directory = Path(tempfile.mkdtemp(prefix="isolated-", dir=self.workspace_root))
        handle = WorkspaceHandle(id=new_id(), run_id=request.run_id, path=directory)
        try:
            directory.chmod(0o755)
            attestation: RuntimeToolchainAttestation | None = None
            attestation_ref: ArtifactRef | None = None
            if run.binding is not None:
                assert self._expected_toolchain is not None
                attestation = self._attest_runtime()
                policy_hash = hashlib.sha256(self.policy.model_dump_json().encode()).hexdigest()
                validate_runtime_toolchain(
                    self._expected_toolchain,
                    attestation,
                    expected_policy_hash=policy_hash,
                )
                attestation_ref = self.store.put(
                    request.run_id,
                    "environment/runtime-attestation.json",
                    attestation.model_dump_json().encode(),
                    self.store.visibility,
                )
            refs = []
            for name, data in snapshots.items():
                self._write_snapshot(directory / name, data)
                refs.append(
                    self.store.put(
                        request.run_id, f"sources/{handle.id}/{name}", data, self.store.visibility
                    )
                )
            self._workspaces[handle.id] = _Workspace(
                handle=handle,
                hashes={name: hashlib.sha256(data).hexdigest() for name, data in snapshots.items()},
            )
            environment = {
                **(
                    {
                        "cuda_nvcc": attestation.cuda_nvcc,
                        "compute_sanitizer": attestation.compute_sanitizer,
                        "target_arch": attestation.target_arch,
                        "compute_capability": attestation.compute_capability,
                    }
                    if attestation is not None
                    else self._versions
                ),
                "backend": "IsolatedGPUBackend",
                "toolchain_lock_hash": (
                    self._expected_toolchain.lock_hash
                    if self._expected_toolchain is not None
                    else "unavailable"
                ),
                "image_id": self._image or "unavailable",
                "base_repo_digest": self._base or "unavailable",
                "policy": self.policy.model_dump_json(),
            }
            if attestation_ref is not None:
                environment["runtime_attestation_sha256"] = attestation_ref.sha256
            self.evidence.save(
                request.run_id,
                EvidenceBundle(
                    environment=environment,
                    source_snapshot=refs,
                    limitations=["Containers share the host kernel and GPU driver."],
                ),
            )
        except BaseException:
            self._workspaces.pop(handle.id, None)
            shutil.rmtree(directory)
            raise
        return handle

    @staticmethod
    def _write_snapshot(path: Path, data: bytes, mode: int = 0o444) -> None:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, mode)
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            # The container UID differs from the host owner, even under umask 077.
            os.fchmod(stream.fileno(), mode)

    def _check_snapshot(self, state: _Workspace) -> None:
        if {p.name for p in state.handle.path.iterdir()} - (SOURCE_NAMES | {"vector_add"}):
            raise ValueError("unexpected workspace files")
        for name, expected in state.hashes.items():
            if (
                hashlib.sha256(read_regular(state.handle.path / name, SOURCE_LIMIT)).hexdigest()
                != expected
            ):
                raise ValueError("snapshot hash mismatch")

    def _docker(
        self,
        args: list[str],
        timeout: float = 15,
        *,
        stdin: bytes = b"",
        limit: int = LOG_LIMIT,
        cancel: Event | None = None,
    ) -> ProcessCapture:
        return self._executor.execute(
            ["docker", *args],
            self.workspace_root,
            timeout,
            limit,
            stdin=stdin,
            env={"PATH": "/usr/local/bin:/usr/bin:/bin", "LC_ALL": "C"},
            cancel=cancel,
        )

    def _policy_create_args(
        self,
        name: str,
        operation_id: str,
        tmpfs: str,
        *,
        mount: Path | None = None,
    ) -> list[str]:
        args = [
            "create",
            "--name",
            name,
            "--label",
            f"{LABEL}={operation_id}",
            "--label",
            f"io.gpu-agent.owner={self._owner}",
            "--user",
            self.policy.user,
            "--network",
            self.policy.network,
            "--cap-drop",
            self.policy.capabilities,
            "--security-opt",
            "no-new-privileges",
            "--read-only",
            "--cpus",
            str(self.policy.cpus),
            "--memory",
            self.policy.memory,
            "--memory-swap",
            self.policy.memory,
            "--pids-limit",
            str(self.policy.pids),
            "--gpus",
            self.policy.gpu,
            "--log-driver",
            self.policy.log_driver,
            "--ulimit",
            "core=0",
            "--shm-size",
            "16m",
            "--tmpfs",
            f"/tmp:rw,nosuid,nodev,size={tmpfs},mode=1777",
        ]
        if mount is not None:
            args.extend(["--mount", f"type=bind,src={mount},dst=/input,readonly"])
        return args

    def _inspect_attestation_container(self, name: str, operation_id: str) -> tuple[str, str]:
        capture = self._docker(["inspect", name], limit=65536)
        if self._runtime_status(capture) != "SUCCESS":
            raise ValueError("runtime container inspection failed")
        try:
            documents = json.loads(capture.stdout)
            if not isinstance(documents, list) or len(documents) != 1:
                raise ValueError
            document = documents[0]
            config = document["Config"]
            host = document["HostConfig"]
            devices = host["DeviceRequests"]
            valid = (
                document["Image"] == self._image
                and config["User"] == self.policy.user
                and config["Labels"][LABEL] == operation_id
                and host["NetworkMode"] == self.policy.network
                and host["CapDrop"] == [self.policy.capabilities]
                and host["SecurityOpt"] == ["no-new-privileges"]
                and host["ReadonlyRootfs"] is self.policy.read_only_root
                and host["NanoCpus"] == self.policy.cpus * 1_000_000_000
                and host["Memory"] == 4 * 1024**3
                and host["MemorySwap"] == 4 * 1024**3
                and host["PidsLimit"] == self.policy.pids
                and host["LogConfig"]["Type"] == self.policy.log_driver
                and host["Tmpfs"]["/tmp"]
                == f"rw,nosuid,nodev,size={self.policy.run_tmpfs},mode=1777"
                and isinstance(devices, list)
                and len(devices) == 1
                and devices[0]["DeviceIDs"] == ["0"]
                and devices[0]["Capabilities"] == [["gpu"]]
                and document["Mounts"] == []
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError("runtime container policy attestation is malformed") from exc
        if not valid:
            raise ValueError("runtime container image or policy mismatch")
        return document["Image"], hashlib.sha256(self.policy.model_dump_json().encode()).hexdigest()

    def _fixed_container_probe(
        self, executable: str, arguments: tuple[str, ...]
    ) -> tuple[bytes, str, str]:
        allowed = {
            ("/usr/local/cuda/bin/nvcc", ("--version",)),
            ("/usr/local/cuda/bin/compute-sanitizer", ("--version",)),
            (
                "/usr/bin/nvidia-smi",
                ("--query-gpu=compute_cap", "--format=csv,noheader"),
            ),
        }
        if (executable, arguments) not in allowed or self._image is None:
            raise ValueError("unsupported runtime attestation command")
        operation_id = new_id()
        name = "gpu-agent-" + operation_id
        created = False
        try:
            argv = self._policy_create_args(name, operation_id, self.policy.run_tmpfs) + [
                "--entrypoint",
                executable,
                self._image,
                *arguments,
            ]
            create = self._docker(argv, limit=65536)
            if self._runtime_status(create) != "SUCCESS":
                raise ValueError("runtime attestation container unavailable")
            created = True
            image_id, policy_hash = self._inspect_attestation_container(name, operation_id)
            capture = self._docker(["start", "--attach", name], limit=65536)
            if (
                self._runtime_status(capture) != "SUCCESS"
                or len(capture.stdout) + len(capture.stderr) > 65536
            ):
                raise ValueError("runtime attestation command failed")
            return capture.stdout + b"\n" + capture.stderr, image_id, policy_hash
        finally:
            if created and not self._remove_container(name, operation_id):
                raise ValueError("runtime attestation container cleanup failed")

    def _attest_runtime(self) -> RuntimeToolchainAttestation:
        if self._expected_toolchain is None:
            raise ValueError("runtime expected configuration is unavailable")
        nvcc, nvcc_image, nvcc_policy = self._fixed_container_probe(
            "/usr/local/cuda/bin/nvcc", ("--version",)
        )
        sanitizer, sanitizer_image, sanitizer_policy = self._fixed_container_probe(
            "/usr/local/cuda/bin/compute-sanitizer", ("--version",)
        )
        device, device_image, device_policy = self._fixed_container_probe(
            "/usr/bin/nvidia-smi",
            ("--query-gpu=compute_cap", "--format=csv,noheader"),
        )
        if (
            len({nvcc_image, sanitizer_image, device_image}) != 1
            or len({nvcc_policy, sanitizer_policy, device_policy}) != 1
        ):
            raise ValueError("runtime attestation containers disagree")
        nvcc_match = re.search(rb"\bV([0-9]+\.[0-9]+\.[0-9]+)\b", nvcc)
        sanitizer_match = re.search(
            rb"\bversion\s+([0-9]+(?:\.[0-9]+){3}(?: \(build [0-9]+\))?)",
            sanitizer,
            re.IGNORECASE,
        )
        capability = device.strip().decode("ascii", errors="strict")
        if (
            nvcc_match is None
            or sanitizer_match is None
            or re.fullmatch(r"[0-9]+\.[0-9]+", capability) is None
        ):
            raise ValueError("runtime attestation output is invalid")
        return RuntimeToolchainAttestation(
            lock_hash=self._expected_toolchain.lock_hash,
            image_id=nvcc_image,
            cuda_nvcc=nvcc_match.group(1).decode("ascii"),
            compute_sanitizer=sanitizer_match.group(1).decode("ascii"),
            compute_capability=capability,
            target_arch="sm_" + capability.replace(".", ""),
            policy_hash=nvcc_policy,
        )

    def _remove_container(self, name: str, operation_id: str) -> bool:
        inspected = self._docker(
            ["inspect", "--format", '{{index .Config.Labels "' + LABEL + '"}}', name]
        )
        if inspected.exit_code != 0:
            # A missing container is clean; other daemon failures are not proof of cleanup.
            return b"No such" in inspected.stderr
        if inspected.stdout.decode().strip() != operation_id:
            return False
        self._docker(["stop", "--time", "0", name])
        removed = self._docker(["rm", "-f", name])
        return removed.exit_code == 0

    def _container(
        self,
        path: Path,
        operation: str,
        timeout: float,
        *,
        stdin: bytes = b"",
        cancel: Event | None = None,
    ) -> tuple[ProcessCapture, bytes, bytes]:
        if self._image is None:
            return (
                ProcessCapture(None, b"", b"", False, tool_error="CONTAINER_UNAVAILABLE"),
                b"",
                b"",
            )
        reject_symlinks(path)
        if path.parent != self.workspace_root or not path.is_dir() or "," in str(path):
            raise ValueError("invalid task mount")
        operation_id = new_id()
        name = "gpu-agent-" + operation_id
        tmpfs = (
            self.policy.build_tmpfs
            if operation in {"build", "build_standalone"}
            else self.policy.run_tmpfs
        )
        args = self._policy_create_args(name, operation_id, tmpfs, mount=path) + [
            "--interactive",
            self._image,
            operation,
        ]
        binary = sanitizer = b""
        try:
            capture = self._docker(args)
            if capture.exit_code != 0 or capture.tool_error or capture.timed_out:
                capture = replace(capture, tool_error="CONTAINER_UNAVAILABLE")
            else:
                capture = self._docker(
                    ["start", "--attach", "--interactive", name],
                    timeout,
                    stdin=stdin,
                    limit=96 * 1024 * 1024,
                    cancel=cancel,
                )
            if capture.exit_code == 0 and not (
                capture.timed_out or capture.cancelled or capture.truncated or capture.tool_error
            ):
                try:
                    envelope = _Envelope.model_validate_json(capture.stdout)
                    output, error, sanitizer, binary = (
                        base64.b64decode(value, validate=True)
                        for value in (
                            envelope.stdout,
                            envelope.stderr,
                            envelope.sanitizer,
                            envelope.binary,
                        )
                    )
                    if (
                        len(output) + len(error) > LOG_LIMIT
                        or len(sanitizer) > LOG_LIMIT
                        or len(binary) > BINARY_LIMIT
                        or (operation not in {"build", "build_standalone"} and binary)
                    ):
                        raise ValueError("export limit")
                    capture = replace(
                        capture,
                        exit_code=envelope.exit_code,
                        stdout=output,
                        stderr=error,
                        truncated=envelope.truncated,
                    )
                except (ValueError, binascii.Error):
                    capture = replace(capture, stdout=b"", stderr=b"", tool_error="INVALID_EXPORT")
                    binary = sanitizer = b""
            else:
                # Incomplete JSON is transport data, not a target log. Never publish it
                # as program output (nor allow the larger binary transport quota to leak).
                unavailable = (
                    b"could not select device driver" in capture.stderr
                    or b"nvidia-container-cli" in capture.stderr
                )
                capture = replace(
                    capture,
                    stdout=b"",
                    stderr=capture.stderr[:LOG_LIMIT],
                    tool_error=capture.tool_error
                    or (
                        None
                        if capture.timed_out or capture.cancelled
                        else ("CONTAINER_UNAVAILABLE" if unavailable else "CONTAINER_ERROR")
                    ),
                )
        finally:
            cleaned = self._remove_container(name, operation_id)
        if not cleaned:
            capture = replace(capture, tool_error="CONTAINER_CLEANUP_FAILED")
        return capture, binary, sanitizer

    def build(self, request: BuildRequest) -> BuildResult:
        state = self._workspace(request.workspace_id)
        state.binary_ref = None
        if request.target_arch != "sm_89":
            raise ValueError("only the locked sm_89 target is supported")
        self._check_snapshot(state)
        binary_path = state.handle.path / "vector_add"
        reject_symlinks(binary_path)
        binary_path.unlink(missing_ok=True)
        request_id = new_id()
        standalone = set(state.hashes) == {"kernel.cu"}
        operation = "build_standalone" if standalone else "build"
        capture, binary, _ = self._container(state.handle.path, operation, request.timeout_seconds)
        if self._runtime_status(capture) == "SUCCESS" and not binary:
            capture = replace(capture, tool_error="BINARY_UNAVAILABLE")
        success = self._runtime_status(capture) == "SUCCESS"
        if success:
            self._write_snapshot(binary_path, binary, 0o555)
            state.binary_ref = self._put(state, f"build/{request_id}/binary", binary)
        stdout = self._put(state, f"build/{request_id}/stdout", capture.stdout)
        stderr = self._put(state, f"build/{request_id}/stderr", capture.stderr)
        argv = [
            "nvcc",
            "-std=c++17",
            "-lineinfo",
            "-arch=sm_89",
            "-ccbin",
            "/usr/bin/g++",
            "/input/kernel.cu",
            "/input/vector_io.cpp",
            "-I",
            "/input",
            "-o",
            "/tmp/vector_add",
        ]
        if standalone:
            argv.remove("/input/vector_io.cpp")
        payload = BuildPayload(
            argv=argv, binary_ref=state.binary_ref, source_manifest=dict(state.hashes)
        )
        result = BuildResult(
            success=success,
            binary_ref=state.binary_ref,
            tool_result=self._tool_result(
                state, "build", capture, payload, stdout, stderr, request_id
            ),
        )
        bundle = self.evidence.view(state.handle.run_id)
        self.evidence.save(state.handle.run_id, bundle.model_copy(update={"build_result": result}))
        return result

    def _put(self, state: _Workspace, name: str, data: bytes) -> ArtifactRef:
        return self.store.put(state.handle.run_id, name, data, self.store.visibility)

    def _input(self, state: _Workspace, ref: ArtifactRef) -> bytes:
        if ref.run_id != state.handle.run_id or ref.visibility != self.store.visibility:
            raise ValueError("input belongs to another run or store")
        if ref.byte_count > 32 * 1024 * 1024:
            raise ValueError("input exceeds process limit")
        return self.store.read(ref)

    def _binary(self, state: _Workspace) -> ArtifactRef:
        self._check_snapshot(state)
        if state.binary_ref is None:
            raise ValueError("workspace has no successful build")
        data = read_regular(state.handle.path / "vector_add", BINARY_LIMIT)
        if hashlib.sha256(data).hexdigest() != state.binary_ref.sha256:
            raise ValueError("binary hash mismatch")
        self.store.read(state.binary_ref)
        return state.binary_ref

    def run(self, request: ExecutionRequest, *, cancel: Event | None = None) -> ExecutionResult:
        state = self._workspace(request.workspace_id)
        binary_ref = self._binary(state)
        stdin = self._input(state, request.stdin_ref)
        request_id = new_id()
        capture, _, _ = self._container(
            state.handle.path, "run", request.timeout_seconds, stdin=stdin, cancel=cancel
        )
        stdout = self._put(state, f"run/{request_id}/stdout", capture.stdout)
        stderr = self._put(state, f"run/{request_id}/stderr", capture.stderr)
        status = self._runtime_status(capture)
        payload = ExecutionPayload(
            runtime_status=status,
            output_ref=stdout,
            stdin_ref=request.stdin_ref,
            binary_ref=binary_ref,
        )
        result = ExecutionResult(
            output_ref=stdout,
            runtime_status=status,
            tool_result=self._tool_result(
                state, "run", capture, payload, stdout, stderr, request_id
            ),
        )
        bundle = self.evidence.view(state.handle.run_id)
        self.evidence.save(
            state.handle.run_id, bundle.model_copy(update={"execution_result": result})
        )
        return result

    def run_sanitizer(
        self, request: SanitizerRequest, *, cancel: Event | None = None
    ) -> SanitizerResult:
        state = self._workspace(request.workspace_id)
        binary_ref = self._binary(state)
        stdin = self._input(state, request.stdin_ref) if request.stdin_ref else b""
        request_id = new_id()
        capture, _, log = self._container(
            state.handle.path,
            request.tool,
            request.timeout_seconds,
            stdin=stdin,
            cancel=cancel,
        )
        stdout = self._put(state, f"sanitizer/{request_id}/program.stdout", capture.stdout)
        stderr = self._put(state, f"sanitizer/{request_id}/program.stderr", capture.stderr)
        raw = self._put(state, f"sanitizer/{request_id}/{request.tool}.log", log)
        parsed = parse_sanitizer(SanitizerTool(request.tool), replace(capture, stderr=log))
        findings = [f.model_copy(update={"raw_ref": raw}) for f in parsed.findings]
        payload = SanitizerPayload(
            status=parsed.status,
            tool=request.tool,
            findings=findings,
            completed=parsed.completed,
            parser_version=parsed.parser_version,
            check_outcome=parsed.check_outcome,
            binary_ref=binary_ref,
            stdin_ref=request.stdin_ref,
            program_output_ref=stdout,
            program_stderr_ref=stderr,
        )
        tool_result = self._tool_result(
            state, "sanitizer", capture, payload, stdout, raw, request_id
        )
        result = parsed.model_copy(
            update={"findings": findings, "tool_result": tool_result, "program_output_ref": stdout}
        )
        bundle = self.evidence.view(state.handle.run_id)
        self.evidence.save(
            state.handle.run_id,
            bundle.model_copy(
                update={
                    "sanitizer_results": [*bundle.sanitizer_results, result],
                    "source_locations": [
                        *bundle.source_locations,
                        *(f.source_location for f in findings if f.source_location),
                    ],
                }
            ),
        )
        return result

    def probe(
        self, kind: ProbeKind, timeout_seconds: float = 10, *, canary_path: Path | None = None
    ) -> ProcessCapture:
        kind = ProbeKind(kind)
        if not 0 < timeout_seconds <= 300:
            raise ValueError("invalid probe timeout")
        directory = Path(tempfile.mkdtemp(prefix="probe-", dir=self.workspace_root))
        directory.chmod(0o755)
        try:
            stdin = json.dumps({"canary_path": str(canary_path)}).encode() if canary_path else b""
            return self._container(directory, kind.value, timeout_seconds, stdin=stdin)[0]
        finally:
            shutil.rmtree(directory)

    def availability(self) -> Availability:
        capture = self.probe(ProbeKind.ISOLATION)
        ready = self._runtime_status(capture) == "SUCCESS"
        return Availability(
            ready=ready,
            reason="READY"
            if ready
            else (
                (capture.tool_error or "CONTAINER_UNAVAILABLE")
                + ": "
                + capture.stderr.decode("utf-8", errors="replace")[:2048]
            ),
        )

    def active_containers(self) -> list[str]:
        result = self._docker(
            ["ps", "--all", "--quiet", "--filter", f"label=io.gpu-agent.owner={self._owner}"]
        )
        if result.exit_code != 0:
            raise RuntimeError("cannot inspect active containers")
        return result.stdout.decode().split()
