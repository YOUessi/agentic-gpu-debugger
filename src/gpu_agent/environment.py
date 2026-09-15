"""Read-only, bounded metadata probes; not a workload execution backend."""

import csv
import hashlib
import json
import platform as host_platform
import re
import subprocess
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from gpu_agent.config import Settings
from gpu_agent.store import read_regular

ProbeRunner = Callable[[list[str], float], subprocess.CompletedProcess[str]]


class LockedToolchain(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    lock_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    image_id: str = Field(pattern=r"^sha256:[a-f0-9]{64}$")
    base_repo_digest: str = Field(pattern=r"^nvidia/cuda@sha256:[a-f0-9]{64}$")
    cuda_nvcc: str = Field(min_length=1, max_length=128)
    compute_sanitizer: str = Field(min_length=1, max_length=128)
    target_arch: str = Field(pattern=r"^sm_[0-9]+$")


def load_toolchain_lock(path: Path) -> LockedToolchain:
    """Load a bounded lock and verify the build inputs it hashes."""
    lock_path = path.absolute()
    raw = read_regular(lock_path, 65536)
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("invalid toolchain lock") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("invalid toolchain lock")
    required = {
        "image_id",
        "base_repo_digest",
        "cuda_nvcc",
        "compute_sanitizer",
        "target_arch",
        "runner_sha256",
        "dockerfile_sha256",
    }
    if any(not isinstance(payload.get(key), str) for key in required):
        raise ValueError("invalid toolchain lock")
    for name, key in (("runner.py", "runner_sha256"), ("Dockerfile", "dockerfile_sha256")):
        actual = hashlib.sha256(read_regular(lock_path.with_name(name), 65536)).hexdigest()
        if actual != payload[key]:
            raise ValueError("toolchain lock input mismatch")
    return LockedToolchain(
        lock_hash=hashlib.sha256(raw).hexdigest(),
        image_id=payload["image_id"],
        base_repo_digest=payload["base_repo_digest"],
        cuda_nvcc=payload["cuda_nvcc"],
        compute_sanitizer=payload["compute_sanitizer"],
        target_arch=payload["target_arch"],
    )


def validate_runtime_toolchain(
    lock: LockedToolchain, environment: dict[str, str], *, expected_policy: str
) -> None:
    """Require recorded isolated-runtime identity to exactly match the verified lock."""
    expected = {
        "backend": "IsolatedGPUBackend",
        "toolchain_lock_hash": lock.lock_hash,
        "image_id": lock.image_id,
        "base_repo_digest": lock.base_repo_digest,
        "cuda_nvcc": lock.cuda_nvcc,
        "compute_sanitizer": lock.compute_sanitizer,
        "target_arch": lock.target_arch,
        "policy": expected_policy,
    }
    if environment != expected:
        raise ValueError("recorded runtime does not match the locked toolchain and policy")


class ProbeEvidence(BaseModel):
    argv: list[str]
    exit_code: int | None = None
    stdout: str = ""
    stderr: str = ""
    error: str | None = None


class GPUInfo(BaseModel):
    name: str
    driver_version: str
    compute_capability: str


class ToolchainManifest(BaseModel):
    cuda_root: Path
    cuda_bin: Path
    host_compiler: Path
    nvidia_smi: Path
    target_arch: str
    python_version: str = Field(default_factory=lambda: host_platform.python_version())
    platform: str = Field(default_factory=host_platform.system)
    machine: str = Field(default_factory=host_platform.machine)
    nvcc_version: str | None = None
    sanitizer_version: str | None = None
    host_compiler_version: str | None = None
    supported_architectures: list[str] = Field(default_factory=list)
    gpus: list[GPUInfo] = Field(default_factory=list)


class EnvironmentReport(BaseModel):
    schema_version: Literal[1] = 1
    policy_version: Literal["cuda-12.8-update1-linux-x86_64-v1"] = (
        "cuda-12.8-update1-linux-x86_64-v1"
    )
    observed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    ready: bool
    readiness_scope: Literal["metadata_only"] = "metadata_only"
    execution_verified: Literal[False] = False
    reason_codes: list[str]
    toolchain: ToolchainManifest
    probes: list[ProbeEvidence]


def run_probe(argv: list[str], timeout: float) -> subprocess.CompletedProcess[str]:
    # These are operator-selected version/query tools, never candidate commands.
    return subprocess.run(
        argv,
        capture_output=True,
        text=True,
        errors="replace",
        timeout=timeout,
        check=False,
    )


def _version(text: str, pattern: str) -> str | None:
    match = re.search(pattern, text, re.IGNORECASE)
    return match.group(1) if match else None


def probe_environment(settings: Settings, *, runner: ProbeRunner = run_probe) -> EnvironmentReport:
    reasons: list[str] = []
    probes: list[ProbeEvidence] = []
    toolchain = ToolchainManifest(
        cuda_root=settings.cuda_root,
        cuda_bin=settings.bin_dir,
        host_compiler=settings.host_compiler,
        nvidia_smi=settings.nvidia_smi,
        target_arch=settings.target_arch,
    )

    def query(code: str, executable: Path, *args: str) -> str | None:
        evidence = ProbeEvidence(argv=[str(executable), *args])
        probes.append(evidence)
        try:
            result = runner(evidence.argv, settings.probe_timeout_seconds)
        except FileNotFoundError:
            evidence.error = f"{code}_MISSING"
        except subprocess.TimeoutExpired:
            evidence.error = f"{code}_TIMEOUT"
        except OSError:
            evidence.error = f"{code}_UNAVAILABLE"
        else:
            evidence.exit_code = result.returncode
            evidence.stdout = result.stdout[:16384]
            evidence.stderr = result.stderr[:16384]
            if len(result.stdout) > 16384 or len(result.stderr) > 16384:
                evidence.error = f"{code}_OUTPUT_TOO_LARGE"
            elif result.returncode != 0:
                evidence.error = f"{code}_FAILED"
        if evidence.error:
            reasons.append(evidence.error)
            return None
        return evidence.stdout + "\n" + evidence.stderr

    nvcc = query("NVCC", settings.bin_dir / "nvcc", "--version")
    if nvcc is not None:
        toolchain.nvcc_version = _version(nvcc, r"\bV(\d+\.\d+\.\d+)\b")
        if toolchain.nvcc_version is None:
            reasons.append("NVCC_VERSION_UNKNOWN")
        elif not toolchain.nvcc_version.startswith("12.8."):
            reasons.append("TOOLKIT_OUTSIDE_BASELINE")

    architectures = query("NVCC_ARCH", settings.bin_dir / "nvcc", "--list-gpu-code")
    if architectures is not None:
        toolchain.supported_architectures = sorted(set(re.findall(r"\bsm_\d+\b", architectures)))
        if settings.target_arch not in toolchain.supported_architectures:
            reasons.append("TARGET_ARCH_UNSUPPORTED")

    sanitizer = query("SANITIZER", settings.bin_dir / "compute-sanitizer", "--version")
    if sanitizer is not None:
        toolchain.sanitizer_version = _version(sanitizer, r"\bversion\s+(\d+(?:\.\d+){2,3})\b")
        if toolchain.sanitizer_version is None:
            reasons.append("SANITIZER_VERSION_UNKNOWN")
        elif not toolchain.sanitizer_version.startswith("2025.1."):
            reasons.append("SANITIZER_OUTSIDE_BASELINE")

    compiler = query("HOST_COMPILER", settings.host_compiler, "-dumpfullversion", "-dumpversion")
    if compiler is not None:
        toolchain.host_compiler_version = _version(compiler.strip(), r"^(\d+\.\d+(?:\.\d+)?)$")
        if toolchain.host_compiler_version is None:
            reasons.append("HOST_COMPILER_VERSION_UNKNOWN")
        elif not 6 <= int(toolchain.host_compiler_version.split(".")[0]) <= 14:
            reasons.append("HOST_COMPILER_UNSUPPORTED")

    gpu_query = query(
        "GPU_QUERY",
        settings.nvidia_smi,
        "--query-gpu=name,driver_version,compute_cap",
        "--format=csv,noheader",
    )
    if gpu_query is not None:
        rows = list(csv.reader(gpu_query.strip().splitlines(), skipinitialspace=True))
        for row in rows:
            if (
                len(row) != 3
                or not row[0].strip()
                or not re.fullmatch(r"\d+\.\d+\.\d+", row[1].strip())
                or not re.fullmatch(r"\d+\.\d+", row[2].strip())
            ):
                reasons.append("GPU_QUERY_INVALID")
                continue
            toolchain.gpus.append(
                GPUInfo(
                    name=row[0].strip(),
                    driver_version=row[1].strip(),
                    compute_capability=row[2].strip(),
                )
            )
        if not rows:
            reasons.append("GPU_QUERY_INVALID")
        matching = [
            gpu
            for gpu in toolchain.gpus
            if "sm_" + gpu.compute_capability.replace(".", "") == settings.target_arch
        ]
        if not matching:
            reasons.append("TARGET_GPU_UNAVAILABLE")
        # Conservative project baseline, not CUDA's lower minor-compatibility floor.
        elif any(
            tuple(map(int, gpu.driver_version.split("."))) < (570, 124, 6) for gpu in matching
        ):
            reasons.append("DRIVER_BELOW_BASELINE")

    if toolchain.platform != "Linux" or toolchain.machine != "x86_64":
        reasons.append("PLATFORM_OUTSIDE_BASELINE")
    if tuple(map(int, toolchain.python_version.split(".")[:2])) not in {(3, 11), (3, 12)}:
        reasons.append("PYTHON_OUTSIDE_BASELINE")
    return EnvironmentReport(
        ready=not reasons,
        reason_codes=list(dict.fromkeys(reasons)),
        toolchain=toolchain,
        probes=probes,
    )
