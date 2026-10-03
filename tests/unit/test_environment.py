import json
import subprocess
from pathlib import Path

import pytest
from typer.testing import CliRunner

from gpu_agent import environment


@pytest.fixture
def fake_probe():
    from gpu_agent.config import Settings

    def probe(overrides=None, **settings):
        outputs = {
            ("/cuda/bin/nvcc", "--version"): "Cuda compilation tools, release 12.8, V12.8.93",
            ("/cuda/bin/nvcc", "--list-gpu-code"): "sm_80\nsm_89\nsm_90\n",
            ("/cuda/bin/compute-sanitizer", "--version"): "Compute Sanitizer version 2025.1.1",
            ("/usr/bin/g++", "-dumpfullversion", "-dumpversion"): "11.4.0\n",
            (
                "/usr/bin/nvidia-smi",
                "--query-gpu=name,driver_version,compute_cap",
                "--format=csv,noheader",
            ): "NVIDIA GPU, 580.178.04, 8.9\n",
        }
        outputs.update(overrides or {})

        def run(argv, timeout):
            value = outputs[tuple(argv)]
            if isinstance(value, Exception):
                raise value
            if isinstance(value, subprocess.CompletedProcess):
                return value
            return subprocess.CompletedProcess(argv, 0, stdout=value, stderr="")

        config = Settings(
            cuda_root=Path("/cuda"),
            host_compiler=Path("/usr/bin/g++"),
            nvidia_smi=Path("/usr/bin/nvidia-smi"),
            **settings,
        )
        return environment.probe_environment(config, runner=run)

    return probe


def test_matching_toolchain_is_ready_but_not_execution_verified(fake_probe):
    report = fake_probe()
    assert report.ready is True
    assert report.reason_codes == []
    assert report.execution_verified is False
    assert report.toolchain.nvcc_version == "12.8.93"
    assert report.toolchain.gpus[0].compute_capability == "8.9"


def test_old_toolkit_cannot_be_replaced_by_torch_runtime(fake_probe, monkeypatch):
    monkeypatch.setenv("CUDA_VERSION", "12.8")
    report = fake_probe(
        {
            ("/cuda/bin/nvcc", "--version"): "Cuda compilation tools, release 11.5, V11.5.119",
            ("/cuda/bin/nvcc", "--list-gpu-code"): "sm_80\nsm_86\n",
        }
    )
    assert report.ready is False
    assert "TOOLKIT_OUTSIDE_BASELINE" in report.reason_codes
    assert "TARGET_ARCH_UNSUPPORTED" in report.reason_codes


@pytest.mark.parametrize(
    "output,reason",
    [
        (FileNotFoundError(), "NVCC_MISSING"),
        (PermissionError(), "NVCC_UNAVAILABLE"),
        (subprocess.TimeoutExpired("nvcc", 5), "NVCC_TIMEOUT"),
        (subprocess.CompletedProcess([], 1, "", "broken"), "NVCC_FAILED"),
        ("nonsense", "NVCC_VERSION_UNKNOWN"),
    ],
)
def test_unusable_compiler_never_reports_ready(fake_probe, output, reason):
    report = fake_probe({("/cuda/bin/nvcc", "--version"): output})
    assert not report.ready
    assert reason in report.reason_codes
    assert report.toolchain.nvcc_version is None


@pytest.mark.parametrize(
    "tool,args,output,reason",
    [
        ("compute-sanitizer", ("--version",), "", "SANITIZER_VERSION_UNKNOWN"),
        (
            "compute-sanitizer",
            ("--version",),
            "Compute Sanitizer version 2021.3.1",
            "SANITIZER_OUTSIDE_BASELINE",
        ),
        ("nvcc", ("--list-gpu-code",), "not sm_89x", "TARGET_ARCH_UNSUPPORTED"),
    ],
)
def test_bad_sanitizer_or_architecture_blocks_readiness(fake_probe, tool, args, output, reason):
    report = fake_probe({(f"/cuda/bin/{tool}", *args): output})
    assert not report.ready
    assert reason in report.reason_codes


@pytest.mark.parametrize(
    "output,reason",
    [
        ("", "GPU_QUERY_INVALID"),
        ("NVIDIA GPU, N/A, N/A", "GPU_QUERY_INVALID"),
        ("NVIDIA GPU, 580.178.04, 8.6", "TARGET_GPU_UNAVAILABLE"),
        ("NVIDIA GPU, 525.60.13, 8.9", "DRIVER_BELOW_BASELINE"),
    ],
)
def test_missing_or_incompatible_gpu_blocks_readiness(fake_probe, output, reason):
    report = fake_probe(
        {
            (
                "/usr/bin/nvidia-smi",
                "--query-gpu=name,driver_version,compute_cap",
                "--format=csv,noheader",
            ): output,
        }
    )
    assert not report.ready
    assert reason in report.reason_codes


@pytest.mark.parametrize(
    "version,reason",
    [
        ("15.1.0", "HOST_COMPILER_UNSUPPORTED"),
        ("", "HOST_COMPILER_VERSION_UNKNOWN"),
    ],
)
def test_host_compiler_baseline_is_checked(fake_probe, version, reason):
    report = fake_probe({("/usr/bin/g++", "-dumpfullversion", "-dumpversion"): version})
    assert not report.ready
    assert reason in report.reason_codes


def test_relative_toolchain_paths_are_rejected():
    from pydantic import ValidationError

    from gpu_agent.config import Settings

    with pytest.raises(ValidationError):
        Settings(cuda_root=Path("relative"))


def test_cli_json_and_exit_status_follow_report(fake_probe, monkeypatch):
    from gpu_agent import cli

    report = fake_probe()
    monkeypatch.setattr(cli, "probe_environment", lambda settings: report)
    result = CliRunner().invoke(cli.app, ["env", "--json", "--cuda-root", "/cuda"])
    assert result.exit_code == 0
    assert json.loads(result.stdout)["execution_verified"] is False
    report = fake_probe({("/cuda/bin/nvcc", "--version"): FileNotFoundError()})
    result = CliRunner().invoke(cli.app, ["env", "--json"])
    assert result.exit_code == 1
    assert "NVCC_MISSING" in json.loads(result.stdout)["reason_codes"]


def test_cli_rejects_relative_cuda_root():
    from gpu_agent.cli import app

    result = CliRunner().invoke(app, ["env", "--cuda-root", "relative", "--json"])
    assert result.exit_code == 2
    assert "Traceback" not in result.output


def test_unsupported_python_is_not_ready(fake_probe, monkeypatch):
    monkeypatch.setattr(environment.host_platform, "python_version", lambda: "3.10.12")
    # The manifest captures the interpreter on construction, not at module import.
    report = fake_probe()
    assert not report.ready
    assert "PYTHON_OUTSIDE_BASELINE" in report.reason_codes


def test_sanitizer_preserves_full_observed_version(fake_probe):
    report = fake_probe(
        {
            (
                "/cuda/bin/compute-sanitizer",
                "--version",
            ): "NVIDIA (R) Compute Sanitizer\nVersion 2025.1.0.0 (build 35583870)\n",
        }
    )
    assert report.toolchain.sanitizer_version == "2025.1.0.0"
