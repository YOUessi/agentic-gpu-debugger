"""Launcher smoke tests do not install anything or start GPU work."""

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_gpu_batch_launcher_is_shipped():
    assert (ROOT / "gpu_batch.sh").is_file()


def test_gpu_batch_launcher_has_valid_bash_syntax():
    result = subprocess.run(
        ["bash", "-n", str(ROOT / "gpu_batch.sh")], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


def test_gpu_batch_help_needs_no_gpu_or_family():
    result = subprocess.run(
        ["bash", str(ROOT / "gpu_batch.sh"), "help"], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert "setup" in result.stdout and "preflight" in result.stdout


def test_gpu_batch_setup_rejects_git_path_overrides(tmp_path):
    env = os.environ.copy()
    env["GPU_BATCH_PYTHON"] = sys.executable
    env["GIT_DIR"] = str(tmp_path / "redirected-git-dir")
    result = subprocess.run(
        ["bash", str(ROOT / "gpu_batch.sh"), "setup"],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "Refusing setup while GIT_DIR is set" in result.stderr
