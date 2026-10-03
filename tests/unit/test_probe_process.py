import subprocess
import sys

import pytest

from gpu_agent.environment import run_probe


def test_probe_captures_real_exit_code_and_both_streams():
    result = run_probe(
        [
            sys.executable,
            "-I",
            "-c",
            "import sys; print('version 1'); print('diagnostic', file=sys.stderr); sys.exit(7)",
        ],
        5,
    )
    assert result.returncode == 7
    assert result.stdout == "version 1\n"
    assert result.stderr == "diagnostic\n"


def test_probe_timeout_is_not_a_successful_empty_response():
    with pytest.raises(subprocess.TimeoutExpired):
        run_probe([sys.executable, "-I", "-c", "import time; time.sleep(3)"], 0.05)


def test_module_entrypoint_works_in_isolated_python():
    result = subprocess.run(
        [sys.executable, "-I", "-m", "gpu_agent", "--help"],
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert result.returncode == 0, result.stderr
    assert "env" in result.stdout
