import sys
import threading
import time

import pytest


def execute(tmp_path, code, **kwargs):
    from gpu_agent.execution.process import ProcessExecutor

    return ProcessExecutor().execute(
        [sys.executable, "-I", "-c", code],
        cwd=tmp_path,
        timeout_seconds=kwargs.pop("timeout_seconds", 5),
        **kwargs,
    )


def test_large_dual_stream_output_is_drained_but_memory_bounded(tmp_path):
    result = execute(
        tmp_path,
        "import os; [(os.write(1,b'x'*4096), os.write(2,b'y'*4096)) for _ in range(1024)]",
        max_log_bytes=1024,
    )
    assert result.exit_code == 0
    assert len(result.stdout) + len(result.stderr) == 1024
    assert result.truncated
    assert not result.timed_out


def test_stdin_streaming_does_not_deadlock_with_output(tmp_path):
    data = b"z" * (512 * 1024)
    result = execute(
        tmp_path,
        "import os,sys; os.write(1,b'x'*131072); data=sys.stdin.buffer.read(); print(len(data))",
        stdin=data,
    )
    assert result.exit_code == 0
    assert result.stdout.endswith(b"524288\n")


def test_non_utf8_and_nonzero_exit_are_preserved(tmp_path):
    result = execute(tmp_path, "import os; os.write(2,b'\\xff'); raise SystemExit(4)")
    assert result.exit_code == 4
    assert result.stderr == b"\xff"
    assert result.elapsed_ms >= 0


def test_timeout_kills_descendant_before_delayed_side_effect(tmp_path):
    child = "import time,pathlib; time.sleep(1); pathlib.Path('escaped').touch()"
    result = execute(
        tmp_path,
        f"import subprocess,sys,time; subprocess.Popen([sys.executable, "
        f"'-I','-c',{child!r}]); time.sleep(3)",
        timeout_seconds=0.2,
    )
    assert result.timed_out
    assert result.exit_code != 0
    time.sleep(1.1)
    assert not (tmp_path / "escaped").exists()


def test_parent_exit_does_not_leave_background_child(tmp_path):
    child = "import time,pathlib; time.sleep(1); pathlib.Path('escaped').touch()"
    result = execute(
        tmp_path,
        f"import subprocess,sys; subprocess.Popen([sys.executable, "
        f"'-I','-c',{child!r}], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)",
    )
    assert result.exit_code == 0
    time.sleep(1.1)
    assert not (tmp_path / "escaped").exists()


def test_parent_exit_cleans_child_that_inherits_output_pipes(tmp_path):
    child = "import time,pathlib; time.sleep(.4); pathlib.Path('escaped').touch()"
    result = execute(
        tmp_path, f"import subprocess,sys; subprocess.Popen([sys.executable, '-I','-c',{child!r}])"
    )
    assert result.exit_code == 0
    assert not result.timed_out
    time.sleep(0.5)
    assert not (tmp_path / "escaped").exists()


def test_cancellation_cleans_up_and_is_not_timeout(tmp_path):
    event = threading.Event()
    timer = threading.Timer(0.1, event.set)
    timer.start()
    try:
        result = execute(tmp_path, "import time; time.sleep(3)", cancel=event)
    finally:
        timer.join()
    assert result.cancelled
    assert not result.timed_out
    assert result.exit_code != 0


def test_missing_executable_returns_typed_error(tmp_path):
    from gpu_agent.execution.process import ProcessExecutor

    result = ProcessExecutor().execute(["/does/not/exist"], cwd=tmp_path, timeout_seconds=1)
    assert result.exit_code is None
    assert result.tool_error == "EXECUTABLE_UNAVAILABLE"


@pytest.mark.parametrize("timeout", [0, -1, float("nan"), float("inf")])
def test_invalid_timeout_cannot_disable_deadline(tmp_path, timeout):
    from gpu_agent.execution.process import ProcessExecutor

    with pytest.raises(ValueError):
        ProcessExecutor().execute(["/does/not/exist"], cwd=tmp_path, timeout_seconds=timeout)
