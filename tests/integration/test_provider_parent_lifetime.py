"""A real detached worker must stop when its owning controller is killed."""

import ctypes
import multiprocessing
import os
import select
import signal
import subprocess
import sys
import time

import pytest

pytestmark = pytest.mark.skipif(
    sys.platform != "linux" or not hasattr(os, "pidfd_open"),
    reason="Linux parent-death signals and pidfds are required",
)


@pytest.fixture
def reap_orphaned_workers():
    """Reap only this test's descendants instead of leaving zombies to PID 1."""
    libc = ctypes.CDLL(None, use_errno=True)
    previous = ctypes.c_int()
    assert libc.prctl(37, ctypes.byref(previous), 0, 0, 0) == 0  # PR_GET_CHILD_SUBREAPER
    assert libc.prctl(36, 1, 0, 0, 0) == 0  # PR_SET_CHILD_SUBREAPER
    try:
        yield
    finally:
        assert libc.prctl(36, previous.value, 0, 0, 0) == 0


def _own_blocking_worker(connection, directory, pause_before_bootstrap):
    from gpu_agent.agent.provider_process import worker_command

    # Keep the real bootstrap and isolated session. Replace only entry into the
    # SDK module with a harmless blocker, so no request or credential is needed.
    code = f"""import os,runpy,signal,time
from pathlib import Path
directory = Path({str(directory)!r})
signal.alarm(30)
def harmless_worker(*args, **kwargs):
    (directory / 'work-started').touch()
    (directory / 'ready').touch()
    while True:
        signal.pause()
runpy.run_module = harmless_worker
if {pause_before_bootstrap!r}:
    (directory / 'ready').touch()
    while not (directory / 'release').exists():
        time.sleep(0.005)
"""
    command = worker_command()
    command[3] = code + command[3]
    worker = subprocess.Popen(
        command,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
        env={"PATH": "/usr/local/bin:/usr/bin:/bin", "LC_ALL": "C"},
    )
    connection.send(worker.pid)
    connection.close()
    while True:
        signal.pause()


def _assert_parent_death_stops_worker(tmp_path, death_signal, *, before_bootstrap=False):
    context = multiprocessing.get_context("fork")
    receiver, sender = context.Pipe(duplex=False)
    owner = context.Process(
        target=_own_blocking_worker,
        args=(sender, tmp_path, before_bootstrap),
    )
    worker_fd = None
    worker_pid = None
    owner.start()
    sender.close()
    try:
        assert receiver.poll(5), "controller did not launch its worker"
        worker_pid = receiver.recv()
        worker_fd = os.pidfd_open(worker_pid)
        deadline = time.monotonic() + 5
        while not (tmp_path / "ready").exists() and time.monotonic() < deadline:
            assert not select.select([worker_fd], [], [], 0.01)[0], "worker exited before ready"
        assert (tmp_path / "ready").exists(), "worker did not reach the controlled boundary"
        assert os.getpgid(worker_pid) == worker_pid  # Reproduce the detached worker boundary.
        assert not select.select([worker_fd], [], [], 0)[0]

        os.kill(owner.pid, death_signal)
        owner.join(5)
        assert owner.exitcode == -death_signal
        if before_bootstrap:
            (tmp_path / "release").touch()

        assert select.select([worker_fd], [], [], 2)[0], (
            "detached SDK worker survived its controller's death"
        )
        if before_bootstrap:
            assert not (tmp_path / "work-started").exists(), "orphan admitted SDK work"
        reaped, status = os.waitpid(worker_pid, 0)
        assert reaped == worker_pid and os.waitstatus_to_exitcode(status) == -signal.SIGKILL
        worker_pid = None
    finally:
        if owner.is_alive():
            owner.kill()
        owner.join(5)
        if worker_fd is not None:
            if not select.select([worker_fd], [], [], 0)[0]:
                signal.pidfd_send_signal(worker_fd, signal.SIGKILL)
            assert select.select([worker_fd], [], [], 5)[0], "owned worker cleanup failed"
            os.close(worker_fd)
        if worker_pid is not None:
            os.waitpid(worker_pid, 0)
        receiver.close()
        owner.close()


@pytest.mark.parametrize("death_signal", [signal.SIGTERM, signal.SIGKILL])
def test_detached_worker_dies_with_controller(tmp_path, reap_orphaned_workers, death_signal):
    _assert_parent_death_stops_worker(tmp_path, death_signal)


def test_worker_refuses_work_if_controller_died_before_guard(tmp_path, reap_orphaned_workers):
    _assert_parent_death_stops_worker(tmp_path, signal.SIGKILL, before_bootstrap=True)
