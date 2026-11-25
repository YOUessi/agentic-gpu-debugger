"""A real detached worker must stop when its owning controller is killed."""

import ctypes
import multiprocessing
import os
import signal
import subprocess
import sys
import time

import pytest

pytestmark = pytest.mark.skipif(
    sys.platform != "linux",
    reason="Linux parent-death signals and subreaper support are required",
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


def _wait_for_owned_worker(worker_pid, timeout):
    """Bounded reap after the dead owner reparents its worker to this subreaper."""
    deadline = time.monotonic() + timeout
    while True:
        reaped, status = os.waitpid(worker_pid, os.WNOHANG)
        if reaped == worker_pid:
            return status
        if time.monotonic() >= deadline:
            return None
        time.sleep(0.005)


def _assert_parent_death_stops_worker(tmp_path, death_signal, *, before_bootstrap=False):
    context = multiprocessing.get_context("fork")
    receiver, sender = context.Pipe(duplex=False)
    owner = context.Process(
        target=_own_blocking_worker,
        args=(sender, tmp_path, before_bootstrap),
    )
    worker_pid = None
    owner.start()
    sender.close()
    try:
        assert receiver.poll(5), "controller did not launch its worker"
        worker_pid = receiver.recv()
        deadline = time.monotonic() + 5
        while not (tmp_path / "ready").exists() and time.monotonic() < deadline:
            assert owner.is_alive(), "controller exited before worker readiness"
            time.sleep(0.005)
        assert (tmp_path / "ready").exists(), "worker did not reach the controlled boundary"
        assert os.getpgid(worker_pid) == worker_pid  # Reproduce the detached worker boundary.

        os.kill(owner.pid, death_signal)
        owner.join(5)
        assert owner.exitcode == -death_signal
        if before_bootstrap:
            (tmp_path / "release").touch()

        status = _wait_for_owned_worker(worker_pid, 2)
        if status is not None:
            worker_pid = None  # Reaped: never signal this numeric PID again.
        assert status is not None, "detached SDK worker survived its controller's death"
        if before_bootstrap:
            assert not (tmp_path / "work-started").exists(), "orphan admitted SDK work"
        assert os.waitstatus_to_exitcode(status) == -signal.SIGKILL
    finally:
        if owner.is_alive():
            owner.kill()
        owner.join(5)
        assert owner.exitcode is not None, "owned controller cleanup failed"
        if worker_pid is not None:
            # waitpid proves ownership before signaling. An unreaped child keeps
            # its PID reserved, including if it exits between this check and kill.
            status = _wait_for_owned_worker(worker_pid, 0)
            if status is None:
                os.kill(worker_pid, signal.SIGKILL)
                status = _wait_for_owned_worker(worker_pid, 5)
            assert status is not None, "owned worker cleanup failed"
        receiver.close()
        owner.close()


@pytest.mark.parametrize("death_signal", [signal.SIGTERM, signal.SIGKILL])
def test_detached_worker_dies_with_controller(tmp_path, reap_orphaned_workers, death_signal):
    _assert_parent_death_stops_worker(tmp_path, death_signal)


def test_worker_refuses_work_if_controller_died_before_guard(tmp_path, reap_orphaned_workers):
    _assert_parent_death_stops_worker(tmp_path, signal.SIGKILL, before_bootstrap=True)
