"""Keep a detached Linux SDK worker within its controller's lifetime."""

import ctypes
import os
import signal
import sys


def bind_parent_lifetime(expected_parent_pid: int) -> None:
    """Arm the guard before SDK imports or request admission, including startup races.

    Other POSIX platforms retain the existing explicit timeout/cancellation cleanup;
    Linux additionally guarantees worker termination when the controller is killed.
    """
    if sys.platform != "linux":
        return
    prctl = ctypes.CDLL(None, use_errno=True).prctl
    prctl.argtypes = [ctypes.c_int, *([ctypes.c_ulong] * 4)]
    prctl.restype = ctypes.c_int
    if prctl(1, signal.SIGKILL, 0, 0, 0) != 0:  # PR_SET_PDEATHSIG
        raise OSError(ctypes.get_errno(), "cannot bind SDK worker to its controller")
    # The parent may die before prctl; the kernel does not replay that death.
    # Compare the PID captured by the controller, not an already-adopted parent.
    if os.getppid() != expected_parent_pid:
        os.kill(os.getpid(), signal.SIGKILL)
