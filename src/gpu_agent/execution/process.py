"""Bounded streaming subprocess runner for trusted controller code on Linux."""

import math
import os
import selectors
import signal
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from threading import Event

from gpu_agent.contracts import now


@dataclass(frozen=True)
class ProcessCapture:
    exit_code: int | None
    stdout: bytes
    stderr: bytes
    timed_out: bool
    elapsed_ms: float
    started_at: datetime
    finished_at: datetime
    truncated: bool = False
    cancelled: bool = False
    tool_error: str | None = None


def _kill_group(pid: int) -> None:
    try:
        os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


class ProcessExecutor:
    def execute(
        self,
        argv: list[str],
        cwd: Path,
        timeout_seconds: float,
        max_log_bytes: int = 2 * 1024 * 1024,
        *,
        stdin: bytes = b"",
        env: dict[str, str] | None = None,
        cancel: Event | None = None,
    ) -> ProcessCapture:
        if (
            not argv
            or not math.isfinite(timeout_seconds)
            or timeout_seconds <= 0
            or max_log_bytes <= 0
            or len(stdin) > 32 * 1024 * 1024
        ):
            raise ValueError("invalid process limits")
        start, started = time.monotonic(), now()
        buffers = {"stdout": bytearray(), "stderr": bytearray()}
        timed_out = cancelled = truncated = False
        try:
            process = subprocess.Popen(
                argv,
                cwd=cwd,
                env=env,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True,
                shell=False,
            )
        except OSError:
            return ProcessCapture(
                None,
                b"",
                b"",
                False,
                (time.monotonic() - start) * 1000,
                started,
                now(),
                tool_error="EXECUTABLE_UNAVAILABLE",
            )
        assert (
            process.stdout is not None and process.stderr is not None and process.stdin is not None
        )
        selector = selectors.DefaultSelector()
        streams = [process.stdout, process.stderr, process.stdin]
        sent = retained = 0
        try:
            for stream, label in [(process.stdout, "stdout"), (process.stderr, "stderr")]:
                os.set_blocking(stream.fileno(), False)
                selector.register(stream, selectors.EVENT_READ, label)
            if stdin:
                os.set_blocking(process.stdin.fileno(), False)
                selector.register(process.stdin, selectors.EVENT_WRITE, "stdin")
            else:
                process.stdin.close()
            while selector.get_map() or process.poll() is None:
                # A descendant can keep pipes open after the direct child exits.
                # Kill that group now, then drain already buffered parent output.
                if process.poll() is not None:
                    _kill_group(process.pid)
                if cancel is not None and cancel.is_set():
                    cancelled = True
                    break
                if time.monotonic() - start >= timeout_seconds:
                    timed_out = True
                    break
                for key, _ in selector.select(timeout=min(0.05, timeout_seconds)):
                    fd = key.fd
                    if key.data == "stdin":
                        try:
                            sent += os.write(fd, stdin[sent : sent + 65536])
                        except BrokenPipeError:
                            sent = len(stdin)
                        if sent == len(stdin):
                            selector.unregister(fd)
                            process.stdin.close()
                    else:
                        chunk = os.read(fd, 65536)
                        if not chunk:
                            selector.unregister(fd)
                            continue
                        keep = chunk[: max_log_bytes - retained]
                        buffers[key.data].extend(keep)
                        retained += len(keep)
                        truncated |= len(keep) != len(chunk)
        finally:
            # Also reap background children on normal parent exit; only our process group.
            _kill_group(process.pid)
            process.wait()
            selector.close()
            for stream in streams:
                stream.close()
        return ProcessCapture(
            process.returncode,
            bytes(buffers["stdout"]),
            bytes(buffers["stderr"]),
            timed_out,
            (time.monotonic() - start) * 1000,
            started,
            now(),
            truncated=truncated,
            cancelled=cancelled,
        )
