"""Bounded IPC to a trusted SDK worker, with an absolute killable deadline."""

import json
import os
import selectors
import signal
import subprocess
import sys
import time
from threading import Event

from gpu_agent.agent.models import ProviderError
from gpu_agent.agent.provider import SDKResult, WorkerRequest

MAX_INPUT = 32 * 1024 * 1024
MAX_OUTPUT = 8 * 1024 * 1024


class ProviderProcessPort:
    def __init__(self, *, cancel: Event | None = None) -> None:
        self.cancel = cancel

    def call(self, request: WorkerRequest) -> SDKResult:
        deadline = time.monotonic() + request.timeout_seconds
        # SecretStr is intentionally excluded from all ordinary serialization.
        data = request.model_dump(mode="json")
        data["api_key"] = request.api_key.get_secret_value()
        payload = json.dumps(data, ensure_ascii=False).encode()
        if len(payload) > MAX_INPUT:
            raise ProviderError("LLM_INVALID_REQUEST")
        if time.monotonic() >= deadline:
            raise ProviderError("LLM_TIMEOUT", state="UNCERTAIN")
        try:
            process = subprocess.Popen(
                [sys.executable, "-I", "-m", "gpu_agent.agent.provider_worker"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True,
                env={"PATH": "/usr/local/bin:/usr/bin:/bin", "LC_ALL": "C"},
            )
        except OSError:
            raise ProviderError("LLM_WORKER_ERROR") from None
        assert process.stdin and process.stdout and process.stderr
        selector = selectors.DefaultSelector()
        output = bytearray()
        stderr_size = 0
        sent = 0
        try:
            for stream in (process.stdin, process.stdout, process.stderr):
                os.set_blocking(stream.fileno(), False)
                selector.register(
                    stream,
                    selectors.EVENT_WRITE if stream is process.stdin else selectors.EVENT_READ,
                )
            while selector.get_map() or process.poll() is None:
                remaining = deadline - time.monotonic()
                if self.cancel is not None and self.cancel.is_set():
                    raise ProviderError("LLM_CANCELLED", state="UNCERTAIN")
                if remaining <= 0:
                    raise ProviderError("LLM_TIMEOUT", state="UNCERTAIN")
                for key, _ in selector.select(min(0.02, remaining)):
                    if key.fileobj is process.stdin:
                        try:
                            sent += os.write(key.fd, payload[sent : sent + 65536])
                        except BrokenPipeError:
                            sent = len(payload)
                        if sent == len(payload):
                            selector.unregister(key.fileobj)
                            process.stdin.close()
                    else:
                        chunk = os.read(key.fd, 65536)
                        if not chunk:
                            selector.unregister(key.fileobj)
                        elif key.fileobj is process.stdout:
                            if len(output) + stderr_size + len(chunk) > MAX_OUTPUT:
                                raise ProviderError("LLM_WORKER_ERROR", state="UNCERTAIN")
                            output.extend(chunk)
                        else:
                            stderr_size += len(chunk)
                            if len(output) + stderr_size > MAX_OUTPUT:
                                raise ProviderError("LLM_WORKER_ERROR", state="UNCERTAIN")
            try:
                process.wait(timeout=max(0.001, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                raise ProviderError("LLM_TIMEOUT", state="UNCERTAIN") from None
            if process.returncode != 0 or stderr_size:
                raise ProviderError("LLM_WORKER_ERROR", state="UNCERTAIN")
            try:
                result = SDKResult.model_validate_json(output)
            except ValueError:
                raise ProviderError("LLM_WORKER_ERROR", state="UNCERTAIN") from None
            if time.monotonic() >= deadline:
                raise ProviderError("LLM_TIMEOUT", state="UNCERTAIN")
            return result
        except (KeyboardInterrupt, SystemExit):
            raise ProviderError("LLM_CANCELLED", state="UNCERTAIN") from None
        except OSError:
            raise ProviderError("LLM_WORKER_ERROR", state="UNCERTAIN") from None
        finally:
            primary_error = sys.exc_info()[0] is not None
            cleanup_failed = False
            # A new session owns this group. Never signal an inherited controller group.
            # Do not use its numeric PGID after poll/wait has reaped the owner.
            try:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            except OSError:
                cleanup_failed = True
            selector.close()
            for stream in (process.stdin, process.stdout, process.stderr):
                try:
                    stream.close()
                except OSError:
                    cleanup_failed = True
            try:
                process.wait(timeout=1)
            except (subprocess.TimeoutExpired, OSError):
                cleanup_failed = True
            if cleanup_failed and not primary_error:
                raise ProviderError("LLM_WORKER_ERROR", state="UNCERTAIN") from None
