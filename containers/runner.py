"""Trusted container entrypoint. This file never runs candidate code on the host.

Only fixed operations are accepted. Files exported in the JSON envelope are read
as bounded regular files; no tar extraction or Docker copy crosses the boundary.
"""

import base64
import json
import os
import resource
import selectors
import signal
import stat
import subprocess
import sys
import time
from pathlib import Path

LOG_LIMIT = 2 * 1024 * 1024
BINARY_LIMIT = 64 * 1024 * 1024


def bounded_file(path: Path, limit: int) -> tuple[bytes, bool]:
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode):
            raise ValueError("not a regular export")
        data = stream.read(limit + 1)
        return data[:limit], len(data) > limit or info.st_size >= limit


def capture(argv: list[str], stdin: bytes) -> tuple[int, bytes, bytes, bool]:
    process = subprocess.Popen(
        argv,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    assert process.stdin and process.stdout and process.stderr
    streams = [process.stdin, process.stdout, process.stderr]
    selector = selectors.DefaultSelector()
    buffers = [bytearray(), bytearray()]
    total = sent = 0
    truncated = False
    for stream, label in [(process.stdout, 0), (process.stderr, 1)]:
        os.set_blocking(stream.fileno(), False)
        selector.register(stream, selectors.EVENT_READ, label)
    if stdin:
        os.set_blocking(process.stdin.fileno(), False)
        selector.register(process.stdin, selectors.EVENT_WRITE, 2)
    else:
        process.stdin.close()
    try:
        while selector.get_map():
            if process.poll() is not None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            for key, _ in selector.select(0.05):
                if key.data == 2:
                    try:
                        sent += os.write(key.fd, stdin[sent : sent + 65536])
                    except BrokenPipeError:
                        sent = len(stdin)
                    if sent == len(stdin):
                        selector.unregister(key.fd)
                        process.stdin.close()
                else:
                    data = os.read(key.fd, 65536)
                    if not data:
                        selector.unregister(key.fd)
                    keep = data[: LOG_LIMIT - total]
                    buffers[key.data].extend(keep)
                    total += len(keep)
                    truncated |= len(keep) != len(data)
        return process.wait(), bytes(buffers[0]), bytes(buffers[1]), truncated
    finally:
        selector.close()
        for stream in streams:
            stream.close()


def isolation_probe(stdin: bytes) -> bytes:
    status = dict(
        line.split(":", 1)
        for line in Path("/proc/self/status").read_text().splitlines()
        if ":" in line
    )

    def readonly(path: str) -> bool:
        try:
            Path(path).write_text("probe")
        except OSError:
            return True
        return False

    canary = json.loads(stdin).get("canary_path", "/outside-canary") if stdin else "/outside-canary"
    return json.dumps(
        {
            "uid": os.getuid(),
            "cap_eff": status["CapEff"].strip(),
            "no_new_privs": status["NoNewPrivs"].strip(),
            "interfaces": sorted(p.name for p in Path("/sys/class/net").iterdir()),
            "root_readonly": readonly("/root-write-probe")
            and bool(os.statvfs("/").f_flag & os.ST_RDONLY),
            "input_readonly": readonly("/input/write-probe")
            and bool(os.statvfs("/input").f_flag & os.ST_RDONLY),
            "canary_visible": Path(canary).exists(),
        }
    ).encode()


def main() -> None:
    operation = sys.argv[1] if len(sys.argv) == 2 else ""
    if operation not in {"build", "run", "memcheck", "isolation", "log_limit", "timeout"}:
        raise ValueError("unsupported typed operation")
    os.chdir("/tmp")
    stdin = sys.stdin.buffer.read(32 * 1024 * 1024 + 1)
    if len(stdin) > 32 * 1024 * 1024:
        raise ValueError("input limit")
    binary = sanitizer = b""
    output = error = b""
    truncated = False
    exit_code = 0
    if operation == "timeout":
        time.sleep(600)
    elif operation == "log_limit":
        exit_code, output, error, truncated = capture(
            [sys.executable, "-I", "-c", "import sys; sys.stdout.write('x' * 4194304)"], b""
        )
    elif operation == "isolation":
        output = isolation_probe(stdin)
    else:
        if operation == "build":
            argv = [
                "/usr/local/cuda/bin/nvcc",
                "-std=c++17",
                "-lineinfo",
                "-arch=sm_89",
                "-ccbin",
                "/usr/bin/g++",
                "/input/kernel.cu",
                "/input/vector_io.cpp",
                "-I",
                "/input",
                "-o",
                "/tmp/vector_add",
            ]
        else:
            argv = ["/input/vector_add"]
            if operation == "memcheck":
                # The log cannot be mixed with program stdout/stderr.
                argv = [
                    "/usr/local/cuda/bin/compute-sanitizer",
                    "--tool",
                    "memcheck",
                    "--error-exitcode",
                    "86",
                    "--log-file",
                    "/tmp/memcheck.log",
                    *argv,
                ]
        # Bound each file on tmpfs, including a sanitizer log emitted by the tool.
        resource.setrlimit(resource.RLIMIT_FSIZE, (BINARY_LIMIT, BINARY_LIMIT))
        exit_code, output, error, truncated = capture(argv, stdin)
        if operation == "build" and exit_code == 0:
            binary, oversized = bounded_file(Path("/tmp/vector_add"), BINARY_LIMIT)
            if oversized or not binary:
                raise ValueError("binary limit")
        if operation == "memcheck":
            try:
                sanitizer, cut = bounded_file(Path("/tmp/memcheck.log"), LOG_LIMIT)
                truncated |= cut
            except FileNotFoundError:
                sanitizer = b""
    envelope = {"exit_code": exit_code, "truncated": truncated}
    for name, value in [
        ("stdout", output),
        ("stderr", error),
        ("sanitizer", sanitizer),
        ("binary", binary),
    ]:
        envelope[name] = base64.b64encode(value).decode("ascii")
    print(json.dumps(envelope), flush=True)


if __name__ == "__main__":
    main()
