#!/usr/bin/env python3
"""Public checker for the sanitized case_0022 workspace.

It validates public functional semantics and generic CUDA runtime/Sanitizer cleanliness.
It does not contain a known CUDA repair or historical fault label.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT / "benchmarks" / "public" / "case_0022" / "public_input"
KERNEL = PUBLIC / "kernel.cu"
TASK = PUBLIC / "task.json"
HARNESS = ROOT / "benchmarks" / "harness"
VENDOR = HARNESS / "vendor"
DEFAULT_SIZES = (1, 31, 32, 33, 127, 128, 129, 256, 257, 1025)


def f32(value: float) -> float:
    return float(struct.unpack("f", struct.pack("f", float(value)))[0])


def make_input(n: int) -> dict[str, object]:
    return {
        "n": n,
        "a": [float(i % 13 - 6) for i in range(n)],
        "b": [float(i % 7 + 1) for i in range(n)],
    }


def encode_input(n: int) -> bytes:
    return (json.dumps(make_input(n), separators=(",", ":")) + "\n").encode()


def reference_segment_scan(a: list[float], b: list[float]) -> list[float]:
    if not a or len(a) != len(b):
        raise ValueError("invalid input")
    a32 = [f32(x) for x in a]
    b32 = [f32(x) for x in b]
    result: list[float] = []
    for start in range(0, len(a32), 128):
        values = [f32(x + y) for x, y in zip(a32[start:start + 128], b32[start:start + 128])]
        offset = 1
        while offset < len(values):
            previous = values
            values = [
                f32(value + (previous[i - offset] if i >= offset else 0.0))
                for i, value in enumerate(previous)
            ]
            offset *= 2
        result.extend(values)
    return result


def parse_output(raw: bytes, n: int) -> list[float]:
    obj = json.loads(raw)
    if not isinstance(obj, dict) or set(obj) != {"dtype", "shape", "values"}:
        raise ValueError("unexpected output schema")
    if obj["dtype"] != "float32" or obj["shape"] != [n]:
        raise ValueError("unexpected dtype or shape")
    values = obj["values"]
    if not isinstance(values, list) or len(values) != n:
        raise ValueError("unexpected values length")
    out: list[float] = []
    for value in values:
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError("output contains non-finite/non-numeric values")
        out.append(float(value))
    return out


def assert_close(actual: list[float], expected: list[float]) -> None:
    if len(actual) != len(expected):
        raise ValueError("shape mismatch")
    for i, (a, e) in enumerate(zip(actual, expected, strict=True)):
        if abs(a - e) > 1e-5 + 1e-5 * abs(e):
            raise ValueError(f"numeric mismatch at index {i}: actual={a}, expected={e}")


def run(argv: list[str], *, stdin: bytes | None = None, timeout: int = 120):
    return subprocess.run(
        argv,
        input=stdin,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=timeout,
    )


def has_gpu() -> bool:
    smi = shutil.which("nvidia-smi")
    if smi is None:
        return False
    try:
        result = run([smi, "-L"], timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0 and bool(result.stdout.strip())


def build(binary: Path) -> tuple[bool, str]:
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        return False, "nvcc not found"
    cmd = [
        nvcc,
        "-std=c++17",
        "-lineinfo",
        "-I", str(HARNESS),
        "-I", str(VENDOR),
        str(KERNEL),
        str(HARNESS / "vector_io.cpp"),
        "-o", str(binary),
    ]
    try:
        result = run(cmd)
    except subprocess.TimeoutExpired:
        return False, "nvcc timed out"
    if result.returncode != 0:
        sys.stderr.buffer.write(result.stdout)
        sys.stderr.buffer.write(result.stderr)
        return False, f"nvcc exited with {result.returncode}"
    return True, ""


def check_one(binary: Path, n: int) -> None:
    data = make_input(n)
    expected = reference_segment_scan(data["a"], data["b"])  # type: ignore[arg-type]
    result = run([str(binary)], stdin=encode_input(n))
    if result.returncode != 0:
        sys.stderr.buffer.write(result.stdout)
        sys.stderr.buffer.write(result.stderr)
        raise RuntimeError(f"program exited with {result.returncode} for n={n}")
    assert_close(parse_output(result.stdout, n), expected)


def check_sanitizers(binary: Path, n: int) -> None:
    sanitizer = shutil.which("compute-sanitizer")
    if sanitizer is None:
        print("Compute Sanitizer: SKIP (compute-sanitizer not found)")
        return
    data = make_input(n)
    expected = reference_segment_scan(data["a"], data["b"])  # type: ignore[arg-type]
    for tool in ("memcheck", "racecheck", "initcheck", "synccheck"):
        cmd = [
            sanitizer,
            f"--tool={tool}",
            "--error-exitcode=99",
            str(binary),
        ]
        try:
            result = run(cmd, stdin=encode_input(n), timeout=180)
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(f"Compute Sanitizer {tool} timed out") from exc
        if result.returncode != 0:
            sys.stderr.buffer.write(result.stdout)
            sys.stderr.buffer.write(result.stderr)
            raise RuntimeError(f"Compute Sanitizer {tool} failed with exit {result.returncode}")
        assert_close(parse_output(result.stdout, n), expected)
        print(f"Compute Sanitizer {tool}: PASS")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", nargs="*", type=int, default=list(DEFAULT_SIZES))
    args = parser.parse_args()

    if not KERNEL.is_file() or not TASK.is_file():
        print("Target task files are missing.", file=sys.stderr)
        return 2

    task = json.loads(TASK.read_text())
    if task.get("algorithm") != "segment-scan-cpu-v1":
        print("Unexpected public task algorithm.", file=sys.stderr)
        return 2

    for n in args.sizes:
        if not 1 <= n <= 65536:
            print(f"Invalid size: {n}", file=sys.stderr)
            return 2
        data = make_input(n)
        expected = reference_segment_scan(data["a"], data["b"])  # type: ignore[arg-type]
        if len(expected) != n or any(not math.isfinite(x) for x in expected):
            print(f"Host reference failed for n={n}", file=sys.stderr)
            return 2
    print("Public specification/reference checks: PASS")

    with tempfile.TemporaryDirectory(prefix="doubao-case0022-") as tmp:
        binary = Path(tmp) / "case0022"
        built, reason = build(binary)
        if not built:
            if reason == "nvcc not found":
                print("CUDA compilation/runtime checks: SKIP (nvcc not found)")
                return 0
            print(f"CUDA compilation: FAIL ({reason})", file=sys.stderr)
            return 1
        print("CUDA compilation: PASS")

        if not has_gpu():
            print("CUDA runtime/Sanitizer checks: SKIP (usable NVIDIA GPU not found)")
            return 0

        try:
            for n in args.sizes:
                check_one(binary, n)
            print("CUDA numeric boundary checks: PASS")
            check_sanitizers(binary, 257)
        except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as exc:
            print(f"Validation failed: {exc}", file=sys.stderr)
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
