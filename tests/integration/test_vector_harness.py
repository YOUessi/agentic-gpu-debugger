"""Exercise the trusted JSON boundary without starting CUDA or requiring a GPU."""

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
HARNESS = ROOT / "benchmarks" / "harness"


@pytest.fixture(scope="module")
def run_harness(tmp_path_factory):
    directory = tmp_path_factory.mktemp("vector-harness")
    stub = directory / "vector_stub.cpp"
    stub.write_text(
        '#include "vector_api.h"\n'
        "#include <cstdlib>\n#include <limits>\n#include <string>\n"
        "int run_vector_add(const float* a, const float* b, float* out, std::size_t n) {\n"
        '  const char* env = std::getenv("VECTOR_TEST_MODE");\n'
        '  const std::string mode = env ? env : "";\n'
        '  if (mode == "error") return 7;\n'
        '  if (mode == "unwritten") return 0;\n'
        "  for (std::size_t i = 0; i < n; ++i) out[i] = a[i] + b[i];\n"
        '  if (mode == "nan") out[n - 1] = std::numeric_limits<float>::quiet_NaN();\n'
        '  if (mode == "inf") out[n - 1] = std::numeric_limits<float>::infinity();\n'
        "  return 0;\n}\n"
    )
    binary = directory / "vector_add"
    compiler = shutil.which("g++")
    assert compiler, "g++ is required for the CPU-only harness integration tests"
    compiled = subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-I",
            str(HARNESS),
            "-I",
            str(HARNESS / "vendor"),
            str(HARNESS / "vector_io.cpp"),
            str(stub),
            "-o",
            str(binary),
        ],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    def run(payload, *, mode=""):
        assert compiled.returncode == 0, compiled.stderr
        env = dict(os.environ, VECTOR_TEST_MODE=mode)
        return subprocess.run(
            [str(binary)], input=payload, capture_output=True, env=env, timeout=10, check=False
        )

    return run


@pytest.mark.parametrize("n", [1, 257, 1025, 65536])
def test_vector_harness_emits_one_float32_result(run_harness, n):
    result = run_harness(json.dumps({"n": n, "a": [1.25] * n, "b": [-0.5] * n}).encode())
    assert result.returncode == 0, result.stderr
    assert result.stderr == b""
    assert result.stdout.endswith(b"\n")
    assert result.stdout.count(b"\n") == 1
    assert json.loads(result.stdout) == {"dtype": "float32", "shape": [n], "values": [0.75] * n}


def test_vector_harness_rounds_inputs_and_outputs_as_float32(run_harness):
    result = run_harness(b'{"n":2,"a":[16777217,-2.5],"b":[0,0.25]}')
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["values"] == [16777216, -2.25]


@pytest.mark.parametrize(
    "payload",
    [
        b"",
        b"null",
        b"[]",
        b"{}",
        b'{"n":1,"a":[0]}',
        b'{"n":1,"a":[0],"b":[0],"extra":0}',
        b'{"n":1,"n":1,"a":[0],"b":[0]}',
        b'{"n":0,"a":[],"b":[]}',
        b'{"n":-1,"a":[0],"b":[0]}',
        b'{"n":65537,"a":[0],"b":[0]}',
        b'{"n":true,"a":[0],"b":[0]}',
        b'{"n":1.0,"a":[0],"b":[0]}',
        b'{"n":18446744073709551615,"a":[0],"b":[0]}',
        b'{"n":1,"a":[],"b":[0]}',
        b'{"n":1,"a":[0],"b":[0,1]}',
        b'{"n":1,"a":0,"b":[0]}',
        b'{"n":1,"a":[true],"b":[0]}',
        b'{"n":1,"a":[0],"b":[false]}',
        b'{"n":1,"a":["0"],"b":[0]}',
        b'{"n":1,"a":[null],"b":[0]}',
        b'{"n":1,"a":[{}],"b":[0]}',
        b'{"n":1,"a":[[0]],"b":[0]}',
        b'{"n":1,"a":[NaN],"b":[0]}',
        b'{"n":1,"a":[Infinity],"b":[0]}',
        b'{"n":1,"a":[1e400],"b":[0]}',
        b'{"n":1,"a":[3.5e38],"b":[0]}',
        b'{"n":1,"a":[0],"b":[-3.5e38]}',
        b'{"n":1,"a":[0],"b":[0]} {}',
        b'{"n":1,"a":[0],"b":[0]} garbage',
        b'{"n":1,"a":[0],"b":[0]}\x00',
        b'{"n":1,"a":[0],"b":[0],}',
        b'{/*comment*/"n":1,"a":[0],"b":[0]}',
    ],
)
def test_vector_harness_rejects_invalid_protocol(run_harness, payload):
    result = run_harness(payload)
    assert result.returncode != 0
    assert result.stdout == b""
    assert result.stderr


def test_vector_harness_accepts_surrounding_json_whitespace(run_harness):
    result = run_harness(b' \t\n{"n":1,"a":[0],"b":[0]}\r\n ')
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["values"] == [0]


@pytest.mark.parametrize("mode", ["error", "nan", "inf", "unwritten"])
def test_vector_harness_rejects_kernel_failure_and_nonfinite_output(run_harness, mode):
    result = run_harness(b'{"n":1,"a":[0],"b":[0]}', mode=mode)
    assert result.returncode != 0
    assert result.stdout == b""
    assert result.stderr


def test_vector_harness_rejects_float32_result_overflow(run_harness):
    result = run_harness(b'{"n":1,"a":[3e38],"b":[3e38]}')
    assert result.returncode != 0
    assert result.stdout == b""
    assert result.stderr


def test_vector_harness_limits_input_bytes_including_whitespace(run_harness):
    result = run_harness(b'{"n":1,"a":[0],"b":[0]}' + b" " * (4 * 1024 * 1024))
    assert result.returncode != 0
    assert result.stdout == b""
    assert result.stderr


def test_vector_harness_rejects_deep_nesting_without_crashing(run_harness):
    result = run_harness(b"[" * 100000 + b"0" + b"]" * 100000)
    assert result.returncode > 0
    assert result.stdout == b""
    assert result.stderr
