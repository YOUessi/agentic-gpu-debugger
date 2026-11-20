"""Generate self-authored HardSet V2 CUDA sources and pinned public hashes.

The generator contains only project-owned, reviewed CUDA/C++ templates.  No
external benchmark data, arbitrary shell command, model output or reference
implementation is ever executed. Re-run with --verify to confirm byte parity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

HEADER = """#include "vector_api.h"
#include <cuda_runtime.h>
#include <cmath>
#include <cstddef>

namespace {
"""
HOST = """
int run_vector_add(const float* a, const float* b, float* out, std::size_t n) {
    if (!a || !b || !out || n == 0 || n > 65536)
        return static_cast<int>(cudaErrorInvalidValue);
    float *device_a = nullptr, *device_b = nullptr, *device_out = nullptr, *scratch = nullptr;
    cudaError_t first_error = cudaSuccess;
    const auto check = [&first_error](cudaError_t status) {
        if (first_error == cudaSuccess && status != cudaSuccess) first_error = status;
        return status == cudaSuccess;
    };
    const std::size_t bytes = n * sizeof(float);
    const unsigned int blocks = static_cast<unsigned int>((n + 31) / 32);
    do {
        if (!check(cudaMalloc(reinterpret_cast<void**>(&device_a), bytes))) break;
        if (!check(cudaMalloc(reinterpret_cast<void**>(&device_b), bytes))) break;
        if (!check(cudaMalloc(reinterpret_cast<void**>(&device_out), bytes))) break;
        __ALLOC__
        if (!check(cudaMemcpy(device_a, a, bytes, cudaMemcpyHostToDevice))) break;
        if (!check(cudaMemcpy(device_b, b, bytes, cudaMemcpyHostToDevice))) break;
        if (!check(cudaMemset(device_out, 0, bytes))) break;
        __LAUNCH__
        if (!check(cudaGetLastError())) break;
        if (!check(cudaDeviceSynchronize())) break;
        if (!check(cudaMemcpy(out, device_out, bytes, cudaMemcpyDeviceToHost))) break;
    } while (false);
    float* allocated[] = {scratch, device_out, device_b, device_a};
    for (float* ptr : allocated) if (ptr) check(cudaFree(ptr));
    return static_cast<int>(first_error);
}
"""
SOFTMAX = """
__global__ void softmax32(const float* a, const float* b, float* out, std::size_t n) {
    const unsigned int lane = threadIdx.x;
    const std::size_t start = static_cast<std::size_t>(blockIdx.x) * 32;
    const std::size_t i = start + lane;
    if (i >= n) return;
    const unsigned int valid = static_cast<unsigned int>((n - start) < 32 ? (n - start) : 32);
    float maximum = -CUDART_INF_F;
    for (unsigned int j = 0; j < valid; ++j) {
        const float x = a[start + j] + b[start + j];
        maximum = fmaxf(maximum, x);
    }
    float denominator = 0.0f;
    for (unsigned int j = 0; j < valid; ++j) {
        const float x = a[start + j] + b[start + j];
        denominator += expf(x - maximum);
    }
    const float own = a[i] + b[i];
    out[i] = expf(own - maximum) / denominator;
}
} // namespace
"""
RMSNORM = """
__global__ void rmsnorm32(const float* a, const float* b, float* out, std::size_t n) {
    const unsigned int lane = threadIdx.x;
    const std::size_t start = static_cast<std::size_t>(blockIdx.x) * 32;
    const std::size_t i = start + lane;
    if (i >= n) return;
    const unsigned int valid = static_cast<unsigned int>((n - start) < 32 ? (n - start) : 32);
    float sum_sq = 0.0f;
    for (unsigned int j = 0; j < valid; ++j) {
        const float x = a[start + j] + b[start + j];
        sum_sq += x * x;
    }
    const float scale = rsqrtf(sum_sq / static_cast<float>(valid) + 1.0e-5f);
    out[i] = (a[i] + b[i]) * scale;
}
} // namespace
"""
GROUP_CENTER = """
__global__ void group_sum(const float* a, const float* b, float* sums, std::size_t n) {
    if (threadIdx.x != 0) return;
    const std::size_t start = static_cast<std::size_t>(blockIdx.x) * 32;
    const unsigned int valid = static_cast<unsigned int>((n - start) < 32 ? (n - start) : 32);
    float total = 0.0f;
    for (unsigned int j = 0; j < valid; ++j) {
        total += a[start + j] + b[start + j];
    }
    sums[blockIdx.x] = total / static_cast<float>(valid);
}
__global__ void group_center(const float* a, const float* b, const float* sums,
                             float* out, std::size_t n) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * 32 + threadIdx.x;
    if (i < n) out[i] = (a[i] + b[i]) - sums[blockIdx.x];
}
} // namespace
"""
CASES = {
    "h001": {
        "oracle_id": "stable-softmax32-v2",
        "difficulty": "numerical-stability",
        "clean_device": SOFTMAX,
        "fault": ("expf(x - maximum)", "expf(x)"),
        "fault2": ("expf(own - maximum)", "expf(own)"),
        "alloc": "",
        "launch": "softmax32<<<blocks, 32>>>(device_a, device_b, device_out, n);",
        "fault_class": "functional_numeric_overflow",
    },
    "h002": {
        "oracle_id": "tail-rmsnorm32-v2",
        "difficulty": "partial-group",
        "clean_device": RMSNORM,
        "fault": ("sum_sq / static_cast<float>(valid)", "sum_sq / 32.0f"),
        "alloc": "",
        "launch": "rmsnorm32<<<blocks, 32>>>(device_a, device_b, device_out, n);",
        "fault_class": "functional_wrong_tail_divisor",
    },
    "h003": {
        "oracle_id": "two-stage-group-center32-v2",
        "difficulty": "multi-kernel-dataflow",
        "clean_device": GROUP_CENTER,
        "fault": ("total += a[start + j] + b[start + j];", "total += a[start + j];"),
        "alloc": (
            "if (!check(cudaMalloc(reinterpret_cast<void**>(&scratch), "
            "blocks * sizeof(float)))) break;"
        ),
        "launch": """group_sum<<<blocks, 32>>>(device_a, device_b, scratch, n);
        if (!check(cudaGetLastError())) break;
        group_center<<<blocks, 32>>>(device_a, device_b, scratch, device_out, n);""",
        "fault_class": "functional_cross_kernel_dependency",
    },
}


def code_for(case: dict[str, str], role: str) -> str:
    device = case["clean_device"]
    if role == "mutant":
        before, after = case["fault"]
        if device.count(before) != 1:
            raise ValueError("mutation pattern not unique")
        device = device.replace(before, after, 1)
        if "fault2" in case:
            before, after = case["fault2"]
            if device.count(before) != 1:
                raise ValueError("second mutation pattern not unique")
            device = device.replace(before, after, 1)
    alloc = case["alloc"]
    launch = case["launch"]
    assembled = HEADER + device + HOST.replace("__ALLOC__", alloc).replace("__LAUNCH__", launch)
    return chr(10).join(line.rstrip() for line in assembled.splitlines()) + chr(10)


def digest(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve() / "benchmarks" / "hardset_v2"
    manifest = {
        "schema_version": "hardset-v2-native-1",
        "status": "NOT_YET_GPU_VALIDATED",
        "cases": [],
    }
    for name, spec in sorted(CASES.items()):
        sha: dict[str, str] = {}
        for role in ("clean", "mutant"):
            path = root / name / role / "kernel.cu"
            code = code_for(spec, role).encode()
            sha[role] = digest(code)
            if args.verify:
                if not path.is_file() or path.is_symlink() or path.read_bytes() != code:
                    raise ValueError(f"fixture differs from pinned generator: {name}/{role}")
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                if path.exists() or path.is_symlink():
                    raise ValueError(f"refuse overwrite: {path}")
                path.write_bytes(code)
        manifest["cases"].append(
            {
                "case_id": name,
                "oracle_id": spec["oracle_id"],
                "difficulty": spec["difficulty"],
                "fault_class": spec["fault_class"],
                "clean_source_sha256": sha["clean"],
                "mutant_source_sha256": sha["mutant"],
                "model_visible_reference": False,
            }
        )
    meta = root / "manifest.json"
    encoded = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    if args.verify:
        if meta.read_bytes() != encoded:
            raise ValueError("hardset manifest no longer matches source template")
    else:
        if meta.exists() or meta.is_symlink():
            raise ValueError("refuse overwrite manifest")
        meta.write_bytes(encoded)
    print(f"hardset-v2 fixture integrity OK, cases={len(CASES)}")


if __name__ == "__main__":
    main()
