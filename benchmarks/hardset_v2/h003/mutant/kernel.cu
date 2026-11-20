#include "vector_api.h"
#include <cuda_runtime.h>
#include <cmath>
#include <cstddef>

namespace {

__global__ void group_sum(const float* a, const float* b, float* sums, std::size_t n) {
    if (threadIdx.x != 0) return;
    const std::size_t start = static_cast<std::size_t>(blockIdx.x) * 32;
    const unsigned int valid = static_cast<unsigned int>((n - start) < 32 ? (n - start) : 32);
    float total = 0.0f;
    for (unsigned int j = 0; j < valid; ++j) {
        total += a[start + j];
    }
    sums[blockIdx.x] = total / static_cast<float>(valid);
}
__global__ void group_center(const float* a, const float* b, const float* sums,
                             float* out, std::size_t n) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * 32 + threadIdx.x;
    if (i < n) out[i] = (a[i] + b[i]) - sums[blockIdx.x];
}
} // namespace

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
        if (!check(cudaMalloc(reinterpret_cast<void**>(&scratch), blocks * sizeof(float)))) break;
        if (!check(cudaMemcpy(device_a, a, bytes, cudaMemcpyHostToDevice))) break;
        if (!check(cudaMemcpy(device_b, b, bytes, cudaMemcpyHostToDevice))) break;
        if (!check(cudaMemset(device_out, 0, bytes))) break;
        group_sum<<<blocks, 32>>>(device_a, device_b, scratch, n);
        if (!check(cudaGetLastError())) break;
        group_center<<<blocks, 32>>>(device_a, device_b, scratch, device_out, n);
        if (!check(cudaGetLastError())) break;
        if (!check(cudaDeviceSynchronize())) break;
        if (!check(cudaMemcpy(out, device_out, bytes, cudaMemcpyDeviceToHost))) break;
    } while (false);
    float* allocated[] = {scratch, device_out, device_b, device_a};
    for (float* ptr : allocated) if (ptr) check(cudaFree(ptr));
    return static_cast<int>(first_error);
}
