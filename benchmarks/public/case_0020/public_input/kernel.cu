#include "vector_api.h"
#include <cuda_runtime.h>
#include <cmath>

namespace {
__global__ void reduce_groups(const float* a, const float* b, float* out, std::size_t n) {
    __shared__ float values[128];
    const unsigned int t = threadIdx.x;
    const unsigned int lane = t % 32;
    const unsigned int base = t - lane;
    const std::size_t i = blockIdx.x * blockDim.x + t;
    values[t] = i < n ? a[i + 1] + b[i] : 0.0f;
    __syncthreads();
    if (lane == 0) {
        float total = 0.0f;
        for (unsigned int j = 0; j < 32; ++j) total += values[base + j];
        values[base] = total;
    }
    __syncthreads();
    if (i < n) out[i] = values[base];
}
}

int run_vector_add(const float* a, const float* b, float* out, std::size_t n) {
    if (!a || !b || !out || n == 0 || n > 65536) return static_cast<int>(cudaErrorInvalidValue);
    float *device_a = nullptr, *device_b = nullptr, *device_out = nullptr, *scratch = nullptr;
    cudaError_t first_error = cudaSuccess;
    const auto check = [&first_error](cudaError_t status) {
        if (first_error == cudaSuccess && status != cudaSuccess) first_error = status;
        return status == cudaSuccess;
    };
    const std::size_t bytes = n * sizeof(float);
    const unsigned int blocks = static_cast<unsigned int>((n + 127) / 128);
    do {
        if (!check(cudaMalloc(reinterpret_cast<void**>(&device_a), bytes))) break;
        if (!check(cudaMalloc(reinterpret_cast<void**>(&device_b), bytes))) break;
        if (!check(cudaMalloc(reinterpret_cast<void**>(&device_out), bytes))) break;
        if (!check(cudaMemcpy(device_a, a, bytes, cudaMemcpyHostToDevice))) break;
        if (!check(cudaMemcpy(device_b, b, bytes, cudaMemcpyHostToDevice))) break;
        if (!check(cudaMemset(device_out, 0xff, bytes))) break;
        reduce_groups<<<blocks, 128>>>(device_a, device_b, device_out, n);
        if (!check(cudaGetLastError())) break;
        if (!check(cudaDeviceSynchronize())) break;
        if (!check(cudaMemcpy(out, device_out, bytes, cudaMemcpyDeviceToHost))) break;
    } while (false);
    float* allocated[] = {scratch, device_out, device_b, device_a};
    for (float* pointer : allocated) if (pointer) check(cudaFree(pointer));
    return static_cast<int>(first_error);
}
