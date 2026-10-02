#include "vector_api.h"
#include <cuda_runtime.h>

namespace {
__global__ void stencil2d(const float* a, const float* b, float* out, std::size_t n) {
    const std::size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    const std::size_t i = row * 32 + col;
    if (col >= 32 || i >= n) return;
    float value = a[i];
    value += col > 0 ? a[i - 1] : 0.0f;
    value += col + 1 < 32 && i + 1 < n ? a[i + 1] : 0.0f;
    value += row > 0 ? a[i - 32] : 0.0f;
    value += i + 32 < n ? a[i + 32] : 0.0f;
    out[i] = value + b[i];
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
    const dim3 threads(8, 8);
    const dim3 blocks(4, static_cast<unsigned int>(((n + 31) / 32 + 7) / 8));
    do {
        if (!check(cudaMalloc(reinterpret_cast<void**>(&device_a), bytes))) break;
        if (!check(cudaMalloc(reinterpret_cast<void**>(&device_b), bytes))) break;
        if (!check(cudaMalloc(reinterpret_cast<void**>(&device_out), bytes))) break;
        if (!check(cudaMemcpy(device_a, a, bytes, cudaMemcpyHostToDevice))) break;
        if (!check(cudaMemcpy(device_b, b, bytes, cudaMemcpyHostToDevice))) break;
        if (!check(cudaMemset(device_out, 0xff, bytes))) break;
        stencil2d<<<blocks, threads>>>(device_a, device_b, device_out, n);
        if (!check(cudaGetLastError())) break;
        if (!check(cudaDeviceSynchronize())) break;
        if (!check(cudaMemcpy(out, device_out, bytes, cudaMemcpyDeviceToHost))) break;
    } while (false);
    float* allocated[] = {scratch, device_out, device_b, device_a};
    for (float* pointer : allocated) if (pointer) check(cudaFree(pointer));
    return static_cast<int>(first_error);
}
