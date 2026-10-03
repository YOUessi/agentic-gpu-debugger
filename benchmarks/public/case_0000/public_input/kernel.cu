#include "vector_api.h"

#include <cuda_runtime.h>

namespace {
__global__ void vector_add(const float* a, const float* b, float* out, std::size_t n) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = a[i] + b[i];
    }
}
}  // namespace

int run_vector_add(const float* a, const float* b, float* out, std::size_t n) {
    if (a == nullptr || b == nullptr || out == nullptr || n == 0 || n > 65536) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    float* device_a = nullptr;
    float* device_b = nullptr;
    float* device_out = nullptr;
    cudaError_t first_error = cudaSuccess;
    const auto check = [&first_error](cudaError_t status) {
        if (first_error == cudaSuccess && status != cudaSuccess) {
            first_error = status;
        }
        return status == cudaSuccess;
    };
    const std::size_t bytes = n * sizeof(float);
    constexpr unsigned int block_size = 256;
    const auto grid_size = static_cast<unsigned int>((n + block_size - 1) / block_size);
    do {
        if (!check(cudaMalloc(reinterpret_cast<void**>(&device_a), bytes))) break;
        if (!check(cudaMalloc(reinterpret_cast<void**>(&device_b), bytes))) break;
        if (!check(cudaMalloc(reinterpret_cast<void**>(&device_out), bytes))) break;
        if (!check(cudaMemcpy(device_a, a, bytes, cudaMemcpyHostToDevice))) break;
        if (!check(cudaMemcpy(device_b, b, bytes, cudaMemcpyHostToDevice))) break;
        vector_add<<<grid_size, block_size>>>(device_a, device_b, device_out, n);
        if (!check(cudaGetLastError())) break;
        if (!check(cudaDeviceSynchronize())) break;
        if (!check(cudaMemcpy(out, device_out, bytes, cudaMemcpyDeviceToHost))) break;
    } while (false);

    // Attempt every allocated buffer's cleanup even after another call/free
    // fails, preserving the first failure as the function's nonzero result.
    float* allocated[] = {device_out, device_b, device_a};
    for (float* pointer : allocated) {
        if (pointer != nullptr) {
            check(cudaFree(pointer));
        }
    }
    return static_cast<int>(first_error);
}
