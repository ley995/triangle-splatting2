#pragma once

#include <functional>
#include <stdexcept>

#include <cuda.h>
#include <cuda_runtime.h>

namespace triangulation {

inline void cuda_check_fn(cudaError_t err, int line, const char *file) {
    if (err != cudaSuccess) {
        std::string msg = "CUDA call at " + std::string(file) + ":" +
                          std::to_string(line) + " failed: ";
        msg = msg + cudaGetErrorString(err);
        throw std::runtime_error(msg);
    }
}

inline void cuda_check_fn(CUresult err, int line, const char *file) {
    if (err != CUDA_SUCCESS) {
        const char *msg;
        cuGetErrorString(err, &msg);
        throw std::runtime_error(std::string("CUDA call at ") + file + ":" +
                                 std::to_string(line) + " failed: " + msg);
    }
}

#define cuda_check(call) triangulation::cuda_check_fn(call, __LINE__, __FILE__)

inline void global_cuda_init() {
    cuda_check(cuInit(0));
    cuda_check(cudaDeviceSetLimit(cudaLimitMallocHeapSize, 1ul << 29ul));
}

} // namespace triangulation
