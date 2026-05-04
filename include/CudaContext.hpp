#pragma once

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>
#include <mutex>

inline const char* cublas_status_to_string(cublasStatus_t status) {
    switch (status) {
        case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE";
        case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH";
        case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED";
        case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR";
        case CUBLAS_STATUS_NOT_SUPPORTED: return "CUBLAS_STATUS_NOT_SUPPORTED";
        case CUBLAS_STATUS_LICENSE_ERROR: return "CUBLAS_STATUS_LICENSE_ERROR";
        default: return "CUBLAS_STATUS_UNKNOWN";
    }
}

inline void cuda_check(cudaError_t err, const char* file, int line, const char* expr) {
    if (err != cudaSuccess) {
        std::ostringstream oss;
        oss << "CUDA error: " << cudaGetErrorString(err)
            << " at " << file << ":" << line << " (" << expr << ")";
        throw std::runtime_error(oss.str());
    }
}

inline void cublas_check(cublasStatus_t status, const char* file, int line, const char* expr) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::ostringstream oss;
        oss << "cuBLAS error: " << cublas_status_to_string(status)
            << " at " << file << ":" << line << " (" << expr << ")";
        throw std::runtime_error(oss.str());
    }
}

#define CUDA_CHECK(expr) ::cuda_check((expr), __FILE__, __LINE__, #expr)
#define CUBLAS_CHECK(expr) ::cublas_check((expr), __FILE__, __LINE__, #expr)

class CudaContext {
public:
    static cublasHandle_t& cublas();
    static cudaStream_t& stream();
    static void set_stream(cudaStream_t stream);
    static void* malloc_bytes(size_t bytes);
    static void free_bytes(void* ptr, size_t bytes);
    static void clear_pool();

private:
    struct Pool {
        std::unordered_map<size_t, std::vector<void*>> free_blocks;
        std::mutex mutex;
    };

    static CudaContext& instance();
    static Pool& pool();
    CudaContext();
    ~CudaContext();
    CudaContext(const CudaContext&) = delete;
    CudaContext& operator=(const CudaContext&) = delete;

    cublasHandle_t handle_{};
    cudaStream_t stream_{};
};
