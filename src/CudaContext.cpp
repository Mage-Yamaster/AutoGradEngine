#include "CudaContext.hpp"

CudaContext::CudaContext() {
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
    CUBLAS_CHECK(cublasCreate(&handle_));
    CUBLAS_CHECK(cublasSetStream(handle_, stream_));
}

CudaContext::~CudaContext() {
    CUBLAS_CHECK(cublasDestroy(handle_));
    CUDA_CHECK(cudaStreamDestroy(stream_));
}

cublasHandle_t& CudaContext::cublas() {
    return instance().handle_;
}

cudaStream_t& CudaContext::stream() {
    return instance().stream_;
}

void CudaContext::set_stream(cudaStream_t stream) {
    CudaContext& ctx = instance();
    ctx.stream_ = stream;
    CUBLAS_CHECK(cublasSetStream(ctx.handle_, stream));
}

CudaContext& CudaContext::instance() {
    static CudaContext ctx;
    return ctx;
}

CudaContext::Pool& CudaContext::pool() {
    static Pool p;
    return p;
}

void* CudaContext::malloc_bytes(size_t bytes) {
    if (bytes == 0) {
        return nullptr;
    }
    Pool& p = pool();
    std::lock_guard<std::mutex> lock(p.mutex);
    auto& bucket = p.free_blocks[bytes];
    if (!bucket.empty()) {
        void* ptr = bucket.back();
        bucket.pop_back();
        return ptr;
    }
    void* ptr = nullptr;
    CUDA_CHECK(cudaMalloc(&ptr, bytes));
    return ptr;
}

void CudaContext::free_bytes(void* ptr, size_t bytes) {
    if (!ptr || bytes == 0) {
        return;
    }
    Pool& p = pool();
    std::lock_guard<std::mutex> lock(p.mutex);
    p.free_blocks[bytes].push_back(ptr);
}

void CudaContext::clear_pool() {
    Pool& p = pool();
    std::lock_guard<std::mutex> lock(p.mutex);
    for (auto& kv : p.free_blocks) {
        for (void* ptr : kv.second) {
            CUDA_CHECK(cudaFree(ptr));
        }
    }
    p.free_blocks.clear();
}
