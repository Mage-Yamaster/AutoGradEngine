#include "CudaKernels.cuh"
#include "CudaContext.hpp"

#include <cuda_runtime.h>

namespace {

constexpr int kThreads = 256;

__host__ __forceinline__ bool is_contiguous_meta(const TensorMetadata& meta) {
    return meta.is_contig != 0;
}

__host__ __forceinline__ bool is_strict_contig_meta(const TensorMetadata& meta) {
    if (meta.is_contig == 0) {
        return false;
    }
    size_t expected = 1;
    for (int d = meta.rank - 1; d >= 0; --d) {
        if (meta.strides[d] != expected) {
            return false;
        }
        expected *= meta.shape[d];
    }
    return true;
}

__host__ __forceinline__ bool shapes_equal_meta(const TensorMetadata& a,
                                                const TensorMetadata& b) {
    if (a.rank != b.rank) {
        return false;
    }
    for (int d = 0; d < a.rank; ++d) {
        if (a.shape[d] != b.shape[d]) {
            return false;
        }
    }
    return true;
}

__host__ __forceinline__ bool is_scalar_meta(const TensorMetadata& meta) {
    if (meta.rank == 0) {
        return true;
    }
    for (int d = 0; d < meta.rank; ++d) {
        if (meta.shape[d] != 1) {
            return false;
        }
    }
    return true;
}

__device__ __forceinline__ void linear_to_indices(size_t linear, const TensorMetadata& meta,
                                                  size_t idx[kMaxDims]) {
    for (int d = meta.rank - 1; d >= 0; --d) {
        size_t dim = meta.shape[d];
        if (dim == 0) {
            idx[d] = 0;
        } else {
            idx[d] = linear % dim;
            linear /= dim;
        }
    }
}

__device__ __forceinline__ size_t offset_from_indices(const TensorMetadata& meta,
                                                      const size_t idx[kMaxDims]) {
    size_t off = 0;
    for (int d = 0; d < meta.rank; ++d) {
        off += idx[d] * meta.strides[d];
    }
    return off;
}

__device__ __forceinline__ size_t offset_from_linear(const TensorMetadata& meta, size_t linear) {
    if (meta.is_contig) {
        return linear;
    }
    size_t idx[kMaxDims]{};
    linear_to_indices(linear, meta, idx);
    return offset_from_indices(meta, idx);
}

__device__ __forceinline__ bool shapes_equal_meta_device(const TensorMetadata& a,
                                                         const TensorMetadata& b) {
    if (a.rank != b.rank) {
        return false;
    }
    for (int d = 0; d < a.rank; ++d) {
        if (a.shape[d] != b.shape[d]) {
            return false;
        }
    }
    return true;
}

__device__ __forceinline__ float warp_reduce_max(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ __forceinline__ float block_reduce_max(float val) {
    __shared__ float shared[32];
    int lane = threadIdx.x & 31;
    int wid = threadIdx.x >> 5;
    val = warp_reduce_max(val);
    if (lane == 0) {
        shared[wid] = val;
    }
    __syncthreads();
    float block_val = (threadIdx.x < (blockDim.x >> 5)) ? shared[lane] : -1e20f;
    block_val = warp_reduce_max(block_val);
    if (threadIdx.x == 0) {
        shared[0] = block_val;
    }
    __syncthreads();
    return shared[0];
}

__device__ __forceinline__ float block_reduce_sum(float val) {
    __shared__ float shared[32];
    int lane = threadIdx.x & 31;
    int wid = threadIdx.x >> 5;
    val = warp_reduce_sum(val);
    if (lane == 0) {
        shared[wid] = val;
    }
    __syncthreads();
    float block_val = (threadIdx.x < (blockDim.x >> 5)) ? shared[lane] : 0.0f;
    block_val = warp_reduce_sum(block_val);
    if (threadIdx.x == 0) {
        shared[0] = block_val;
    }
    __syncthreads();
    return shared[0];
}

__global__ void relu_forward_kernel(const float* in, float* out,
                                    TensorMetadata in_meta,
                                    TensorMetadata out_meta,
                                    size_t numel) {
    size_t linear = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (linear < numel) {
        size_t idx[kMaxDims]{};
        linear_to_indices(linear, out_meta, idx);
        size_t in_off = offset_from_indices(in_meta, idx);
        size_t out_off = offset_from_indices(out_meta, idx);
        float v = in[in_off];
        out[out_off] = v > 0.0f ? v : 0.0f;
    }
}

__global__ void relu_forward_contig_kernel(const float* in, float* out, size_t numel) {
    size_t linear = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (linear < numel) {
        float v = in[linear];
        out[linear] = v > 0.0f ? v : 0.0f;
    }
}

__global__ void relu_backward_kernel(const float* in, const float* out_grad, float* in_grad,
                                     TensorMetadata in_meta,
                                     TensorMetadata out_meta,
                                     size_t numel) {
    size_t linear = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (linear < numel) {
        size_t idx[kMaxDims]{};
        linear_to_indices(linear, out_meta, idx);
        size_t in_off = offset_from_indices(in_meta, idx);
        size_t out_off = offset_from_indices(out_meta, idx);
        float g = in[in_off] > 0.0f ? out_grad[out_off] : 0.0f;
        in_grad[in_off] += g;
    }
}

__global__ void relu_backward_contig_kernel(const float* in, const float* out_grad, float* in_grad,
                                            size_t numel) {
    size_t linear = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (linear < numel) {
        float g = in[linear] > 0.0f ? out_grad[linear] : 0.0f;
        in_grad[linear] += g;
    }
}

__global__ void sigmoid_forward_kernel(const float* in, float* out,
                                       TensorMetadata in_meta,
                                       TensorMetadata out_meta,
                                       size_t numel) {
    size_t linear = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (linear < numel) {
        size_t idx[kMaxDims]{};
        linear_to_indices(linear, out_meta, idx);
        size_t in_off = offset_from_indices(in_meta, idx);
        size_t out_off = offset_from_indices(out_meta, idx);
        float v = in[in_off];
        out[out_off] = 1.0f / (1.0f + expf(-v));
    }
}

__global__ void sigmoid_forward_contig_kernel(const float* in, float* out, size_t numel) {
    size_t linear = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (linear < numel) {
        float v = in[linear];
        out[linear] = 1.0f / (1.0f + expf(-v));
    }
}

__global__ void sigmoid_backward_kernel(const float* out, const float* out_grad, float* in_grad,
                                        TensorMetadata out_meta,
                                        TensorMetadata in_meta,
                                        size_t numel) {
    size_t linear = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (linear < numel) {
        size_t idx[kMaxDims]{};
        linear_to_indices(linear, out_meta, idx);
        size_t out_off = offset_from_indices(out_meta, idx);
        size_t in_off = offset_from_indices(in_meta, idx);
        float y = out[out_off];
        in_grad[in_off] += out_grad[out_off] * y * (1.0f - y);
    }
}

__global__ void sigmoid_backward_contig_kernel(const float* out, const float* out_grad, float* in_grad,
                                               size_t numel) {
    size_t linear = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (linear < numel) {
        float y = out[linear];
        in_grad[linear] += out_grad[linear] * y * (1.0f - y);
    }
}

__global__ void mse_sum_kernel(const float* pred, const float* target, float* out,
                               TensorMetadata pred_meta,
                               TensorMetadata target_meta,
                               size_t numel,
                               float scale) {
    __shared__ float sdata[kThreads];
    size_t linear = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    float val = 0.0f;
    if (linear < numel) {
        size_t p_off = 0;
        size_t t_off = 0;
        if (pred_meta.is_contig && target_meta.is_contig) {
            p_off = linear;
            t_off = linear;
        } else {
            size_t idx[kMaxDims]{};
            linear_to_indices(linear, pred_meta, idx);
            p_off = offset_from_indices(pred_meta, idx);
            t_off = offset_from_indices(target_meta, idx);
        }
        float diff = pred[p_off] - target[t_off];
        val = diff * diff * scale;
    }
    sdata[threadIdx.x] = val;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        atomicAdd(out, sdata[0]);
    }
}

__global__ void mse_backward_kernel(const float* pred, const float* target, const float* out_grad,
                                    float* pred_grad, float* target_grad,
                                    TensorMetadata pred_meta,
                                    TensorMetadata target_meta,
                                    size_t numel) {
    size_t linear = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (linear < numel) {
        size_t p_off = 0;
        size_t t_off = 0;
        if (pred_meta.is_contig && target_meta.is_contig) {
            p_off = linear;
            t_off = linear;
        } else {
            size_t idx[kMaxDims]{};
            linear_to_indices(linear, pred_meta, idx);
            p_off = offset_from_indices(pred_meta, idx);
            t_off = offset_from_indices(target_meta, idx);
        }
        float diff = pred[p_off] - target[t_off];
        float g = out_grad[0] * (2.0f / static_cast<float>(numel)) * diff;
        if (pred_grad) {
            pred_grad[p_off] += g;
        }
        if (target_grad) {
            target_grad[t_off] -= g;
        }
    }
}

// Deprecated: use cuBLAS in Operators.cpp for matmul forward/backward.
__global__ void matmul_forward_kernel(const float* a, const float* b, float* out,
                                      size_t m, size_t k, size_t n,
                                      size_t a_s0, size_t a_s1,
                                      size_t b_s0, size_t b_s1,
                                      size_t out_s0, size_t out_s1) {
    size_t row = static_cast<size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    size_t col = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row < m && col < n) {
        float sum = 0.0f;
        for (size_t p = 0; p < k; ++p) {
            size_t a_off = row * a_s0 + p * a_s1;
            size_t b_off = p * b_s0 + col * b_s1;
            sum += a[a_off] * b[b_off];
        }
        size_t out_off = row * out_s0 + col * out_s1;
        out[out_off] = sum;
    }
}

__global__ void matmul_backward_a_kernel(const float* out_grad, const float* b, float* a_grad,
                                         size_t m, size_t k, size_t n,
                                         size_t out_s0, size_t out_s1,
                                         size_t b_s0, size_t b_s1,
                                         size_t a_s0, size_t a_s1) {
    size_t row = static_cast<size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    size_t col = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row < m && col < k) {
        float sum = 0.0f;
        for (size_t j = 0; j < n; ++j) {
            size_t out_off = row * out_s0 + j * out_s1;
            size_t b_off = col * b_s0 + j * b_s1;
            sum += out_grad[out_off] * b[b_off];
        }
        size_t a_off = row * a_s0 + col * a_s1;
        a_grad[a_off] += sum;
    }
}

__global__ void matmul_backward_b_kernel(const float* a, const float* out_grad, float* b_grad,
                                         size_t m, size_t k, size_t n,
                                         size_t a_s0, size_t a_s1,
                                         size_t out_s0, size_t out_s1,
                                         size_t b_s0, size_t b_s1) {
    size_t row = static_cast<size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    size_t col = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row < k && col < n) {
        float sum = 0.0f;
        for (size_t i = 0; i < m; ++i) {
            size_t a_off = i * a_s0 + row * a_s1;
            size_t out_off = i * out_s0 + col * out_s1;
            sum += a[a_off] * out_grad[out_off];
        }
        size_t b_off = row * b_s0 + col * b_s1;
        b_grad[b_off] += sum;
    }
}

__global__ void add_broadcast_nd_forward_kernel(const float* a, const float* b, float* out,
                                                TensorMetadata a_meta,
                                                TensorMetadata b_meta,
                                                TensorMetadata out_meta,
                                                size_t out_numel) {
    size_t linear = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (linear < out_numel) {
        size_t idx[kMaxDims]{};
        linear_to_indices(linear, out_meta, idx);
        size_t a_off = 0;
        size_t b_off = 0;
        size_t out_off = 0;
        for (int d = 0; d < out_meta.rank; ++d) {
            size_t out_idx = idx[d];
            size_t a_idx = (a_meta.shape[d] == 1) ? 0 : out_idx;
            size_t b_idx = (b_meta.shape[d] == 1) ? 0 : out_idx;
            a_off += a_idx * a_meta.strides[d];
            b_off += b_idx * b_meta.strides[d];
            out_off += out_idx * out_meta.strides[d];
        }
        out[out_off] = a[a_off] + b[b_off];
    }
}

__global__ void add_nobroadcast_contig_forward_kernel(const float* a, const float* b, float* out,
                                                      size_t out_numel) {
    size_t linear = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (linear < out_numel) {
        out[linear] = a[linear] + b[linear];
    }
}

__global__ void add_broadcast_nd_backward_kernel(const float* out_grad, float* a_grad, float* b_grad,
                                                 TensorMetadata a_meta,
                                                 TensorMetadata b_meta,
                                                 TensorMetadata out_meta,
                                                 size_t out_numel) {
    size_t linear = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (linear < out_numel) {
        size_t idx[kMaxDims]{};
        linear_to_indices(linear, out_meta, idx);
        size_t a_off = 0;
        size_t b_off = 0;
        size_t out_off = 0;
        for (int d = 0; d < out_meta.rank; ++d) {
            size_t out_idx = idx[d];
            size_t a_idx = (a_meta.shape[d] == 1) ? 0 : out_idx;
            size_t b_idx = (b_meta.shape[d] == 1) ? 0 : out_idx;
            a_off += a_idx * a_meta.strides[d];
            b_off += b_idx * b_meta.strides[d];
            out_off += out_idx * out_meta.strides[d];
        }
        float og = out_grad[out_off];
        if (a_grad) {
            atomicAdd(&a_grad[a_off], og);
        }
        if (b_grad) {
            atomicAdd(&b_grad[b_off], og);
        }
    }
}

__global__ void reduce_sum_kernel(const float* in, float* out,
                                  TensorMetadata in_meta,
                                  size_t numel) {
    __shared__ float sdata[kThreads];
    size_t linear = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    float val = 0.0f;
    if (linear < numel) {
        size_t off = offset_from_linear(in_meta, linear);
        val = in[off];
    }
    sdata[threadIdx.x] = val;
    __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        atomicAdd(out, sdata[0]);
    }
}

__global__ void reduce_sum_mul_kernel(const float* out_grad, const float* other, float* out,
                                      TensorMetadata out_meta,
                                      TensorMetadata other_meta,
                                      size_t numel) {
    __shared__ float sdata[kThreads];
    size_t linear = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    float val = 0.0f;
    if (linear < numel) {
        size_t out_off = offset_from_linear(out_meta, linear);
        size_t other_off = offset_from_linear(other_meta, linear);
        val = out_grad[out_off] * other[other_off];
    }
    sdata[threadIdx.x] = val;
    __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        atomicAdd(out, sdata[0]);
    }
}

__global__ void add_backward_nobroadcast_kernel(const float* out_grad, float* in_grad,
                                                TensorMetadata out_meta,
                                                TensorMetadata in_meta,
                                                size_t numel) {
    size_t linear = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (linear < numel) {
        size_t idx[kMaxDims]{};
        linear_to_indices(linear, out_meta, idx);
        size_t out_off = offset_from_indices(out_meta, idx);
        size_t in_off = offset_from_indices(in_meta, idx);
        in_grad[in_off] += out_grad[out_off];
    }
}

__global__ void add_backward_nobroadcast_contig_kernel(const float* out_grad, float* in_grad,
                                                       size_t numel) {
    size_t linear = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (linear < numel) {
        in_grad[linear] += out_grad[linear];
    }
}

__global__ void mul_broadcast_nd_forward_kernel(const float* a, const float* b, float* out,
                                                TensorMetadata a_meta,
                                                TensorMetadata b_meta,
                                                TensorMetadata out_meta,
                                                size_t out_numel) {
    size_t linear = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (linear < out_numel) {
        size_t idx[kMaxDims]{};
        linear_to_indices(linear, out_meta, idx);
        size_t a_off = 0;
        size_t b_off = 0;
        size_t out_off = 0;
        for (int d = 0; d < out_meta.rank; ++d) {
            size_t out_idx = idx[d];
            size_t a_idx = (a_meta.shape[d] == 1) ? 0 : out_idx;
            size_t b_idx = (b_meta.shape[d] == 1) ? 0 : out_idx;
            a_off += a_idx * a_meta.strides[d];
            b_off += b_idx * b_meta.strides[d];
            out_off += out_idx * out_meta.strides[d];
        }
        out[out_off] = a[a_off] * b[b_off];
    }
}

__global__ void mul_nobroadcast_contig_forward_kernel(const float* a, const float* b, float* out,
                                                      size_t out_numel) {
    size_t linear = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (linear < out_numel) {
        out[linear] = a[linear] * b[linear];
    }
}

__global__ void mul_backward_nobroadcast_kernel(const float* out_grad, const float* a, const float* b,
                                                float* a_grad, float* b_grad,
                                                TensorMetadata out_meta,
                                                TensorMetadata a_meta,
                                                TensorMetadata b_meta,
                                                size_t numel) {
    size_t linear = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (linear < numel) {
        size_t idx[kMaxDims]{};
        linear_to_indices(linear, out_meta, idx);
        size_t out_off = offset_from_indices(out_meta, idx);
        size_t a_off = offset_from_indices(a_meta, idx);
        size_t b_off = offset_from_indices(b_meta, idx);
        float og = out_grad[out_off];
        if (a_grad) {
            a_grad[a_off] += og * b[b_off];
        }
        if (b_grad) {
            b_grad[b_off] += og * a[a_off];
        }
    }
}

__global__ void mul_backward_nobroadcast_contig_kernel(const float* out_grad, const float* a, const float* b,
                                                       float* a_grad, float* b_grad,
                                                       size_t numel) {
    size_t linear = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (linear < numel) {
        float og = out_grad[linear];
        if (a_grad) {
            a_grad[linear] += og * b[linear];
        }
        if (b_grad) {
            b_grad[linear] += og * a[linear];
        }
    }
}

__global__ void mul_broadcast_nd_backward_kernel(const float* out_grad, const float* a, const float* b,
                                                 float* a_grad, float* b_grad,
                                                 TensorMetadata a_meta,
                                                 TensorMetadata b_meta,
                                                 TensorMetadata out_meta,
                                                 size_t out_numel) {
    size_t linear = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (linear < out_numel) {
        size_t idx[kMaxDims]{};
        linear_to_indices(linear, out_meta, idx);
        size_t a_off = 0;
        size_t b_off = 0;
        size_t out_off = 0;
        for (int d = 0; d < out_meta.rank; ++d) {
            size_t out_idx = idx[d];
            size_t a_idx = (a_meta.shape[d] == 1) ? 0 : out_idx;
            size_t b_idx = (b_meta.shape[d] == 1) ? 0 : out_idx;
            a_off += a_idx * a_meta.strides[d];
            b_off += b_idx * b_meta.strides[d];
            out_off += out_idx * out_meta.strides[d];
        }
        float og = out_grad[out_off];
        float av = a[a_off];
        float bv = b[b_off];
        if (a_grad) {
            atomicAdd(&a_grad[a_off], og * bv);
        }
        if (b_grad) {
            atomicAdd(&b_grad[b_off], og * av);
        }
    }
}

__global__ void add_inplace_nd_kernel(const float* src, float* dst,
                                      TensorMetadata meta, size_t numel) {
    size_t linear = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (linear < numel) {
        size_t off = offset_from_linear(meta, linear);
        dst[off] += src[linear];
    }
}

__global__ void add_contig_to_strided_kernel(const float* src_contig, float* dst_strided,
                                             TensorMetadata dst_meta, size_t numel) {
    size_t linear = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (linear < numel) {
        size_t dst_off = offset_from_linear(dst_meta, linear);
        dst_strided[dst_off] += src_contig[linear];
    }
}

__global__ void copy_contig_to_strided_kernel(const float* src_contig, float* dst_strided,
                                              TensorMetadata dst_meta, size_t numel) {
    size_t linear = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (linear < numel) {
        size_t dst_off = offset_from_linear(dst_meta, linear);
        dst_strided[dst_off] = src_contig[linear];
    }
}

__global__ void copy_strided_to_contig_kernel(const float* src_strided, float* dst_contig,
                                              TensorMetadata src_meta, size_t numel) {
    size_t linear = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (linear < numel) {
        size_t src_off = offset_from_linear(src_meta, linear);
        dst_contig[linear] = src_strided[src_off];
    }
}

__global__ void fill_kernel(float* data, float value, size_t numel) {
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < numel) {
        data[idx] = value;
    }
}

__global__ void add_scalar_kernel(float* data, float value, size_t numel) {
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < numel) {
        data[idx] += value;
    }
}

__global__ void add_inplace_broadcast_kernel(const float* b, float* a,
                                             TensorMetadata b_meta,
                                             TensorMetadata a_meta,
                                             size_t out_numel) {
    size_t linear = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (linear < out_numel) {
        size_t idx[kMaxDims]{};
        linear_to_indices(linear, a_meta, idx);
        size_t a_off = offset_from_indices(a_meta, idx);
        size_t b_off = 0;
        for (int d = 0; d < a_meta.rank; ++d) {
            size_t out_idx = idx[d];
            size_t b_idx = (b_meta.shape[d] == 1) ? 0 : out_idx;
            b_off += b_idx * b_meta.strides[d];
        }
        a[a_off] += b[b_off];
    }
}

__global__ void mul_inplace_broadcast_kernel(const float* b, float* a,
                                             TensorMetadata b_meta,
                                             TensorMetadata a_meta,
                                             size_t out_numel) {
    size_t linear = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (linear < out_numel) {
        size_t idx[kMaxDims]{};
        linear_to_indices(linear, a_meta, idx);
        size_t a_off = offset_from_indices(a_meta, idx);
        size_t b_off = 0;
        for (int d = 0; d < a_meta.rank; ++d) {
            size_t out_idx = idx[d];
            size_t b_idx = (b_meta.shape[d] == 1) ? 0 : out_idx;
            b_off += b_idx * b_meta.strides[d];
        }
        a[a_off] *= b[b_off];
    }
}

__global__ void relu_inplace_kernel(float* data, TensorMetadata meta, size_t numel) {
    size_t linear = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (linear < numel) {
        size_t idx[kMaxDims]{};
        linear_to_indices(linear, meta, idx);
        size_t off = offset_from_indices(meta, idx);
        float v = data[off];
        data[off] = v > 0.0f ? v : 0.0f;
    }
}

__global__ void relu_inplace_contig_kernel(float* data, size_t numel) {
    size_t linear = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (linear < numel) {
        float v = data[linear];
        data[linear] = v > 0.0f ? v : 0.0f;
    }
}

__device__ __forceinline__ void outer_index_to_indices(size_t outer,
                                                       const TensorMetadata& meta,
                                                       size_t idx[kMaxDims]) {
    for (int d = meta.rank - 2; d >= 0; --d) {
        size_t dim = meta.shape[d];
        if (dim == 0) {
            idx[d] = 0;
        } else {
            idx[d] = outer % dim;
            outer /= dim;
        }
    }
}

__global__ void softmax_forward_kernel(const float* in, float* out,
                                       TensorMetadata in_meta,
                                       TensorMetadata out_meta,
                                       size_t outer, size_t cols) {
    size_t row = static_cast<size_t>(blockIdx.x);
    if (row >= outer) {
        return;
    }
    bool contig = in_meta.is_contig && out_meta.is_contig;
    size_t idx[kMaxDims]{};
    if (!contig) {
        outer_index_to_indices(row, in_meta, idx);
    }
    float local_max = -1e20f;
    for (size_t c = threadIdx.x; c < cols; c += blockDim.x) {
        size_t in_off = 0;
        if (contig) {
            in_off = row * cols + c;
        } else {
            idx[in_meta.rank - 1] = c;
            in_off = offset_from_indices(in_meta, idx);
        }
        float v = in[in_off];
        if (v > local_max) {
            local_max = v;
        }
    }
    float row_max = block_reduce_max(local_max);
    float local_sum = 0.0f;
    for (size_t c = threadIdx.x; c < cols; c += blockDim.x) {
        size_t in_off = 0;
        if (contig) {
            in_off = row * cols + c;
        } else {
            idx[in_meta.rank - 1] = c;
            in_off = offset_from_indices(in_meta, idx);
        }
        local_sum += expf(in[in_off] - row_max);
    }
    float row_sum = block_reduce_sum(local_sum);
    row_sum = fmaxf(row_sum, 1e-20f);
    float inv_sum = 1.0f / row_sum;
    for (size_t c = threadIdx.x; c < cols; c += blockDim.x) {
        size_t in_off = 0;
        size_t out_off = 0;
        if (contig) {
            in_off = row * cols + c;
            out_off = row * cols + c;
        } else {
            idx[in_meta.rank - 1] = c;
            in_off = offset_from_indices(in_meta, idx);
            idx[out_meta.rank - 1] = c;
            out_off = offset_from_indices(out_meta, idx);
        }
        out[out_off] = expf(in[in_off] - row_max) * inv_sum;
    }
}

__global__ void softmax_backward_kernel(const float* out, const float* out_grad, float* in_grad,
                                        TensorMetadata out_meta,
                                        TensorMetadata in_meta,
                                        size_t outer, size_t cols) {
    size_t row = static_cast<size_t>(blockIdx.x);
    if (row >= outer) {
        return;
    }
    bool contig = out_meta.is_contig && in_meta.is_contig;
    size_t idx[kMaxDims]{};
    if (!contig) {
        outer_index_to_indices(row, out_meta, idx);
    }
    float local_dot = 0.0f;
    for (size_t c = threadIdx.x; c < cols; c += blockDim.x) {
        size_t out_off = 0;
        if (contig) {
            out_off = row * cols + c;
        } else {
            idx[out_meta.rank - 1] = c;
            out_off = offset_from_indices(out_meta, idx);
        }
        float y = out[out_off];
        float g = out_grad[out_off];
        local_dot += y * g;
    }
    float dot = block_reduce_sum(local_dot);
    for (size_t c = threadIdx.x; c < cols; c += blockDim.x) {
        size_t out_off = 0;
        size_t in_off = 0;
        if (contig) {
            out_off = row * cols + c;
            in_off = row * cols + c;
        } else {
            idx[out_meta.rank - 1] = c;
            out_off = offset_from_indices(out_meta, idx);
            idx[in_meta.rank - 1] = c;
            in_off = offset_from_indices(in_meta, idx);
        }
        float y = out[out_off];
        float g = out_grad[out_off];
        in_grad[in_off] += y * (g - dot);
    }
}

__global__ void cross_entropy_forward_kernel(const float* logits, const float* target, float* out,
                                             TensorMetadata logits_meta,
                                             TensorMetadata target_meta,
                                             size_t outer, size_t cols,
                                             float scale) {
    size_t row = static_cast<size_t>(blockIdx.x);
    if (row >= outer) {
        return;
    }
    bool contig = logits_meta.is_contig;
    size_t idx[kMaxDims]{};
    if (!contig) {
        outer_index_to_indices(row, logits_meta, idx);
    }
    float local_max = -1e20f;
    for (size_t c = threadIdx.x; c < cols; c += blockDim.x) {
        size_t off = 0;
        if (contig) {
            off = row * cols + c;
        } else {
            idx[logits_meta.rank - 1] = c;
            off = offset_from_indices(logits_meta, idx);
        }
        float v = logits[off];
        if (v > local_max) {
            local_max = v;
        }
    }
    float row_max = block_reduce_max(local_max);
    float local_sum = 0.0f;
    for (size_t c = threadIdx.x; c < cols; c += blockDim.x) {
        size_t off = 0;
        if (contig) {
            off = row * cols + c;
        } else {
            idx[logits_meta.rank - 1] = c;
            off = offset_from_indices(logits_meta, idx);
        }
        local_sum += expf(logits[off] - row_max);
    }
    float row_sum = block_reduce_sum(local_sum);
    row_sum = fmaxf(row_sum, 1e-20f);
    size_t tgt_idx[kMaxDims]{};
    size_t t_off = 0;
    if (target_meta.is_contig) {
        t_off = row;
    } else {
        linear_to_indices(row, target_meta, tgt_idx);
        t_off = offset_from_indices(target_meta, tgt_idx);
    }
    int label = static_cast<int>(target[t_off]);
    if (label < 0) {
        label = 0;
    } else if (label >= static_cast<int>(cols)) {
        label = static_cast<int>(cols) - 1;
    }
    size_t logit_off = 0;
    if (contig) {
        logit_off = row * cols + static_cast<size_t>(label);
    } else {
        idx[logits_meta.rank - 1] = static_cast<size_t>(label);
        logit_off = offset_from_indices(logits_meta, idx);
    }
    float logit = logits[logit_off];
    float log_sum_exp = logf(row_sum) + row_max;
    float loss = (log_sum_exp - logit) * scale;
    if (threadIdx.x == 0) {
        atomicAdd(out, loss);
    }
}

__global__ void cross_entropy_backward_kernel(const float* logits, const float* target, const float* out_grad,
                                              float* logits_grad,
                                              TensorMetadata logits_meta,
                                              TensorMetadata target_meta,
                                              size_t outer, size_t cols) {
    size_t row = static_cast<size_t>(blockIdx.x);
    if (row >= outer) {
        return;
    }
    bool contig = logits_meta.is_contig;
    size_t idx[kMaxDims]{};
    if (!contig) {
        outer_index_to_indices(row, logits_meta, idx);
    }
    float local_max = -1e20f;
    for (size_t c = threadIdx.x; c < cols; c += blockDim.x) {
        size_t off = 0;
        if (contig) {
            off = row * cols + c;
        } else {
            idx[logits_meta.rank - 1] = c;
            off = offset_from_indices(logits_meta, idx);
        }
        float v = logits[off];
        if (v > local_max) {
            local_max = v;
        }
    }
    float row_max = block_reduce_max(local_max);
    float local_sum = 0.0f;
    for (size_t c = threadIdx.x; c < cols; c += blockDim.x) {
        size_t off = 0;
        if (contig) {
            off = row * cols + c;
        } else {
            idx[logits_meta.rank - 1] = c;
            off = offset_from_indices(logits_meta, idx);
        }
        local_sum += expf(logits[off] - row_max);
    }
    float row_sum = block_reduce_sum(local_sum);
    row_sum = fmaxf(row_sum, 1e-20f);
    size_t tgt_idx[kMaxDims]{};
    size_t t_off = 0;
    if (target_meta.is_contig) {
        t_off = row;
    } else {
        linear_to_indices(row, target_meta, tgt_idx);
        t_off = offset_from_indices(target_meta, tgt_idx);
    }
    int label = static_cast<int>(target[t_off]);
    if (label < 0) {
        label = 0;
    } else if (label >= static_cast<int>(cols)) {
        label = static_cast<int>(cols) - 1;
    }
    float g = out_grad[0] / static_cast<float>(outer);
    for (size_t c = threadIdx.x; c < cols; c += blockDim.x) {
        size_t off = 0;
        if (contig) {
            off = row * cols + c;
        } else {
            idx[logits_meta.rank - 1] = c;
            off = offset_from_indices(logits_meta, idx);
        }
        float p = expf(logits[off] - row_max) / row_sum;
        float grad = (static_cast<int>(c) == label) ? (p - 1.0f) : p;
        logits_grad[off] += g * grad;
    }
}

__global__ void sgd_step_nd_kernel(float* data, const float* grad,
                                   TensorMetadata data_meta,
                                   TensorMetadata grad_meta,
                                   size_t numel,
                                   float lr) {
    size_t linear = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (linear < numel) {
        size_t data_off = 0;
        size_t grad_off = 0;
        if (data_meta.is_contig && grad_meta.is_contig) {
            data_off = linear;
            grad_off = linear;
        } else {
            size_t idx[kMaxDims]{};
            linear_to_indices(linear, data_meta, idx);
            data_off = offset_from_indices(data_meta, idx);
            grad_off = offset_from_indices(grad_meta, idx);
        }
        data[data_off] -= lr * grad[grad_off];
    }
}

__global__ void adam_step_nd_kernel(float* data, const float* grad, float* m, float* v,
                                    TensorMetadata data_meta,
                                    TensorMetadata grad_meta,
                                    TensorMetadata m_meta,
                                    TensorMetadata v_meta,
                                    size_t numel,
                                    float lr, float beta1, float beta2, float eps,
                                    float bc1, float bc2) {
    size_t linear = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (linear < numel) {
        size_t data_off = 0;
        size_t grad_off = 0;
        size_t m_off = 0;
        size_t v_off = 0;
        if (data_meta.is_contig && grad_meta.is_contig &&
            m_meta.is_contig && v_meta.is_contig) {
            data_off = linear;
            grad_off = linear;
            m_off = linear;
            v_off = linear;
        } else {
            size_t idx[kMaxDims]{};
            linear_to_indices(linear, data_meta, idx);
            data_off = offset_from_indices(data_meta, idx);
            grad_off = offset_from_indices(grad_meta, idx);
            m_off = offset_from_indices(m_meta, idx);
            v_off = offset_from_indices(v_meta, idx);
        }
        float g = grad[grad_off];
        float m_val = beta1 * m[m_off] + (1.0f - beta1) * g;
        float v_val = beta2 * v[v_off] + (1.0f - beta2) * g * g;
        m[m_off] = m_val;
        v[v_off] = v_val;
        float m_hat = m_val / bc1;
        float v_hat = v_val / bc2;
        data[data_off] -= lr * m_hat / (sqrtf(v_hat) + eps);
    }
}

}  // namespace

void launch_relu_forward(const float* in, float* out,
                         const TensorMetadata& in_meta,
                         const TensorMetadata& out_meta,
                         size_t numel) {
    int blocks = static_cast<int>((numel + kThreads - 1) / kThreads);
    cudaStream_t stream = CudaContext::stream();
    bool fast = is_strict_contig_meta(in_meta) &&
                is_strict_contig_meta(out_meta) &&
                shapes_equal_meta(in_meta, out_meta);
    if (fast) {
        relu_forward_contig_kernel<<<blocks, kThreads, 0, stream>>>(in, out, numel);
        CUDA_CHECK(cudaGetLastError());
    } else {
        relu_forward_kernel<<<blocks, kThreads, 0, stream>>>(in, out, in_meta, out_meta, numel);
        CUDA_CHECK(cudaGetLastError());
    }
}

void launch_relu_backward(const float* in, const float* out_grad, float* in_grad,
                          const TensorMetadata& in_meta,
                          const TensorMetadata& out_meta,
                          size_t numel) {
    int blocks = static_cast<int>((numel + kThreads - 1) / kThreads);
    cudaStream_t stream = CudaContext::stream();
    bool fast = is_strict_contig_meta(in_meta) &&
                is_strict_contig_meta(out_meta) &&
                shapes_equal_meta(in_meta, out_meta);
    if (fast) {
        relu_backward_contig_kernel<<<blocks, kThreads, 0, stream>>>(in, out_grad, in_grad, numel);
        CUDA_CHECK(cudaGetLastError());
    } else {
        relu_backward_kernel<<<blocks, kThreads, 0, stream>>>(in, out_grad, in_grad, in_meta, out_meta, numel);
        CUDA_CHECK(cudaGetLastError());
    }
}

void launch_sigmoid_forward(const float* in, float* out,
                            const TensorMetadata& in_meta,
                            const TensorMetadata& out_meta,
                            size_t numel) {
    int blocks = static_cast<int>((numel + kThreads - 1) / kThreads);
    cudaStream_t stream = CudaContext::stream();
    bool fast = is_strict_contig_meta(in_meta) &&
                is_strict_contig_meta(out_meta) &&
                shapes_equal_meta(in_meta, out_meta);
    if (fast) {
        sigmoid_forward_contig_kernel<<<blocks, kThreads, 0, stream>>>(in, out, numel);
        CUDA_CHECK(cudaGetLastError());
    } else {
        sigmoid_forward_kernel<<<blocks, kThreads, 0, stream>>>(in, out, in_meta, out_meta, numel);
        CUDA_CHECK(cudaGetLastError());
    }
}

void launch_sigmoid_backward(const float* out, const float* out_grad, float* in_grad,
                             const TensorMetadata& out_meta,
                             const TensorMetadata& in_meta,
                             size_t numel) {
    int blocks = static_cast<int>((numel + kThreads - 1) / kThreads);
    cudaStream_t stream = CudaContext::stream();
    bool fast = is_strict_contig_meta(in_meta) &&
                is_strict_contig_meta(out_meta) &&
                shapes_equal_meta(in_meta, out_meta);
    if (fast) {
        sigmoid_backward_contig_kernel<<<blocks, kThreads, 0, stream>>>(out, out_grad, in_grad, numel);
        CUDA_CHECK(cudaGetLastError());
    } else {
        sigmoid_backward_kernel<<<blocks, kThreads, 0, stream>>>(out, out_grad, in_grad, out_meta, in_meta, numel);
        CUDA_CHECK(cudaGetLastError());
    }
}

void launch_mse_forward(const float* pred, const float* target, float* out,
                        const TensorMetadata& pred_meta,
                        const TensorMetadata& target_meta,
                        size_t numel) {
    cudaStream_t stream = CudaContext::stream();
    CUDA_CHECK(cudaMemsetAsync(out, 0, sizeof(float), stream));
    int blocks = static_cast<int>((numel + kThreads - 1) / kThreads);
    float scale = 1.0f / static_cast<float>(numel);
    mse_sum_kernel<<<blocks, kThreads, 0, stream>>>(
        pred, target, out, pred_meta, target_meta, numel, scale);
    CUDA_CHECK(cudaGetLastError());
}

void launch_mse_backward(const float* pred, const float* target, const float* out_grad,
                         float* pred_grad, float* target_grad,
                         const TensorMetadata& pred_meta,
                         const TensorMetadata& target_meta,
                         size_t numel) {
    int blocks = static_cast<int>((numel + kThreads - 1) / kThreads);
    cudaStream_t stream = CudaContext::stream();
    mse_backward_kernel<<<blocks, kThreads, 0, stream>>>(
        pred, target, out_grad, pred_grad, target_grad, pred_meta, target_meta, numel);
    CUDA_CHECK(cudaGetLastError());
}

void launch_matmul_forward(const float* a, const float* b, float* out,
                           size_t m, size_t k, size_t n,
                           size_t a_stride0, size_t a_stride1,
                           size_t b_stride0, size_t b_stride1,
                           size_t out_stride0, size_t out_stride1) {
    dim3 threads(16, 16);
    dim3 blocks((n + threads.x - 1) / threads.x, (m + threads.y - 1) / threads.y);
    cudaStream_t stream = CudaContext::stream();
    matmul_forward_kernel<<<blocks, threads, 0, stream>>>(a, b, out, m, k, n,
                                                                      a_stride0, a_stride1,
                                                                      b_stride0, b_stride1,
                                                                      out_stride0, out_stride1);
    CUDA_CHECK(cudaGetLastError());
}

void launch_matmul_backward_a(const float* out_grad, const float* b, float* a_grad,
                              size_t m, size_t k, size_t n,
                              size_t out_stride0, size_t out_stride1,
                              size_t b_stride0, size_t b_stride1,
                              size_t a_stride0, size_t a_stride1) {
    dim3 threads(16, 16);
    dim3 blocks((k + threads.x - 1) / threads.x, (m + threads.y - 1) / threads.y);
    cudaStream_t stream = CudaContext::stream();
    matmul_backward_a_kernel<<<blocks, threads, 0, stream>>>(out_grad, b, a_grad, m, k, n,
                                                                         out_stride0, out_stride1,
                                                                         b_stride0, b_stride1,
                                                                         a_stride0, a_stride1);
    CUDA_CHECK(cudaGetLastError());
}

void launch_matmul_backward_b(const float* a, const float* out_grad, float* b_grad,
                              size_t m, size_t k, size_t n,
                              size_t a_stride0, size_t a_stride1,
                              size_t out_stride0, size_t out_stride1,
                              size_t b_stride0, size_t b_stride1) {
    dim3 threads(16, 16);
    dim3 blocks((n + threads.x - 1) / threads.x, (k + threads.y - 1) / threads.y);
    cudaStream_t stream = CudaContext::stream();
    matmul_backward_b_kernel<<<blocks, threads, 0, stream>>>(a, out_grad, b_grad, m, k, n,
                                                                         a_stride0, a_stride1,
                                                                         out_stride0, out_stride1,
                                                                         b_stride0, b_stride1);
    CUDA_CHECK(cudaGetLastError());
}

void launch_add_broadcast_forward(const float* a, const float* b, float* out,
                                  const TensorMetadata& a_meta,
                                  const TensorMetadata& b_meta,
                                  const TensorMetadata& out_meta,
                                  size_t out_numel) {
    int blocks = static_cast<int>((out_numel + kThreads - 1) / kThreads);
    cudaStream_t stream = CudaContext::stream();
    bool fast = is_strict_contig_meta(a_meta) &&
                is_strict_contig_meta(b_meta) &&
                is_strict_contig_meta(out_meta) &&
                shapes_equal_meta(a_meta, out_meta) &&
                shapes_equal_meta(b_meta, out_meta);
    if (fast) {
        add_nobroadcast_contig_forward_kernel<<<blocks, kThreads, 0, stream>>>(
            a, b, out, out_numel);
        CUDA_CHECK(cudaGetLastError());
    } else {
        add_broadcast_nd_forward_kernel<<<blocks, kThreads, 0, stream>>>(
            a, b, out, a_meta, b_meta, out_meta, out_numel);
        CUDA_CHECK(cudaGetLastError());
    }
}

void launch_add_broadcast_backward(const float* out_grad, float* a_grad, float* b_grad,
                                   const TensorMetadata& a_meta,
                                   const TensorMetadata& b_meta,
                                   const TensorMetadata& out_meta,
                                   size_t out_numel) {
    int blocks = static_cast<int>((out_numel + kThreads - 1) / kThreads);
    cudaStream_t stream = CudaContext::stream();
    if (a_grad && is_scalar_meta(a_meta)) {
        reduce_sum_kernel<<<blocks, kThreads, 0, stream>>>(
            out_grad, a_grad, out_meta, out_numel);
        CUDA_CHECK(cudaGetLastError());
        a_grad = nullptr;
    }
    if (b_grad && is_scalar_meta(b_meta)) {
        reduce_sum_kernel<<<blocks, kThreads, 0, stream>>>(
            out_grad, b_grad, out_meta, out_numel);
        CUDA_CHECK(cudaGetLastError());
        b_grad = nullptr;
    }
    if (a_grad || b_grad) {
        add_broadcast_nd_backward_kernel<<<blocks, kThreads, 0, stream>>>(
            out_grad, a_grad, b_grad, a_meta, b_meta, out_meta, out_numel);
        CUDA_CHECK(cudaGetLastError());
    }
}

void launch_add_backward_nobroadcast(const float* out_grad, float* in_grad,
                                     const TensorMetadata& out_meta,
                                     const TensorMetadata& in_meta,
                                     size_t numel) {
    int blocks = static_cast<int>((numel + kThreads - 1) / kThreads);
    cudaStream_t stream = CudaContext::stream();
    bool fast = is_strict_contig_meta(out_meta) &&
                is_strict_contig_meta(in_meta) &&
                shapes_equal_meta(out_meta, in_meta);
    if (fast) {
        add_backward_nobroadcast_contig_kernel<<<blocks, kThreads, 0, stream>>>(
            out_grad, in_grad, numel);
        CUDA_CHECK(cudaGetLastError());
    } else {
        add_backward_nobroadcast_kernel<<<blocks, kThreads, 0, stream>>>(
            out_grad, in_grad, out_meta, in_meta, numel);
        CUDA_CHECK(cudaGetLastError());
    }
}

void launch_mul_broadcast_forward(const float* a, const float* b, float* out,
                                  const TensorMetadata& a_meta,
                                  const TensorMetadata& b_meta,
                                  const TensorMetadata& out_meta,
                                  size_t out_numel) {
    int blocks = static_cast<int>((out_numel + kThreads - 1) / kThreads);
    cudaStream_t stream = CudaContext::stream();
    bool fast = is_strict_contig_meta(a_meta) &&
                is_strict_contig_meta(b_meta) &&
                is_strict_contig_meta(out_meta) &&
                shapes_equal_meta(a_meta, out_meta) &&
                shapes_equal_meta(b_meta, out_meta);
    if (fast) {
        mul_nobroadcast_contig_forward_kernel<<<blocks, kThreads, 0, stream>>>(
            a, b, out, out_numel);
        CUDA_CHECK(cudaGetLastError());
    } else {
        mul_broadcast_nd_forward_kernel<<<blocks, kThreads, 0, stream>>>(
            a, b, out, a_meta, b_meta, out_meta, out_numel);
        CUDA_CHECK(cudaGetLastError());
    }
}

void launch_mul_broadcast_backward(const float* out_grad, const float* a, const float* b,
                                   float* a_grad, float* b_grad,
                                   const TensorMetadata& a_meta,
                                   const TensorMetadata& b_meta,
                                   const TensorMetadata& out_meta,
                                   size_t out_numel) {
    int blocks = static_cast<int>((out_numel + kThreads - 1) / kThreads);
    cudaStream_t stream = CudaContext::stream();
    if (a_grad && is_scalar_meta(a_meta)) {
        reduce_sum_mul_kernel<<<blocks, kThreads, 0, stream>>>(
            out_grad, b, a_grad, out_meta, out_meta, out_numel);
        CUDA_CHECK(cudaGetLastError());
        a_grad = nullptr;
    }
    if (b_grad && is_scalar_meta(b_meta)) {
        reduce_sum_mul_kernel<<<blocks, kThreads, 0, stream>>>(
            out_grad, a, b_grad, out_meta, out_meta, out_numel);
        CUDA_CHECK(cudaGetLastError());
        b_grad = nullptr;
    }
    if (a_grad || b_grad) {
        mul_broadcast_nd_backward_kernel<<<blocks, kThreads, 0, stream>>>(
            out_grad, a, b, a_grad, b_grad, a_meta, b_meta, out_meta, out_numel);
        CUDA_CHECK(cudaGetLastError());
    }
}

void launch_mul_backward_nobroadcast(const float* out_grad, const float* a, const float* b,
                                     float* a_grad, float* b_grad,
                                     const TensorMetadata& out_meta,
                                     const TensorMetadata& a_meta,
                                     const TensorMetadata& b_meta,
                                     size_t numel) {
    int blocks = static_cast<int>((numel + kThreads - 1) / kThreads);
    cudaStream_t stream = CudaContext::stream();
    bool fast = is_strict_contig_meta(out_meta) &&
                is_strict_contig_meta(a_meta) &&
                is_strict_contig_meta(b_meta) &&
                shapes_equal_meta(out_meta, a_meta) &&
                shapes_equal_meta(out_meta, b_meta);
    if (fast) {
        mul_backward_nobroadcast_contig_kernel<<<blocks, kThreads, 0, stream>>>(
            out_grad, a, b, a_grad, b_grad, numel);
        CUDA_CHECK(cudaGetLastError());
    } else {
        mul_backward_nobroadcast_kernel<<<blocks, kThreads, 0, stream>>>(
            out_grad, a, b, a_grad, b_grad, out_meta, a_meta, b_meta, numel);
        CUDA_CHECK(cudaGetLastError());
    }
}

void launch_add_inplace_nd(const float* src, float* dst,
                           const TensorMetadata& meta,
                           size_t numel) {
    int blocks = static_cast<int>((numel + kThreads - 1) / kThreads);
    cudaStream_t stream = CudaContext::stream();
    add_inplace_nd_kernel<<<blocks, kThreads, 0, stream>>>(src, dst, meta, numel);
    CUDA_CHECK(cudaGetLastError());
}

void launch_add_contig_to_strided(const float* src_contig, float* dst_strided,
                                  const TensorMetadata& dst_meta,
                                  size_t numel) {
    int blocks = static_cast<int>((numel + kThreads - 1) / kThreads);
    cudaStream_t stream = CudaContext::stream();
    add_contig_to_strided_kernel<<<blocks, kThreads, 0, stream>>>(
        src_contig, dst_strided, dst_meta, numel);
    CUDA_CHECK(cudaGetLastError());
}

void launch_copy_contig_to_strided(const float* src_contig, float* dst_strided,
                                   const TensorMetadata& dst_meta,
                                   size_t numel) {
    int blocks = static_cast<int>((numel + kThreads - 1) / kThreads);
    cudaStream_t stream = CudaContext::stream();
    copy_contig_to_strided_kernel<<<blocks, kThreads, 0, stream>>>(
        src_contig, dst_strided, dst_meta, numel);
    CUDA_CHECK(cudaGetLastError());
}

void launch_copy_strided_to_contig(const float* src_strided, float* dst_contig,
                                   const TensorMetadata& src_meta,
                                   size_t numel) {
    int blocks = static_cast<int>((numel + kThreads - 1) / kThreads);
    cudaStream_t stream = CudaContext::stream();
    copy_strided_to_contig_kernel<<<blocks, kThreads, 0, stream>>>(
        src_strided, dst_contig, src_meta, numel);
    CUDA_CHECK(cudaGetLastError());
}

void launch_fill(float* data, float value, size_t numel) {
    int blocks = static_cast<int>((numel + kThreads - 1) / kThreads);
    cudaStream_t stream = CudaContext::stream();
    fill_kernel<<<blocks, kThreads, 0, stream>>>(data, value, numel);
    CUDA_CHECK(cudaGetLastError());
}

void launch_add_scalar(float* data, float value, size_t numel) {
    int blocks = static_cast<int>((numel + kThreads - 1) / kThreads);
    cudaStream_t stream = CudaContext::stream();
    add_scalar_kernel<<<blocks, kThreads, 0, stream>>>(data, value, numel);
    CUDA_CHECK(cudaGetLastError());
}

void launch_add_inplace_broadcast(const float* b, float* a,
                                  const TensorMetadata& b_meta,
                                  const TensorMetadata& a_meta,
                                  size_t out_numel) {
    int blocks = static_cast<int>((out_numel + kThreads - 1) / kThreads);
    cudaStream_t stream = CudaContext::stream();
    add_inplace_broadcast_kernel<<<blocks, kThreads, 0, stream>>>(
        b, a, b_meta, a_meta, out_numel);
    CUDA_CHECK(cudaGetLastError());
}

void launch_mul_inplace_broadcast(const float* b, float* a,
                                  const TensorMetadata& b_meta,
                                  const TensorMetadata& a_meta,
                                  size_t out_numel) {
    int blocks = static_cast<int>((out_numel + kThreads - 1) / kThreads);
    cudaStream_t stream = CudaContext::stream();
    mul_inplace_broadcast_kernel<<<blocks, kThreads, 0, stream>>>(
        b, a, b_meta, a_meta, out_numel);
    CUDA_CHECK(cudaGetLastError());
}

void launch_relu_inplace(const float* in, float* out,
                         const TensorMetadata& meta,
                         size_t numel) {
    (void)in;
    int blocks = static_cast<int>((numel + kThreads - 1) / kThreads);
    cudaStream_t stream = CudaContext::stream();
    if (is_strict_contig_meta(meta)) {
        relu_inplace_contig_kernel<<<blocks, kThreads, 0, stream>>>(out, numel);
        CUDA_CHECK(cudaGetLastError());
    } else {
        relu_inplace_kernel<<<blocks, kThreads, 0, stream>>>(out, meta, numel);
        CUDA_CHECK(cudaGetLastError());
    }
}

void launch_softmax_forward(const float* in, float* out,
                            const TensorMetadata& in_meta,
                            const TensorMetadata& out_meta,
                            size_t outer, size_t cols) {
    cudaStream_t stream = CudaContext::stream();
    softmax_forward_kernel<<<static_cast<int>(outer), kThreads, 0, stream>>>(
        in, out, in_meta, out_meta, outer, cols);
    CUDA_CHECK(cudaGetLastError());
}

void launch_softmax_backward(const float* out, const float* out_grad, float* in_grad,
                             const TensorMetadata& out_meta,
                             const TensorMetadata& in_meta,
                             size_t outer, size_t cols) {
    cudaStream_t stream = CudaContext::stream();
    softmax_backward_kernel<<<static_cast<int>(outer), kThreads, 0, stream>>>(
        out, out_grad, in_grad, out_meta, in_meta, outer, cols);
    CUDA_CHECK(cudaGetLastError());
}

void launch_cross_entropy_forward(const float* logits, const float* target, float* out,
                                  const TensorMetadata& logits_meta,
                                  const TensorMetadata& target_meta,
                                  size_t outer, size_t cols) {
    cudaStream_t stream = CudaContext::stream();
    CUDA_CHECK(cudaMemsetAsync(out, 0, sizeof(float), stream));
    float scale = 1.0f / static_cast<float>(outer);
    cross_entropy_forward_kernel<<<static_cast<int>(outer), kThreads, 0, stream>>>(
        logits, target, out, logits_meta, target_meta, outer, cols, scale);
    CUDA_CHECK(cudaGetLastError());
}

void launch_cross_entropy_backward(const float* logits, const float* target, const float* out_grad,
                                   float* logits_grad,
                                   const TensorMetadata& logits_meta,
                                   const TensorMetadata& target_meta,
                                   size_t outer, size_t cols) {
    cudaStream_t stream = CudaContext::stream();
    cross_entropy_backward_kernel<<<static_cast<int>(outer), kThreads, 0, stream>>>(
        logits, target, out_grad, logits_grad, logits_meta, target_meta, outer, cols);
    CUDA_CHECK(cudaGetLastError());
}
void launch_sgd_step(float* data, const float* grad,
                     const TensorMetadata& data_meta,
                     const TensorMetadata& grad_meta,
                     size_t numel,
                     float lr) {
    int blocks = static_cast<int>((numel + kThreads - 1) / kThreads);
    cudaStream_t stream = CudaContext::stream();
    sgd_step_nd_kernel<<<blocks, kThreads, 0, stream>>>(data, grad, data_meta, grad_meta, numel, lr);
    CUDA_CHECK(cudaGetLastError());
}

void launch_adam_step(float* data, const float* grad, float* m, float* v,
                      const TensorMetadata& data_meta,
                      const TensorMetadata& grad_meta,
                      const TensorMetadata& m_meta,
                      const TensorMetadata& v_meta,
                      size_t numel,
                      float lr, float beta1, float beta2, float eps,
                      float bias_correction1, float bias_correction2) {
    int blocks = static_cast<int>((numel + kThreads - 1) / kThreads);
    cudaStream_t stream = CudaContext::stream();
    adam_step_nd_kernel<<<blocks, kThreads, 0, stream>>>(
        data, grad, m, v,
        data_meta, grad_meta, m_meta, v_meta,
        numel,
        lr, beta1, beta2, eps, bias_correction1, bias_correction2);
    CUDA_CHECK(cudaGetLastError());
}
