#include "Operators.hpp"
#include "CudaKernels.cuh"
#include "CudaContext.hpp"

#include <algorithm>
#include <cublas_v2.h>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace {

size_t numel_from_shape(const std::vector<size_t>& shape) {
    if (shape.empty()) {
        return 1;
    }
    return std::accumulate(shape.begin(), shape.end(), static_cast<size_t>(1),
                           std::multiplies<size_t>());
}

std::vector<size_t> broadcast_shape(const std::vector<size_t>& a,
                                    const std::vector<size_t>& b) {
    size_t rank = std::max(a.size(), b.size());
    std::vector<size_t> out(rank, 1);
    for (size_t i = 0; i < rank; ++i) {
        size_t a_dim = (i < rank - a.size()) ? 1 : a[i - (rank - a.size())];
        size_t b_dim = (i < rank - b.size()) ? 1 : b[i - (rank - b.size())];
        if (a_dim != b_dim && a_dim != 1 && b_dim != 1) {
            throw std::runtime_error("broadcast_shape: incompatible shapes");
        }
        out[i] = std::max(a_dim, b_dim);
    }
    return out;
}

void linear_to_indices(size_t linear,
                       const std::vector<size_t>& shape,
                       const std::vector<size_t>& strides,
                       std::vector<size_t>& indices_out) {
    indices_out.assign(shape.size(), 0);
    if (shape.empty()) {
        return;
    }
    for (size_t i = 0; i < shape.size(); ++i) {
        size_t stride = strides[i];
        indices_out[i] = linear / stride;
        linear %= stride;
    }
}

size_t indices_to_linear(const std::vector<size_t>& indices,
                         const std::vector<size_t>& strides) {
    size_t linear = 0;
    for (size_t i = 0; i < indices.size(); ++i) {
        linear += indices[i] * strides[i];
    }
    return linear;
}


size_t numel_excluding_last(const std::vector<size_t>& shape) {
    if (shape.empty()) {
        return 1;
    }
    if (shape.size() == 1) {
        return 1;
    }
    return std::accumulate(shape.begin(), shape.end() - 1, static_cast<size_t>(1),
                           std::multiplies<size_t>());
}


enum class MatrixLayout {
    RowMajor,
    ColMajor,
    Unsupported
};

MatrixLayout infer_layout(size_t rows, size_t cols, size_t s0, size_t s1) {
    if (s1 == 1 && s0 >= cols) {
        return MatrixLayout::RowMajor;
    }
    if (s0 == 1 && s1 >= rows) {
        return MatrixLayout::ColMajor;
    }
    return MatrixLayout::Unsupported;
}

cublasOperation_t toggle_op(cublasOperation_t op) {
    return op == CUBLAS_OP_N ? CUBLAS_OP_T : CUBLAS_OP_N;
}

void cublas_gemm_row_major(const float* a, const float* b, float* c,
                           int m, int n, int k,
                           size_t a_s0, size_t a_s1,
                           size_t b_s0, size_t b_s1,
                           size_t c_s0, size_t c_s1,
                           cublasOperation_t op_a, cublasOperation_t op_b,
                           float alpha, float beta) {
    cublasHandle_t handle = CudaContext::cublas();
    int a_rows = (op_a == CUBLAS_OP_N) ? m : k;
    int a_cols = (op_a == CUBLAS_OP_N) ? k : m;
    int b_rows = (op_b == CUBLAS_OP_N) ? k : n;
    int b_cols = (op_b == CUBLAS_OP_N) ? n : k;
    MatrixLayout a_layout = infer_layout(static_cast<size_t>(a_rows), static_cast<size_t>(a_cols), a_s0, a_s1);
    MatrixLayout b_layout = infer_layout(static_cast<size_t>(b_rows), static_cast<size_t>(b_cols), b_s0, b_s1);
    MatrixLayout c_layout = infer_layout(static_cast<size_t>(m), static_cast<size_t>(n), c_s0, c_s1);
    if (a_layout == MatrixLayout::Unsupported ||
        b_layout == MatrixLayout::Unsupported ||
        c_layout == MatrixLayout::Unsupported) {
        throw std::runtime_error("cublas_gemm_row_major: unsupported strided layout");
    }

    int lda = (a_layout == MatrixLayout::RowMajor) ? static_cast<int>(a_s0) : static_cast<int>(a_s1);
    int ldb = (b_layout == MatrixLayout::RowMajor) ? static_cast<int>(b_s0) : static_cast<int>(b_s1);
    if (c_layout == MatrixLayout::RowMajor) {
        cublasOperation_t op_a_eff = (a_layout == MatrixLayout::RowMajor) ? op_a : toggle_op(op_a);
        cublasOperation_t op_b_eff = (b_layout == MatrixLayout::RowMajor) ? op_b : toggle_op(op_b);
        int ldc = static_cast<int>(c_s0);
        CUBLAS_CHECK(cublasSgemm(handle,
                                 op_b_eff, op_a_eff,
                                 n, m, k,
                                 &alpha,
                                 b, ldb,
                                 a, lda,
                                 &beta,
                                 c, ldc));
    } else {
        cublasOperation_t op_a_eff = (a_layout == MatrixLayout::RowMajor) ? toggle_op(op_a) : op_a;
        cublasOperation_t op_b_eff = (b_layout == MatrixLayout::RowMajor) ? toggle_op(op_b) : op_b;
        int ldc = static_cast<int>(c_s1);
        CUBLAS_CHECK(cublasSgemm(handle,
                                 op_a_eff, op_b_eff,
                                 m, n, k,
                                 &alpha,
                                 a, lda,
                                 b, ldb,
                                 &beta,
                                 c, ldc));
    }
}

struct TensorView {
    std::shared_ptr<float> data_base;
    std::shared_ptr<float> grad_base;
    size_t storage_offset = 0;
    size_t grad_offset = 0;
    std::vector<size_t> shape;
    std::vector<size_t> strides;
    Device device = Device::CPU;
    bool requires_grad = false;

    size_t numel() const { return numel_from_shape(shape); }
    float* data_ptr() const { return data_base.get() + storage_offset; }
    float* grad_ptr() const { return grad_base ? grad_base.get() + grad_offset : nullptr; }
};

std::vector<float> reduce_grad_to_shape(const std::vector<float>& out_grad,
                                        const TensorView& out,
                                        const TensorView& in) {
    const std::vector<size_t>& out_shape = out.shape;
    const std::vector<size_t>& in_shape = in.shape;
    const std::vector<size_t>& out_strides = out.strides;
    const std::vector<size_t>& in_strides = in.strides;
    size_t out_numel = numel_from_shape(out_shape);
    size_t in_numel = numel_from_shape(in_shape);
    std::vector<float> reduced(in_numel, 0.0f);

    size_t rank_out = out_shape.size();
    size_t rank_in = in_shape.size();
    size_t offset = rank_out - rank_in;

    std::vector<size_t> out_idx;
    std::vector<size_t> in_idx(rank_in, 0);

    for (size_t linear = 0; linear < out_numel; ++linear) {
        linear_to_indices(linear, out_shape, out_strides, out_idx);
        for (size_t i = 0; i < rank_in; ++i) {
            size_t out_dim_idx = i + offset;
            size_t dim = in_shape[i];
            if (dim == 1) {
                in_idx[i] = 0;
            } else {
                in_idx[i] = out_idx[out_dim_idx];
            }
        }
        size_t in_linear = indices_to_linear(in_idx, in_strides);
        reduced[in_linear] += out_grad[linear];
    }
    return reduced;
}

void add_inplace(const TensorView& dest, const std::vector<float>& src) {
    if (!dest.grad_base) {
        return;
    }
    float* grad_ptr = dest.grad_ptr();
    std::vector<size_t> logical_strides = Tensor::compute_contiguous_strides(dest.shape);
    std::vector<size_t> idx;
    for (size_t linear = 0; linear < src.size(); ++linear) {
        linear_to_indices(linear, dest.shape, logical_strides, idx);
        size_t offset = indices_to_linear(idx, dest.strides);
        Tensor::accumulate_grad_value(grad_ptr + offset, src[linear]);
    }
}

TensorMetadata make_metadata(const Tensor& t, size_t out_rank, bool broadcast) {
    if (out_rank > static_cast<size_t>(kMaxDims)) {
        throw std::runtime_error("make_metadata: rank exceeds kMaxDims");
    }
    TensorMetadata meta{};
    meta.rank = static_cast<int>(out_rank);
    size_t rank = t.shape.size();
    size_t offset = out_rank - rank;
    for (size_t i = 0; i < out_rank; ++i) {
        if (i < offset) {
            meta.shape[i] = 1;
            meta.strides[i] = 0;
        } else {
            size_t dim = t.shape[i - offset];
            size_t stride = t.strides[i - offset];
            meta.shape[i] = dim;
            meta.strides[i] = (broadcast && dim == 1) ? 0 : stride;
        }
    }
    std::vector<size_t> contig = Tensor::compute_contiguous_strides(
        std::vector<size_t>(meta.shape, meta.shape + out_rank));
    bool is_contig = true;
    for (size_t i = 0; i < out_rank; ++i) {
        if (meta.strides[i] != contig[i]) {
            is_contig = false;
            break;
        }
    }
    meta.is_contig = is_contig ? 1 : 0;
    return meta;
}

TensorMetadata make_metadata(const TensorView& t, size_t out_rank, bool broadcast) {
    if (out_rank > static_cast<size_t>(kMaxDims)) {
        throw std::runtime_error("make_metadata: rank exceeds kMaxDims");
    }
    TensorMetadata meta{};
    meta.rank = static_cast<int>(out_rank);
    size_t rank = t.shape.size();
    size_t offset = out_rank - rank;
    for (size_t i = 0; i < out_rank; ++i) {
        if (i < offset) {
            meta.shape[i] = 1;
            meta.strides[i] = 0;
        } else {
            size_t dim = t.shape[i - offset];
            size_t stride = t.strides[i - offset];
            meta.shape[i] = dim;
            meta.strides[i] = (broadcast && dim == 1) ? 0 : stride;
        }
    }
    std::vector<size_t> contig = Tensor::compute_contiguous_strides(
        std::vector<size_t>(meta.shape, meta.shape + out_rank));
    bool is_contig = true;
    for (size_t i = 0; i < out_rank; ++i) {
        if (meta.strides[i] != contig[i]) {
            is_contig = false;
            break;
        }
    }
    meta.is_contig = is_contig ? 1 : 0;
    return meta;
}

}  // namespace

Tensor add_cpu_impl(const Tensor& a, const Tensor& b) {
    std::vector<size_t> out_shape = broadcast_shape(a.shape, b.shape);
    Tensor out(out_shape, a.device, a.requires_grad || b.requires_grad);

    const std::vector<size_t>& out_strides = out.strides;
    const std::vector<size_t>& a_strides = a.strides;
    const std::vector<size_t>& b_strides = b.strides;

    size_t out_numel = out.numel();
    size_t rank_out = out_shape.size();
    size_t rank_a = a.shape.size();
    size_t rank_b = b.shape.size();
    size_t offset_a = rank_out - rank_a;
    size_t offset_b = rank_out - rank_b;

    std::vector<size_t> out_idx;
    std::vector<size_t> a_idx(rank_a, 0);
    std::vector<size_t> b_idx(rank_b, 0);

    for (size_t linear = 0; linear < out_numel; ++linear) {
        linear_to_indices(linear, out_shape, out_strides, out_idx);
        for (size_t i = 0; i < rank_a; ++i) {
            size_t out_dim_idx = i + offset_a;
            a_idx[i] = (a.shape[i] == 1) ? 0 : out_idx[out_dim_idx];
        }
        for (size_t i = 0; i < rank_b; ++i) {
            size_t out_dim_idx = i + offset_b;
            b_idx[i] = (b.shape[i] == 1) ? 0 : out_idx[out_dim_idx];
        }
        size_t a_linear = indices_to_linear(a_idx, a_strides);
        size_t b_linear = indices_to_linear(b_idx, b_strides);
        size_t out_linear = indices_to_linear(out_idx, out_strides);
        out.get_raw_pointer()[out_linear] = a.get_raw_pointer()[a_linear] + b.get_raw_pointer()[b_linear];
    }

    if (out.requires_grad && Tensor::is_grad_enabled()) {
        out.creator_node = std::make_shared<ComputeNode>();
        Tensor::add_input_nodes(out.creator_node, {a, b});
        auto out_grad_base = out.grad_base();
        size_t out_grad_offset = out.grad_offset;
        auto a_grad_base = a.grad_base();
        size_t a_grad_offset = a.grad_offset;
        auto b_grad_base = b.grad_base();
        size_t b_grad_offset = b.grad_offset;
        auto out_shape = out.shape;
        auto out_strides = out.strides;
        auto a_shape = a.shape;
        auto a_strides = a.strides;
        auto b_shape = b.shape;
        auto b_strides = b.strides;
        bool a_req = a.requires_grad;
        bool b_req = b.requires_grad;
        out.creator_node->backward_fn = [out_grad_base, out_grad_offset,
                                         a_grad_base, a_grad_offset,
                                         b_grad_base, b_grad_offset,
                                         out_shape, out_strides,
                                         a_shape, a_strides,
                                         b_shape, b_strides,
                                         a_req, b_req]() mutable {
            if (!out_grad_base) {
                return;
            }
            size_t out_numel = numel_from_shape(out_shape);
            float* out_grad_ptr = out_grad_base.get() + out_grad_offset;
            std::vector<float> out_grad(out_numel);
            std::copy(out_grad_ptr, out_grad_ptr + out_numel, out_grad.begin());

            TensorView out_view;
            out_view.shape = out_shape;
            out_view.strides = out_strides;

            if (a_req && a_grad_base) {
                TensorView a_view;
                a_view.grad_base = a_grad_base;
                a_view.grad_offset = a_grad_offset;
                a_view.shape = a_shape;
                a_view.strides = a_strides;
                std::vector<float> reduced = reduce_grad_to_shape(out_grad, out_view, a_view);
                add_inplace(a_view, reduced);
            }
            if (b_req && b_grad_base) {
                TensorView b_view;
                b_view.grad_base = b_grad_base;
                b_view.grad_offset = b_grad_offset;
                b_view.shape = b_shape;
                b_view.strides = b_strides;
                std::vector<float> reduced = reduce_grad_to_shape(out_grad, out_view, b_view);
                add_inplace(b_view, reduced);
            }
        };
    }
    return out;
}

Tensor add_cuda_impl(const Tensor& a, const Tensor& b) {
    std::vector<size_t> out_shape = broadcast_shape(a.shape, b.shape);
    Tensor out(out_shape, a.device, a.requires_grad || b.requires_grad);

    size_t out_rank = out_shape.size();
    check_rank_within_kmaxdims(out_rank, "add CUDA: rank exceeds kMaxDims");
    TensorMetadata out_meta = make_metadata(out, out_rank, false);
    TensorMetadata a_meta = make_metadata(a, out_rank, true);
    TensorMetadata b_meta = make_metadata(b, out_rank, true);
    launch_add_broadcast_forward(
        a.get_raw_pointer(), b.get_raw_pointer(), out.get_raw_pointer(),
        a_meta, b_meta, out_meta, out.numel());

    if (out.requires_grad && Tensor::is_grad_enabled()) {
        out.creator_node = std::make_shared<ComputeNode>();
        Tensor::add_input_nodes(out.creator_node, {a, b});
        auto out_grad_base = out.grad_base();
        size_t out_grad_offset = out.grad_offset;
        auto a_grad_base = a.grad_base();
        size_t a_grad_offset = a.grad_offset;
        auto b_grad_base = b.grad_base();
        size_t b_grad_offset = b.grad_offset;
        auto out_shape = out.shape;
        auto out_strides = out.strides;
        auto a_shape = a.shape;
        auto a_strides = a.strides;
        auto b_shape = b.shape;
        auto b_strides = b.strides;
        bool a_req = a.requires_grad;
        bool b_req = b.requires_grad;
        out.creator_node->backward_fn = [out_grad_base, out_grad_offset,
                                         a_grad_base, a_grad_offset,
                                         b_grad_base, b_grad_offset,
                                         out_shape, out_strides,
                                         a_shape, a_strides,
                                         b_shape, b_strides,
                                         a_req, b_req]() mutable {
            if (!out_grad_base) {
                return;
            }
            size_t out_numel = numel_from_shape(out_shape);
            float* out_grad_ptr = out_grad_base.get() + out_grad_offset;
            size_t out_rank = out_shape.size();
            check_rank_within_kmaxdims(out_rank, "add backward CUDA: rank exceeds kMaxDims");
            TensorView out_view;
            out_view.shape = out_shape;
            out_view.strides = out_strides;
            TensorView a_view;
            a_view.shape = a_shape;
            a_view.strides = a_strides;
            TensorView b_view;
            b_view.shape = b_shape;
            b_view.strides = b_strides;
            TensorMetadata out_meta = make_metadata(out_view, out_rank, false);
            TensorMetadata a_meta = make_metadata(a_view, out_rank, true);
            TensorMetadata b_meta = make_metadata(b_view, out_rank, true);
            bool a_nobroadcast = (a_shape == out_shape);
            bool b_nobroadcast = (b_shape == out_shape);
            float* a_grad_ptr = a_grad_base ? (a_grad_base.get() + a_grad_offset) : nullptr;
            float* b_grad_ptr = b_grad_base ? (b_grad_base.get() + b_grad_offset) : nullptr;
            if (a_req && a_nobroadcast && a_grad_ptr) {
                TensorMetadata a_meta_nb = make_metadata(a_view, out_rank, false);
                launch_add_backward_nobroadcast(out_grad_ptr, a_grad_ptr, out_meta, a_meta_nb, out_numel);
            }
            if (b_req && b_nobroadcast && b_grad_ptr) {
                TensorMetadata b_meta_nb = make_metadata(b_view, out_rank, false);
                launch_add_backward_nobroadcast(out_grad_ptr, b_grad_ptr, out_meta, b_meta_nb, out_numel);
            }
            if ((a_req && !a_nobroadcast) || (b_req && !b_nobroadcast)) {
                launch_add_broadcast_backward(
                    out_grad_ptr,
                    (a_req && !a_nobroadcast) ? a_grad_ptr : nullptr,
                    (b_req && !b_nobroadcast) ? b_grad_ptr : nullptr,
                    a_meta, b_meta, out_meta, out_numel);
            }
        };
    }
    return out;
}

Tensor add(const Tensor& a, const Tensor& b) {
    if (a.device != b.device) {
        throw std::runtime_error("Device mismatch");
    }
    if (a.device == Device::CPU) {
        return add_cpu_impl(a, b);
    }
    if (a.device == Device::CUDA) {
        return add_cuda_impl(a, b);
    }
    throw std::runtime_error("add: unsupported device");
}

Tensor mul_cpu_impl(const Tensor& a, const Tensor& b) {
    std::vector<size_t> out_shape = broadcast_shape(a.shape, b.shape);
    Tensor out(out_shape, a.device, a.requires_grad || b.requires_grad);

    const std::vector<size_t>& out_strides = out.strides;
    const std::vector<size_t>& a_strides = a.strides;
    const std::vector<size_t>& b_strides = b.strides;

    size_t out_numel = out.numel();
    size_t rank_out = out_shape.size();
    size_t rank_a = a.shape.size();
    size_t rank_b = b.shape.size();
    size_t offset_a = rank_out - rank_a;
    size_t offset_b = rank_out - rank_b;

    std::vector<size_t> out_idx;
    std::vector<size_t> a_idx(rank_a, 0);
    std::vector<size_t> b_idx(rank_b, 0);

    for (size_t linear = 0; linear < out_numel; ++linear) {
        linear_to_indices(linear, out_shape, out_strides, out_idx);
        for (size_t i = 0; i < rank_a; ++i) {
            size_t out_dim_idx = i + offset_a;
            a_idx[i] = (a.shape[i] == 1) ? 0 : out_idx[out_dim_idx];
        }
        for (size_t i = 0; i < rank_b; ++i) {
            size_t out_dim_idx = i + offset_b;
            b_idx[i] = (b.shape[i] == 1) ? 0 : out_idx[out_dim_idx];
        }
        size_t a_linear = indices_to_linear(a_idx, a_strides);
        size_t b_linear = indices_to_linear(b_idx, b_strides);
        size_t out_linear = indices_to_linear(out_idx, out_strides);
        out.get_raw_pointer()[out_linear] = a.get_raw_pointer()[a_linear] * b.get_raw_pointer()[b_linear];
    }

    if (out.requires_grad && Tensor::is_grad_enabled()) {
        out.creator_node = std::make_shared<ComputeNode>();
        Tensor::add_input_nodes(out.creator_node, {a, b});
        auto out_grad_base = out.grad_base();
        size_t out_grad_offset = out.grad_offset;
        auto a_data_base = a.data_base();
        size_t a_data_offset = a.storage_offset;
        auto b_data_base = b.data_base();
        size_t b_data_offset = b.storage_offset;
        auto a_grad_base = a.grad_base();
        size_t a_grad_offset = a.grad_offset;
        auto b_grad_base = b.grad_base();
        size_t b_grad_offset = b.grad_offset;
        auto out_shape = out.shape;
        auto out_strides = out.strides;
        auto a_shape = a.shape;
        auto a_strides = a.strides;
        auto b_shape = b.shape;
        auto b_strides = b.strides;
        bool a_req = a.requires_grad;
        bool b_req = b.requires_grad;
        out.creator_node->backward_fn = [out_grad_base, out_grad_offset,
                                         a_data_base, a_data_offset,
                                         b_data_base, b_data_offset,
                                         a_grad_base, a_grad_offset,
                                         b_grad_base, b_grad_offset,
                                         out_shape, out_strides,
                                         a_shape, a_strides,
                                         b_shape, b_strides,
                                         a_req, b_req]() mutable {
            if (!out_grad_base) {
                return;
            }
            size_t out_numel = numel_from_shape(out_shape);
            float* out_grad_ptr = out_grad_base.get() + out_grad_offset;
            std::vector<float> out_grad(out_numel);
            std::copy(out_grad_ptr, out_grad_ptr + out_numel, out_grad.begin());

            size_t rank_out = out_shape.size();
            size_t rank_a = a_shape.size();
            size_t rank_b = b_shape.size();
            size_t offset_a = rank_out - rank_a;
            size_t offset_b = rank_out - rank_b;

            std::vector<size_t> out_idx;
            std::vector<size_t> a_idx(rank_a, 0);
            std::vector<size_t> b_idx(rank_b, 0);

            std::vector<float> grad_a_out(out_numel, 0.0f);
            std::vector<float> grad_b_out(out_numel, 0.0f);

            float* a_data_ptr = a_data_base.get() + a_data_offset;
            float* b_data_ptr = b_data_base.get() + b_data_offset;

            for (size_t linear = 0; linear < out_numel; ++linear) {
                linear_to_indices(linear, out_shape, out_strides, out_idx);
                for (size_t i = 0; i < rank_a; ++i) {
                    size_t out_dim_idx = i + offset_a;
                    a_idx[i] = (a_shape[i] == 1) ? 0 : out_idx[out_dim_idx];
                }
                for (size_t i = 0; i < rank_b; ++i) {
                    size_t out_dim_idx = i + offset_b;
                    b_idx[i] = (b_shape[i] == 1) ? 0 : out_idx[out_dim_idx];
                }
                size_t a_linear = indices_to_linear(a_idx, a_strides);
                size_t b_linear = indices_to_linear(b_idx, b_strides);
                float og = out_grad[linear];
                grad_a_out[linear] = og * b_data_ptr[b_linear];
                grad_b_out[linear] = og * a_data_ptr[a_linear];
            }

            TensorView out_view;
            out_view.shape = out_shape;
            out_view.strides = out_strides;

            if (a_req && a_grad_base) {
                TensorView a_view;
                a_view.grad_base = a_grad_base;
                a_view.grad_offset = a_grad_offset;
                a_view.shape = a_shape;
                a_view.strides = a_strides;
                std::vector<float> reduced = reduce_grad_to_shape(grad_a_out, out_view, a_view);
                add_inplace(a_view, reduced);
            }
            if (b_req && b_grad_base) {
                TensorView b_view;
                b_view.grad_base = b_grad_base;
                b_view.grad_offset = b_grad_offset;
                b_view.shape = b_shape;
                b_view.strides = b_strides;
                std::vector<float> reduced = reduce_grad_to_shape(grad_b_out, out_view, b_view);
                add_inplace(b_view, reduced);
            }
        };
    }
    return out;
}

Tensor mul_cuda_impl(const Tensor& a, const Tensor& b) {
    std::vector<size_t> out_shape = broadcast_shape(a.shape, b.shape);
    Tensor out(out_shape, a.device, a.requires_grad || b.requires_grad);

    size_t out_rank = out_shape.size();
    check_rank_within_kmaxdims(out_rank, "mul CUDA: rank exceeds kMaxDims");
    TensorMetadata out_meta = make_metadata(out, out_rank, false);
    TensorMetadata a_meta = make_metadata(a, out_rank, true);
    TensorMetadata b_meta = make_metadata(b, out_rank, true);
    launch_mul_broadcast_forward(
        a.get_raw_pointer(), b.get_raw_pointer(), out.get_raw_pointer(),
        a_meta, b_meta, out_meta, out.numel());

    if (out.requires_grad && Tensor::is_grad_enabled()) {
        out.creator_node = std::make_shared<ComputeNode>();
        Tensor::add_input_nodes(out.creator_node, {a, b});
        auto out_grad_base = out.grad_base();
        size_t out_grad_offset = out.grad_offset;
        auto a_data_base = a.data_base();
        size_t a_data_offset = a.storage_offset;
        auto b_data_base = b.data_base();
        size_t b_data_offset = b.storage_offset;
        auto a_grad_base = a.grad_base();
        size_t a_grad_offset = a.grad_offset;
        auto b_grad_base = b.grad_base();
        size_t b_grad_offset = b.grad_offset;
        auto out_shape = out.shape;
        auto out_strides = out.strides;
        auto a_shape = a.shape;
        auto a_strides = a.strides;
        auto b_shape = b.shape;
        auto b_strides = b.strides;
        bool a_req = a.requires_grad;
        bool b_req = b.requires_grad;
        out.creator_node->backward_fn = [out_grad_base, out_grad_offset,
                                         a_data_base, a_data_offset,
                                         b_data_base, b_data_offset,
                                         a_grad_base, a_grad_offset,
                                         b_grad_base, b_grad_offset,
                                         out_shape, out_strides,
                                         a_shape, a_strides,
                                         b_shape, b_strides,
                                         a_req, b_req]() mutable {
            if (!out_grad_base) {
                return;
            }
            size_t out_numel = numel_from_shape(out_shape);
            float* out_grad_ptr = out_grad_base.get() + out_grad_offset;
            size_t out_rank = out_shape.size();
            check_rank_within_kmaxdims(out_rank, "mul backward CUDA: rank exceeds kMaxDims");
            TensorView out_view;
            out_view.shape = out_shape;
            out_view.strides = out_strides;
            TensorView a_view;
            a_view.shape = a_shape;
            a_view.strides = a_strides;
            TensorView b_view;
            b_view.shape = b_shape;
            b_view.strides = b_strides;
            TensorMetadata out_meta = make_metadata(out_view, out_rank, false);
            TensorMetadata a_meta = make_metadata(a_view, out_rank, true);
            TensorMetadata b_meta = make_metadata(b_view, out_rank, true);
            bool a_nobroadcast = (a_shape == out_shape);
            bool b_nobroadcast = (b_shape == out_shape);
            float* a_data_ptr = a_data_base.get() + a_data_offset;
            float* b_data_ptr = b_data_base.get() + b_data_offset;
            float* a_grad_ptr = a_grad_base ? (a_grad_base.get() + a_grad_offset) : nullptr;
            float* b_grad_ptr = b_grad_base ? (b_grad_base.get() + b_grad_offset) : nullptr;
            if (a_req && b_req && a_nobroadcast && b_nobroadcast && a_grad_ptr && b_grad_ptr) {
                TensorMetadata a_meta_nb = make_metadata(a_view, out_rank, false);
                TensorMetadata b_meta_nb = make_metadata(b_view, out_rank, false);
                launch_mul_backward_nobroadcast(out_grad_ptr, a_data_ptr, b_data_ptr,
                                                a_grad_ptr, b_grad_ptr,
                                                out_meta, a_meta_nb, b_meta_nb, out_numel);
            } else {
                launch_mul_broadcast_backward(
                    out_grad_ptr,
                    a_data_ptr,
                    b_data_ptr,
                    a_req ? a_grad_ptr : nullptr,
                    b_req ? b_grad_ptr : nullptr,
                    a_meta, b_meta, out_meta, out_numel);
            }
        };
    }
    return out;
}

Tensor mul(const Tensor& a, const Tensor& b) {
    if (a.device != b.device) {
        throw std::runtime_error("Device mismatch");
    }
    if (a.device == Device::CPU) {
        return mul_cpu_impl(a, b);
    }
    if (a.device == Device::CUDA) {
        return mul_cuda_impl(a, b);
    }
    throw std::runtime_error("mul: unsupported device");
}

Tensor matmul_cpu_impl(const Tensor& a, const Tensor& b) {
    if (a.shape.size() != 2 || b.shape.size() != 2) {
        throw std::runtime_error("matmul: only 2D tensors supported");
    }
    if (a.shape[1] != b.shape[0]) {
        throw std::runtime_error("matmul: shape mismatch");
    }
    size_t m = a.shape[0];
    size_t k = a.shape[1];
    size_t n = b.shape[1];
    Tensor out({m, n}, a.device, a.requires_grad || b.requires_grad);
    size_t a_s0 = a.strides[0];
    size_t a_s1 = a.strides[1];
    size_t b_s0 = b.strides[0];
    size_t b_s1 = b.strides[1];
    size_t out_s0 = out.strides[0];
    size_t out_s1 = out.strides[1];
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            float sum = 0.0f;
            for (size_t p = 0; p < k; ++p) {
                size_t a_off = i * a_s0 + p * a_s1;
                size_t b_off = p * b_s0 + j * b_s1;
                sum += a.get_raw_pointer()[a_off] * b.get_raw_pointer()[b_off];
            }
            out.get_raw_pointer()[i * out_s0 + j * out_s1] = sum;
        }
    }

    if (out.requires_grad && Tensor::is_grad_enabled()) {
        out.creator_node = std::make_shared<ComputeNode>();
        Tensor::add_input_nodes(out.creator_node, {a, b});
        auto out_grad_base = out.grad_base();
        size_t out_grad_offset = out.grad_offset;
        auto a_data_base = a.data_base();
        size_t a_data_offset = a.storage_offset;
        auto b_data_base = b.data_base();
        size_t b_data_offset = b.storage_offset;
        auto a_grad_base = a.grad_base();
        size_t a_grad_offset = a.grad_offset;
        auto b_grad_base = b.grad_base();
        size_t b_grad_offset = b.grad_offset;
        auto a_strides = a.strides;
        auto b_strides = b.strides;
        auto out_strides = out.strides;
        bool a_req = a.requires_grad;
        bool b_req = b.requires_grad;
        out.creator_node->backward_fn = [out_grad_base, out_grad_offset,
                                         a_data_base, a_data_offset,
                                         b_data_base, b_data_offset,
                                         a_grad_base, a_grad_offset,
                                         b_grad_base, b_grad_offset,
                                         a_strides, b_strides, out_strides,
                                         a_req, b_req, m, k, n]() mutable {
            if (!out_grad_base) {
                return;
            }
            float* out_grad_ptr = out_grad_base.get() + out_grad_offset;
            float* a_data_ptr = a_data_base.get() + a_data_offset;
            float* b_data_ptr = b_data_base.get() + b_data_offset;
            float* a_grad_ptr = a_grad_base ? (a_grad_base.get() + a_grad_offset) : nullptr;
            float* b_grad_ptr = b_grad_base ? (b_grad_base.get() + b_grad_offset) : nullptr;
            if (a_req && a_grad_ptr) {
                size_t a_s0 = a_strides[0];
                size_t a_s1 = a_strides[1];
                size_t b_s0 = b_strides[0];
                size_t b_s1 = b_strides[1];
                size_t out_s0 = out_strides[0];
                size_t out_s1 = out_strides[1];
                for (size_t i = 0; i < m; ++i) {
                    for (size_t p = 0; p < k; ++p) {
                        float sum = 0.0f;
                        for (size_t j = 0; j < n; ++j) {
                            size_t out_off = i * out_s0 + j * out_s1;
                            size_t b_off = p * b_s0 + j * b_s1;
                            sum += out_grad_ptr[out_off] * b_data_ptr[b_off];
                        }
                        size_t a_off = i * a_s0 + p * a_s1;
                        Tensor::accumulate_grad_value(a_grad_ptr + a_off, sum);
                    }
                }
            }
            if (b_req && b_grad_ptr) {
                size_t a_s0 = a_strides[0];
                size_t a_s1 = a_strides[1];
                size_t b_s0 = b_strides[0];
                size_t b_s1 = b_strides[1];
                size_t out_s0 = out_strides[0];
                size_t out_s1 = out_strides[1];
                for (size_t p = 0; p < k; ++p) {
                    for (size_t j = 0; j < n; ++j) {
                        float sum = 0.0f;
                        for (size_t i = 0; i < m; ++i) {
                            size_t a_off = i * a_s0 + p * a_s1;
                            size_t out_off = i * out_s0 + j * out_s1;
                            sum += a_data_ptr[a_off] * out_grad_ptr[out_off];
                        }
                        size_t b_off = p * b_s0 + j * b_s1;
                        Tensor::accumulate_grad_value(b_grad_ptr + b_off, sum);
                    }
                }
            }
        };
    }
    return out;
}

Tensor matmul_cuda_impl(const Tensor& a, const Tensor& b) {
    if (a.shape.size() != 2 || b.shape.size() != 2) {
        throw std::runtime_error("matmul: only 2D tensors supported");
    }
    if (a.shape[1] != b.shape[0]) {
        throw std::runtime_error("matmul: shape mismatch");
    }
    size_t m = a.shape[0];
    size_t k = a.shape[1];
    size_t n = b.shape[1];

    Tensor a_contig = a;
    Tensor b_contig = b;
    if (infer_layout(m, k, a.strides[0], a.strides[1]) == MatrixLayout::Unsupported) {
        a_contig = a.contiguous();
    }
    if (infer_layout(k, n, b.strides[0], b.strides[1]) == MatrixLayout::Unsupported) {
        b_contig = b.contiguous();
    }
    Tensor out({m, n}, a.device, a_contig.requires_grad || b_contig.requires_grad);
    float alpha = 1.0f;
    float beta = 0.0f;
    cublas_gemm_row_major(a_contig.get_raw_pointer(), b_contig.get_raw_pointer(), out.get_raw_pointer(),
                          static_cast<int>(m), static_cast<int>(n), static_cast<int>(k),
                          a_contig.strides[0], a_contig.strides[1],
                          b_contig.strides[0], b_contig.strides[1],
                          out.strides[0], out.strides[1],
                          CUBLAS_OP_N, CUBLAS_OP_N, alpha, beta);

    if (out.requires_grad && Tensor::is_grad_enabled()) {
        out.creator_node = std::make_shared<ComputeNode>();
        Tensor::add_input_nodes(out.creator_node, {a_contig, b_contig});
        auto out_grad_base = out.grad_base();
        size_t out_grad_offset = out.grad_offset;
        auto a_data_base = a_contig.data_base();
        size_t a_data_offset = a_contig.storage_offset;
        auto b_data_base = b_contig.data_base();
        size_t b_data_offset = b_contig.storage_offset;
        auto a_grad_base = a_contig.grad_base();
        size_t a_grad_offset = a_contig.grad_offset;
        auto b_grad_base = b_contig.grad_base();
        size_t b_grad_offset = b_contig.grad_offset;
        auto a_strides = a_contig.strides;
        auto b_strides = b_contig.strides;
        auto out_strides = out.strides;
        bool a_req = a_contig.requires_grad;
        bool b_req = b_contig.requires_grad;
        out.creator_node->backward_fn = [out_grad_base, out_grad_offset,
                                         a_data_base, a_data_offset,
                                         b_data_base, b_data_offset,
                                         a_grad_base, a_grad_offset,
                                         b_grad_base, b_grad_offset,
                                         a_strides, b_strides, out_strides,
                                         a_req, b_req, m, k, n]() mutable {
            if (!out_grad_base) {
                return;
            }
            float* out_grad_ptr = out_grad_base.get() + out_grad_offset;
            float* a_data_ptr = a_data_base.get() + a_data_offset;
            float* b_data_ptr = b_data_base.get() + b_data_offset;
            float* a_grad_ptr = a_grad_base ? (a_grad_base.get() + a_grad_offset) : nullptr;
            float* b_grad_ptr = b_grad_base ? (b_grad_base.get() + b_grad_offset) : nullptr;
            float alpha = 1.0f;
            float beta = 1.0f;
            if (a_req && a_grad_ptr) {
                cublas_gemm_row_major(out_grad_ptr, b_data_ptr, a_grad_ptr,
                                      static_cast<int>(m), static_cast<int>(k), static_cast<int>(n),
                                      out_strides[0], out_strides[1],
                                      b_strides[0], b_strides[1],
                                      a_strides[0], a_strides[1],
                                      CUBLAS_OP_N, CUBLAS_OP_T, alpha, beta);
            }
            if (b_req && b_grad_ptr) {
                cublas_gemm_row_major(a_data_ptr, out_grad_ptr, b_grad_ptr,
                                      static_cast<int>(k), static_cast<int>(n), static_cast<int>(m),
                                      a_strides[0], a_strides[1],
                                      out_strides[0], out_strides[1],
                                      b_strides[0], b_strides[1],
                                      CUBLAS_OP_T, CUBLAS_OP_N, alpha, beta);
            }
        };
    }
    return out;
}

Tensor matmul(const Tensor& a, const Tensor& b) {
    if (a.device != b.device) {
        throw std::runtime_error("Device mismatch");
    }
    if (a.device == Device::CPU) {
        return matmul_cpu_impl(a, b);
    }
    if (a.device == Device::CUDA) {
        return matmul_cuda_impl(a, b);
    }
    throw std::runtime_error("matmul: unsupported device");
}

Tensor relu_cpu_impl(const Tensor& a) {
    Tensor out(a.shape, a.device, a.requires_grad);
    std::vector<size_t> logical_strides = Tensor::compute_contiguous_strides(a.shape);
    std::vector<size_t> idx;
    for (size_t linear = 0; linear < a.numel(); ++linear) {
        linear_to_indices(linear, a.shape, logical_strides, idx);
        size_t a_off = indices_to_linear(idx, a.strides);
        size_t out_off = indices_to_linear(idx, out.strides);
        out.get_raw_pointer()[out_off] = std::max(0.0f, a.get_raw_pointer()[a_off]);
    }
    if (out.requires_grad && Tensor::is_grad_enabled()) {
        out.creator_node = std::make_shared<ComputeNode>();
        Tensor::add_input_nodes(out.creator_node, {a});
        auto out_grad_base = out.grad_base();
        size_t out_grad_offset = out.grad_offset;
        auto a_data_base = a.data_base();
        size_t a_data_offset = a.storage_offset;
        auto a_grad_base = a.grad_base();
        size_t a_grad_offset = a.grad_offset;
        auto a_shape = a.shape;
        auto a_strides = a.strides;
        auto out_strides = out.strides;
        out.creator_node->backward_fn = [out_grad_base, out_grad_offset,
                                         a_data_base, a_data_offset,
                                         a_grad_base, a_grad_offset,
                                         a_shape, a_strides, out_strides]() mutable {
            if (!out_grad_base || !a_grad_base) {
                return;
            }
            float* out_grad_ptr = out_grad_base.get() + out_grad_offset;
            float* a_data_ptr = a_data_base.get() + a_data_offset;
            float* a_grad_ptr = a_grad_base.get() + a_grad_offset;
            std::vector<size_t> logical_strides = Tensor::compute_contiguous_strides(a_shape);
            std::vector<size_t> idx;
            size_t a_numel = numel_from_shape(a_shape);
            for (size_t linear = 0; linear < a_numel; ++linear) {
                linear_to_indices(linear, a_shape, logical_strides, idx);
                size_t a_off = indices_to_linear(idx, a_strides);
                size_t out_off = indices_to_linear(idx, out_strides);
                float grad_val = (a_data_ptr[a_off] > 0.0f) ? out_grad_ptr[out_off] : 0.0f;
                Tensor::accumulate_grad_value(a_grad_ptr + a_off, grad_val);
            }
        };
    }
    return out;
}

Tensor relu_cuda_impl(const Tensor& a) {
    Tensor out(a.shape, a.device, a.requires_grad);
    size_t rank = a.shape.size();
    check_rank_within_kmaxdims(rank, "relu CUDA: rank exceeds kMaxDims");
    TensorMetadata in_meta = make_metadata(a, rank, false);
    TensorMetadata out_meta = make_metadata(out, rank, false);
    launch_relu_forward(a.get_raw_pointer(), out.get_raw_pointer(),
                        in_meta, out_meta, a.numel());
    if (out.requires_grad && Tensor::is_grad_enabled()) {
        out.creator_node = std::make_shared<ComputeNode>();
        Tensor::add_input_nodes(out.creator_node, {a});
        auto out_grad_base = out.grad_base();
        size_t out_grad_offset = out.grad_offset;
        auto a_data_base = a.data_base();
        size_t a_data_offset = a.storage_offset;
        auto a_grad_base = a.grad_base();
        size_t a_grad_offset = a.grad_offset;
        auto a_shape = a.shape;
        auto a_strides = a.strides;
        auto out_strides = out.strides;
        out.creator_node->backward_fn = [out_grad_base, out_grad_offset,
                                         a_data_base, a_data_offset,
                                         a_grad_base, a_grad_offset,
                                         a_shape, a_strides, out_strides]() mutable {
            if (!out_grad_base || !a_grad_base) {
                return;
            }
            float* out_grad_ptr = out_grad_base.get() + out_grad_offset;
            float* a_data_ptr = a_data_base.get() + a_data_offset;
            float* a_grad_ptr = a_grad_base.get() + a_grad_offset;
            size_t rank = a_shape.size();
            check_rank_within_kmaxdims(rank, "relu backward CUDA: rank exceeds kMaxDims");
            TensorView in_view;
            in_view.shape = a_shape;
            in_view.strides = a_strides;
            TensorView out_view;
            out_view.shape = a_shape;
            out_view.strides = out_strides;
            TensorMetadata in_meta = make_metadata(in_view, rank, false);
            TensorMetadata out_meta = make_metadata(out_view, rank, false);
            launch_relu_backward(a_data_ptr, out_grad_ptr, a_grad_ptr,
                                 in_meta, out_meta, numel_from_shape(a_shape));
        };
    }
    return out;
}

Tensor relu(const Tensor& a) {
    if (a.device == Device::CPU) {
        return relu_cpu_impl(a);
    }
    if (a.device == Device::CUDA) {
        return relu_cuda_impl(a);
    }
    throw std::runtime_error("relu: unsupported device");
}

Tensor sigmoid_cpu_impl(const Tensor& a) {
    Tensor out(a.shape, a.device, a.requires_grad);
    std::vector<size_t> logical_strides = Tensor::compute_contiguous_strides(a.shape);
    std::vector<size_t> idx;
    for (size_t linear = 0; linear < a.numel(); ++linear) {
        linear_to_indices(linear, a.shape, logical_strides, idx);
        size_t a_off = indices_to_linear(idx, a.strides);
        size_t out_off = indices_to_linear(idx, out.strides);
        float v = a.get_raw_pointer()[a_off];
        out.get_raw_pointer()[out_off] = 1.0f / (1.0f + std::exp(-v));
    }
    if (out.requires_grad && Tensor::is_grad_enabled()) {
        out.creator_node = std::make_shared<ComputeNode>();
        Tensor::add_input_nodes(out.creator_node, {a});
        auto out_data_base = out.data_base();
        size_t out_data_offset = out.storage_offset;
        auto out_grad_base = out.grad_base();
        size_t out_grad_offset = out.grad_offset;
        auto a_grad_base = a.grad_base();
        size_t a_grad_offset = a.grad_offset;
        auto a_shape = a.shape;
        auto a_strides = a.strides;
        auto out_strides = out.strides;
        out.creator_node->backward_fn = [out_data_base, out_data_offset,
                                         out_grad_base, out_grad_offset,
                                         a_grad_base, a_grad_offset,
                                         a_shape, a_strides, out_strides]() mutable {
            if (!out_grad_base || !out_data_base || !a_grad_base) {
                return;
            }
            float* out_data_ptr = out_data_base.get() + out_data_offset;
            float* out_grad_ptr = out_grad_base.get() + out_grad_offset;
            float* a_grad_ptr = a_grad_base.get() + a_grad_offset;
            std::vector<size_t> logical_strides = Tensor::compute_contiguous_strides(a_shape);
            std::vector<size_t> idx;
            size_t a_numel = numel_from_shape(a_shape);
            for (size_t linear = 0; linear < a_numel; ++linear) {
                linear_to_indices(linear, a_shape, logical_strides, idx);
                size_t a_off = indices_to_linear(idx, a_strides);
                size_t out_off = indices_to_linear(idx, out_strides);
                float y = out_data_ptr[out_off];
                Tensor::accumulate_grad_value(a_grad_ptr + a_off,
                                              out_grad_ptr[out_off] * y * (1.0f - y));
            }
        };
    }
    return out;
}

Tensor sigmoid_cuda_impl(const Tensor& a) {
    Tensor out(a.shape, a.device, a.requires_grad);
    size_t rank = a.shape.size();
    check_rank_within_kmaxdims(rank, "sigmoid CUDA: rank exceeds kMaxDims");
    TensorMetadata in_meta = make_metadata(a, rank, false);
    TensorMetadata out_meta = make_metadata(out, rank, false);
    launch_sigmoid_forward(a.get_raw_pointer(), out.get_raw_pointer(),
                           in_meta, out_meta, a.numel());
    if (out.requires_grad && Tensor::is_grad_enabled()) {
        out.creator_node = std::make_shared<ComputeNode>();
        Tensor::add_input_nodes(out.creator_node, {a});
        auto out_data_base = out.data_base();
        size_t out_data_offset = out.storage_offset;
        auto out_grad_base = out.grad_base();
        size_t out_grad_offset = out.grad_offset;
        auto a_grad_base = a.grad_base();
        size_t a_grad_offset = a.grad_offset;
        auto a_shape = a.shape;
        auto a_strides = a.strides;
        auto out_strides = out.strides;
        out.creator_node->backward_fn = [out_data_base, out_data_offset,
                                         out_grad_base, out_grad_offset,
                                         a_grad_base, a_grad_offset,
                                         a_shape, a_strides, out_strides]() mutable {
            if (!out_grad_base || !out_data_base || !a_grad_base) {
                return;
            }
            float* out_data_ptr = out_data_base.get() + out_data_offset;
            float* out_grad_ptr = out_grad_base.get() + out_grad_offset;
            float* a_grad_ptr = a_grad_base.get() + a_grad_offset;
            size_t rank = a_shape.size();
            check_rank_within_kmaxdims(rank, "sigmoid backward CUDA: rank exceeds kMaxDims");
            TensorView out_view;
            out_view.shape = a_shape;
            out_view.strides = out_strides;
            TensorView in_view;
            in_view.shape = a_shape;
            in_view.strides = a_strides;
            TensorMetadata out_meta = make_metadata(out_view, rank, false);
            TensorMetadata in_meta = make_metadata(in_view, rank, false);
            launch_sigmoid_backward(out_data_ptr, out_grad_ptr, a_grad_ptr,
                                    out_meta, in_meta, numel_from_shape(a_shape));
        };
    }
    return out;
}

Tensor sigmoid(const Tensor& a) {
    if (a.device == Device::CPU) {
        return sigmoid_cpu_impl(a);
    }
    if (a.device == Device::CUDA) {
        return sigmoid_cuda_impl(a);
    }
    throw std::runtime_error("sigmoid: unsupported device");
}

Tensor softmax_cpu_impl(const Tensor& a) {
    if (a.shape.empty()) {
        throw std::runtime_error("softmax: input must have at least 1 dimension");
    }
    size_t cols = a.shape.back();
    size_t outer = numel_excluding_last(a.shape);
    Tensor out(a.shape, a.device, a.requires_grad);
    std::vector<size_t> logical_strides = Tensor::compute_contiguous_strides(a.shape);
    std::vector<size_t> idx;
    for (size_t row = 0; row < outer; ++row) {
        float max_val = -1e20f;
        for (size_t c = 0; c < cols; ++c) {
            size_t linear = row * cols + c;
            linear_to_indices(linear, a.shape, logical_strides, idx);
            size_t a_off = indices_to_linear(idx, a.strides);
            float v = a.get_raw_pointer()[a_off];
            if (v > max_val) {
                max_val = v;
            }
        }
        float sum = 0.0f;
        for (size_t c = 0; c < cols; ++c) {
            size_t linear = row * cols + c;
            linear_to_indices(linear, a.shape, logical_strides, idx);
            size_t a_off = indices_to_linear(idx, a.strides);
            sum += std::exp(a.get_raw_pointer()[a_off] - max_val);
        }
        for (size_t c = 0; c < cols; ++c) {
            size_t linear = row * cols + c;
            linear_to_indices(linear, a.shape, logical_strides, idx);
            size_t a_off = indices_to_linear(idx, a.strides);
            size_t out_off = indices_to_linear(idx, out.strides);
            out.get_raw_pointer()[out_off] = std::exp(a.get_raw_pointer()[a_off] - max_val) / sum;
        }
    }

    if (out.requires_grad && Tensor::is_grad_enabled()) {
        out.creator_node = std::make_shared<ComputeNode>();
        Tensor::add_input_nodes(out.creator_node, {a});
        auto out_data_base = out.data_base();
        size_t out_data_offset = out.storage_offset;
        auto out_grad_base = out.grad_base();
        size_t out_grad_offset = out.grad_offset;
        auto a_grad_base = a.grad_base();
        size_t a_grad_offset = a.grad_offset;
        auto a_shape = a.shape;
        auto a_strides = a.strides;
        auto out_strides = out.strides;
        out.creator_node->backward_fn = [out_data_base, out_data_offset,
                                         out_grad_base, out_grad_offset,
                                         a_grad_base, a_grad_offset,
                                         a_shape, a_strides, out_strides,
                                         outer, cols]() mutable {
            if (!out_grad_base || !out_data_base || !a_grad_base) {
                return;
            }
            float* out_data_ptr = out_data_base.get() + out_data_offset;
            float* out_grad_ptr = out_grad_base.get() + out_grad_offset;
            float* a_grad_ptr = a_grad_base.get() + a_grad_offset;
            std::vector<size_t> logical_strides = Tensor::compute_contiguous_strides(a_shape);
            std::vector<size_t> idx;
            for (size_t row = 0; row < outer; ++row) {
                float dot = 0.0f;
                for (size_t c = 0; c < cols; ++c) {
                    size_t linear = row * cols + c;
                    linear_to_indices(linear, a_shape, logical_strides, idx);
                    size_t out_off = indices_to_linear(idx, out_strides);
                    float y = out_data_ptr[out_off];
                    float g = out_grad_ptr[out_off];
                    dot += y * g;
                }
                for (size_t c = 0; c < cols; ++c) {
                    size_t linear = row * cols + c;
                    linear_to_indices(linear, a_shape, logical_strides, idx);
                    size_t out_off = indices_to_linear(idx, out_strides);
                    size_t a_off = indices_to_linear(idx, a_strides);
                    float y = out_data_ptr[out_off];
                    float g = out_grad_ptr[out_off];
                    Tensor::accumulate_grad_value(a_grad_ptr + a_off, y * (g - dot));
                }
            }
        };
    }
    return out;
}

Tensor softmax_cuda_impl(const Tensor& a) {
    if (a.shape.empty()) {
        throw std::runtime_error("softmax: input must have at least 1 dimension");
    }
    size_t cols = a.shape.back();
    size_t outer = numel_excluding_last(a.shape);
    Tensor out(a.shape, a.device, a.requires_grad);
    size_t rank = a.shape.size();
    check_rank_within_kmaxdims(rank, "softmax CUDA: rank exceeds kMaxDims");
    TensorMetadata in_meta = make_metadata(a, rank, false);
    TensorMetadata out_meta = make_metadata(out, rank, false);
    launch_softmax_forward(a.get_raw_pointer(), out.get_raw_pointer(),
                           in_meta, out_meta, outer, cols);

    if (out.requires_grad && Tensor::is_grad_enabled()) {
        out.creator_node = std::make_shared<ComputeNode>();
        Tensor::add_input_nodes(out.creator_node, {a});
        auto out_data_base = out.data_base();
        size_t out_data_offset = out.storage_offset;
        auto out_grad_base = out.grad_base();
        size_t out_grad_offset = out.grad_offset;
        auto a_grad_base = a.grad_base();
        size_t a_grad_offset = a.grad_offset;
        auto a_shape = a.shape;
        auto a_strides = a.strides;
        auto out_strides = out.strides;
        out.creator_node->backward_fn = [out_data_base, out_data_offset,
                                         out_grad_base, out_grad_offset,
                                         a_grad_base, a_grad_offset,
                                         a_shape, a_strides, out_strides,
                                         outer, cols]() mutable {
            if (!out_grad_base || !out_data_base || !a_grad_base) {
                return;
            }
            float* out_data_ptr = out_data_base.get() + out_data_offset;
            float* out_grad_ptr = out_grad_base.get() + out_grad_offset;
            float* a_grad_ptr = a_grad_base.get() + a_grad_offset;
            size_t rank = a_shape.size();
            check_rank_within_kmaxdims(rank, "softmax backward CUDA: rank exceeds kMaxDims");
            TensorView out_view;
            out_view.shape = a_shape;
            out_view.strides = out_strides;
            TensorView in_view;
            in_view.shape = a_shape;
            in_view.strides = a_strides;
            TensorMetadata out_meta = make_metadata(out_view, rank, false);
            TensorMetadata in_meta = make_metadata(in_view, rank, false);
            launch_softmax_backward(out_data_ptr, out_grad_ptr, a_grad_ptr,
                                    out_meta, in_meta, outer, cols);
        };
    }
    return out;
}

Tensor softmax(const Tensor& a) {
    if (a.device == Device::CPU) {
        return softmax_cpu_impl(a);
    }
    if (a.device == Device::CUDA) {
        return softmax_cuda_impl(a);
    }
    throw std::runtime_error("softmax: unsupported device");
}

Tensor cross_entropy_loss_cpu_impl(const Tensor& logits, const Tensor& target,
                                   size_t outer, size_t cols) {
    Tensor out({1}, logits.device, logits.requires_grad);
    std::vector<size_t> logical_strides = Tensor::compute_contiguous_strides(logits.shape);
    std::vector<size_t> target_strides = Tensor::compute_contiguous_strides(target.shape);
    std::vector<size_t> idx;
    std::vector<size_t> t_idx;
    float sum_loss = 0.0f;
    for (size_t row = 0; row < outer; ++row) {
        float max_val = -1e20f;
        for (size_t c = 0; c < cols; ++c) {
            size_t linear = row * cols + c;
            linear_to_indices(linear, logits.shape, logical_strides, idx);
            size_t off = indices_to_linear(idx, logits.strides);
            float v = logits.get_raw_pointer()[off];
            if (v > max_val) {
                max_val = v;
            }
        }
        float sum = 0.0f;
        for (size_t c = 0; c < cols; ++c) {
            size_t linear = row * cols + c;
            linear_to_indices(linear, logits.shape, logical_strides, idx);
            size_t off = indices_to_linear(idx, logits.strides);
            sum += std::exp(logits.get_raw_pointer()[off] - max_val);
        }
        linear_to_indices(row, target.shape, target_strides, t_idx);
        size_t t_off = indices_to_linear(t_idx, target.strides);
        int label = static_cast<int>(target.get_raw_pointer()[t_off]);
        if (label < 0) {
            label = 0;
        } else if (label >= static_cast<int>(cols)) {
            label = static_cast<int>(cols) - 1;
        }
        size_t linear = row * cols + static_cast<size_t>(label);
        linear_to_indices(linear, logits.shape, logical_strides, idx);
        size_t logit_off = indices_to_linear(idx, logits.strides);
        float logit = logits.get_raw_pointer()[logit_off];
        float loss = -(logit - max_val) + std::log(sum);
        sum_loss += loss;
    }
    out.get_raw_pointer()[0] = sum_loss / static_cast<float>(outer);

    if (out.requires_grad && Tensor::is_grad_enabled()) {
        out.creator_node = std::make_shared<ComputeNode>();
        Tensor::add_input_nodes(out.creator_node, {logits, target});
        auto out_grad_base = out.grad_base();
        size_t out_grad_offset = out.grad_offset;
        auto logits_data_base = logits.data_base();
        size_t logits_data_offset = logits.storage_offset;
        auto logits_grad_base = logits.grad_base();
        size_t logits_grad_offset = logits.grad_offset;
        auto target_data_base = target.data_base();
        size_t target_data_offset = target.storage_offset;
        auto logits_shape = logits.shape;
        auto logits_strides = logits.strides;
        auto target_shape = target.shape;
        auto target_strides_actual = target.strides;
        out.creator_node->backward_fn = [out_grad_base, out_grad_offset,
                                         logits_data_base, logits_data_offset,
                                         logits_grad_base, logits_grad_offset,
                                         target_data_base, target_data_offset,
                                         logits_shape, logits_strides,
                                         target_shape, target_strides_actual,
                                         outer, cols]() mutable {
            if (!out_grad_base || !logits_grad_base || !logits_data_base || !target_data_base) {
                return;
            }
            float* out_grad_ptr = out_grad_base.get() + out_grad_offset;
            float* logits_data_ptr = logits_data_base.get() + logits_data_offset;
            float* logits_grad_ptr = logits_grad_base.get() + logits_grad_offset;
            float* target_data_ptr = target_data_base.get() + target_data_offset;
            std::vector<size_t> logical_strides = Tensor::compute_contiguous_strides(logits_shape);
            std::vector<size_t> target_strides_logical = Tensor::compute_contiguous_strides(target_shape);
            std::vector<size_t> idx;
            std::vector<size_t> t_idx;
            float g = out_grad_ptr[0] / static_cast<float>(outer);
            for (size_t row = 0; row < outer; ++row) {
                float max_val = -1e20f;
                for (size_t c = 0; c < cols; ++c) {
                    size_t linear = row * cols + c;
                    linear_to_indices(linear, logits_shape, logical_strides, idx);
                    size_t off = indices_to_linear(idx, logits_strides);
                    float v = logits_data_ptr[off];
                    if (v > max_val) {
                        max_val = v;
                    }
                }
                float sum = 0.0f;
                for (size_t c = 0; c < cols; ++c) {
                    size_t linear = row * cols + c;
                    linear_to_indices(linear, logits_shape, logical_strides, idx);
                    size_t off = indices_to_linear(idx, logits_strides);
                    sum += std::exp(logits_data_ptr[off] - max_val);
                }
                linear_to_indices(row, target_shape, target_strides_logical, t_idx);
                size_t t_off = indices_to_linear(t_idx, target_strides_actual);
                int label = static_cast<int>(target_data_ptr[t_off]);
                if (label < 0) {
                    label = 0;
                } else if (label >= static_cast<int>(cols)) {
                    label = static_cast<int>(cols) - 1;
                }
                for (size_t c = 0; c < cols; ++c) {
                    size_t linear = row * cols + c;
                    linear_to_indices(linear, logits_shape, logical_strides, idx);
                    size_t off = indices_to_linear(idx, logits_strides);
                    float p = std::exp(logits_data_ptr[off] - max_val) / sum;
                    float grad = (static_cast<int>(c) == label) ? (p - 1.0f) : p;
                    Tensor::accumulate_grad_value(logits_grad_ptr + off, g * grad);
                }
            }
        };
    }
    return out;
}

Tensor cross_entropy_loss_cuda_impl(const Tensor& logits, const Tensor& target,
                                    size_t outer, size_t cols) {
    Tensor out({1}, logits.device, logits.requires_grad);
    size_t rank = logits.shape.size();
    check_rank_within_kmaxdims(rank, "cross_entropy_loss CUDA: rank exceeds kMaxDims");
    TensorMetadata logits_meta = make_metadata(logits, rank, false);
    TensorMetadata target_meta = make_metadata(target, rank - 1, false);
    launch_cross_entropy_forward(logits.get_raw_pointer(), target.get_raw_pointer(), out.get_raw_pointer(),
                                 logits_meta, target_meta, outer, cols);

    if (out.requires_grad && Tensor::is_grad_enabled()) {
        out.creator_node = std::make_shared<ComputeNode>();
        Tensor::add_input_nodes(out.creator_node, {logits, target});
        auto out_grad_base = out.grad_base();
        size_t out_grad_offset = out.grad_offset;
        auto logits_data_base = logits.data_base();
        size_t logits_data_offset = logits.storage_offset;
        auto logits_grad_base = logits.grad_base();
        size_t logits_grad_offset = logits.grad_offset;
        auto target_data_base = target.data_base();
        size_t target_data_offset = target.storage_offset;
        auto logits_shape = logits.shape;
        auto logits_strides = logits.strides;
        auto target_shape = target.shape;
        auto target_strides_actual = target.strides;
        out.creator_node->backward_fn = [out_grad_base, out_grad_offset,
                                         logits_data_base, logits_data_offset,
                                         logits_grad_base, logits_grad_offset,
                                         target_data_base, target_data_offset,
                                         logits_shape, logits_strides,
                                         target_shape, target_strides_actual,
                                         outer, cols]() mutable {
            if (!out_grad_base || !logits_grad_base || !logits_data_base || !target_data_base) {
                return;
            }
            float* out_grad_ptr = out_grad_base.get() + out_grad_offset;
            float* logits_data_ptr = logits_data_base.get() + logits_data_offset;
            float* logits_grad_ptr = logits_grad_base.get() + logits_grad_offset;
            float* target_data_ptr = target_data_base.get() + target_data_offset;
            size_t rank = logits_shape.size();
            check_rank_within_kmaxdims(rank, "cross_entropy_loss backward CUDA: rank exceeds kMaxDims");
            TensorView logits_view;
            logits_view.shape = logits_shape;
            logits_view.strides = logits_strides;
            TensorView target_view;
            target_view.shape = target_shape;
            target_view.strides = target_strides_actual;
            TensorMetadata logits_meta = make_metadata(logits_view, rank, false);
            TensorMetadata target_meta = make_metadata(target_view, rank - 1, false);
            launch_cross_entropy_backward(logits_data_ptr, target_data_ptr, out_grad_ptr,
                                          logits_grad_ptr, logits_meta, target_meta, outer, cols);
        };
    }
    return out;
}

Tensor cross_entropy_loss(const Tensor& logits, const Tensor& target) {
    if (logits.device != target.device) {
        throw std::runtime_error("cross_entropy_loss: device mismatch");
    }
    if (logits.shape.empty()) {
        throw std::runtime_error("cross_entropy_loss: logits must have at least 1 dimension");
    }
    size_t cols = logits.shape.back();
    size_t outer = numel_excluding_last(logits.shape);
    std::vector<size_t> outer_shape = logits.shape;
    outer_shape.pop_back();
    if (target.shape != outer_shape) {
        throw std::runtime_error("cross_entropy_loss: target shape must match logits shape excluding last dim");
    }
    if (logits.device == Device::CPU) {
        return cross_entropy_loss_cpu_impl(logits, target, outer, cols);
    }
    if (logits.device == Device::CUDA) {
        return cross_entropy_loss_cuda_impl(logits, target, outer, cols);
    }
    throw std::runtime_error("cross_entropy_loss: unsupported device");
}

Tensor mse_loss_cpu_impl(const Tensor& pred, const Tensor& target) {
    Tensor out({1}, pred.device, pred.requires_grad || target.requires_grad);
    float sum = 0.0f;
    std::vector<size_t> logical_strides = Tensor::compute_contiguous_strides(pred.shape);
    std::vector<size_t> idx;
    for (size_t linear = 0; linear < pred.numel(); ++linear) {
        linear_to_indices(linear, pred.shape, logical_strides, idx);
        size_t p_off = indices_to_linear(idx, pred.strides);
        size_t t_off = indices_to_linear(idx, target.strides);
        float diff = pred.get_raw_pointer()[p_off] - target.get_raw_pointer()[t_off];
        sum += diff * diff;
    }
    float mean = sum / static_cast<float>(pred.numel());
    out.get_raw_pointer()[0] = mean;

    if (out.requires_grad && Tensor::is_grad_enabled()) {
        out.creator_node = std::make_shared<ComputeNode>();
        Tensor::add_input_nodes(out.creator_node, {pred, target});
        auto out_grad_base = out.grad_base();
        size_t out_grad_offset = out.grad_offset;
        auto pred_data_base = pred.data_base();
        size_t pred_data_offset = pred.storage_offset;
        auto target_data_base = target.data_base();
        size_t target_data_offset = target.storage_offset;
        auto pred_grad_base = pred.grad_base();
        size_t pred_grad_offset = pred.grad_offset;
        auto target_grad_base = target.grad_base();
        size_t target_grad_offset = target.grad_offset;
        auto pred_shape = pred.shape;
        auto pred_strides = pred.strides;
        auto target_shape = target.shape;
        auto target_strides = target.strides;
        bool pred_req = pred.requires_grad;
        bool target_req = target.requires_grad;
        out.creator_node->backward_fn = [out_grad_base, out_grad_offset,
                                         pred_data_base, pred_data_offset,
                                         target_data_base, target_data_offset,
                                         pred_grad_base, pred_grad_offset,
                                         target_grad_base, target_grad_offset,
                                         pred_shape, pred_strides,
                                         target_shape, target_strides,
                                         pred_req, target_req]() mutable {
            if (!out_grad_base || !pred_data_base || !target_data_base) {
                return;
            }
            float* out_grad_ptr = out_grad_base.get() + out_grad_offset;
            float* pred_data_ptr = pred_data_base.get() + pred_data_offset;
            float* target_data_ptr = target_data_base.get() + target_data_offset;
            float* pred_grad_ptr = pred_grad_base ? (pred_grad_base.get() + pred_grad_offset) : nullptr;
            float* target_grad_ptr = target_grad_base ? (target_grad_base.get() + target_grad_offset) : nullptr;
            float g = out_grad_ptr[0];
            float scale = 2.0f / static_cast<float>(numel_from_shape(pred_shape));

            if (pred_req && pred_grad_ptr) {
                std::vector<size_t> logical_strides = Tensor::compute_contiguous_strides(pred_shape);
                std::vector<size_t> idx;
                size_t pred_numel = numel_from_shape(pred_shape);
                for (size_t linear = 0; linear < pred_numel; ++linear) {
                    linear_to_indices(linear, pred_shape, logical_strides, idx);
                    size_t p_off = indices_to_linear(idx, pred_strides);
                    size_t t_off = indices_to_linear(idx, target_strides);
                    float diff = pred_data_ptr[p_off] - target_data_ptr[t_off];
                    Tensor::accumulate_grad_value(pred_grad_ptr + p_off, g * scale * diff);
                }
            }
            if (target_req && target_grad_ptr) {
                std::vector<size_t> logical_strides = Tensor::compute_contiguous_strides(target_shape);
                std::vector<size_t> idx;
                size_t target_numel = numel_from_shape(target_shape);
                for (size_t linear = 0; linear < target_numel; ++linear) {
                    linear_to_indices(linear, target_shape, logical_strides, idx);
                    size_t p_off = indices_to_linear(idx, pred_strides);
                    size_t t_off = indices_to_linear(idx, target_strides);
                    float diff = pred_data_ptr[p_off] - target_data_ptr[t_off];
                    target_grad_ptr[t_off] -= g * scale * diff;
                }
            }
        };
    }
    return out;
}

Tensor mse_loss_cuda_impl(const Tensor& pred, const Tensor& target) {
    Tensor out({1}, pred.device, pred.requires_grad || target.requires_grad);
    size_t out_rank = std::max(pred.shape.size(), target.shape.size());
    check_rank_within_kmaxdims(out_rank, "mse_loss CUDA: rank exceeds kMaxDims");
    TensorMetadata p_meta = make_metadata(pred, out_rank, false);
    TensorMetadata t_meta = make_metadata(target, out_rank, false);
    launch_mse_forward(pred.get_raw_pointer(), target.get_raw_pointer(), out.get_raw_pointer(),
                       p_meta, t_meta, pred.numel());

    if (out.requires_grad && Tensor::is_grad_enabled()) {
        out.creator_node = std::make_shared<ComputeNode>();
        Tensor::add_input_nodes(out.creator_node, {pred, target});
        auto out_grad_base = out.grad_base();
        size_t out_grad_offset = out.grad_offset;
        auto pred_data_base = pred.data_base();
        size_t pred_data_offset = pred.storage_offset;
        auto target_data_base = target.data_base();
        size_t target_data_offset = target.storage_offset;
        auto pred_grad_base = pred.grad_base();
        size_t pred_grad_offset = pred.grad_offset;
        auto target_grad_base = target.grad_base();
        size_t target_grad_offset = target.grad_offset;
        auto pred_shape = pred.shape;
        auto pred_strides = pred.strides;
        auto target_shape = target.shape;
        auto target_strides = target.strides;
        bool pred_req = pred.requires_grad;
        bool target_req = target.requires_grad;
        out.creator_node->backward_fn = [out_grad_base, out_grad_offset,
                                         pred_data_base, pred_data_offset,
                                         target_data_base, target_data_offset,
                                         pred_grad_base, pred_grad_offset,
                                         target_grad_base, target_grad_offset,
                                         pred_shape, pred_strides,
                                         target_shape, target_strides,
                                         pred_req, target_req]() mutable {
            if (!out_grad_base || !pred_data_base || !target_data_base) {
                return;
            }
            float* out_grad_ptr = out_grad_base.get() + out_grad_offset;
            float* pred_data_ptr = pred_data_base.get() + pred_data_offset;
            float* target_data_ptr = target_data_base.get() + target_data_offset;
            float* pred_grad_ptr = pred_grad_base ? (pred_grad_base.get() + pred_grad_offset) : nullptr;
            float* target_grad_ptr = target_grad_base ? (target_grad_base.get() + target_grad_offset) : nullptr;
            size_t out_rank = std::max(pred_shape.size(), target_shape.size());
            check_rank_within_kmaxdims(out_rank, "mse_loss backward CUDA: rank exceeds kMaxDims");
            TensorView p_view;
            p_view.shape = pred_shape;
            p_view.strides = pred_strides;
            TensorView t_view;
            t_view.shape = target_shape;
            t_view.strides = target_strides;
            TensorMetadata p_meta = make_metadata(p_view, out_rank, false);
            TensorMetadata t_meta = make_metadata(t_view, out_rank, false);
            launch_mse_backward(pred_data_ptr, target_data_ptr, out_grad_ptr,
                                pred_req ? pred_grad_ptr : nullptr,
                                target_req ? target_grad_ptr : nullptr,
                                p_meta, t_meta, numel_from_shape(pred_shape));
        };
    }
    return out;
}

Tensor mse_loss(const Tensor& pred, const Tensor& target) {
    if (pred.device != target.device) {
        throw std::runtime_error("Device mismatch");
    }
    if (pred.numel() != target.numel()) {
        throw std::runtime_error("mse_loss: size mismatch");
    }
    if (pred.device == Device::CPU) {
        return mse_loss_cpu_impl(pred, target);
    }
    if (pred.device == Device::CUDA) {
        return mse_loss_cuda_impl(pred, target);
    }
    throw std::runtime_error("mse_loss: unsupported device");
}

Tensor transpose_cpu_impl(const Tensor& a, size_t dim0, size_t dim1) {
    Tensor out = a.transpose(dim0, dim1);
    if (out.requires_grad && Tensor::is_grad_enabled()) {
        out.creator_node = std::make_shared<ComputeNode>();
        Tensor::add_input_nodes(out.creator_node, {a});
        auto out_grad_base = out.grad_base();
        size_t out_grad_offset = out.grad_offset;
        auto a_grad_base = a.grad_base();
        size_t a_grad_offset = a.grad_offset;
        auto a_shape = a.shape;
        auto a_strides = a.strides;
        out.creator_node->backward_fn = [out_grad_base, out_grad_offset,
                                         a_grad_base, a_grad_offset,
                                         a_shape, a_strides]() mutable {
            if (!out_grad_base || !a_grad_base) {
                return;
            }
            float* out_grad_ptr = out_grad_base.get() + out_grad_offset;
            float* a_grad_ptr = a_grad_base.get() + a_grad_offset;
            std::vector<size_t> logical_strides = Tensor::compute_contiguous_strides(a_shape);
            std::vector<size_t> idx;
            size_t a_numel = numel_from_shape(a_shape);
            for (size_t linear = 0; linear < a_numel; ++linear) {
                linear_to_indices(linear, a_shape, logical_strides, idx);
                size_t off = indices_to_linear(idx, a_strides);
                Tensor::accumulate_grad_value(a_grad_ptr + off, out_grad_ptr[off]);
            }
        };
    }
    return out;
}

Tensor transpose_cuda_impl(const Tensor& a, size_t dim0, size_t dim1) {
    Tensor out = a.transpose(dim0, dim1);
    if (out.requires_grad && Tensor::is_grad_enabled()) {
        out.creator_node = std::make_shared<ComputeNode>();
        Tensor::add_input_nodes(out.creator_node, {a});
        auto out_grad_base = out.grad_base();
        size_t out_grad_offset = out.grad_offset;
        auto a_grad_base = a.grad_base();
        size_t a_grad_offset = a.grad_offset;
        auto a_shape = a.shape;
        auto a_strides = a.strides;
        out.creator_node->backward_fn = [out_grad_base, out_grad_offset,
                                         a_grad_base, a_grad_offset,
                                         a_shape, a_strides]() mutable {
            if (!out_grad_base || !a_grad_base) {
                return;
            }
            float* out_grad_ptr = out_grad_base.get() + out_grad_offset;
            float* a_grad_ptr = a_grad_base.get() + a_grad_offset;
            size_t rank = a_shape.size();
            check_rank_within_kmaxdims(rank, "transpose backward CUDA: rank exceeds kMaxDims");
            TensorView a_view;
            a_view.shape = a_shape;
            a_view.strides = a_strides;
            TensorMetadata meta = make_metadata(a_view, rank, false);
            launch_add_inplace_nd(out_grad_ptr, a_grad_ptr, meta, numel_from_shape(a_shape));
        };
    }
    return out;
}

Tensor transpose(const Tensor& a, size_t dim0, size_t dim1) {
    if (a.device == Device::CPU) {
        return transpose_cpu_impl(a, dim0, dim1);
    }
    if (a.device == Device::CUDA) {
        return transpose_cuda_impl(a, dim0, dim1);
    }
    throw std::runtime_error("transpose: unsupported device");
}

Tensor& add_(Tensor& a, const Tensor& b) {
    if (a.device != b.device) {
        throw std::runtime_error("add_: device mismatch");
    }
    std::vector<size_t> out_shape = broadcast_shape(a.shape, b.shape);
    if (out_shape != a.shape) {
        throw std::runtime_error("add_: broadcast result must match lhs shape");
    }
    if (a.device == Device::CPU) {
        std::vector<size_t> logical_strides = Tensor::compute_contiguous_strides(a.shape);
        std::vector<size_t> idx;
        std::vector<size_t> b_idx(b.shape.size(), 0);
        size_t rank_a = a.shape.size();
        size_t rank_b = b.shape.size();
        size_t offset_b = rank_a - rank_b;
        for (size_t linear = 0; linear < a.numel(); ++linear) {
            linear_to_indices(linear, a.shape, logical_strides, idx);
            for (size_t i = 0; i < rank_b; ++i) {
                size_t out_dim_idx = i + offset_b;
                b_idx[i] = (b.shape[i] == 1) ? 0 : idx[out_dim_idx];
            }
            size_t a_off = indices_to_linear(idx, a.strides);
            size_t b_off = indices_to_linear(b_idx, b.strides);
            a.get_raw_pointer()[a_off] += b.get_raw_pointer()[b_off];
        }
    } else if (a.device == Device::CUDA) {
        size_t rank = a.shape.size();
        check_rank_within_kmaxdims(rank, "add_ CUDA: rank exceeds kMaxDims");
        TensorMetadata a_meta = make_metadata(a, rank, false);
        TensorMetadata b_meta = make_metadata(b, rank, true);
        launch_add_inplace_broadcast(b.get_raw_pointer(), a.get_raw_pointer(),
                                     b_meta, a_meta, a.numel());
    } else {
        throw std::runtime_error("add_: unsupported device");
    }
    a.bump_version();
    return a;
}

Tensor& mul_(Tensor& a, const Tensor& b) {
    if (a.device != b.device) {
        throw std::runtime_error("mul_: device mismatch");
    }
    std::vector<size_t> out_shape = broadcast_shape(a.shape, b.shape);
    if (out_shape != a.shape) {
        throw std::runtime_error("mul_: broadcast result must match lhs shape");
    }
    if (a.device == Device::CPU) {
        std::vector<size_t> logical_strides = Tensor::compute_contiguous_strides(a.shape);
        std::vector<size_t> idx;
        std::vector<size_t> b_idx(b.shape.size(), 0);
        size_t rank_a = a.shape.size();
        size_t rank_b = b.shape.size();
        size_t offset_b = rank_a - rank_b;
        for (size_t linear = 0; linear < a.numel(); ++linear) {
            linear_to_indices(linear, a.shape, logical_strides, idx);
            for (size_t i = 0; i < rank_b; ++i) {
                size_t out_dim_idx = i + offset_b;
                b_idx[i] = (b.shape[i] == 1) ? 0 : idx[out_dim_idx];
            }
            size_t a_off = indices_to_linear(idx, a.strides);
            size_t b_off = indices_to_linear(b_idx, b.strides);
            a.get_raw_pointer()[a_off] *= b.get_raw_pointer()[b_off];
        }
    } else if (a.device == Device::CUDA) {
        size_t rank = a.shape.size();
        check_rank_within_kmaxdims(rank, "mul_ CUDA: rank exceeds kMaxDims");
        TensorMetadata a_meta = make_metadata(a, rank, false);
        TensorMetadata b_meta = make_metadata(b, rank, true);
        launch_mul_inplace_broadcast(b.get_raw_pointer(), a.get_raw_pointer(),
                                     b_meta, a_meta, a.numel());
    } else {
        throw std::runtime_error("mul_: unsupported device");
    }
    a.bump_version();
    return a;
}

Tensor& relu_(Tensor& a) {
    if (a.device == Device::CPU) {
        std::vector<size_t> logical_strides = Tensor::compute_contiguous_strides(a.shape);
        std::vector<size_t> idx;
        for (size_t linear = 0; linear < a.numel(); ++linear) {
            linear_to_indices(linear, a.shape, logical_strides, idx);
            size_t a_off = indices_to_linear(idx, a.strides);
            float v = a.get_raw_pointer()[a_off];
            a.get_raw_pointer()[a_off] = v > 0.0f ? v : 0.0f;
        }
    } else if (a.device == Device::CUDA) {
        size_t rank = a.shape.size();
        check_rank_within_kmaxdims(rank, "relu_ CUDA: rank exceeds kMaxDims");
        TensorMetadata meta = make_metadata(a, rank, false);
        launch_relu_inplace(a.get_raw_pointer(), a.get_raw_pointer(), meta, a.numel());
    } else {
        throw std::runtime_error("relu_: unsupported device");
    }
    a.bump_version();
    return a;
}

Tensor operator+(const Tensor& a, const Tensor& b) { return add(a, b); }
Tensor operator*(const Tensor& a, const Tensor& b) { return mul(a, b); }
