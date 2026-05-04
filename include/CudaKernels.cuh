#pragma once

#include <cstddef>
#include <stdexcept>

constexpr int kMaxDims = 8;

struct TensorMetadata {
    int rank = 0;
    size_t shape[kMaxDims]{};
    size_t strides[kMaxDims]{};
    int is_contig = 0;
};

inline void check_rank_within_kmaxdims(size_t rank, const char* what) {
    if (rank > static_cast<size_t>(kMaxDims)) {
        throw std::runtime_error(what);
    }
}

void launch_relu_forward(const float* in, float* out,
                         const TensorMetadata& in_meta,
                         const TensorMetadata& out_meta,
                         size_t numel);
void launch_relu_backward(const float* in, const float* out_grad, float* in_grad,
                          const TensorMetadata& in_meta,
                          const TensorMetadata& out_meta,
                          size_t numel);

void launch_sigmoid_forward(const float* in, float* out,
                            const TensorMetadata& in_meta,
                            const TensorMetadata& out_meta,
                            size_t numel);
void launch_sigmoid_backward(const float* out, const float* out_grad, float* in_grad,
                             const TensorMetadata& out_meta,
                             const TensorMetadata& in_meta,
                             size_t numel);

void launch_mse_forward(const float* pred, const float* target, float* out,
                        const TensorMetadata& pred_meta,
                        const TensorMetadata& target_meta,
                        size_t numel);
void launch_mse_backward(const float* pred, const float* target, const float* out_grad,
                         float* pred_grad, float* target_grad,
                         const TensorMetadata& pred_meta,
                         const TensorMetadata& target_meta,
                         size_t numel);

void launch_add_broadcast_forward(const float* a, const float* b, float* out,
                                  const TensorMetadata& a_meta,
                                  const TensorMetadata& b_meta,
                                  const TensorMetadata& out_meta,
                                  size_t out_numel);
void launch_add_broadcast_backward(const float* out_grad, float* a_grad, float* b_grad,
                                   const TensorMetadata& a_meta,
                                   const TensorMetadata& b_meta,
                                   const TensorMetadata& out_meta,
                                   size_t out_numel);
void launch_add_backward_nobroadcast(const float* out_grad, float* in_grad,
                                     const TensorMetadata& out_meta,
                                     const TensorMetadata& in_meta,
                                     size_t numel);

void launch_mul_broadcast_forward(const float* a, const float* b, float* out,
                                  const TensorMetadata& a_meta,
                                  const TensorMetadata& b_meta,
                                  const TensorMetadata& out_meta,
                                  size_t out_numel);
void launch_mul_broadcast_backward(const float* out_grad, const float* a, const float* b,
                                   float* a_grad, float* b_grad,
                                   const TensorMetadata& a_meta,
                                   const TensorMetadata& b_meta,
                                   const TensorMetadata& out_meta,
                                   size_t out_numel);
void launch_mul_backward_nobroadcast(const float* out_grad, const float* a, const float* b,
                                     float* a_grad, float* b_grad,
                                     const TensorMetadata& out_meta,
                                     const TensorMetadata& a_meta,
                                     const TensorMetadata& b_meta,
                                     size_t numel);

void launch_add_inplace_nd(const float* src, float* dst,
                           const TensorMetadata& meta,
                           size_t numel);

void launch_add_contig_to_strided(const float* src_contig, float* dst_strided,
                                  const TensorMetadata& dst_meta,
                                  size_t numel);

void launch_copy_contig_to_strided(const float* src_contig, float* dst_strided,
                                   const TensorMetadata& dst_meta,
                                   size_t numel);
void launch_copy_strided_to_contig(const float* src_strided, float* dst_contig,
                                   const TensorMetadata& src_meta,
                                   size_t numel);

void launch_fill(float* data, float value, size_t numel);
void launch_add_scalar(float* data, float value, size_t numel);

void launch_add_inplace_broadcast(const float* b, float* a,
                                  const TensorMetadata& b_meta,
                                  const TensorMetadata& a_meta,
                                  size_t out_numel);
void launch_mul_inplace_broadcast(const float* b, float* a,
                                  const TensorMetadata& b_meta,
                                  const TensorMetadata& a_meta,
                                  size_t out_numel);
void launch_relu_inplace(const float* in, float* out,
                         const TensorMetadata& meta,
                         size_t numel);

void launch_softmax_forward(const float* in, float* out,
                            const TensorMetadata& in_meta,
                            const TensorMetadata& out_meta,
                            size_t outer, size_t cols);
void launch_softmax_backward(const float* out, const float* out_grad, float* in_grad,
                             const TensorMetadata& out_meta,
                             const TensorMetadata& in_meta,
                             size_t outer, size_t cols);

void launch_cross_entropy_forward(const float* logits, const float* target, float* out,
                                  const TensorMetadata& logits_meta,
                                  const TensorMetadata& target_meta,
                                  size_t outer, size_t cols);
void launch_cross_entropy_backward(const float* logits, const float* target, const float* out_grad,
                                   float* logits_grad,
                                   const TensorMetadata& logits_meta,
                                   const TensorMetadata& target_meta,
                                   size_t outer, size_t cols);

void launch_sgd_step(float* data, const float* grad,
                     const TensorMetadata& data_meta,
                     const TensorMetadata& grad_meta,
                     size_t numel,
                     float lr);
void launch_adam_step(float* data, const float* grad, float* m, float* v,
                      const TensorMetadata& data_meta,
                      const TensorMetadata& grad_meta,
                      const TensorMetadata& m_meta,
                      const TensorMetadata& v_meta,
                      size_t numel,
                      float lr, float beta1, float beta2, float eps,
                      float bias_correction1, float bias_correction2);
