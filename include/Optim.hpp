#pragma once

#include "Tensor.hpp"
#include "CudaKernels.cuh"
#include "CudaContext.hpp"

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#include <cuda_runtime.h>

class Optimizer {
public:
    Optimizer(const std::vector<Tensor>& params_) : params(params_) {}
    virtual ~Optimizer() = default;

    virtual void step() = 0;

    virtual void zero_grad() {
        for (auto& p : params) {
            p.zero_grad();
        }
    }

protected:
    std::vector<Tensor> params;

    static TensorMetadata make_meta(const Tensor& t) {
        TensorMetadata meta{};
        meta.rank = static_cast<int>(t.shape.size());
        for (size_t i = 0; i < t.shape.size(); ++i) {
            meta.shape[i] = t.shape[i];
            meta.strides[i] = t.strides[i];
        }
        meta.is_contig = (t.strides == Tensor::compute_contiguous_strides(t.shape)) ? 1 : 0;
        return meta;
    }

    static TensorMetadata make_contig_meta(const Tensor& t) {
        TensorMetadata meta{};
        meta.rank = static_cast<int>(t.shape.size());
        std::vector<size_t> contig = Tensor::compute_contiguous_strides(t.shape);
        for (size_t i = 0; i < t.shape.size(); ++i) {
            meta.shape[i] = t.shape[i];
            meta.strides[i] = contig[i];
        }
        meta.is_contig = 1;
        return meta;
    }
};

class SGD : public Optimizer {
public:
    SGD(const std::vector<Tensor>& params_, float lr_) : Optimizer(params_), lr(lr_) {}

    void step() override {
        for (const auto& p : params) {
            if (p.device == Device::CUDA) {
                CUDA_CHECK(cudaStreamSynchronize(CudaContext::stream()));
                break;
            }
        }
        for (auto& p : params) {
            if (!p.grad_storage) {
                continue;
            }
            if (p.device == Device::CPU) {
                for (size_t i = 0; i < p.numel(); ++i) {
                    size_t data_off = p.offset_from_linear(i);
                    size_t grad_off = i;
                    p.get_raw_pointer()[data_off] -= lr * p.get_raw_grad_pointer()[grad_off];
                }
            } else if (p.device == Device::CUDA) {
                TensorMetadata p_meta = make_meta(p);
                TensorMetadata g_meta = make_contig_meta(p);
                launch_sgd_step(p.get_raw_pointer(), p.get_raw_grad_pointer(),
                                p_meta, g_meta,
                                p.numel(),
                                lr);
            } else {
                throw std::runtime_error("SGD::step: unsupported device");
            }
        }
    }

private:
    float lr = 0.01f;
};

class Adam : public Optimizer {
public:
    Adam(const std::vector<Tensor>& params_,
         float lr_,
         float beta1_ = 0.9f,
         float beta2_ = 0.999f,
         float eps_ = 1e-8f)
        : Optimizer(params_), lr(lr_), beta1(beta1_), beta2(beta2_), eps(eps_) {}

    void step() override {
        for (const auto& p : params) {
            if (p.device == Device::CUDA) {
                CUDA_CHECK(cudaStreamSynchronize(CudaContext::stream()));
                break;
            }
        }
        ++t;
        float bc1 = 1.0f - std::pow(beta1, static_cast<float>(t));
        float bc2 = 1.0f - std::pow(beta2, static_cast<float>(t));

        for (auto& p : params) {
            if (!p.grad_storage) {
                continue;
            }
            Tensor& m = get_state(m_state, p);
            Tensor& v = get_state(v_state, p);

            if (p.device == Device::CPU) {
                for (size_t i = 0; i < p.numel(); ++i) {
                    size_t data_off = p.offset_from_linear(i);
                    size_t grad_off = i;
                    float g = p.get_raw_grad_pointer()[grad_off];
                    float m_val = beta1 * m.get_raw_pointer()[grad_off] + (1.0f - beta1) * g;
                    float v_val = beta2 * v.get_raw_pointer()[grad_off] + (1.0f - beta2) * g * g;
                    m.get_raw_pointer()[grad_off] = m_val;
                    v.get_raw_pointer()[grad_off] = v_val;
                    float m_hat = m_val / bc1;
                    float v_hat = v_val / bc2;
                    p.get_raw_pointer()[data_off] -= lr * m_hat / (std::sqrt(v_hat) + eps);
                }
            } else if (p.device == Device::CUDA) {
                TensorMetadata p_meta = make_meta(p);
                TensorMetadata g_meta = make_contig_meta(p);
                TensorMetadata m_meta = make_contig_meta(m);
                TensorMetadata v_meta = make_contig_meta(v);
                launch_adam_step(p.get_raw_pointer(), p.get_raw_grad_pointer(),
                                 m.get_raw_pointer(), v.get_raw_pointer(),
                                 p_meta, g_meta, m_meta, v_meta,
                                 p.numel(),
                                 lr, beta1, beta2, eps, bc1, bc2);
            } else {
                throw std::runtime_error("Adam::step: unsupported device");
            }
        }
    }

private:
    Tensor& get_state(std::unordered_map<uint64_t, Tensor>& state, const Tensor& p) {
        uint64_t key = p.id();
        if (key == 0) {
            throw std::runtime_error("Adam::get_state: invalid tensor id");
        }
        auto it = state.find(key);
        if (it == state.end()) {
            Tensor t_state(p.shape, p.device, false);
            if (p.device == Device::CUDA) {
                cudaError_t err = cudaMemset(t_state.get_raw_pointer(), 0, p.numel() * sizeof(float));
                if (err != cudaSuccess) {
                    throw std::runtime_error("cudaMemset failed for Adam state");
                }
            }
            auto res = state.emplace(key, std::move(t_state));
            return res.first->second;
        }
        return it->second;
    }

private:
    float lr = 0.001f;
    float beta1 = 0.9f;
    float beta2 = 0.999f;
    float eps = 1e-8f;
    size_t t = 0;
    std::unordered_map<uint64_t, Tensor> m_state;
    std::unordered_map<uint64_t, Tensor> v_state;
};
