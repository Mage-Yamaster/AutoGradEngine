#include "Tensor.hpp"

#include <algorithm>
#include <numeric>
#include <stdexcept>

#include <cuda_runtime.h>

#include "CudaKernels.cuh"
#include "CudaContext.hpp"

namespace {

size_t numel_from_shape(const std::vector<size_t>& shape) {
    if (shape.empty()) {
        return 1;
    }
    return std::accumulate(shape.begin(), shape.end(), static_cast<size_t>(1),
                           std::multiplies<size_t>());
}

std::vector<size_t> compute_contiguous_strides_impl(const std::vector<size_t>& shape) {
    std::vector<size_t> strides(shape.size(), 1);
    if (shape.empty()) {
        return strides;
    }
    for (int i = static_cast<int>(shape.size()) - 2; i >= 0; --i) {
        strides[static_cast<size_t>(i)] =
            strides[static_cast<size_t>(i + 1)] * shape[static_cast<size_t>(i + 1)];
    }
    return strides;
}

void linear_to_indices(size_t linear,
                       const std::vector<size_t>& shape,
                       const std::vector<size_t>& logical_strides,
                       std::vector<size_t>& indices_out) {
    indices_out.assign(shape.size(), 0);
    if (shape.empty()) {
        return;
    }
    for (size_t i = 0; i < shape.size(); ++i) {
        size_t stride = logical_strides[i];
        indices_out[i] = linear / stride;
        linear %= stride;
    }
}

TensorMetadata build_metadata(const Tensor& t) {
    check_rank_within_kmaxdims(t.shape.size(), "Tensor: rank exceeds kMaxDims");
    TensorMetadata meta{};
    meta.rank = static_cast<int>(t.shape.size());
    for (size_t i = 0; i < t.shape.size(); ++i) {
        meta.shape[i] = t.shape[i];
        meta.strides[i] = t.strides[i];
    }
    meta.is_contig = (t.strides == compute_contiguous_strides_impl(t.shape)) ? 1 : 0;
    return meta;
}

void check_inplace_versions(const std::shared_ptr<ComputeNode>& node) {
    if (!node) {
        return;
    }
    for (const auto& snap : node->version_snapshots) {
        if (!snap.is_leaf) {
            continue;
        }
        auto storage = snap.storage.lock();
        if (!storage) {
            continue;
        }
        uint64_t current = storage->version_counter.load();
        if (current != snap.version) {
            throw std::runtime_error(
                "RuntimeError: a leaf Variable that requires grad is being used in an in-place operation");
        }
    }
}

}  // namespace

thread_local bool Tensor::grad_mode_enabled = true;
std::atomic<uint64_t> Storage::next_id{1};

std::shared_ptr<Storage> Storage::create(size_t numel, Device device) {
    auto storage = std::make_shared<Storage>();
    storage->numel = numel;
    storage->device = device;
    storage->id = next_id.fetch_add(1);
    if (device == Device::CPU) {
        storage->data = std::shared_ptr<float>(new float[numel](), [](float* p) { delete[] p; });
    } else if (device == Device::CUDA) {
        float* ptr = static_cast<float*>(CudaContext::malloc_bytes(numel * sizeof(float)));
        storage->data = std::shared_ptr<float>(ptr, [numel](float* p) {
            CudaContext::free_bytes(p, numel * sizeof(float));
        });
    } else {
        throw std::runtime_error("Storage::create: unsupported device");
    }
    return storage;
}

Tensor::~Tensor() {
}

Tensor::Tensor(const std::vector<size_t>& shape_, Device device_, bool requires_grad_)
    : shape(shape_),
      strides(compute_contiguous_strides_impl(shape_)),
      device(device_),
      requires_grad(requires_grad_) {
    allocate_data();
    if (requires_grad) {
        allocate_grad();
    }
}

Tensor::Tensor(const std::vector<float>& values,
               const std::vector<size_t>& shape_,
               Device device_,
               bool requires_grad_)
    : shape(shape_),
      strides(compute_contiguous_strides_impl(shape_)),
      device(device_),
      requires_grad(requires_grad_) {
    if (values.size() != numel_from_shape(shape)) {
        throw std::runtime_error("Tensor: values size does not match shape");
    }
    allocate_data();
    if (requires_grad) {
        allocate_grad();
    }
    std::copy(values.begin(), values.end(), get_raw_pointer());
}

float* Tensor::get_raw_pointer() const {
    if (!storage || !storage->data) {
        return nullptr;
    }
    return storage->data.get() + storage_offset;
}

float* Tensor::get_raw_grad_pointer() const {
    if (!grad_storage || !grad_storage->data) {
        return nullptr;
    }
    return grad_storage->data.get() + grad_offset;
}

std::shared_ptr<float> Tensor::data_base() const {
    return storage ? storage->data : std::shared_ptr<float>();
}

std::shared_ptr<float> Tensor::grad_base() const {
    return grad_storage ? grad_storage->data : std::shared_ptr<float>();
}

size_t Tensor::numel() const {
    return numel_from_shape(shape);
}

bool Tensor::is_contiguous() const {
    return strides == compute_contiguous_strides_impl(shape);
}

size_t Tensor::offset(const std::vector<size_t>& indices) const {
    if (indices.size() != shape.size()) {
        throw std::runtime_error("offset: rank mismatch");
    }
    size_t linear = 0;
    for (size_t i = 0; i < indices.size(); ++i) {
        if (indices[i] >= shape[i]) {
            throw std::runtime_error("offset: index out of bounds");
        }
        linear += indices[i] * strides[i];
    }
    return linear;
}

size_t Tensor::offset_from_linear(size_t linear) const {
    std::vector<size_t> logical_strides = compute_contiguous_strides_impl(shape);
    std::vector<size_t> idx;
    linear_to_indices(linear, shape, logical_strides, idx);
    return offset(idx);
}

float* Tensor::data_ptr(const std::vector<size_t>& indices) const {
    return get_raw_pointer() + offset(indices);
}

void Tensor::set_data(const std::vector<float>& values) {
    if (values.size() != numel()) {
        throw std::runtime_error("set_data: size mismatch");
    }
    if (device == Device::CPU) {
        if (is_contiguous()) {
            std::copy(values.begin(), values.end(), get_raw_pointer());
        } else {
            std::vector<size_t> logical_strides = compute_contiguous_strides_impl(shape);
            std::vector<size_t> idx;
            for (size_t linear = 0; linear < values.size(); ++linear) {
                linear_to_indices(linear, shape, logical_strides, idx);
                get_raw_pointer()[offset(idx)] = values[linear];
            }
        }
    } else {
        cudaStream_t stream = CudaContext::stream();
        if (is_contiguous()) {
            CUDA_CHECK(cudaMemcpyAsync(get_raw_pointer(), values.data(),
                                       numel() * sizeof(float),
                                       cudaMemcpyHostToDevice,
                                       stream));
        } else {
            float* tmp = nullptr;
            tmp = static_cast<float*>(CudaContext::malloc_bytes(numel() * sizeof(float)));
            CUDA_CHECK(cudaMemcpyAsync(tmp, values.data(), numel() * sizeof(float),
                                       cudaMemcpyHostToDevice, stream));
            TensorMetadata meta = build_metadata(*this);
            launch_copy_contig_to_strided(tmp, get_raw_pointer(), meta, numel());
            CudaContext::free_bytes(tmp, numel() * sizeof(float));
        }
    }
}

std::vector<float> Tensor::get_data() const {
    std::vector<float> out(numel());
    if (device == Device::CPU) {
        if (is_contiguous()) {
            std::copy(get_raw_pointer(), get_raw_pointer() + numel(), out.begin());
        } else {
            std::vector<size_t> logical_strides = compute_contiguous_strides_impl(shape);
            std::vector<size_t> idx;
            for (size_t linear = 0; linear < out.size(); ++linear) {
                linear_to_indices(linear, shape, logical_strides, idx);
                out[linear] = get_raw_pointer()[offset(idx)];
            }
        }
    } else {
        cudaStream_t stream = CudaContext::stream();
        if (is_contiguous()) {
            CUDA_CHECK(cudaMemcpyAsync(out.data(), get_raw_pointer(),
                                       numel() * sizeof(float),
                                       cudaMemcpyDeviceToHost,
                                       stream));
        } else {
            float* tmp = nullptr;
            tmp = static_cast<float*>(CudaContext::malloc_bytes(numel() * sizeof(float)));
            TensorMetadata meta = build_metadata(*this);
            launch_copy_strided_to_contig(get_raw_pointer(), tmp, meta, numel());
            CUDA_CHECK(cudaMemcpyAsync(out.data(), tmp, numel() * sizeof(float),
                                       cudaMemcpyDeviceToHost, stream));
            CudaContext::free_bytes(tmp, numel() * sizeof(float));
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    return out;
}

void Tensor::zero_grad() {
    if (!grad_storage) {
        allocate_grad();
    }
    if (device == Device::CPU) {
        std::fill(get_raw_grad_pointer(), get_raw_grad_pointer() + numel(), 0.0f);
    } else {
        cudaStream_t stream = CudaContext::stream();
        CUDA_CHECK(cudaMemsetAsync(get_raw_grad_pointer(), 0, numel() * sizeof(float), stream));
    }
}

void Tensor::backward(bool clear_graph_) {
    backward_impl(nullptr, clear_graph_);
}

void Tensor::backward(const Tensor& gradient, bool clear_graph_) {
    backward_impl(&gradient, clear_graph_);
}

void Tensor::backward_impl(const Tensor* gradient, bool clear_graph_) {
    if (!requires_grad) {
        return;
    }
    if (!grad_storage) {
        allocate_grad();
    }
    if (!gradient) {
        if (numel() != 1) {
            throw std::runtime_error("backward: gradient must be provided for non-scalar tensor");
        }
        if (device == Device::CPU) {
            for (size_t i = 0; i < numel(); ++i) {
                size_t off = offset_from_linear(i);
                Tensor::accumulate_grad_value(get_raw_grad_pointer() + off, 1.0f);
            }
        } else {
            if (is_contiguous()) {
                launch_add_scalar(get_raw_grad_pointer(), 1.0f, numel());
            } else {
                float* tmp = nullptr;
                tmp = static_cast<float*>(CudaContext::malloc_bytes(numel() * sizeof(float)));
                launch_fill(tmp, 1.0f, numel());
                TensorMetadata meta = build_metadata(*this);
                launch_add_contig_to_strided(tmp, get_raw_grad_pointer(), meta, numel());
                CudaContext::free_bytes(tmp, numel() * sizeof(float));
            }
        }
    } else {
        if (gradient->device != device) {
            throw std::runtime_error("backward: gradient device mismatch");
        }
        if (gradient->numel() != numel() || gradient->shape != shape) {
            throw std::runtime_error("backward: gradient shape mismatch");
        }
        if (device == Device::CPU) {
            std::vector<size_t> logical_strides = compute_contiguous_strides_impl(shape);
            std::vector<size_t> idx;
            for (size_t linear = 0; linear < numel(); ++linear) {
                linear_to_indices(linear, shape, logical_strides, idx);
                size_t self_off = offset(idx);
                size_t grad_off = gradient->offset(idx);
                Tensor::accumulate_grad_value(get_raw_grad_pointer() + self_off,
                                              gradient->get_raw_pointer()[grad_off]);
            }
        } else if (device == Device::CUDA) {
            TensorMetadata self_meta = build_metadata(*this);
            if (gradient->is_contiguous()) {
                launch_add_contig_to_strided(gradient->get_raw_pointer(),
                                             get_raw_grad_pointer(),
                                             self_meta, numel());
            } else {
                float* tmp = nullptr;
                tmp = static_cast<float*>(CudaContext::malloc_bytes(numel() * sizeof(float)));
                TensorMetadata grad_meta = build_metadata(*gradient);
                launch_copy_strided_to_contig(gradient->get_raw_pointer(), tmp, grad_meta, numel());
                launch_add_contig_to_strided(tmp, get_raw_grad_pointer(), self_meta, numel());
                CudaContext::free_bytes(tmp, numel() * sizeof(float));
            }
        } else {
            throw std::runtime_error("backward: unsupported device");
        }
    }

    std::vector<std::shared_ptr<ComputeNode>> topo;
    std::unordered_set<ComputeNode*> visited;
    build_topo(creator_node, topo, visited);

    for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
        check_inplace_versions(*it);
        if ((*it)->backward_fn) {
            (*it)->backward_fn();
        }
        if (clear_graph_) {
            (*it)->backward_fn = nullptr;
            (*it)->inputs.clear();
        }
    }

    if (clear_graph_) {
        creator_node.reset();
    }

    if (device == Device::CUDA) {
        CUDA_CHECK(cudaStreamSynchronize(CudaContext::stream()));
    }
}

bool Tensor::is_grad_enabled() {
    return grad_mode_enabled;
}

void Tensor::set_grad_enabled(bool enabled) {
    grad_mode_enabled = enabled;
}

std::vector<size_t> Tensor::compute_contiguous_strides(const std::vector<size_t>& shape) {
    return compute_contiguous_strides_impl(shape);
}

Tensor Tensor::transpose(size_t dim0, size_t dim1) const {
    if (dim0 >= shape.size() || dim1 >= shape.size()) {
        throw std::runtime_error("transpose: dim out of range");
    }
    Tensor out;
    out.storage = storage;
    out.grad_storage = grad_storage;
    out.storage_offset = storage_offset;
    out.grad_offset = grad_offset;
    out.shape = shape;
    out.strides = strides;
    std::swap(out.shape[dim0], out.shape[dim1]);
    std::swap(out.strides[dim0], out.strides[dim1]);
    out.device = device;
    out.requires_grad = requires_grad;
    out.creator_node = nullptr;
    return out;
}

Tensor Tensor::contiguous() const {
    if (is_contiguous()) {
        return *this;
    }
    Tensor out(shape, device, requires_grad);
    if (device == Device::CPU) {
        std::vector<size_t> logical_strides = compute_contiguous_strides_impl(shape);
        std::vector<size_t> idx;
        for (size_t linear = 0; linear < numel(); ++linear) {
            linear_to_indices(linear, shape, logical_strides, idx);
            size_t in_off = offset(idx);
            out.get_raw_pointer()[linear] = get_raw_pointer()[in_off];
        }
    } else if (device == Device::CUDA) {
        TensorMetadata meta = build_metadata(*this);
        launch_copy_strided_to_contig(get_raw_pointer(), out.get_raw_pointer(), meta, numel());
    } else {
        throw std::runtime_error("contiguous: unsupported device");
    }

    if (requires_grad && Tensor::is_grad_enabled()) {
        Tensor self = *this;
        out.creator_node = std::make_shared<ComputeNode>();
        Tensor::add_input_nodes(out.creator_node, {self});
        auto out_grad = out.grad_base();
        auto self_grad = self.grad_base();
        size_t out_grad_offset = out.grad_offset;
        size_t self_grad_offset = self.grad_offset;
        auto self_shape = self.shape;
        auto self_strides = self.strides;
        Device self_device = self.device;
        size_t self_numel = self.numel();
        out.creator_node->backward_fn = [out_grad, self_grad, self_shape, self_strides,
                                         out_grad_offset, self_grad_offset,
                                         self_device, self_numel]() mutable {
            if (!out_grad) {
                return;
            }
            if (self_device == Device::CPU) {
                std::vector<size_t> logical_strides = compute_contiguous_strides_impl(self_shape);
                std::vector<size_t> idx;
                for (size_t linear = 0; linear < self_numel; ++linear) {
                    linear_to_indices(linear, self_shape, logical_strides, idx);
                    size_t in_off = 0;
                    for (size_t i = 0; i < idx.size(); ++i) {
                        in_off += idx[i] * self_strides[i];
                    }
                    Tensor::accumulate_grad_value(self_grad.get() + self_grad_offset + in_off,
                                                  out_grad.get()[out_grad_offset + linear]);
                }
            } else if (self_device == Device::CUDA) {
                TensorMetadata meta{};
                check_rank_within_kmaxdims(self_shape.size(), "contiguous backward: rank exceeds kMaxDims");
                meta.rank = static_cast<int>(self_shape.size());
                for (size_t i = 0; i < self_shape.size(); ++i) {
                    meta.shape[i] = self_shape[i];
                    meta.strides[i] = self_strides[i];
                }
                launch_add_contig_to_strided(out_grad.get() + out_grad_offset,
                                             self_grad.get() + self_grad_offset,
                                             meta, self_numel);
            } else {
                throw std::runtime_error("contiguous backward: unsupported device");
            }
        };
    }
    return out;
}

void Tensor::allocate_data() {
    size_t n = numel_from_shape(shape);
    storage = Storage::create(n, device);
    storage_offset = 0;
}

void Tensor::allocate_grad() {
    size_t n = numel_from_shape(shape);
    grad_storage = Storage::create(n, device);
    grad_offset = 0;
    if (device == Device::CUDA) {
        cudaStream_t stream = CudaContext::stream();
        CUDA_CHECK(cudaMemsetAsync(get_raw_grad_pointer(), 0, n * sizeof(float), stream));
    }
}

void Tensor::clear_graph() {
    if (!creator_node) {
        return;
    }
    std::vector<std::shared_ptr<ComputeNode>> topo;
    std::unordered_set<ComputeNode*> visited;
    build_topo(creator_node, topo, visited);
    for (auto& node : topo) {
        node->backward_fn = nullptr;
        node->inputs.clear();
    }
    creator_node.reset();
}

void Tensor::build_topo(const std::shared_ptr<ComputeNode>& node,
                        std::vector<std::shared_ptr<ComputeNode>>& topo,
                        std::unordered_set<ComputeNode*>& visited) {
    if (!node) {
        return;
    }
    if (visited.count(node.get()) != 0) {
        return;
    }
    visited.insert(node.get());
    for (const auto& input_node : node->inputs) {
        build_topo(input_node, topo, visited);
    }
    topo.push_back(node);
}

void Tensor::add_input_node(const std::shared_ptr<ComputeNode>& node, const Tensor& input) {
    if (!node) {
        return;
    }
    if (input.creator_node) {
        node->inputs.emplace_back(input.creator_node);
    } else if (input.requires_grad) {
        if (!input.leaf_node) {
            input.leaf_node = std::make_shared<ComputeNode>();
        }
        node->inputs.emplace_back(input.leaf_node);
    }
    if (input.requires_grad && !input.creator_node && input.storage) {
        ComputeNode::VersionSnapshot snap;
        snap.storage = input.storage;
        snap.version = input.storage->version_counter.load();
        snap.is_leaf = true;
        node->version_snapshots.push_back(snap);
    }
}

void Tensor::add_input_nodes(const std::shared_ptr<ComputeNode>& node,
                             std::initializer_list<std::reference_wrapper<const Tensor>> inputs) {
    for (const Tensor& input : inputs) {
        add_input_node(node, input);
    }
}

NoGradGuard::NoGradGuard() : prev_(Tensor::is_grad_enabled()) {
    Tensor::set_grad_enabled(false);
}

NoGradGuard::~NoGradGuard() {
    Tensor::set_grad_enabled(prev_);
}

uint64_t Tensor::version() const {
    return storage ? storage->version_counter.load() : 0;
}

void Tensor::bump_version() {
    if (storage) {
        storage->version_counter.fetch_add(1);
    }
}

void Tensor::accumulate_grad_value(float* grad_ptr, float value) {
    if (!grad_ptr) {
        return;
    }
    if (Tensor::is_grad_enabled()) {
        // TODO: route through Tensor ops for higher-order grads when supported.
        *grad_ptr += value;
    } else {
        *grad_ptr += value;
    }
}
