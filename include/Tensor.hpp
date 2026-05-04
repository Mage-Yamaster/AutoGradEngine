#pragma once

#include <functional>
#include <memory>
#include <atomic>
#include <cstdint>
#include <unordered_set>
#include <vector>

enum class Device { CPU, CUDA };

class Tensor;

class Storage {
public:
    std::shared_ptr<float> data;
    size_t numel = 0;
    Device device = Device::CPU;
    uint64_t id = 0;
    std::atomic<uint64_t> version_counter{0};

    static std::shared_ptr<Storage> create(size_t numel, Device device);

private:
    static std::atomic<uint64_t> next_id;
};

struct ComputeNode {
    std::vector<std::shared_ptr<ComputeNode>> inputs;
    std::function<void()> backward_fn;
    struct VersionSnapshot {
        std::weak_ptr<Storage> storage;
        uint64_t version = 0;
        bool is_leaf = false;
    };
    std::vector<VersionSnapshot> version_snapshots;
};

class Tensor {
public:
    std::shared_ptr<Storage> storage;
    std::shared_ptr<Storage> grad_storage;
    size_t storage_offset = 0;
    size_t grad_offset = 0;
    std::vector<size_t> shape;
    std::vector<size_t> strides;
    Device device = Device::CPU;
    bool requires_grad = false;
    std::shared_ptr<ComputeNode> creator_node;
    mutable std::shared_ptr<ComputeNode> leaf_node;

    Tensor() = default;
    ~Tensor();
    Tensor(const std::vector<size_t>& shape_, Device device_ = Device::CPU, bool requires_grad_ = false);
    Tensor(const std::vector<float>& values,
           const std::vector<size_t>& shape_,
           Device device_ = Device::CPU,
           bool requires_grad_ = false);

    float* get_raw_pointer() const;
    float* get_raw_grad_pointer() const;
    std::shared_ptr<float> data_base() const;
    std::shared_ptr<float> grad_base() const;
    uint64_t id() const { return storage ? storage->id : 0; }
    size_t numel() const;
    bool is_contiguous() const;
    size_t offset(const std::vector<size_t>& indices) const;
    size_t offset_from_linear(size_t linear) const;
    float* data_ptr(const std::vector<size_t>& indices) const;
    void set_data(const std::vector<float>& values);
    std::vector<float> get_data() const;
    void zero_grad();
    void backward(bool clear_graph_ = true);
    void backward(const Tensor& gradient, bool clear_graph_ = true);
    void clear_graph();
    Tensor transpose(size_t dim0 = 0, size_t dim1 = 1) const;
    Tensor contiguous() const;
    uint64_t version() const;
    void bump_version();

    static bool is_grad_enabled();
    static void set_grad_enabled(bool enabled);
    static std::vector<size_t> compute_contiguous_strides(const std::vector<size_t>& shape);
    static void add_input_node(const std::shared_ptr<ComputeNode>& node, const Tensor& input);
    static void add_input_nodes(const std::shared_ptr<ComputeNode>& node,
                                std::initializer_list<std::reference_wrapper<const Tensor>> inputs);
    static void accumulate_grad_value(float* grad_ptr, float value);

private:
    void allocate_data();
    void allocate_grad();
    void backward_impl(const Tensor* gradient, bool clear_graph_);
    static void build_topo(const std::shared_ptr<ComputeNode>& node,
                           std::vector<std::shared_ptr<ComputeNode>>& topo,
                           std::unordered_set<ComputeNode*>& visited);

    static thread_local bool grad_mode_enabled;
};

class NoGradGuard {
public:
    NoGradGuard();
    ~NoGradGuard();

private:
    bool prev_;
};
