#pragma once

#include "backend/tensor/tensor.hpp"
#include <any>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace neollm::graph {

// Compute graph node operation types
enum class OpType {
    INPUT,
    ADD,
    ARGMAX,
    EMBEDDING,
    LINEAR,
    RMS_NORM,
    ROPE,
    SELF_ATTENTION,
    SWIGLU
};

// Represents a node in the computation graph
class GraphNode {
public:
    GraphNode(OpType op, std::string name)
        : op_type_(op), name_(std::move(name)) {}

    OpType op_type() const { return op_type_; }
    const std::string &name() const { return name_; }

    // Input management
    void add_input(std::shared_ptr<GraphNode> node) { inputs_.push_back(node); }
    const std::vector<std::shared_ptr<GraphNode>> &inputs() const { return inputs_; }

    // Weight tensors (for operators like Linear)
    void set_weight(tensor_t weight) { weight_ = weight; }
    tensor_t weight() const { return weight_; }
    void set_bias(tensor_t bias) { bias_ = bias; }
    tensor_t bias() const { return bias_; }

    // Parameter storage for static configuration
    template <typename T>
    void set_param(const std::string &key, const T &value) {
        params_[key] = value;
    }

    template <typename T>
    T get_param(const std::string &key) const {
        auto it = params_.find(key);
        if (it != params_.end()) {
            return std::any_cast<T>(it->second);
        }
        throw std::runtime_error("Parameter not found: " + key);
    }

    bool has_param(const std::string &key) const {
        return params_.find(key) != params_.end();
    }

private:
    OpType op_type_;
    std::string name_;
    std::vector<std::shared_ptr<GraphNode>> inputs_;
    tensor_t weight_;
    tensor_t bias_;
    std::vector<size_t> output_shape_;
    std::unordered_map<std::string, std::any> params_;
};

using graph_node_t = std::shared_ptr<GraphNode>;

// Computation graph for representing neural network operations
class ComputeGraph {
public:
    ComputeGraph() = default;

    // Graph construction
    graph_node_t add_node(OpType op, const std::string &name);

    // Input/output node management
    void set_input(const std::string &name, graph_node_t node);
    void set_output(const std::string &name, graph_node_t node);

    // Query operations
    graph_node_t get_input(const std::string &name) const;
    graph_node_t get_output(const std::string &name) const;

    // Get nodes in topological order
    std::vector<graph_node_t> get_execution_order() const;

    // Graph optimization (fusion, constant folding, etc.)
    void optimize();

    // Debug print
    void print() const;

private:
    std::vector<graph_node_t> nodes_;
    std::unordered_map<std::string, graph_node_t> inputs_;
    std::unordered_map<std::string, graph_node_t> outputs_;

    // Topological sort implementation
    void topological_sort(std::vector<graph_node_t> &sorted) const;

    // Optimization passes
    void fuse_operators();
    void fold_constants();
    void eliminate_dead_code();
    void optimize_memory();
};

using compute_graph_t = std::shared_ptr<ComputeGraph>;

} // namespace neollm::graph