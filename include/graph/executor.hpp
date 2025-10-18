#pragma once

#include "backend/tensor/tensor.hpp"
#include "graph/graph.hpp"
#include "kvcache/base.hpp"
#include <unordered_map>
#include <vector>

namespace neollm::graph {

/**
 * Runtime configuration for executor.
 * Inherits device settings from KVCache to avoid duplication.
 */
struct ExecutorConfig {
    NeollmDeviceType_t device_type;
    int device_id;
    NeollmDataType_t dtype; // Default dtype for activations

    static ExecutorConfig from_kvcache_config(const kvcache::KVCacheConfig &kv_config) {
        return ExecutorConfig{
            .device_type = kv_config.device_type,
            .device_id = kv_config.device_id,
            .dtype = kv_config.dtype};
    }
};

/**
 * Memory pool for activation tensors.
 *
 * Features:
 * - Shape-based tensor lookup and reuse
 * - Batch release after each forward pass
 */
class ActivationPool {
public:
    explicit ActivationPool(const ExecutorConfig &config);

    /**
     * Acquire tensor from pool (reused or newly allocated).
     * @param shape Tensor shape
     * @return Tensor matching the requested shape
     */
    tensor_t acquire(const std::vector<size_t> &shape);

    /** Mark tensor as available for reuse */
    void release(tensor_t tensor);

    /** Release all tensors (call after each forward) */
    void release_all();

    /** Pool statistics */
    size_t total_tensors() const { return pool_.size(); }
    size_t tensors_in_use() const;
    size_t total_memory() const;
    void print_stats() const;

private:
    struct TensorSlot {
        tensor_t tensor;
        std::vector<size_t> shape;
        bool in_use;
    };

    ExecutorConfig config_;
    std::vector<TensorSlot> pool_;
};

/**
 * Pre-allocated position IDs cache to avoid repeated allocation.
 */
class PositionIDsCache {
public:
    PositionIDsCache(int max_seq_len, const ExecutorConfig &config);

    /**
     * Get position_ids slice [start_pos, start_pos+1, ..., start_pos+seq_len-1].
     * @return Tensor of shape [seq_len]
     */
    tensor_t get_slice(int start_pos, int seq_len);

    int max_seq_len() const { return max_seq_len_; }

private:
    int max_seq_len_;
    ExecutorConfig config_;
    tensor_t cache_; // Pre-allocated [0, 1, 2, ..., max_seq_len-1]

    void initialize_cache();
};

/**
 * Computation graph executor.
 */
class GraphExecutor {
public:
    /**
     * Forward inference.
     * @param input_ids Input token sequence
     * @param past_len Length of cached history
     * @return Output logits
     */
    tensor_t forward(
        const std::vector<int> &input_ids,
        int past_len = 0);

    const ExecutorConfig &config() const { return config_; }
    void print_stats() const;

    static std::unique_ptr<GraphExecutor> create(
        compute_graph_t graph,
        kvcache::kvcache_t kv_cache);

private:
    GraphExecutor(
        compute_graph_t graph,
        kvcache::kvcache_t kv_cache,
        const ExecutorConfig &config);

    compute_graph_t graph_;
    kvcache::kvcache_t kv_cache_;
    ExecutorConfig config_;

    std::unique_ptr<ActivationPool> activation_pool_;
    std::unique_ptr<PositionIDsCache> position_ids_cache_;

    /** Execute single graph node */
    void execute_node(
        graph_node_t node,
        std::unordered_map<graph_node_t, tensor_t> &activations,
        int past_len,
        int seq_len);

    /** Extract layer index from node name */
    int extract_layer_idx(const std::string &node_name);
};

} // namespace neollm::graph