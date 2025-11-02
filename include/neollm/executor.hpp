#pragma once

#include "backend/core/storage/storage.hpp" // IWYU pragma: keep
#include "backend/kvcache/base.hpp"
#include "backend/tensor/tensor.hpp"
#include "frontend/graph/graph.hpp"
#include "neollm/activation.hpp"

#include <memory>
#include <unordered_map>
#include <vector>

namespace neollm {

/**
 * Executes computation graph with optimized memory management.
 * Uses arena allocation for prefill and pooling for decode phases.
 */
class GraphExecutor {
public:
    // Factory method
    static std::unique_ptr<GraphExecutor> create(
        graph::compute_graph_t graph,
        kvcache::kvcache_t kv_cache,
        ExecutorConfig config);

    // Core inference interface
    tensor_t forward(const std::vector<int> &input_ids, int past_len = 0);

    // Configuration and diagnostics
    const ExecutorConfig &config() const { return config_; }
    void print_stats() const;

private:
    // Private constructor - use create() factory method
    GraphExecutor(
        graph::compute_graph_t graph,
        kvcache::kvcache_t kv_cache,
        const ExecutorConfig &config);

    // Core components
    graph::compute_graph_t graph_;
    kvcache::kvcache_t kv_cache_;
    ExecutorConfig config_;

    // Memory allocators
    std::unique_ptr<PrefillArena> prefill_arena_;
    std::unique_ptr<DecodePool> decode_pool_;
    std::unique_ptr<PositionIDsCache> position_ids_cache_;

    // Helper methods
    void execute_node(
        graph::graph_node_t node,
        std::unordered_map<graph::graph_node_t, tensor_t> &activations,
        int past_len,
        int seq_len);

    int extract_layer_idx(const std::string &node_name);
};

} // namespace neollm