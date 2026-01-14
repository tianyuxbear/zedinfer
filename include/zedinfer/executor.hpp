#pragma once

#include "backend/core/storage/storage.hpp" // IWYU pragma: keep
#include "backend/kvcache/base.hpp"
#include "backend/tensor/tensor.hpp"
#include "frontend/graph/graph.hpp"
#include "frontend/graph/shape.hpp"
#include "zedinfer/activation.hpp"

#include <memory>
#include <unordered_map>
#include <vector>

namespace zedinfer {

/**
 * Computation graph executor with optimized memory management
 * - Prefill phase: Arena allocation for batch processing
 * - Decode phase: Pooled allocation for token-by-token generation
 */
class GraphExecutor {
public:
    /**
     * Factory method to create executor instance
     * @param graph Computation graph to execute
     * @param config Executor configuration
     * @return Shared pointer to executor
     */
    static std::shared_ptr<GraphExecutor> create(
        graph::compute_graph_t graph,
        ExecutorConfig config);

    /**
     * Execute forward pass through the graph
     * @param kvcache KV cache for attention layers
     * @param input_ids Input token sequence
     * @param past_len Number of previously cached tokens
     * @return Output logits tensor
     */
    tensor_t forward(
        kvcache::KVCache &kvcache,
        const std::vector<int> &input_ids,
        int past_len = 0);

    /**
     * Get executor configuration
     */
    const ExecutorConfig &config() const { return config_; }

    /**
     * Print execution statistics and diagnostics
     */
    void print_stats() const;

private:
    // Private constructor - use create() factory
    GraphExecutor(
        graph::compute_graph_t graph,
        const ExecutorConfig &config);

    // Core components
    graph::compute_graph_t graph_;
    ExecutorConfig config_;

    std::unique_ptr<PositionIDsCache> position_ids_cache_; // Cached position embeddings

    // Internal execution helpers
    void execute_node(
        kvcache::KVCache &kvcache,
        graph::graph_node_t node,
        std::unordered_map<graph::graph_node_t, tensor_t> &activations,
        graph::ExecutionContext &ctx);

    void execute_op(
        graph::graph_node_t node,
        const std::vector<tensor_t> &inputs,
        tensor_t output);

    /**
     * Parse layer index from node name (e.g., "layer_12" -> 12)
     */
    int extract_layer_idx(const std::string &node_name);
};

} // namespace zedinfer