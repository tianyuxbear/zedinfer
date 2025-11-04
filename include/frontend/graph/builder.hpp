#pragma once

#include "frontend/graph/graph.hpp"
#include "frontend/models/base.hpp"

#include <cstddef>

namespace neollm::graph {

// Base class for building computation graphs from model definitions
class GraphBuilder {
public:
    virtual ~GraphBuilder() = default;

    virtual compute_graph_t build(const model::Model *model) = 0;

    /**
     * Factory method to create builder based on model type
     */
    static std::shared_ptr<GraphBuilder> create(const std::string &model_type);
};

// Graph builder implementation for Qwen2 architecture
class Qwen2GraphBuilder : public GraphBuilder {
public:
    compute_graph_t build(const model::Model *model) override;

private:
    // Build a single transformer layer subgraph
    graph_node_t build_transformer_layer(
        compute_graph_t graph,
        graph_node_t input,
        const model::Model *model,
        int layer_idx,
        graph_node_t k_cache,
        graph_node_t v_cache,
        graph_node_t position_ids,
        size_t &numel);

    // Build multi-head self-attention subgraph
    graph_node_t build_attention(
        compute_graph_t graph,
        graph_node_t hidden_states,
        const model::Model *model,
        int layer_idx,
        graph_node_t k_cache,
        graph_node_t v_cache,
        graph_node_t position_ids,
        size_t &numel);

    // Build MLP (feed-forward) subgraph with SwiGLU activation
    graph_node_t build_mlp(
        compute_graph_t graph,
        graph_node_t hidden_states,
        const model::Model *model,
        int layer_idx,
        size_t &numel);
};

} // namespace neollm::graph