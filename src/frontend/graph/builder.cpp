#include "frontend/graph/builder.hpp"
#include "frontend/graph/graph.hpp"
#include <cstddef>

namespace zedinfer::graph {

std::shared_ptr<GraphBuilder> GraphBuilder::create(const std::string &model_type) {
    if (model_type == "qwen2") {
        return std::make_unique<Qwen2GraphBuilder>();
    } else if (model_type == "qwen3") {
        return std::make_unique<Qwen3GraphBuilder>();
    } else {
        throw std::runtime_error("Unsupported model type: " + model_type);
    }
}

compute_graph_t Qwen2GraphBuilder::build(const model::Model *model) {
    auto graph = std::make_shared<ComputeGraph>();
    size_t per_token_activation_numel = 0;

    const auto &config = model->config();

    // Create input nodes
    auto input_ids = graph->add_node(OpType::INPUT, "input_ids");
    auto position_ids = graph->add_node(OpType::INPUT, "position_ids");

    graph->set_input("input_ids", input_ids);
    graph->set_input("position_ids", position_ids);

    // Token embedding layer
    auto embedding_node = graph->add_node(OpType::EMBEDDING, "embedding");
    embedding_node->add_input(input_ids);
    embedding_node->set_weight(model->weights().get_tensor(
        model->get_embedding_weight_name()));
    embedding_node->set_param("hidden_size", config.hidden_size);
    per_token_activation_numel += config.hidden_size;

    // Build transformer layers
    auto hidden_states = embedding_node;

    for (size_t i = 0; i < config.num_hidden_layers; ++i) {
        // Create KV cache input nodes for each layer
        auto k_cache_name = "layer_" + std::to_string(i) + "_k_cache";
        auto v_cache_name = "layer_" + std::to_string(i) + "_v_cache";

        auto k_cache_node = graph->add_node(OpType::INPUT, k_cache_name);
        auto v_cache_node = graph->add_node(OpType::INPUT, v_cache_name);

        graph->set_input(k_cache_name, k_cache_node);
        graph->set_input(v_cache_name, v_cache_node);

        hidden_states = build_transformer_layer(
            graph, hidden_states, model, i, k_cache_node, v_cache_node, position_ids, per_token_activation_numel);
    }

    // Final normalization layer
    auto norm_node = graph->add_node(OpType::RMS_NORM, "output_norm");
    norm_node->add_input(hidden_states);
    norm_node->set_weight(model->weights().get_tensor(
        model->get_output_norm_weight_name()));
    norm_node->set_param("hidden_size", config.hidden_size);
    norm_node->set_param("eps", config.rms_norm_eps);
    per_token_activation_numel += config.hidden_size;

    // Language model head (vocabulary projection)
    auto lm_head = graph->add_node(OpType::LINEAR, "lm_head");
    lm_head->add_input(norm_node);
    lm_head->set_weight(model->weights().get_tensor(
        model->get_output_weight_name()));
    lm_head->set_bias(nullptr);
    lm_head->set_param("vocab_size", config.vocab_size);
    per_token_activation_numel += config.vocab_size;

    graph->set_output("logits", lm_head);

    graph->set_per_token_activation_numel(per_token_activation_numel);

    return graph;
}

graph_node_t Qwen2GraphBuilder::build_transformer_layer(
    compute_graph_t graph,
    graph_node_t input,
    const model::Model *model,
    int layer_idx,
    graph_node_t k_cache,
    graph_node_t v_cache,
    graph_node_t position_ids, size_t &numel) {

    const auto &config = model->config();
    auto layer_weights = model->get_layer_weight_names(layer_idx);

    // Pre-attention normalization
    auto input_norm = graph->add_node(OpType::RMS_NORM,
                                      "layer_" + std::to_string(layer_idx) + "_input_norm");
    input_norm->add_input(input);
    input_norm->set_weight(model->weights().get_tensor(layer_weights[10]));
    input_norm->set_param("hidden_size", config.hidden_size);
    input_norm->set_param("eps", config.rms_norm_eps);
    numel += config.hidden_size;

    // Multi-head self-attention
    auto attn_output = build_attention(
        graph, input_norm, model, layer_idx, k_cache, v_cache, position_ids, numel);

    // First residual connection (attention output)
    auto residual_1 = graph->add_node(OpType::ADD,
                                      "layer_" + std::to_string(layer_idx) + "_residual_1");
    residual_1->add_input(input);
    residual_1->add_input(attn_output);
    residual_1->set_param("hidden_size", config.hidden_size);
    numel += config.hidden_size;

    // Pre-MLP normalization
    auto post_attn_norm = graph->add_node(OpType::RMS_NORM,
                                          "layer_" + std::to_string(layer_idx) + "_post_attn_norm");
    post_attn_norm->add_input(residual_1);
    post_attn_norm->set_weight(model->weights().get_tensor(layer_weights[11]));
    post_attn_norm->set_param("hidden_size", config.hidden_size);
    post_attn_norm->set_param("eps", config.rms_norm_eps);
    numel += config.hidden_size;

    // Feed-forward network
    auto mlp_output = build_mlp(graph, post_attn_norm, model, layer_idx, numel);

    // Second residual connection (MLP output)
    auto residual_2 = graph->add_node(OpType::ADD,
                                      "layer_" + std::to_string(layer_idx) + "_residual_2");
    residual_2->add_input(residual_1);
    residual_2->add_input(mlp_output);
    residual_2->set_param("hidden_size", config.hidden_size);
    numel += config.hidden_size;

    return residual_2;
}

graph_node_t Qwen2GraphBuilder::build_attention(
    compute_graph_t graph,
    graph_node_t hidden_states,
    const model::Model *model,
    int layer_idx,
    graph_node_t k_cache,
    graph_node_t v_cache,
    graph_node_t position_ids,
    size_t &numel) {

    const auto &config = model->config();
    const auto &weights = model->weights();
    auto weight_names = model->get_layer_weight_names(layer_idx);
    std::string prefix = "layer_" + std::to_string(layer_idx) + "_";

    // Query projection
    auto q_proj = graph->add_node(OpType::LINEAR, prefix + "q_proj");
    q_proj->add_input(hidden_states);
    q_proj->set_weight(weights.get_tensor(weight_names[0]));
    q_proj->set_bias(weights.get_tensor(weight_names[1]));
    q_proj->set_param("hidden_size", config.hidden_size);
    numel += config.hidden_size;

    // Key projection (for GQA, uses fewer heads)
    auto k_proj = graph->add_node(OpType::LINEAR, prefix + "k_proj");
    k_proj->add_input(hidden_states);
    k_proj->set_weight(weights.get_tensor(weight_names[2]));
    k_proj->set_bias(weights.get_tensor(weight_names[3]));
    auto hidden_dim = config.hidden_size / config.num_attention_heads * config.num_key_value_heads;
    k_proj->set_param("hidden_dim", hidden_dim);
    numel += hidden_dim;

    // Value projection (for GQA, uses fewer heads)
    auto v_proj = graph->add_node(OpType::LINEAR, prefix + "v_proj");
    v_proj->add_input(hidden_states);
    v_proj->set_weight(weights.get_tensor(weight_names[4]));
    v_proj->set_bias(weights.get_tensor(weight_names[5]));
    v_proj->set_param("hidden_dim", hidden_dim);

    // Apply rotary position encoding to queries
    auto q_rope = graph->add_node(OpType::ROPE, prefix + "q_rope");
    q_rope->add_input(q_proj);
    q_rope->add_input(position_ids);
    q_rope->set_param("theta", config.rope_theta);
    q_rope->set_param("nhead", config.num_attention_heads);
    auto head_dim = config.hidden_size / config.num_attention_heads;
    q_rope->set_param("head_dim", head_dim);
    numel += config.num_attention_heads * head_dim;

    // Apply rotary position encoding to keys
    auto k_rope = graph->add_node(OpType::ROPE, prefix + "k_rope");
    k_rope->add_input(k_proj);
    k_rope->add_input(position_ids);
    k_rope->set_param("theta", config.rope_theta);
    k_rope->set_param("nkvhead", config.num_key_value_heads);
    k_rope->set_param("head_dim", head_dim);

    // Grouped query attention computation with KV cache
    auto attn_node = graph->add_node(OpType::SELF_ATTENTION, prefix + "self_attn");
    attn_node->add_input(q_rope);
    attn_node->add_input(k_cache);
    attn_node->add_input(v_cache);
    attn_node->set_param("nhead", config.num_attention_heads);
    attn_node->set_param("head_dim", head_dim);
    numel += config.num_attention_heads * head_dim;

    // Output projection
    auto o_proj = graph->add_node(OpType::LINEAR, prefix + "o_proj");
    o_proj->add_input(attn_node);
    o_proj->set_weight(weights.get_tensor(weight_names[6]));
    o_proj->set_bias(nullptr);
    o_proj->set_param("hidden_size", config.hidden_size);
    numel += config.hidden_size;

    return o_proj;
}

graph_node_t Qwen2GraphBuilder::build_mlp(
    compute_graph_t graph,
    graph_node_t hidden_states,
    const model::Model *model,
    int layer_idx, size_t &numel) {

    const auto &config = model->config();
    const auto &weights = model->weights();
    auto weight_names = model->get_layer_weight_names(layer_idx);
    std::string prefix = "layer_" + std::to_string(layer_idx) + "_";

    // Gate projection for SwiGLU
    auto gate_proj = graph->add_node(OpType::LINEAR, prefix + "gate_proj");
    gate_proj->add_input(hidden_states);
    gate_proj->set_weight(weights.get_tensor(weight_names[7]));
    gate_proj->set_bias(nullptr);
    gate_proj->set_param("hidden_size", config.hidden_size);
    gate_proj->set_param("intermediate_size", config.intermediate_size);
    numel += config.intermediate_size;

    // Up projection for SwiGLU
    auto up_proj = graph->add_node(OpType::LINEAR, prefix + "up_proj");
    up_proj->add_input(hidden_states);
    up_proj->set_weight(weights.get_tensor(weight_names[8]));
    up_proj->set_bias(nullptr);
    up_proj->set_param("hidden_size", config.hidden_size);
    up_proj->set_param("intermediate_size", config.intermediate_size);
    numel += config.intermediate_size;

    // SwiGLU activation: SiLU(gate) * up
    auto swiglu_node = graph->add_node(OpType::SWIGLU, prefix + "swiglu");
    swiglu_node->add_input(gate_proj);
    swiglu_node->add_input(up_proj);
    swiglu_node->set_param("intermediate_size", config.intermediate_size);
    numel += config.intermediate_size;

    // Down projection back to hidden size
    auto down_proj = graph->add_node(OpType::LINEAR, prefix + "down_proj");
    down_proj->add_input(swiglu_node);
    down_proj->set_weight(weights.get_tensor(weight_names[9]));
    down_proj->set_bias(nullptr);
    down_proj->set_param("hidden_size", config.hidden_size);
    down_proj->set_param("intermediate_size", config.intermediate_size);
    numel += config.hidden_size;

    return down_proj;
}

compute_graph_t Qwen3GraphBuilder::build(const model::Model *model) {
    auto graph = std::make_shared<ComputeGraph>();
    size_t per_token_activation_numel = 0;

    const auto &config = model->config();

    // Create input nodes
    auto input_ids = graph->add_node(OpType::INPUT, "input_ids");
    auto position_ids = graph->add_node(OpType::INPUT, "position_ids");

    graph->set_input("input_ids", input_ids);
    graph->set_input("position_ids", position_ids);

    // Token embedding layer
    auto embedding_node = graph->add_node(OpType::EMBEDDING, "embedding");
    embedding_node->add_input(input_ids);
    embedding_node->set_weight(model->weights().get_tensor(
        model->get_embedding_weight_name()));
    embedding_node->set_param("hidden_size", config.hidden_size);
    per_token_activation_numel += config.hidden_size;

    // Build transformer layers
    auto hidden_states = embedding_node;

    for (size_t i = 0; i < config.num_hidden_layers; ++i) {
        // Create KV cache input nodes for each layer
        auto k_cache_name = "layer_" + std::to_string(i) + "_k_cache";
        auto v_cache_name = "layer_" + std::to_string(i) + "_v_cache";

        auto k_cache_node = graph->add_node(OpType::INPUT, k_cache_name);
        auto v_cache_node = graph->add_node(OpType::INPUT, v_cache_name);

        graph->set_input(k_cache_name, k_cache_node);
        graph->set_input(v_cache_name, v_cache_node);

        hidden_states = build_transformer_layer(
            graph, hidden_states, model, i, k_cache_node, v_cache_node, position_ids, per_token_activation_numel);
    }

    // Final normalization layer
    auto norm_node = graph->add_node(OpType::RMS_NORM, "output_norm");
    norm_node->add_input(hidden_states);
    norm_node->set_weight(model->weights().get_tensor(
        model->get_output_norm_weight_name()));
    norm_node->set_param("hidden_size", config.hidden_size);
    norm_node->set_param("eps", config.rms_norm_eps);
    per_token_activation_numel += config.hidden_size;

    // Language model head (vocabulary projection)
    auto lm_head = graph->add_node(OpType::LINEAR, "lm_head");
    lm_head->add_input(norm_node);
    lm_head->set_weight(model->weights().get_tensor(
        model->get_output_weight_name()));
    lm_head->set_bias(nullptr);
    lm_head->set_param("vocab_size", config.vocab_size);
    per_token_activation_numel += config.vocab_size;

    graph->set_output("logits", lm_head);

    graph->set_per_token_activation_numel(per_token_activation_numel);

    return graph;
}

graph_node_t Qwen3GraphBuilder::build_transformer_layer(
    compute_graph_t graph,
    graph_node_t input,
    const model::Model *model,
    int layer_idx,
    graph_node_t k_cache,
    graph_node_t v_cache,
    graph_node_t position_ids, size_t &numel) {

    const auto &config = model->config();
    auto layer_weights = model->get_layer_weight_names(layer_idx);

    // Pre-attention normalization
    auto input_norm = graph->add_node(OpType::RMS_NORM,
                                      "layer_" + std::to_string(layer_idx) + "_input_norm");
    input_norm->add_input(input);
    input_norm->set_weight(model->weights().get_tensor(layer_weights[9]));
    input_norm->set_param("hidden_size", config.hidden_size);
    input_norm->set_param("eps", config.rms_norm_eps);
    numel += config.hidden_size;

    // Multi-head self-attention
    auto attn_output = build_attention(
        graph, input_norm, model, layer_idx, k_cache, v_cache, position_ids, numel);

    // First residual connection (attention output)
    auto residual_1 = graph->add_node(OpType::ADD,
                                      "layer_" + std::to_string(layer_idx) + "_residual_1");
    residual_1->add_input(input);
    residual_1->add_input(attn_output);
    residual_1->set_param("hidden_size", config.hidden_size);
    numel += config.hidden_size;

    // Pre-MLP normalization
    auto post_attn_norm = graph->add_node(OpType::RMS_NORM,
                                          "layer_" + std::to_string(layer_idx) + "_post_attn_norm");
    post_attn_norm->add_input(residual_1);
    post_attn_norm->set_weight(model->weights().get_tensor(layer_weights[10]));
    post_attn_norm->set_param("hidden_size", config.hidden_size);
    post_attn_norm->set_param("eps", config.rms_norm_eps);
    numel += config.hidden_size;

    // Feed-forward network
    auto mlp_output = build_mlp(graph, post_attn_norm, model, layer_idx, numel);

    // Second residual connection (MLP output)
    auto residual_2 = graph->add_node(OpType::ADD,
                                      "layer_" + std::to_string(layer_idx) + "_residual_2");
    residual_2->add_input(residual_1);
    residual_2->add_input(mlp_output);
    residual_2->set_param("hidden_size", config.hidden_size);
    numel += config.hidden_size;

    return residual_2;
}

graph_node_t Qwen3GraphBuilder::build_attention(
    compute_graph_t graph,
    graph_node_t hidden_states,
    const model::Model *model,
    int layer_idx,
    graph_node_t k_cache,
    graph_node_t v_cache,
    graph_node_t position_ids,
    size_t &numel) {

    const auto &config = model->config();
    const auto &weights = model->weights();
    auto weight_names = model->get_layer_weight_names(layer_idx);
    std::string prefix = "layer_" + std::to_string(layer_idx) + "_";

    // Query projection
    auto q_proj = graph->add_node(OpType::LINEAR, prefix + "q_proj");
    q_proj->add_input(hidden_states);
    q_proj->set_weight(weights.get_tensor(weight_names[0]));
    q_proj->set_param("hidden_size", config.hidden_size);
    numel += config.hidden_size;

    // Key projection (for GQA, uses fewer heads)
    auto k_proj = graph->add_node(OpType::LINEAR, prefix + "k_proj");
    k_proj->add_input(hidden_states);
    k_proj->set_weight(weights.get_tensor(weight_names[2]));
    auto hidden_dim = config.hidden_size / config.num_attention_heads * config.num_key_value_heads;
    k_proj->set_param("hidden_dim", hidden_dim);
    numel += hidden_dim;

    // Value projection (for GQA, uses fewer heads)
    auto v_proj = graph->add_node(OpType::LINEAR, prefix + "v_proj");
    v_proj->add_input(hidden_states);
    v_proj->set_weight(weights.get_tensor(weight_names[4]));
    v_proj->set_param("hidden_dim", hidden_dim);

    auto head_dim = config.hidden_size / config.num_attention_heads;

    // Per-head RMS normalization for query projections
    auto q_norm = graph->add_node(OpType::RMS_NORM, prefix + "q_norm");
    q_norm->add_input(q_proj);
    q_norm->set_weight(weights.get_tensor(weight_names[1]));
    q_norm->set_param("hidden_size", config.hidden_size);
    q_norm->set_param("nhead", config.num_attention_heads);
    q_norm->set_param("head_dim", head_dim);
    q_norm->set_param("eps", config.rms_norm_eps);
    numel += config.num_attention_heads * head_dim;

    // Per-head RMS normalization for key projections
    auto k_norm = graph->add_node(OpType::RMS_NORM, prefix + "k_norm");
    k_norm->add_input(k_proj);
    k_norm->set_weight(weights.get_tensor(weight_names[3]));
    k_norm->set_param("hidden_size", config.hidden_size);
    k_norm->set_param("nkvhead", config.num_key_value_heads);
    k_norm->set_param("head_dim", head_dim);
    k_norm->set_param("eps", config.rms_norm_eps);
    numel += hidden_dim;

    // Apply rotary position encoding to queries
    auto q_rope = graph->add_node(OpType::ROPE, prefix + "q_rope");
    q_rope->add_input(q_norm);
    q_rope->add_input(position_ids);
    q_rope->set_param("theta", config.rope_theta);
    q_rope->set_param("nhead", config.num_attention_heads);
    q_rope->set_param("head_dim", head_dim);
    numel += config.num_attention_heads * head_dim;

    // Apply rotary position encoding to keys
    auto k_rope = graph->add_node(OpType::ROPE, prefix + "k_rope");
    k_rope->add_input(k_norm);
    k_rope->add_input(position_ids);
    k_rope->set_param("theta", config.rope_theta);
    k_rope->set_param("nkvhead", config.num_key_value_heads);
    k_rope->set_param("head_dim", head_dim);

    // Grouped query attention computation with KV cache
    auto attn_node = graph->add_node(OpType::SELF_ATTENTION, prefix + "self_attn");
    attn_node->add_input(q_rope);
    attn_node->add_input(k_cache);
    attn_node->add_input(v_cache);
    attn_node->set_param("nhead", config.num_attention_heads);
    attn_node->set_param("head_dim", head_dim);
    numel += config.num_attention_heads * head_dim;

    // Output projection
    auto o_proj = graph->add_node(OpType::LINEAR, prefix + "o_proj");
    o_proj->add_input(attn_node);
    o_proj->set_weight(weights.get_tensor(weight_names[5]));
    o_proj->set_bias(nullptr);
    o_proj->set_param("hidden_size", config.hidden_size);
    numel += config.hidden_size;

    return o_proj;
}

graph_node_t Qwen3GraphBuilder::build_mlp(
    compute_graph_t graph,
    graph_node_t hidden_states,
    const model::Model *model,
    int layer_idx, size_t &numel) {

    const auto &config = model->config();
    const auto &weights = model->weights();
    auto weight_names = model->get_layer_weight_names(layer_idx);
    std::string prefix = "layer_" + std::to_string(layer_idx) + "_";

    // Gate projection for SwiGLU
    auto gate_proj = graph->add_node(OpType::LINEAR, prefix + "gate_proj");
    gate_proj->add_input(hidden_states);
    gate_proj->set_weight(weights.get_tensor(weight_names[6]));
    gate_proj->set_bias(nullptr);
    gate_proj->set_param("hidden_size", config.hidden_size);
    gate_proj->set_param("intermediate_size", config.intermediate_size);
    numel += config.intermediate_size;

    // Up projection for SwiGLU
    auto up_proj = graph->add_node(OpType::LINEAR, prefix + "up_proj");
    up_proj->add_input(hidden_states);
    up_proj->set_weight(weights.get_tensor(weight_names[7]));
    up_proj->set_bias(nullptr);
    up_proj->set_param("hidden_size", config.hidden_size);
    up_proj->set_param("intermediate_size", config.intermediate_size);
    numel += config.intermediate_size;

    // SwiGLU activation: SiLU(gate) * up
    auto swiglu_node = graph->add_node(OpType::SWIGLU, prefix + "swiglu");
    swiglu_node->add_input(gate_proj);
    swiglu_node->add_input(up_proj);
    swiglu_node->set_param("intermediate_size", config.intermediate_size);
    numel += config.intermediate_size;

    // Down projection back to hidden size
    auto down_proj = graph->add_node(OpType::LINEAR, prefix + "down_proj");
    down_proj->add_input(swiglu_node);
    down_proj->set_weight(weights.get_tensor(weight_names[8]));
    down_proj->set_bias(nullptr);
    down_proj->set_param("hidden_size", config.hidden_size);
    down_proj->set_param("intermediate_size", config.intermediate_size);
    numel += config.hidden_size;

    return down_proj;
}

} // namespace zedinfer::graph