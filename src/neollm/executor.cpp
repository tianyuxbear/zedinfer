#include "neollm/executor.hpp"
#include "backend/ops/ops.hpp"
#include "neollm.h"
#include "neollm/activation.hpp"
#include "utils/types.hpp"

#include <cmath>
#include <iostream>
#include <memory>
#include <plog/Log.h>
#include <sstream>

namespace neollm {

using namespace graph;

// ==================== GraphExecutor ====================
GraphExecutor::GraphExecutor(
    compute_graph_t graph,
    kvcache::kvcache_t kv_cache,
    const ExecutorConfig &config)
    : graph_(graph),
      kv_cache_(kv_cache),
      config_(config) {
    // Allocate arena memory for prefill-phase activations.
    const size_t per_token_bytes = graph_->get_per_token_activation_numel() * utils::dsize(config_.data_type);
    const size_t capacity = per_token_bytes * config_.max_prefill_len;
    prefill_arena_ = std::make_unique<PrefillArena>(config_, capacity);

    // Initialize tensor pool for decode-phase activations.
    decode_pool_ = std::make_unique<DecodePool>(config_);

    // Pre-allocate reusable position ID tensor.
    position_ids_cache_ = std::make_unique<PositionIDsCache>(config_);

    graph->optimize();
}

std::unique_ptr<GraphExecutor> GraphExecutor::create(
    compute_graph_t graph,
    kvcache::kvcache_t kv_cache,
    ExecutorConfig config) {
    // Validate input arguments to ensure safe construction.
    if (!graph) {
        throw std::invalid_argument("Graph must not be null.");
    }
    if (!kv_cache) {
        throw std::invalid_argument("KV cache must not be null.");
    }
    if (config.max_prefill_len <= 0) {
        throw std::invalid_argument("max_prefill_len must be positive.");
    }
    if (config.max_seq_len <= 0) {
        throw std::invalid_argument("max_seq_len must be positive.");
    }

    // Construct GraphExecutor safely with validated inputs.
    return std::unique_ptr<GraphExecutor>(
        new GraphExecutor(graph, kv_cache, config));
}

tensor_t GraphExecutor::forward(
    const std::vector<int> &input_ids,
    int past_len) {

    int seq_len = input_ids.size();

    if (seq_len == 0) {
        throw std::invalid_argument("input_ids cannot be empty");
    }

    // Activation storage
    std::unordered_map<graph_node_t, tensor_t> activations;

    // ==================== 1. Prepare input_ids ====================
    auto input_ids_node = graph_->get_input("input_ids");
    if (!input_ids_node) {
        throw std::runtime_error("Graph input 'input_ids' not found");
    }

    auto input_ids_tensor = Tensor::create(
        {static_cast<size_t>(seq_len)},
        NEOLLM_DTYPE_I32,
        config_.device_type,
        config_.device_id,
        false,
        nullptr);
    input_ids_tensor->load(input_ids.data());
    activations[input_ids_node] = input_ids_tensor;

    // ==================== 2. Prepare position_ids ====================
    auto position_ids_node = graph_->get_input("position_ids");
    if (!position_ids_node) {
        throw std::runtime_error("Graph input 'position_ids' not found");
    }

    // Zero-copy from cache
    auto position_ids_tensor = position_ids_cache_->get_slice(past_len, seq_len);
    activations[position_ids_node] = position_ids_tensor;

    // ==================== 3. Prepare KV cache ====================
    for (int layer_idx = 0; layer_idx < kv_cache_->config().num_layers; ++layer_idx) {
        auto k_cache_node = graph_->get_input(
            "layer_" + std::to_string(layer_idx) + "_k_cache");
        auto v_cache_node = graph_->get_input(
            "layer_" + std::to_string(layer_idx) + "_v_cache");
        activations[k_cache_node] = kv_cache_->get_k_cache_slice(layer_idx, past_len + seq_len);
        activations[v_cache_node] = kv_cache_->get_v_cache_slice(layer_idx, past_len + seq_len);
    }

    // ==================== 4. Execute graph ====================
    auto execution_order = graph_->get_execution_order();

    // Prepare execution context
    ExecutionContext ctx{1, seq_len, past_len};

    for (auto &node : execution_order) {
        if (activations.find(node) != activations.end()) {
            continue; // Skip input nodes
        }

        execute_node(node, activations, ctx);
    }

    // ==================== 5. Release activations ====================
    prefill_arena_->reset();
    decode_pool_->reset();

    // ==================== 6. Update KV cache length ====================
    kv_cache_->update_seq_len(seq_len);

    // ==================== 7. Return logits ====================
    auto logits_node = graph_->get_output("logits");
    if (!logits_node) {
        throw std::runtime_error("Graph output 'logits' not found");
    }

    return activations[logits_node];
}

void GraphExecutor::execute_node(
    graph_node_t node,
    std::unordered_map<graph_node_t, tensor_t> &activations,
    ExecutionContext &ctx) {

    // Select activation allocator based on phase
    ActivationAllocator *activation_allocator;
    if (ctx.seq_len == 1) {
        activation_allocator = decode_pool_.get();
    } else {
        activation_allocator = prefill_arena_.get();
    }

    // Collect input tensors
    std::vector<tensor_t> inputs;
    for (const auto &input_node : node->inputs()) {
        auto it = activations.find(input_node);
        if (it == activations.end()) {
            throw std::runtime_error(
                "Input activation not found for node: " + node->name());
        }
        inputs.push_back(it->second);
    }

    tensor_t output;
    std::vector<size_t> shape;
    std::string node_name = node->name();

    // Handle special case: write directly to KV cache
    bool is_v_proj = node_name.find("v_proj") != std::string::npos;
    bool is_k_rope = node_name.find("k_rope") != std::string::npos;
    bool write_to_kv_cache = is_k_rope || is_v_proj;

    if (write_to_kv_cache) {
        int layer_idx = extract_layer_idx(node_name);

        output = is_v_proj
                   ? kv_cache_->get_v_cache_write_slice(layer_idx, ctx.past_len, ctx.seq_len)
                   : kv_cache_->get_k_cache_write_slice(layer_idx, ctx.past_len, ctx.seq_len);

        // Resolve shape from template and reshape
        shape = node->get_output_shape_template().resolve(ctx);
        output = output->view(shape);
    } else {
        // Resolve shape from cached template
        shape = node->get_output_shape_template().resolve(ctx);
        output = activation_allocator->acquire(shape);
    }

    // Reshape ROPE input to match expected 3D format
    bool is_rope = node_name.find("rope") != std::string::npos;
    if (is_rope) {
        inputs[0] = inputs[0]->view(shape);
    }

    // Execute operator
    execute_op(node, inputs, output);

    // Flatten SELF_ATTENTION output
    if (node->op_type() == OpType::SELF_ATTENTION) {
        size_t nhead = node->get_param<size_t>("nhead");
        size_t head_dim = node->get_param<size_t>("head_dim");
        output = output->view({static_cast<size_t>(ctx.seq_len), nhead * head_dim});
    }

    activations[node] = output;
}

void GraphExecutor::execute_op(
    graph_node_t node,
    const std::vector<tensor_t> &inputs,
    tensor_t output) {

    switch (node->op_type()) {
    case OpType::ADD:
        ops::add(output, inputs[0], inputs[1]);
        break;

    case OpType::EMBEDDING:
        ops::embedding(output, inputs[0], node->weight());
        break;

    case OpType::LINEAR:
        ops::linear(output, inputs[0], node->weight(), node->bias());
        break;

    case OpType::RMS_NORM: {
        float eps = node->get_param<float>("eps");
        ops::rms_norm(output, inputs[0], node->weight(), eps);
        break;
    }

    case OpType::ROPE: {
        float theta = node->get_param<float>("theta");
        ops::rope(output, inputs[0], inputs[1], theta);
        break;
    }

    case OpType::SELF_ATTENTION: {
        float scale = 1.0f / std::sqrt(static_cast<float>(node->get_param<size_t>("head_dim")));
        ops::self_attention(output, inputs[0], inputs[1], inputs[2], scale);
        break;
    }

    case OpType::SWIGLU:
        ops::swiglu(output, inputs[0], inputs[1]);
        break;

    default:
        throw std::runtime_error(
            "Unsupported op type: " + std::to_string(static_cast<int>(node->op_type())));
    }
}

int GraphExecutor::extract_layer_idx(const std::string &node_name) {
    size_t layer_pos = node_name.find("layer_");
    if (layer_pos == std::string::npos) {
        throw std::runtime_error("Cannot extract layer_idx from: " + node_name);
    }

    size_t start = layer_pos + 6;
    size_t end = node_name.find("_", start);

    std::string layer_str = node_name.substr(start, end - start);
    return std::stoi(layer_str);
}

void GraphExecutor::print_stats() const {
    std::ostringstream oss;

    oss << "\n=== GraphExecutor Statistics ===\n";
    oss << "  Device: " << (config_.device_type == NEOLLM_DEVICE_CPU ? "CPU" : "GPU")
        << " (ID: " << config_.device_id << ")" << std::endl;
    oss << "  Default dtype: ";
    switch (config_.data_type) {
    case NEOLLM_DTYPE_BF16:
        oss << "BF16";
        break;
    case NEOLLM_DTYPE_F16:
        oss << "FP16";
        break;
    case NEOLLM_DTYPE_F32:
        oss << "FP32";
        break;
    default:
        oss << "Unknown";
    }
    oss << std::endl;

    LOGI << oss.str();

    // Pool statistics
    LOGI << prefill_arena_->get_stats();
    LOGI << decode_pool_->get_stats();

    // KV Cache statistics
    LOGI << kv_cache_->get_stats();
}

} // namespace neollm