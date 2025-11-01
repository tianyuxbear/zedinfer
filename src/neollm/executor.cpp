#include "neollm/executor.hpp"
#include "backend/ops/ops.hpp"
#include "neollm.h"
#include "utils/types.hpp"

#include <cmath>
#include <iomanip>
#include <iostream>
#include <map>

namespace neollm::graph {

// ==================== ActivationPool ====================

ActivationPool::ActivationPool(const ExecutorConfig &config)
    : config_(config) {
    pool_.reserve(128);
}

tensor_t ActivationPool::acquire(
    const std::vector<size_t> &shape) {

    // Find available tensor with matching shape
    for (auto &slot : pool_) {
        if (!slot.in_use && slot.shape == shape) {
            slot.in_use = true;
            return slot.tensor;
        }
    }

    // Allocate new tensor if none available
    auto tensor = Tensor::create(
        shape,
        config_.dtype,
        config_.device_type,
        config_.device_id,
        false,
        nullptr);

    TensorSlot slot;
    slot.tensor = tensor;
    slot.shape = shape;
    slot.in_use = true;

    pool_.push_back(slot);

    return tensor;
}

void ActivationPool::release(tensor_t tensor) {
    for (auto &slot : pool_) {
        if (slot.tensor == tensor) {
            slot.in_use = false;
            return;
        }
    }
}

void ActivationPool::release_all() {
    for (auto &slot : pool_) {
        slot.in_use = false;
    }
}

size_t ActivationPool::tensors_in_use() const {
    size_t count = 0;
    for (const auto &slot : pool_) {
        if (slot.in_use) {
            count++;
        }
    }
    return count;
}

size_t ActivationPool::total_memory() const {
    size_t total = 0;
    for (const auto &slot : pool_) {
        total += slot.tensor->numel() * utils::dsize(config_.dtype);
    }
    return total;
}

void ActivationPool::print_stats() const {
    std::cout << "\n=== Activation Pool Statistics ===" << std::endl;
    std::cout << "  Total tensors: " << pool_.size() << std::endl;
    std::cout << "  Tensors in use: " << tensors_in_use() << std::endl;
    std::cout << "  Total memory: " << std::fixed << std::setprecision(2)
              << (total_memory() / 1024.0 / 1024.0) << " MB" << std::endl;

    // Shape distribution
    std::map<std::vector<size_t>, int> shape_counts;
    for (const auto &slot : pool_) {
        shape_counts[slot.shape]++;
    }

    if (!shape_counts.empty()) {
        std::cout << "  Shape distribution:" << std::endl;
        for (const auto &[shape, count] : shape_counts) {
            std::cout << "    [";
            for (size_t i = 0; i < shape.size(); ++i) {
                std::cout << shape[i];
                if (i < shape.size() - 1) {
                    std::cout << ", ";
                }
            }
            std::cout << "]: " << count << " tensors" << std::endl;
        }
    }
}

// ==================== PositionIDsCache ====================

PositionIDsCache::PositionIDsCache(int max_seq_len, const ExecutorConfig &config)
    : max_seq_len_(max_seq_len), config_(config) {

    initialize_cache();
}

void PositionIDsCache::initialize_cache() {
    // Create [0, 1, 2, ..., max_seq_len-1]
    cache_ = Tensor::create(
        {static_cast<size_t>(max_seq_len_)},
        NEOLLM_DTYPE_I64,
        config_.device_type,
        config_.device_id,
        false,
        nullptr);

    std::vector<int64_t> positions(max_seq_len_);
    for (int i = 0; i < max_seq_len_; ++i) {
        positions[i] = i;
    }

    cache_->load(positions.data());
}

tensor_t PositionIDsCache::get_slice(int start_pos, int seq_len) {
    if (start_pos < 0 || seq_len < 0) {
        throw std::invalid_argument("start_pos and seq_len must be non-negative");
    }

    if (start_pos + seq_len > max_seq_len_) {
        throw std::out_of_range(
            "Position range exceeds cache size: " + std::to_string(start_pos + seq_len) + " > " + std::to_string(max_seq_len_));
    }

    // Zero-copy slice
    return cache_->slice(0, start_pos, start_pos + seq_len);
}

// ==================== GraphExecutor ====================

GraphExecutor::GraphExecutor(
    compute_graph_t graph,
    kvcache::kvcache_t kv_cache,
    const ExecutorConfig &config)
    : graph_(graph),
      kv_cache_(kv_cache),
      config_(config) {

    // Initialize activation pool
    activation_pool_ = std::make_unique<ActivationPool>(config_);

    // Initialize position_ids cache with generous default
    int pos_cache_len = 8192;
    position_ids_cache_ = std::make_unique<PositionIDsCache>(pos_cache_len, config_);

    std::cout << "[GraphExecutor] Initialized:" << std::endl;
    std::cout << "  Device: " << (config_.device_type == NEOLLM_DEVICE_CPU ? "CPU" : "GPU")
              << " (ID: " << config_.device_id << ")" << std::endl;
    std::cout << "  Dtype: ";
    switch (config_.dtype) {
    case NEOLLM_DTYPE_BF16:
        std::cout << "BF16";
        break;
    case NEOLLM_DTYPE_F16:
        std::cout << "FP16";
        break;
    case NEOLLM_DTYPE_F32:
        std::cout << "FP32";
        break;
    default:
        std::cout << "Unknown";
    }
    std::cout << std::endl;
    std::cout << "  Activation pool: ON" << std::endl;
    std::cout << "  Position IDs cache: ON (max_len=" << pos_cache_len << ")" << std::endl;
}

std::unique_ptr<GraphExecutor> GraphExecutor::create(
    compute_graph_t graph,
    kvcache::kvcache_t kv_cache) {
    auto config = ExecutorConfig::from_kvcache_config(kv_cache->config());
    // Static member can access private constructor
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

    for (auto &node : execution_order) {
        if (activations.find(node) != activations.end()) {
            continue; // Skip input nodes
        }

        execute_node(node, activations, past_len, seq_len);
    }

    // ==================== 5. Release activations ====================
    activation_pool_->release_all();

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
    int past_len,
    int seq_len) {

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

    // Dispatch by OpType
    switch (node->op_type()) {
    case OpType::ADD: {
        shape = inputs[0]->shape();
        output = activation_pool_->acquire(shape);
        ops::add(output, inputs[0], inputs[1]);
        break;
    }
    case OpType::EMBEDDING: {
        shape = {static_cast<size_t>(seq_len), node->get_param<size_t>("hidden_size")};
        output = activation_pool_->acquire(shape);
        auto weight = node->weight();
        ops::embedding(output, inputs[0], weight);
        break;
    }
    case OpType::LINEAR: {
        bool is_v_proj = node_name.find("v_proj") != std::string::npos;
        if (is_v_proj) {
            output = kv_cache_->get_v_cache_write_slice(extract_layer_idx(node_name), past_len, seq_len);
            shape = {static_cast<size_t>(seq_len), node->get_param<size_t>("hidden_dim")};
            output = output->view(shape);
        } else {
            if (node_name == "lm_head") {
                shape = {static_cast<size_t>(seq_len), node->get_param<size_t>("vocab_size")};
            } else if (node_name.find("q_proj") != std::string::npos || node_name.find("o_proj") != std::string::npos || node_name.find("down_proj") != std::string::npos) {
                shape = {static_cast<size_t>(seq_len), node->get_param<size_t>("hidden_size")};
            } else if (node_name.find("k_proj") != std::string::npos) {
                shape = {static_cast<size_t>(seq_len), node->get_param<size_t>("hidden_dim")};
            } else if (node_name.find("gate_proj") != std::string::npos || node_name.find("up_proj") != std::string::npos) {
                shape = {static_cast<size_t>(seq_len), node->get_param<size_t>("intermediate_size")};
            }
            output = activation_pool_->acquire(shape);
        }
        auto weight = node->weight();
        auto bias = node->bias();
        ops::linear(output, inputs[0], weight, bias);
        break;
    }

    case OpType::RMS_NORM: {
        shape = {static_cast<size_t>(seq_len), node->get_param<size_t>("hidden_size")};
        output = activation_pool_->acquire(shape);
        auto weight = node->weight();
        float eps = node->get_param<float>("eps");
        ops::rms_norm(output, inputs[0], weight, eps);
        break;
    }

    case OpType::ROPE: {
        // Check if K's RoPE (write directly to KV cache)
        bool is_k_rope = (node_name.find("k_rope") != std::string::npos);
        if (is_k_rope) {
            shape = {static_cast<size_t>(seq_len), node->get_param<size_t>("nkvhead"), node->get_param<size_t>("head_dim")};
            int layer_idx = extract_layer_idx(node_name);
            output = kv_cache_->get_k_cache_write_slice(layer_idx, past_len, seq_len);
        } else {
            shape = {static_cast<size_t>(seq_len), node->get_param<size_t>("nhead"), node->get_param<size_t>("head_dim")};
            output = activation_pool_->acquire(shape);
        }

        float theta = node->get_param<float>("theta");
        ops::rope(output, inputs[0]->view(shape), inputs[1], theta);
        break;
    }

    case OpType::SELF_ATTENTION: {
        shape = {static_cast<size_t>(seq_len), node->get_param<size_t>("nhead"), node->get_param<size_t>("head_dim")};
        output = activation_pool_->acquire(shape);
        float scale = 1.0f / std::sqrt(static_cast<float>(node->get_param<size_t>("head_dim")));
        ops::self_attention(output, inputs[0], inputs[1], inputs[2], scale);
        output = output->view({static_cast<size_t>(seq_len), node->get_param<size_t>("nhead") * node->get_param<size_t>("head_dim")});
        break;
    }

    case OpType::SWIGLU: {
        shape = {static_cast<size_t>(seq_len), node->get_param<size_t>("intermediate_size")};
        output = activation_pool_->acquire(shape);
        ops::swiglu(output, inputs[0], inputs[1]);
        break;
    }

    default:
        throw std::runtime_error(
            "Unsupported op type: " + std::to_string(static_cast<int>(node->op_type())));
    }

    activations[node] = output;
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
    std::cout << "\n=== GraphExecutor Statistics ===" << std::endl;
    std::cout << "  Device: " << (config_.device_type == NEOLLM_DEVICE_CPU ? "CPU" : "GPU")
              << " (ID: " << config_.device_id << ")" << std::endl;
    std::cout << "  Default dtype: ";
    switch (config_.dtype) {
    case NEOLLM_DTYPE_BF16:
        std::cout << "BF16";
        break;
    case NEOLLM_DTYPE_F16:
        std::cout << "FP16";
        break;
    case NEOLLM_DTYPE_F32:
        std::cout << "FP32";
        break;
    default:
        std::cout << "Unknown";
    }
    std::cout << std::endl;

    // Pool statistics
    activation_pool_->print_stats();

    // Cache statistics
    kv_cache_->print_stats();
}

} // namespace neollm::graph