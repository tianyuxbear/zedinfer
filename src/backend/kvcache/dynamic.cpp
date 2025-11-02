#include "backend/kvcache/dynamic.hpp"
#include "backend/core/context/context.hpp"
#include "neollm.h"
#include "utils/check.hpp"
#include "utils/types.hpp"
#include <chrono>
#include <cstring>
#include <iostream>
#include <sstream>

namespace neollm::kvcache {

void DynamicKVCacheConfig::validate() const {
    KVCacheConfig::validate();

    if (initial_capacity <= 0) {
        throw std::invalid_argument("initial_capacity must be positive");
    }
    if (model_max_seq_len < initial_capacity) {
        throw std::invalid_argument("model_max_seq_len must be >= initial_capacity");
    }
    if (aggressive_target_capacity > model_max_seq_len) {
        throw std::invalid_argument("aggressive_target_capacity must be <= model_max_seq_len");
    }
}

DynamicKVCacheManager::DynamicKVCacheManager(const DynamicKVCacheConfig &config)
    : KVCacheManager(config),
      dynamic_config_(config),
      allocated_capacity_(config.initial_capacity),
      growth_count_(0),
      total_growth_time_ms_(0.0) {

    dynamic_config_.validate();
    allocate_cache(dynamic_config_.initial_capacity);

#ifdef DEBUG
    std::cout << "[DynamicKVCache] Initialized:" << std::endl;
    std::cout << "  Initial capacity: " << allocated_capacity_ << " tokens" << std::endl;
    std::cout << "  Model max length: " << dynamic_config_.model_max_seq_len << std::endl;
    std::cout << "  Memory: " << (memory_usage() / 1024.0 / 1024.0) << " MB" << std::endl;
#endif
}

void DynamicKVCacheManager::allocate_cache(int capacity) {
    k_caches_.clear();
    v_caches_.clear();

    std::vector<size_t> cache_shape = {
        static_cast<size_t>(capacity),
        static_cast<size_t>(config_.num_kv_heads),
        static_cast<size_t>(config_.head_dim)};

    for (int layer = 0; layer < config_.num_layers; ++layer) {
        k_caches_.push_back(Tensor::create(
            cache_shape, config_.dtype, config_.device_type,
            config_.device_id, false, nullptr));
        v_caches_.push_back(Tensor::create(
            cache_shape, config_.dtype, config_.device_type,
            config_.device_id, false, nullptr));
    }

    allocated_capacity_ = capacity;
}

void DynamicKVCacheManager::grow_cache(int required_capacity) {
    int new_capacity = calculate_new_capacity(required_capacity);

    std::cout << "\n[Growth] " << allocated_capacity_
              << " -> " << new_capacity << " tokens" << std::endl;

    auto start = std::chrono::high_resolution_clock::now();

    auto old_k = std::move(k_caches_);
    auto old_v = std::move(v_caches_);

    allocate_cache(new_capacity);

    if (current_length_ > 0) {
        for (int layer = 0; layer < config_.num_layers; ++layer) {
            copy_cache_data(old_k[layer], k_caches_[layer], current_length_);
            copy_cache_data(old_v[layer], v_caches_[layer], current_length_);
        }
    }

    growth_count_++;

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration<double, std::milli>(end - start).count();
    total_growth_time_ms_ += duration;

    std::cout << "  Time: " << duration << " ms" << std::endl;
}

int DynamicKVCacheManager::calculate_new_capacity(int required) const {
    int new_cap = allocated_capacity_;

    switch (dynamic_config_.growth_strategy) {
    case DynamicKVCacheConfig::GrowthStrategy::DOUBLE:
        while (new_cap < required) {
            new_cap *= 2;
        }
        break;
    case DynamicKVCacheConfig::GrowthStrategy::AGGRESSIVE:
        if (growth_count_ >= dynamic_config_.aggressive_threshold) {
            new_cap = std::max(required, dynamic_config_.aggressive_target_capacity);
        } else {
            while (new_cap < required) {
                new_cap *= 2;
            }
        }
        break;
    case DynamicKVCacheConfig::GrowthStrategy::CONSERVATIVE:
        while (new_cap < required) {
            new_cap += dynamic_config_.initial_capacity;
        }
        break;
    }

    return std::min(new_cap, dynamic_config_.model_max_seq_len);
}

void DynamicKVCacheManager::copy_cache_data(tensor_t src, tensor_t dst, int valid_len) {
    auto src_slice = src->slice(0, 0, valid_len);
    auto dst_slice = dst->slice(0, 0, valid_len);

    size_t copy_size = valid_len * config_.num_kv_heads * config_.head_dim * utils::dsize(config_.dtype);

    if (config_.device_type == NEOLLM_DEVICE_CPU) {
        std::memcpy(dst_slice->data(), src_slice->data(), copy_size);
    } else {
        // GPU: cudaMemcpy or fallback
        core::context().setDevice(config_.device_type, config_.device_id);
        core::context().runtime().api()->memcpy_sync(dst_slice->data(), src_slice->data(), copy_size, NEOLLM_MEMCPY_D2D);
    }
}

void DynamicKVCacheManager::ensure_capacity(int required) {
    if (required > allocated_capacity_) {
        grow_cache(required);
    }
}

tensor_t DynamicKVCacheManager::get_k_cache(int layer_idx) {
    validate_layer_idx(layer_idx);
    return k_caches_[layer_idx];
}

tensor_t DynamicKVCacheManager::get_v_cache(int layer_idx) {
    validate_layer_idx(layer_idx);
    return v_caches_[layer_idx];
}

tensor_t DynamicKVCacheManager::get_k_cache_slice(int layer_idx, int total_len) {
    validate_layer_idx(layer_idx);

    ASSERT(total_len != 0, "total_len must be greater than 0 to slice KV cache");

    return k_caches_[layer_idx]->slice(0, 0, total_len);
}

tensor_t DynamicKVCacheManager::get_v_cache_slice(int layer_idx, int total_len) {
    validate_layer_idx(layer_idx);

    ASSERT(total_len != 0, "total_len must be greater than 0 to slice KV cache");

    return v_caches_[layer_idx]->slice(0, 0, total_len);
}

tensor_t DynamicKVCacheManager::get_k_cache_write_slice(
    int layer_idx, int past_len, int seq_len) {

    validate_layer_idx(layer_idx);
    validate_length_params(past_len, seq_len);

    int required = past_len + seq_len;
    ensure_capacity(required);

    if (required > dynamic_config_.model_max_seq_len) {
        throw std::runtime_error("Exceeds model_max_seq_len");
    }

    return k_caches_[layer_idx]->slice(0, past_len, past_len + seq_len);
}

tensor_t DynamicKVCacheManager::get_v_cache_write_slice(
    int layer_idx, int past_len, int seq_len) {

    validate_layer_idx(layer_idx);
    validate_length_params(past_len, seq_len);

    int required = past_len + seq_len;
    ensure_capacity(required);

    if (required > dynamic_config_.model_max_seq_len) {
        throw std::runtime_error("Exceeds model_max_seq_len");
    }

    return v_caches_[layer_idx]->slice(0, past_len, past_len + seq_len);
}

void DynamicKVCacheManager::reset() {
    KVCacheManager::reset();

    if (dynamic_config_.auto_shrink && should_shrink()) {
        shrink();
    }
}

bool DynamicKVCacheManager::should_shrink() const {
    return allocated_capacity_ > dynamic_config_.initial_capacity * dynamic_config_.shrink_capacity_multiplier && utilization() < dynamic_config_.shrink_utilization_threshold;
}

bool DynamicKVCacheManager::shrink() {
    if (!should_shrink()) {
        return false;
    }

    int new_cap = dynamic_config_.initial_capacity * 2;
    std::cout << "[Shrink] " << allocated_capacity_ << " -> " << new_cap << std::endl;

    allocate_cache(new_cap);
    current_length_ = 0;
    growth_count_ = 0;
    total_growth_time_ms_ = 0.0;

    return true;
}

void DynamicKVCacheManager::reserve(int capacity) {
    if (capacity <= allocated_capacity_) {
        return;
    }
    if (capacity > dynamic_config_.model_max_seq_len) {
        throw std::invalid_argument("Cannot reserve beyond model_max_seq_len");
    }

    if (current_length_ > 0) {
        grow_cache(capacity);
    } else {
        allocate_cache(capacity);
    }
}

size_t DynamicKVCacheManager::memory_usage() const {
    size_t total = 0;
    for (const auto &cache : k_caches_) {
        total += cache->numel() * utils::dsize(cache->dtype());
    }
    for (const auto &cache : v_caches_) {
        total += cache->numel() * utils::dsize(cache->dtype());
    }
    return total;
}

float DynamicKVCacheManager::utilization() const {
    return allocated_capacity_ > 0 ? static_cast<float>(current_length_) / allocated_capacity_ : 0.0f;
}

double DynamicKVCacheManager::average_growth_time_ms() const {
    return growth_count_ > 0 ? total_growth_time_ms_ / growth_count_ : 0.0;
}

std::string DynamicKVCacheManager::get_stats() const {
    std::ostringstream oss;

    oss << KVCacheManager::get_stats();

    oss << "  --- Dynamic Stats ---" << std::endl;
    oss << "  Growth count: " << growth_count_ << std::endl;
    if (growth_count_ > 0) {
        oss << "  Avg growth time: " << average_growth_time_ms() << " ms" << std::endl;
    }

    return oss.str();
}

} // namespace neollm::kvcache