#include "backend/kvcache/dynamic.hpp"
#include "backend/core/context/context.hpp"
#include "utils/types.hpp"
#include "zedinfer.h"

#include <chrono>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <plog/Log.h>
#include <sstream>

namespace zedinfer::kvcache {

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

DynamicKVCache::DynamicKVCache(const DynamicKVCacheConfig &config)
    : KVCache(config),
      dynamic_config_(config),
      allocated_capacity_(0),
      growth_count_(0),
      total_growth_time_ms_(0.0) {

    dynamic_config_.validate();
    allocate_cache(dynamic_config_.initial_capacity);
}

void DynamicKVCache::allocate_cache(int capacity) {
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

void DynamicKVCache::grow_cache(int required_capacity) {
    int new_capacity = calculate_new_capacity(required_capacity);

    std::ostringstream oss;

    oss << "\n=== KVCache Growth ===\n";
    oss << "  Capacity: " << allocated_capacity_
        << " -> " << new_capacity << " tokens\n";

    auto start = std::chrono::high_resolution_clock::now();

    // Save old caches
    auto old_k = std::move(k_caches_);
    auto old_v = std::move(v_caches_);

    // Allocate new caches
    allocate_cache(new_capacity);

    // Copy existing data
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

    oss << "  Time: " << std::fixed << std::setprecision(2)
        << duration << " ms\n";

    LOGI << oss.str();
}

int DynamicKVCache::calculate_new_capacity(int required) const {
    int new_cap = allocated_capacity_;

    switch (dynamic_config_.growth_strategy) {
    case DynamicKVCacheConfig::GrowthStrategy::DOUBLE:
        // Double until >= required
        while (new_cap < required) {
            new_cap *= 2;
        }
        break;

    case DynamicKVCacheConfig::GrowthStrategy::AGGRESSIVE:
        // Double initially, then jump to target after threshold
        if (growth_count_ >= dynamic_config_.aggressive_threshold) {
            new_cap = std::max(required, dynamic_config_.aggressive_target_capacity);
        } else {
            while (new_cap < required) {
                new_cap *= 2;
            }
        }
        break;

    case DynamicKVCacheConfig::GrowthStrategy::CONSERVATIVE:
        // Linear growth by initial_capacity
        while (new_cap < required) {
            new_cap += dynamic_config_.initial_capacity;
        }
        break;
    }

    // Cap at model limit
    return std::min(new_cap, dynamic_config_.model_max_seq_len);
}

void DynamicKVCache::copy_cache_data(tensor_t src, tensor_t dst, int valid_len) {
    auto src_slice = src->slice(0, 0, valid_len);
    auto dst_slice = dst->slice(0, 0, valid_len);

    size_t copy_size = valid_len * config_.num_kv_heads * config_.head_dim
                     * utils::dsize(config_.dtype);

    if (config_.device_type == ZEDINFER_DEVICE_CPU) {
        std::memcpy(dst_slice->data(), src_slice->data(), copy_size);
    } else {
        // GPU memory copy
        core::context().setDevice(config_.device_type, config_.device_id);
        core::context().runtime().api()->memcpy_sync(
            dst_slice->data(), src_slice->data(), copy_size, ZEDINFER_MEMCPY_D2D);
    }
}

void DynamicKVCache::ensure_capacity(int required_capacity) {
    if (required_capacity < 0) {
        throw std::invalid_argument("required_capacity must be non-negative");
    }
    if (required_capacity > dynamic_config_.model_max_seq_len) {
        throw std::invalid_argument("required_capacity exceeds model_max_seq_len");
    }

    if (required_capacity > allocated_capacity_) {
        if (current_length_ > 0) {
            // Has data: grow with copy
            grow_cache(required_capacity);
        } else {
            // Empty cache: direct allocation (no copy overhead)
            allocate_cache(required_capacity);
        }
    }
}

tensor_t DynamicKVCache::get_k_cache(int layer_idx) {
    validate_layer_idx(layer_idx);
    return k_caches_[layer_idx];
}

tensor_t DynamicKVCache::get_v_cache(int layer_idx) {
    validate_layer_idx(layer_idx);
    return v_caches_[layer_idx];
}

tensor_t DynamicKVCache::get_k_cache_slice(int layer_idx, int total_len) {
    validate_layer_idx(layer_idx);

    if (total_len > dynamic_config_.model_max_seq_len) {
        throw std::runtime_error("total_len exceeds model_max_seq_len");
    }

    ensure_capacity(total_len);
    return k_caches_[layer_idx]->slice(0, 0, total_len);
}

tensor_t DynamicKVCache::get_v_cache_slice(int layer_idx, int total_len) {
    validate_layer_idx(layer_idx);

    if (total_len > dynamic_config_.model_max_seq_len) {
        throw std::runtime_error("total_len exceeds model_max_seq_len");
    }

    ensure_capacity(total_len);
    return v_caches_[layer_idx]->slice(0, 0, total_len);
}

tensor_t DynamicKVCache::get_k_cache_slice(int layer_idx, int past_len, int seq_len) {
    validate_layer_idx(layer_idx);
    validate_length_params(past_len, seq_len);

    int total_len = past_len + seq_len;
    if (total_len > dynamic_config_.model_max_seq_len) {
        throw std::runtime_error("total_len exceeds model_max_seq_len");
    }

    ensure_capacity(total_len);
    return k_caches_[layer_idx]->slice(0, past_len, past_len + seq_len);
}

tensor_t DynamicKVCache::get_v_cache_slice(int layer_idx, int past_len, int seq_len) {
    validate_layer_idx(layer_idx);
    validate_length_params(past_len, seq_len);

    int total_len = past_len + seq_len;
    if (total_len > dynamic_config_.model_max_seq_len) {
        throw std::runtime_error("total_len exceeds model_max_seq_len");
    }

    ensure_capacity(total_len);
    return v_caches_[layer_idx]->slice(0, past_len, past_len + seq_len);
}

size_t DynamicKVCache::memory_usage() const {
    size_t total = 0;
    for (const auto &cache : k_caches_) {
        total += cache->numel() * utils::dsize(cache->dtype());
    }
    for (const auto &cache : v_caches_) {
        total += cache->numel() * utils::dsize(cache->dtype());
    }
    return total;
}

float DynamicKVCache::utilization() const {
    return allocated_capacity_ > 0
             ? static_cast<float>(current_length_) / allocated_capacity_
             : 0.0f;
}

double DynamicKVCache::average_growth_time_ms() const {
    return growth_count_ > 0
             ? total_growth_time_ms_ / growth_count_
             : 0.0;
}

std::string DynamicKVCache::get_stats() const {
    std::ostringstream oss;

    oss << KVCache::get_stats();
    oss << "  --- Dynamic Stats ---" << std::endl;
    oss << "  Growth count: " << growth_count_ << std::endl;

    if (growth_count_ > 0) {
        oss << "  Avg growth time: " << std::fixed << std::setprecision(2)
            << average_growth_time_ms() << " ms" << std::endl;
    }

    return oss.str();
}

} // namespace zedinfer::kvcache