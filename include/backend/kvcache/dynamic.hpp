#pragma once

#include "backend/kvcache/base.hpp"
#include "backend/tensor/tensor.hpp"

#include <memory>
#include <string>
#include <vector>

namespace zedinfer::kvcache {

struct DynamicKVCacheConfig : public KVCacheConfig {
    int initial_capacity = 256;
    int model_max_seq_len = 8192;

    enum class GrowthStrategy { DOUBLE, AGGRESSIVE, CONSERVATIVE };
    GrowthStrategy growth_strategy = GrowthStrategy::DOUBLE;

    int aggressive_threshold = 3;
    int aggressive_target_capacity = 8192;

    void validate() const;
};

/**
 * Standalone dynamic KV cache with automatic growth.
 * Used only by warmup() and profile(). Not a polymorphic base.
 */
class DynamicKVCache {
public:
    explicit DynamicKVCache(const DynamicKVCacheConfig &config);
    ~DynamicKVCache() = default;

    const DynamicKVCacheConfig &dyn_config() const { return dynamic_config_; }
    const KVCacheConfig &kv_config() const { return dynamic_config_; }

    int current_length() const { return current_length_; }
    int allocated_capacity() const { return allocated_capacity_; }

    tensor_t get_k_cache(int layer_idx);
    tensor_t get_v_cache(int layer_idx);
    tensor_t get_k_cache_slice(int layer_idx, int total_len);
    tensor_t get_v_cache_slice(int layer_idx, int total_len);
    tensor_t get_k_cache_slice(int layer_idx, int past_len, int seq_len);
    tensor_t get_v_cache_slice(int layer_idx, int past_len, int seq_len);

    void update_seq_len(int new_tokens);
    void reset();

    size_t memory_usage() const;
    float utilization() const;

    void ensure_capacity(int required_capacity);

    static std::unique_ptr<DynamicKVCache> create(const DynamicKVCacheConfig &config) {
        return std::make_unique<DynamicKVCache>(config);
    }

private:
    DynamicKVCacheConfig dynamic_config_;
    int current_length_ = 0;
    int allocated_capacity_ = 0;
    int growth_count_ = 0;
    double total_growth_time_ms_ = 0.0;

    std::vector<tensor_t> k_caches_;
    std::vector<tensor_t> v_caches_;

    void allocate_cache(int capacity);
    void grow_cache(int required_capacity);
    int calculate_new_capacity(int required) const;
    void copy_cache_data(tensor_t src, tensor_t dst, int valid_len);
    void validate_layer_idx(int layer_idx) const;
};

} // namespace zedinfer::kvcache
