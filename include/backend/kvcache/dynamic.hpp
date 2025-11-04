#pragma once

#include "backend/kvcache/base.hpp"

namespace neollm::kvcache {

/**
 * @brief Configuration for dynamic KV cache
 */
struct DynamicKVCacheConfig : public KVCacheConfig {
    // Initial allocation size (tokens)
    int initial_capacity = 256;

    // Maximum sequence length supported by model
    int model_max_seq_len = 8192;

    // Growth strategy for cache expansion
    enum class GrowthStrategy {
        DOUBLE,      // 2x growth each time
        AGGRESSIVE,  // Jump to target after N growths
        CONSERVATIVE // Linear growth by initial_capacity
    };
    GrowthStrategy growth_strategy = GrowthStrategy::DOUBLE;

    // Aggressive strategy parameters
    int aggressive_threshold = 3;          // Trigger after N growths
    int aggressive_target_capacity = 8192; // Jump to this capacity

    void validate() const;
};

/**
 * @brief Dynamic KV cache with automatic growth
 *
 * Each session owns a dedicated cache instance. No shrinking needed
 * as cache is destroyed when session ends.
 */
class DynamicKVCache : public KVCache {
public:
    explicit DynamicKVCache(const DynamicKVCacheConfig &config);
    ~DynamicKVCache() override = default;

    const DynamicKVCacheConfig &config() const { return dynamic_config_; }

    // KVCache interface
    int allocated_capacity() const override { return allocated_capacity_; }
    tensor_t get_k_cache(int layer_idx) override;
    tensor_t get_v_cache(int layer_idx) override;
    tensor_t get_k_cache_slice(int layer_idx, int total_len) override;
    tensor_t get_v_cache_slice(int layer_idx, int total_len) override;
    tensor_t get_k_cache_slice(int layer_idx, int past_len, int seq_len) override;
    tensor_t get_v_cache_slice(int layer_idx, int past_len, int seq_len) override;
    size_t memory_usage() const override;
    float utilization() const override;
    std::string get_stats() const override;

    // Growth statistics
    int growth_count() const { return growth_count_; }
    double total_growth_time_ms() const { return total_growth_time_ms_; }
    double average_growth_time_ms() const;

    // Ensure cache has sufficient capacity
    void ensure_capacity(int required_capacity);

    // Factory method
    static std::unique_ptr<KVCache> create_dynamic_kvcache(
        const DynamicKVCacheConfig &config) {
        return std::make_unique<DynamicKVCache>(config);
    }

private:
    DynamicKVCacheConfig dynamic_config_;
    int allocated_capacity_;
    int growth_count_;
    double total_growth_time_ms_;

    // Internal memory management
    void allocate_cache(int capacity);
    void grow_cache(int required_capacity);
    int calculate_new_capacity(int required) const;
    void copy_cache_data(tensor_t src, tensor_t dst, int valid_len);
};

} // namespace neollm::kvcache