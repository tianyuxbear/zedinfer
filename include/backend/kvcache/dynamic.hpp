#pragma once

#include "backend/kvcache/base.hpp"

namespace neollm::kvcache {

/**
 * 动态KV Cache配置
 */
struct DynamicKVCacheConfig : public KVCacheConfig {
    int initial_capacity = 256;
    int model_max_seq_len = 8192;

    enum class GrowthStrategy {
        DOUBLE,
        AGGRESSIVE,
        CONSERVATIVE
    };
    GrowthStrategy growth_strategy = GrowthStrategy::DOUBLE;

    int aggressive_threshold = 3;
    int aggressive_target_capacity = 4096;

    bool auto_shrink = false;
    int shrink_capacity_multiplier = 4;
    float shrink_utilization_threshold = 0.1f;

    void validate() const;
};

/**
 * 动态增长的KV Cache管理器
 */
class DynamicKVCacheManager : public KVCacheManager {
public:
    explicit DynamicKVCacheManager(const DynamicKVCacheConfig &config);
    ~DynamicKVCacheManager() override = default;

    const DynamicKVCacheConfig &config() const { return dynamic_config_; }

    // 基类接口实现
    int allocated_capacity() const override { return allocated_capacity_; }
    tensor_t get_k_cache(int layer_idx) override;
    tensor_t get_v_cache(int layer_idx) override;
    tensor_t get_k_cache_slice(int layer_idx, int past_len) override;
    tensor_t get_v_cache_slice(int layer_idx, int past_len) override;
    tensor_t get_k_cache_write_slice(int layer_idx, int past_len, int seq_len) override;
    tensor_t get_v_cache_write_slice(int layer_idx, int past_len, int seq_len) override;
    void reset() override;
    size_t memory_usage() const override;
    float utilization() const override;
    std::string get_stats() const override;

    // 动态特有接口
    int growth_count() const { return growth_count_; }
    double total_growth_time_ms() const { return total_growth_time_ms_; }
    double average_growth_time_ms() const;
    bool shrink();
    void reserve(int capacity);

    static inline std::shared_ptr<DynamicKVCacheManager> create_dynamic_kvcache(
        const DynamicKVCacheConfig &config) {
        return std::make_shared<DynamicKVCacheManager>(config);
    }

private:
    DynamicKVCacheConfig dynamic_config_;
    int allocated_capacity_;
    int growth_count_;
    double total_growth_time_ms_;

    void allocate_cache(int capacity);
    void grow_cache(int required_capacity);
    int calculate_new_capacity(int required) const;
    void copy_cache_data(tensor_t src, tensor_t dst, int valid_len);
    void ensure_capacity(int required_capacity);
    bool should_shrink() const;
};
} // namespace neollm::kvcache