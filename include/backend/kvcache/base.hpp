#pragma once

#include "backend/tensor/tensor.hpp"
#include "neollm.h"
#include <memory>
#include <vector>

namespace neollm::kvcache {

/**
 * KV Cache基础配置
 */
struct KVCacheConfig {
    // 模型结构参数
    int num_layers;
    int num_kv_heads;
    int head_dim;

    // 设备参数
    NeollmDeviceType_t device_type = NEOLLM_DEVICE_CPU;
    int device_id = 0;

    // 数据类型
    NeollmDataType_t dtype = NEOLLM_DTYPE_BF16;

    void validate() const;
    size_t bytes_per_token() const;
};

/**
 * KV Cache管理器基类
 */
class KVCacheManager {
public:
    explicit KVCacheManager(const KVCacheConfig &config);
    virtual ~KVCacheManager() = default;

    // 基本信息
    const KVCacheConfig &config() const { return config_; }
    int current_length() const { return current_length_; }
    virtual int allocated_capacity() const = 0;

    // 核心接口（纯虚函数）
    virtual tensor_t get_k_cache(int layer_idx) = 0;
    virtual tensor_t get_v_cache(int layer_idx) = 0;
    virtual tensor_t get_k_cache_slice(int layer_idx, int total_len) = 0;
    virtual tensor_t get_v_cache_slice(int layer_idx, int total_len) = 0;
    virtual tensor_t get_k_cache_write_slice(int layer_idx, int past_len, int seq_len) = 0;
    virtual tensor_t get_v_cache_write_slice(int layer_idx, int past_len, int seq_len) = 0;

    // 状态管理
    void update_seq_len(int new_tokens);
    virtual void reset();

    // 统计信息
    virtual size_t memory_usage() const = 0;
    virtual float utilization() const = 0;
    virtual std::string get_stats() const;

protected:
    KVCacheConfig config_;
    int current_length_;
    std::vector<tensor_t> k_caches_;
    std::vector<tensor_t> v_caches_;

    // 辅助方法
    void validate_layer_idx(int layer_idx) const;
    void validate_length_params(int past_len, int seq_len) const;
};

using kvcache_t = std::shared_ptr<KVCacheManager>;

} // namespace neollm::kvcache