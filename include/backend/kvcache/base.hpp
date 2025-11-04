#pragma once

#include "backend/tensor/tensor.hpp"
#include "zedinfer.h"
#include <memory>
#include <vector>

namespace zedinfer::kvcache {

/**
 * Configuration for KV cache initialization
 */
struct KVCacheConfig {
    // Model architecture
    int num_layers;
    int num_kv_heads;
    int head_dim;

    // Device settings
    zedinferDeviceType_t device_type = ZEDINFER_DEVICE_CPU;
    int device_id = 0;

    // Data type
    zedinferDataType_t dtype = ZEDINFER_DTYPE_BF16;

    void validate() const;
    size_t bytes_per_token() const;
};

/**
 * Base class for KV cache management
 */
class KVCache {
public:
    explicit KVCache(const KVCacheConfig &config);
    virtual ~KVCache() = default;

    // Getters
    const KVCacheConfig &config() const { return config_; }
    int current_length() const { return current_length_; }
    virtual int allocated_capacity() const = 0;

    // Cache access interface
    virtual tensor_t get_k_cache(int layer_idx) = 0;
    virtual tensor_t get_v_cache(int layer_idx) = 0;
    virtual tensor_t get_k_cache_slice(int layer_idx, int total_len) = 0;
    virtual tensor_t get_v_cache_slice(int layer_idx, int total_len) = 0;
    virtual tensor_t get_k_cache_slice(int layer_idx, int past_len, int seq_len) = 0;
    virtual tensor_t get_v_cache_slice(int layer_idx, int past_len, int seq_len) = 0;

    // State management
    void update_seq_len(int new_tokens);
    virtual void reset();

    // Statistics
    virtual size_t memory_usage() const = 0;
    virtual float utilization() const = 0;
    virtual std::string get_stats() const;

protected:
    KVCacheConfig config_;
    int current_length_;
    std::vector<tensor_t> k_caches_;
    std::vector<tensor_t> v_caches_;

    // Helper methods
    void validate_layer_idx(int layer_idx) const;
    void validate_length_params(int past_len, int seq_len) const;
};

using kvcache_t = std::unique_ptr<KVCache>;

} // namespace zedinfer::kvcache