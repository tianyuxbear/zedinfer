#pragma once

#include "zedinfer.h"
#include <cstddef>

namespace zedinfer::kvcache {

/**
 * Configuration for KV cache initialization
 */
struct KVCacheConfig {
    int num_layers;
    int num_kv_heads;
    int head_dim;

    zedinferDeviceType_t device_type = ZEDINFER_DEVICE_CPU;
    int device_id = 0;
    zedinferDataType_t dtype = ZEDINFER_DTYPE_BF16;

    void validate() const;
    size_t bytes_per_token() const;
};

} // namespace zedinfer::kvcache
