#pragma once

#include "backend/core/storage/storage.hpp" // IWYU pragma: keep
#include "backend/tensor/tensor.hpp"
#include "zedinfer.h"

#include <cstddef>

namespace zedinfer {

/**
 * Configuration for model execution environment.
 */
struct ExecutorConfig {
    zedinferDeviceType_t device_type;
    int device_id;
    zedinferDataType_t data_type; // Default dtype for activations
    size_t max_seq_len;

    ExecutorConfig(zedinferDeviceType_t device_type = ZEDINFER_DEVICE_CPU,
                   int device_id = 0,
                   zedinferDataType_t data_type = ZEDINFER_DTYPE_F32,
                   size_t max_seq_len = 16384)
        : device_type(device_type),
          device_id(device_id),
          data_type(data_type),
          max_seq_len(max_seq_len) {}
};

/**
 * Pre-allocated position IDs cache to avoid repeated allocation.
 */
class PositionIDsCache {
public:
    PositionIDsCache(const ExecutorConfig &config);

    /**
     * Get position_ids slice [start_pos, start_pos+1, ..., start_pos+seq_len-1].
     * @return Tensor of shape [seq_len]
     */
    tensor_t get_slice(int start_pos, int seq_len);

    int max_seq_len() const { return config_.max_seq_len; }

private:
    ExecutorConfig config_;
    tensor_t cache_; // Pre-allocated [0, 1, 2, ..., max_seq_len-1]

    void initialize_cache();
};

} // namespace zedinfer