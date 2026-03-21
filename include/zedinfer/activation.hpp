#pragma once

#include "zedinfer.h"

#include <cstddef>

namespace zedinfer {

/**
 * Configuration for model execution environment.
 */
struct ExecutorConfig {
    zedinferDeviceType_t device_type;
    int device_id;
    zedinferDataType_t data_type;
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

} // namespace zedinfer
