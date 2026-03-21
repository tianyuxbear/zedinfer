#pragma once

#ifdef USE_CUDNN_FLASH

#include "zedinfer.h"

#include <cstddef>

namespace zedinfer::ops::nvidia::cudnn_flash {

// cuDNN FlashAttention for prefill (seqlen > 1).
// Returns true on success, false if cuDNN doesn't support this config
// (caller should fallback to custom kernel).
bool flash_attention_prefill(
    std::byte *output, const std::byte *q, const std::byte *k, const std::byte *v,
    float scale, zedinferDataType_t type,
    int seqlen, int nhead, int head_dim, int total_len, int nkvhead);

} // namespace zedinfer::ops::nvidia::cudnn_flash

#endif // USE_CUDNN_FLASH
