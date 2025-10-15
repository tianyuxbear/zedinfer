#pragma once

#include "neollm.h"

#include <cstddef>

namespace neollm::ops::cpu {
void self_attention(std::byte *attn_val, const std::byte *q, const std::byte *k, const std::byte *v, float scale, NeollmDataType_t type, size_t seqlen, size_t nhead, size_t dv, size_t total_len, size_t nkvhead, size_t d);
}