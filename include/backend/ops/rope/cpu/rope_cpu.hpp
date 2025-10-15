#pragma once

#include "neollm.h"

#include <cstddef>

namespace neollm::ops::cpu {
void rope(std::byte *out, const std::byte *in, const std::byte *pos_ids, float theta, NeollmDataType_t type, size_t seqlen, size_t nhead, size_t d);
}