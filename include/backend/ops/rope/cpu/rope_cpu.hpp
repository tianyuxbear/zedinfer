#pragma once

#include "zedinfer.h"

#include <cstddef>

namespace zedinfer::ops::cpu {
void rope(std::byte *out, const std::byte *in, const std::byte *pos_ids, float theta, zedinferDataType_t type, size_t seqlen, size_t nhead, size_t d);
}