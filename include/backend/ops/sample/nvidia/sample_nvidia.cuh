#pragma once

#include "zedinfer.h"

#include <cstddef>
#include <cstdint>

namespace zedinfer::ops::nvidia {

size_t sample_sort_workspace_bytes(size_t vocab_size);

void sample_token(std::byte* out_token, const std::byte* logits, float* work_logits, int32_t* work_ids,
                  float* sorted_logits, int32_t* sorted_ids, void* sort_temp, size_t sort_temp_bytes,
                  const int32_t* recent_tokens, size_t recent_count, zedinferDataType_t dtype, size_t vocab_size,
                  float temperature, int top_k, float top_p, float repetition_penalty, float random);

} // namespace zedinfer::ops::nvidia
