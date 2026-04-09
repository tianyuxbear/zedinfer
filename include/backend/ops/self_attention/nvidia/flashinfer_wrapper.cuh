#pragma once

#include "backend/ops/attention_params.hpp"

namespace zedinfer::ops::nvidia {

void flashinfer_attention_decode(const AttentionParams& params);
void flashinfer_attention_prefill(const AttentionParams& params);

} // namespace zedinfer::ops::nvidia
