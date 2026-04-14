#pragma once

#include "backend/ops/attention_params.hpp"

namespace zedinfer::ops::nvidia {

bool flashinfer_prepare_single_decode_plan(const AttentionConfig& config, int total_pages, FlashInferDecodePlan& plan);
void flashinfer_attention_decode(const AttentionParams& params);
void flashinfer_attention_prefill(const AttentionParams& params);

} // namespace zedinfer::ops::nvidia
