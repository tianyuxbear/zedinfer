#pragma once

#include "backend/tensor/tensor.hpp"

namespace zedinfer::ops {

// In-place scatter of vision-tower outputs into the LLM's hidden state at
// <|image_pad|> token positions.
//
//   hidden       : [N_total, H]   (in/out, BF16/FP16/FP32)
//   input_ids    : [N_total]      (int32 device tensor with token ids)
//   image_embeds : [N_img, H]     (BF16/FP16/FP32; dtype must match hidden)
//   image_token_id: int           (e.g. <|image_pad|> id)
//
// Positions of `image_token_id` in `input_ids` are scanned in order; the i-th
// such position receives image_embeds[i, :]. The caller must guarantee that
// image_embeds has exactly as many rows as `input_ids` has occurrences of
// `image_token_id`. On NVIDIA the kernel runs on the runtime compute stream.
void scatter_image_embeds(tensor_t hidden, tensor_t input_ids, tensor_t image_embeds, int image_token_id);

} // namespace zedinfer::ops
