#pragma once

#include "zedinfer.h"

#include <cstddef>

namespace zedinfer::ops::nvidia {

// hidden [N_total, H]; input_ids [N_total] (int32); image_embeds [N_img, H].
// All buffers on device; in-place modification of `hidden`.
void scatter_image_embeds(std::byte* hidden, const std::byte* input_ids, const std::byte* image_embeds,
                          zedinferDataType_t dtype, int N_total, int H, int image_token_id);

} // namespace zedinfer::ops::nvidia
