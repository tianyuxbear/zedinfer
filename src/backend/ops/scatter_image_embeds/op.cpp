#include "backend/core/context/context.hpp"
#include "backend/core/core.hpp"
#include "backend/ops/scatter_image_embeds/nvidia/scatter_image_embeds_nvidia.cuh"
#include "backend/ops/scatter_image_embeds/scatter_image_embeds.hpp"
#include "utils/check.hpp"

namespace zedinfer::ops {

void scatter_image_embeds(tensor_t hidden, tensor_t input_ids, tensor_t image_embeds, int image_token_id) {
    CHECK_SAME_DEVICE(hidden, input_ids, image_embeds);
    CHECK_SAME_DTYPE(hidden->dtype(), image_embeds->dtype());
    ASSERT(hidden->shape().size() == 2 && image_embeds->shape().size() == 2,
           "scatter_image_embeds: hidden and image_embeds must be 2-D [N, H].");
    ASSERT(input_ids->dtype() == ZEDINFER_DTYPE_I32, "scatter_image_embeds: input_ids must be int32.");
    ASSERT(hidden->dim(1) == image_embeds->dim(1),
           "scatter_image_embeds: hidden and image_embeds must share hidden dim H.");

    const int N = static_cast<int>(hidden->dim(0));
    const int H = static_cast<int>(hidden->dim(1));

    core::context().setDevice(hidden->deviceType(), hidden->deviceId());

    switch (hidden->deviceType()) {
#ifdef ENABLE_NVIDIA_API
        case ZEDINFER_DEVICE_NVIDIA:
            return nvidia::scatter_image_embeds(hidden->data(), input_ids->data(), image_embeds->data(),
                                                hidden->dtype(), N, H, image_token_id);
#endif
        default:
            EXCEPTION_UNSUPPORTED_DEVICE;
    }
}

} // namespace zedinfer::ops
