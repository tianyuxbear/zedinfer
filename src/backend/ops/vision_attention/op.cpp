#include "backend/core/context/context.hpp"
#include "backend/core/core.hpp"
#include "backend/ops/vision_attention/nvidia/vision_attention_nvidia.cuh"
#include "backend/ops/vision_attention/vision_attention.hpp"
#include "utils/check.hpp"

namespace zedinfer::ops {

void vision_attention(const VisionAttentionParams& p) {
    CHECK_SAME_DEVICE(p.q, p.k, p.v, p.out);
    CHECK_SAME_DTYPE(p.q->dtype(), p.k->dtype(), p.v->dtype(), p.out->dtype());
    ASSERT(p.q->shape().size() == 3 && p.k->shape().size() == 3 && p.v->shape().size() == 3
               && p.out->shape().size() == 3,
           "vision_attention: q/k/v/out must be 3-D [N, H, D].");
    const int N = static_cast<int>(p.q->shape()[0]);
    const int H = static_cast<int>(p.q->shape()[1]);
    const int D = static_cast<int>(p.q->shape()[2]);

    core::context().setDevice(p.out->deviceType(), p.out->deviceId());

    switch (p.out->deviceType()) {
#ifdef ENABLE_NVIDIA_API
        case ZEDINFER_DEVICE_NVIDIA:
            return nvidia::vision_attention(p.out->data(), p.q->data(), p.k->data(), p.v->data(), p.q->dtype(), N, H, D,
                                            p.scale);
#endif
        default:
            EXCEPTION_UNSUPPORTED_DEVICE;
    }
}

} // namespace zedinfer::ops
