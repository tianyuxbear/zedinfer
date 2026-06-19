#include "backend/core/context/context.hpp"
#include "backend/core/core.hpp"
#include "backend/ops/apply_rotary_emb/apply_rotary_emb.hpp"
#include "backend/ops/apply_rotary_emb/nvidia/apply_rotary_emb_nvidia.cuh"
#include "utils/check.hpp"

namespace zedinfer::ops {

void apply_rotary_emb_inplace(tensor_t q, tensor_t cos, tensor_t sin) {
    CHECK_SAME_DEVICE(q, cos, sin);
    CHECK_SAME_DTYPE(q->dtype(), cos->dtype(), sin->dtype());
    ASSERT(q->shape().size() == 3, "apply_rotary_emb_inplace: q must be 3-D [N, H, D]");
    ASSERT(cos->shape().size() == 2 && sin->shape().size() == 2,
           "apply_rotary_emb_inplace: cos/sin must be 2-D [N, D]");
    const int N = static_cast<int>(q->dim(0));
    const int H = static_cast<int>(q->dim(1));
    const int D = static_cast<int>(q->dim(2));
    ASSERT(D % 2 == 0, "apply_rotary_emb_inplace: head_dim must be even");
    ASSERT(static_cast<int>(cos->dim(0)) == N && static_cast<int>(cos->dim(1)) == D,
           "apply_rotary_emb_inplace: cos shape must be [N, D]");

    core::context().setDevice(q->deviceType(), q->deviceId());

    switch (q->deviceType()) {
#ifdef ENABLE_NVIDIA_API
        case ZEDINFER_DEVICE_NVIDIA:
            return nvidia::apply_rotary_emb_inplace(q->data(), cos->data(), sin->data(), q->dtype(), N, H, D);
#endif
        default:
            EXCEPTION_UNSUPPORTED_DEVICE;
    }
}

} // namespace zedinfer::ops
