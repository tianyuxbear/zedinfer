#include "backend/core/context/context.hpp"
#include "backend/ops/ops.hpp"
#include "utils/check.hpp"

#ifdef ENABLE_NVIDIA_API
#include "backend/ops/sample/nvidia/sample_nvidia.cuh"
#endif

#include <climits>

namespace zedinfer::ops {

size_t sample_sort_workspace_bytes(zedinferDeviceType_t device_type, size_t vocab_size) {
    ASSERT(vocab_size <= static_cast<size_t>(INT_MAX), "sample_sort_workspace_bytes: vocab size exceeds INT_MAX.");
    switch (device_type) {
#ifdef ENABLE_NVIDIA_API
        case ZEDINFER_DEVICE_NVIDIA:
            return nvidia::sample_sort_workspace_bytes(vocab_size);
#endif
        default:
            EXCEPTION_UNSUPPORTED_DEVICE;
    }
}

void sample_token(tensor_t out_token, tensor_t logits, tensor_t work_logits, tensor_t work_ids, tensor_t sorted_logits,
                  tensor_t sorted_ids, tensor_t sort_temp, tensor_t recent_tokens, float temperature, int top_k,
                  float top_p, float repetition_penalty, float random) {
    ASSERT(out_token && logits && work_logits, "sample_token: output, logits and work_logits must be non-null.");
    ASSERT(out_token->isContiguous() && logits->isContiguous() && work_logits->isContiguous(),
           "sample_token: tensors must be contiguous.");
    ASSERT(out_token->dtype() == ZEDINFER_DTYPE_I64, "sample_token: out_token must be int64.");
    ASSERT(work_logits->dtype() == ZEDINFER_DTYPE_F32, "sample_token: work_logits must be float32.");
    ASSERT(out_token->numel() == 1, "sample_token: out_token must have one element.");
    ASSERT(logits->numel() == work_logits->numel(), "sample_token: logits/work_logits size mismatch.");
    ASSERT(logits->numel() <= static_cast<size_t>(INT_MAX), "sample_token: vocab size exceeds INT_MAX.");
    ASSERT(temperature > 0.0f, "sample_token: temperature must be positive.");
    ASSERT(top_k >= 0, "sample_token: top_k must be non-negative.");
    ASSERT(top_p > 0.0f && top_p <= 1.0f, "sample_token: top_p must be in (0, 1].");
    ASSERT(repetition_penalty > 0.0f, "sample_token: repetition_penalty must be positive.");
    ASSERT(random >= 0.0f && random < 1.0f, "sample_token: random must be in [0, 1).");

    CHECK_SAME_DEVICE(logits, out_token, work_logits);

    const size_t vocab_size = logits->numel();
    const bool has_top_k_filter = top_k > 0 && static_cast<size_t>(top_k) < vocab_size;
    const bool needs_sort = (has_top_k_filter && top_k > 32) || (!has_top_k_filter && top_p < 1.0f);
    if (needs_sort) {
        ASSERT(work_ids && sorted_logits && sorted_ids && sort_temp,
               "sample_token: work_ids, sorted_logits, sorted_ids and sort_temp are required for sorted sampling.");
        ASSERT(work_ids->isContiguous() && sorted_logits->isContiguous() && sorted_ids->isContiguous()
                   && sort_temp->isContiguous(),
               "sample_token: sorted sampling tensors must be contiguous.");
        ASSERT(work_ids->dtype() == ZEDINFER_DTYPE_I32 && sorted_ids->dtype() == ZEDINFER_DTYPE_I32,
               "sample_token: token id workspaces must be int32.");
        ASSERT(sorted_logits->dtype() == ZEDINFER_DTYPE_F32, "sample_token: sorted_logits must be float32.");
        ASSERT(sort_temp->dtype() == ZEDINFER_DTYPE_U8, "sample_token: sort_temp must be uint8.");
        ASSERT(work_ids->numel() >= vocab_size && sorted_logits->numel() >= vocab_size
                   && sorted_ids->numel() >= vocab_size,
               "sample_token: sorted sampling workspace is smaller than vocab size.");
        ASSERT(sort_temp->numel() > 0, "sample_token: sort_temp workspace is empty.");
        CHECK_SAME_DEVICE(logits, work_ids, sorted_logits, sorted_ids, sort_temp);
    }

    const int32_t* recent_ptr = nullptr;
    size_t recent_count = 0;
    if (recent_tokens) {
        ASSERT(recent_tokens->isContiguous(), "sample_token: recent_tokens must be contiguous.");
        ASSERT(recent_tokens->dtype() == ZEDINFER_DTYPE_I32, "sample_token: recent_tokens must be int32.");
        CHECK_SAME_DEVICE(logits, recent_tokens);
        recent_ptr = reinterpret_cast<const int32_t*>(recent_tokens->data());
        recent_count = recent_tokens->numel();
    }

    zedinfer::core::context().setDevice(logits->deviceType(), logits->deviceId());

    switch (logits->deviceType()) {
#ifdef ENABLE_NVIDIA_API
        case ZEDINFER_DEVICE_NVIDIA:
            return nvidia::sample_token(
                out_token->data(), logits->data(), reinterpret_cast<float*>(work_logits->data()),
                work_ids ? reinterpret_cast<int32_t*>(work_ids->data()) : nullptr,
                sorted_logits ? reinterpret_cast<float*>(sorted_logits->data()) : nullptr,
                sorted_ids ? reinterpret_cast<int32_t*>(sorted_ids->data()) : nullptr,
                sort_temp ? sort_temp->data() : nullptr, sort_temp ? sort_temp->numel() : 0, recent_ptr, recent_count,
                logits->dtype(), logits->numel(), temperature, top_k, top_p, repetition_penalty, random);
#endif
        default:
            EXCEPTION_UNSUPPORTED_DEVICE;
    }
}

} // namespace zedinfer::ops
