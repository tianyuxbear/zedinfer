#include "backend/core/context/context.hpp"
#include "backend/core/core.hpp"
#include "backend/ops/ops.hpp"
#include "backend/ops/self_attention/cpu/self_attention_cpu.hpp"
#include "backend/ops/self_attention/cpu/paged_attention_cpu.hpp"
#include "backend/ops/self_attention/nvidia/self_attention_nvidia.cuh"
#ifdef ENABLE_NVIDIA_API
#include "backend/ops/self_attention/nvidia/paged_attention_nvidia.cuh"
#endif
#include "utils/check.hpp"

namespace zedinfer::ops {

void self_attention(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale) {
    CHECK_SAME_DEVICE(attn_val, q, k, v);
    CHECK_SAME_DTYPE(attn_val->dtype(), q->dtype(), k->dtype(), v->dtype());
    ASSERT(attn_val->isContiguous() && q->isContiguous() && k->isContiguous() && v->isContiguous(), "SelfAttention: all tensors must be contiguous.");

    if (attn_val->deviceType() == ZEDINFER_DEVICE_CPU) {
        return cpu::self_attention(attn_val->data(), q->data(), k->data(), v->data(), scale, attn_val->dtype(), attn_val->dim(0), attn_val->dim(1), attn_val->dim(2), k->dim(0), k->dim(1), k->dim(2));
    }

    zedinfer::core::context().setDevice(attn_val->deviceType(), attn_val->deviceId());

    switch (attn_val->deviceType()) {
    case ZEDINFER_DEVICE_CPU:
        return cpu::self_attention(attn_val->data(), q->data(), k->data(), v->data(), scale, attn_val->dtype(), attn_val->dim(0), attn_val->dim(1), attn_val->dim(2), k->dim(0), k->dim(1), k->dim(2));
#ifdef ENABLE_NVIDIA_API
    case ZEDINFER_DEVICE_NVIDIA:
        return nvidia::self_attention(attn_val->data(), q->data(), k->data(), v->data(), scale, attn_val->dtype(), attn_val->dim(0), attn_val->dim(1), attn_val->dim(2), k->dim(0), k->dim(1), k->dim(2));
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}

void paged_attention_decode(
    tensor_t attn_val, tensor_t q,
    const void *pool_base,
    const int *k_block_table, const int *v_block_table,
    int seq_len, float scale,
    zedinferDataType_t dtype,
    zedinferDeviceType_t device_type,
    int device_id,
    int nhead, int nkvhead, int head_dim, int block_size) {

    if (device_type == ZEDINFER_DEVICE_CPU) {
        return cpu::paged_attention_decode(
            attn_val->data(), q->data(),
            reinterpret_cast<const std::byte *>(pool_base),
            k_block_table, v_block_table,
            seq_len, scale, dtype,
            nhead, nkvhead, head_dim, block_size);
    }

    zedinfer::core::context().setDevice(device_type, device_id);

    switch (device_type) {
#ifdef ENABLE_NVIDIA_API
    case ZEDINFER_DEVICE_NVIDIA:
        return nvidia::paged_attention_decode(
            attn_val->data(), q->data(),
            reinterpret_cast<const std::byte *>(pool_base),
            k_block_table, v_block_table,
            seq_len, scale, dtype,
            nhead, nkvhead, head_dim, block_size);
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}

void paged_attention_prefill(
    tensor_t attn_val, tensor_t q,
    const void *pool_base,
    const int *k_block_table, const int *v_block_table,
    int seqlen_q, int past_len,
    float scale,
    zedinferDataType_t dtype,
    zedinferDeviceType_t device_type,
    int device_id,
    int nhead, int nkvhead, int head_dim, int block_size) {

    zedinfer::core::context().setDevice(device_type, device_id);

    switch (device_type) {
    case ZEDINFER_DEVICE_CPU:
        // CPU prefill: fall back to gather + self_attention (no paged prefill kernel for CPU yet)
        ASSERT(false, "CPU paged prefill not implemented — use gather path");
        break;
#ifdef ENABLE_NVIDIA_API
    case ZEDINFER_DEVICE_NVIDIA:
        return nvidia::paged_attention_prefill(
            attn_val->data(), q->data(),
            reinterpret_cast<const std::byte *>(pool_base),
            k_block_table, v_block_table,
            seqlen_q, past_len, scale, dtype,
            nhead, nkvhead, head_dim, block_size);
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}

void paged_attention_decode_batched(
    tensor_t attn_val, tensor_t q,
    const void *pool_base,
    const int *k_block_tables, const int *v_block_tables,
    const int *seq_lens,
    int num_reqs, int max_blocks_per_seq,
    float scale,
    zedinferDataType_t dtype,
    zedinferDeviceType_t device_type,
    int device_id,
    int nhead, int nkvhead, int head_dim, int block_size) {

    zedinfer::core::context().setDevice(device_type, device_id);

    switch (device_type) {
#ifdef ENABLE_NVIDIA_API
    case ZEDINFER_DEVICE_NVIDIA:
        return nvidia::paged_attention_decode_batched(
            attn_val->data(), q->data(),
            reinterpret_cast<const std::byte *>(pool_base),
            k_block_tables, v_block_tables, seq_lens,
            num_reqs, max_blocks_per_seq,
            scale, dtype, nhead, nkvhead, head_dim, block_size);
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}

} // namespace zedinfer::ops
