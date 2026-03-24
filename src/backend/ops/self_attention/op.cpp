#include "backend/core/context/context.hpp"
#include "backend/core/core.hpp"
#include "backend/ops/ops.hpp"
#include "backend/ops/self_attention/cpu/paged_attention_cpu.hpp"
#ifdef ENABLE_NVIDIA_API
#include "backend/ops/self_attention/nvidia/paged_attention_nvidia.cuh"
#endif
#include "utils/check.hpp"

namespace zedinfer::ops {

// ============================================================================
// Paged decode — single request
// ============================================================================

static void dispatch_paged_decode(const AttentionParams &p) {
    auto &c = p.config;

    if (c.device_type == ZEDINFER_DEVICE_CPU) {
        return cpu::paged_attention_decode(
            p.out->data(), p.q->data(),
            reinterpret_cast<const std::byte *>(p.pool_base),
            p.k_block_table, p.v_block_table,
            p.seq_len, c.scale, c.dtype,
            c.nhead, c.nkvhead, c.head_dim, c.block_size);
    }

    core::context().setDevice(c.device_type, c.device_id);

    switch (c.device_type) {
#ifdef ENABLE_NVIDIA_API
    case ZEDINFER_DEVICE_NVIDIA:
        return nvidia::paged_attention_decode(
            p.out->data(), p.q->data(),
            reinterpret_cast<const std::byte *>(p.pool_base),
            p.k_block_table, p.v_block_table,
            p.seq_len, c.scale, c.dtype,
            c.nhead, c.nkvhead, c.head_dim, c.block_size);
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}

// ============================================================================
// Paged decode — batched (multiple requests)
// ============================================================================

static void dispatch_paged_decode_batched(const AttentionParams &p) {
    auto &c = p.config;

    if (c.device_type == ZEDINFER_DEVICE_CPU) {
        return cpu::paged_attention_decode_batched(
            p.out->data(), p.q->data(),
            reinterpret_cast<const std::byte *>(p.pool_base),
            p.batched_k_block_tables, p.batched_v_block_tables,
            p.batched_seq_lens,
            p.num_requests, p.max_blocks_per_seq,
            c.scale, c.dtype,
            c.nhead, c.nkvhead, c.head_dim, c.block_size);
    }

    core::context().setDevice(c.device_type, c.device_id);

    switch (c.device_type) {
#ifdef ENABLE_NVIDIA_API
    case ZEDINFER_DEVICE_NVIDIA:
        return nvidia::paged_attention_decode_batched(
            p.out->data(), p.q->data(),
            reinterpret_cast<const std::byte *>(p.pool_base),
            p.batched_k_block_tables, p.batched_v_block_tables,
            p.batched_seq_lens,
            p.num_requests, p.max_blocks_per_seq,
            c.scale, c.dtype,
            c.nhead, c.nkvhead, c.head_dim, c.block_size);
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}

// ============================================================================
// Paged prefill — single request
// ============================================================================

static void dispatch_paged_prefill(const AttentionParams &p) {
    auto &c = p.config;

    if (c.device_type == ZEDINFER_DEVICE_CPU) {
        return cpu::paged_attention_prefill(
            p.out->data(), p.q->data(),
            reinterpret_cast<const std::byte *>(p.pool_base),
            p.k_block_table, p.v_block_table,
            p.seqlen_q, p.past_len, c.scale, c.dtype,
            c.nhead, c.nkvhead, c.head_dim, c.block_size);
    }

    core::context().setDevice(c.device_type, c.device_id);

    switch (c.device_type) {
#ifdef ENABLE_NVIDIA_API
    case ZEDINFER_DEVICE_NVIDIA:
        return nvidia::paged_attention_prefill(
            p.out->data(), p.q->data(),
            reinterpret_cast<const std::byte *>(p.pool_base),
            p.k_block_table, p.v_block_table,
            p.seqlen_q, p.past_len, c.scale, c.dtype,
            c.nhead, c.nkvhead, c.head_dim, c.block_size);
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}

// ============================================================================
// Unified dispatch
// ============================================================================

void attention(const AttentionParams &params) {
    if (params.is_batched()) {
        return dispatch_paged_decode_batched(params);
    }
    if (params.is_decode()) {
        return dispatch_paged_decode(params);
    }
    return dispatch_paged_prefill(params);
}

} // namespace zedinfer::ops
