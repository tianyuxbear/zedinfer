#include "backend/core/context/context.hpp"
#include "backend/core/core.hpp"
#include "backend/ops/ops.hpp"
#include "backend/ops/self_attention/cpu/paged_attention_cpu.hpp"
#ifdef ENABLE_NVIDIA_API
#include "backend/ops/self_attention/nvidia/paged_attention_nvidia.cuh"
#ifdef USE_FLASHINFER
#include "backend/ops/self_attention/nvidia/flashinfer_wrapper.cuh"
#endif
#endif
#include "utils/check.hpp"

namespace zedinfer::ops {

// ============================================================================
// Paged decode — single request
// ============================================================================

static void dispatch_paged_decode(const AttentionParams& p) {
    auto& c = p.config;

    if (c.device_type == ZEDINFER_DEVICE_CPU) {
        return cpu::paged_attention_decode(p.out->data(), p.q->data(),
                                           reinterpret_cast<const std::byte*>(p.k_pool_base),
                                           reinterpret_cast<const std::byte*>(p.v_pool_base), p.page_table, p.seq_len,
                                           c.scale, c.dtype, c.nhead, c.nkvhead, c.head_dim, c.block_size);
    }

    core::context().setDevice(c.device_type, c.device_id);

    switch (c.device_type) {
#ifdef ENABLE_NVIDIA_API
        case ZEDINFER_DEVICE_NVIDIA:
            return nvidia::paged_attention_decode(
                p.out->data(), p.q->data(), reinterpret_cast<const std::byte*>(p.k_pool_base),
                reinterpret_cast<const std::byte*>(p.v_pool_base), p.page_table, p.seq_len, c.scale, c.dtype, c.nhead,
                c.nkvhead, c.head_dim, c.block_size);
#endif
        default:
            EXCEPTION_UNSUPPORTED_DEVICE;
    }
}

// ============================================================================
// Paged decode — batched (multiple requests)
// ============================================================================

static void dispatch_paged_decode_batched(const AttentionParams& p) {
    auto& c = p.config;

    if (c.device_type == ZEDINFER_DEVICE_CPU) {
        return cpu::paged_attention_decode_batched(
            p.out->data(), p.q->data(), reinterpret_cast<const std::byte*>(p.k_pool_base),
            reinterpret_cast<const std::byte*>(p.v_pool_base), p.batched_page_tables, p.batched_seq_lens,
            p.num_requests, p.max_blocks_per_seq, c.scale, c.dtype, c.nhead, c.nkvhead, c.head_dim, c.block_size);
    }

    core::context().setDevice(c.device_type, c.device_id);

    switch (c.device_type) {
#ifdef ENABLE_NVIDIA_API
        case ZEDINFER_DEVICE_NVIDIA:
            return nvidia::paged_attention_decode_batched(
                p.out->data(), p.q->data(), reinterpret_cast<const std::byte*>(p.k_pool_base),
                reinterpret_cast<const std::byte*>(p.v_pool_base), p.batched_page_tables, p.batched_seq_lens,
                p.num_requests, p.max_blocks_per_seq, c.scale, c.dtype, c.nhead, c.nkvhead, c.head_dim, c.block_size);
#endif
        default:
            EXCEPTION_UNSUPPORTED_DEVICE;
    }
}

// ============================================================================
// Paged prefill — single request
// ============================================================================

static void dispatch_paged_prefill(const AttentionParams& p) {
    auto& c = p.config;

    if (c.device_type == ZEDINFER_DEVICE_CPU) {
        return cpu::paged_attention_prefill(p.out->data(), p.q->data(),
                                            reinterpret_cast<const std::byte*>(p.k_pool_base),
                                            reinterpret_cast<const std::byte*>(p.v_pool_base), p.page_table, p.seqlen_q,
                                            p.past_len, c.scale, c.dtype, c.nhead, c.nkvhead, c.head_dim, c.block_size);
    }

    core::context().setDevice(c.device_type, c.device_id);

    switch (c.device_type) {
#ifdef ENABLE_NVIDIA_API
        case ZEDINFER_DEVICE_NVIDIA: {
            // Spec-decode verify (n_q ∈ {2, 3, 4}) goes through a kernel tuned
            // for the "almost-decode" shape: one CTA per head, KV cache loaded
            // once and scored against every query. The general prefill kernel
            // launches n_q separate blocks per head and rereads K/V each time
            // — that's ~4.4x slower at n_q=2. The small_nq kernel returns true
            // when it handled the call; otherwise we fall back to prefill.
            if (p.seqlen_q >= 2 && p.seqlen_q <= 4
                && std::getenv("ZEDINFER_DISABLE_SMALL_NQ") == nullptr) {
                if (nvidia::paged_attention_small_nq(
                        p.out->data(), p.q->data(), reinterpret_cast<const std::byte*>(p.k_pool_base),
                        reinterpret_cast<const std::byte*>(p.v_pool_base), p.page_table, p.seqlen_q, p.past_len,
                        c.scale, c.dtype, c.nhead, c.nkvhead, c.head_dim, c.block_size)) {
                    return;
                }
            }
            return nvidia::paged_attention_prefill(
                p.out->data(), p.q->data(), reinterpret_cast<const std::byte*>(p.k_pool_base),
                reinterpret_cast<const std::byte*>(p.v_pool_base), p.page_table, p.seqlen_q, p.past_len, c.scale,
                c.dtype, c.nhead, c.nkvhead, c.head_dim, c.block_size);
        }
#endif
        default:
            EXCEPTION_UNSUPPORTED_DEVICE;
    }
}

// ============================================================================
// Unified dispatch
// ============================================================================

void attention(const AttentionParams& params) {
#if defined(ENABLE_NVIDIA_API) && defined(USE_FLASHINFER)
    if (params.use_flashinfer && params.config.device_type == ZEDINFER_DEVICE_NVIDIA) {
        auto& c = params.config;
        core::context().setDevice(c.device_type, c.device_id);
        if (params.qo_indptr != nullptr) {
            return nvidia::flashinfer_attention_prefill(params);
        }
        return nvidia::flashinfer_attention_decode(params);
    }
#endif
    if (params.is_batched()) {
        return dispatch_paged_decode_batched(params);
    }
    if (params.is_decode()) {
        return dispatch_paged_decode(params);
    }
    return dispatch_paged_prefill(params);
}

} // namespace zedinfer::ops
