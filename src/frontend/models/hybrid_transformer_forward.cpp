#include "frontend/models/hybrid_transformer_forward.hpp"

#include "backend/core/context/context.hpp"
#include "backend/device/runtime_api.hpp"
#include "backend/ops/attn_output_gate/attn_output_gate.hpp"
#include "backend/ops/mamba/causal_conv1d.hpp"
#include "backend/ops/mamba/gdn.hpp"
#include "backend/ops/mamba/qk_l2norm.hpp"
#include "backend/ops/mamba/ssu.hpp"
#include "backend/ops/mrope/mrope_3d.hpp"
#include "backend/ops/ops.hpp"
#include "frontend/models/decode_scratch.hpp"
#include "frontend/models/moe_forward.hpp"
#include "frontend/models/paged_forward_context.hpp"
#include "utils/types.hpp"
#include "zedinfer/activation.hpp"
#include "zedinfer/request.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace zedinfer::model {

// Persistent per-thread scratch for N=1 decode through forward_linear_attn_layer.
// 30 of the 40 layers in Qwen3.5-A3B run this path, and the original code did 11
// Tensor::create() calls per layer against the populated MoE pool; folding them
// into a thread_local growable struct eliminates 330 pool round-trips per token.
//
// Shapes depend on the model's linear-attn config (Hk, Dk, Hv, Dv) which is
// constant across layers; we (re)allocate when the cached shape differs.
struct LinearAttnDecodeScratch {
    // Activations / projection outputs
    tensor_t qkv;      // {1, qkv_dim}
    tensor_t z;        // {1, Hv*Dv}
    tensor_t a;        // {1, Hv}
    tensor_t b;        // {1, Hv}
    tensor_t qkv_conv; // {1, qkv_dim}
    tensor_t q_ssm;    // {1, Hk*Dk}
    tensor_t k_ssm;    // {1, Hk*Dk}
    tensor_t v_ssm;    // {1, Hv*Dv}
    tensor_t y;        // {1, Hv*Dv}
    tensor_t y_normed; // {1, Hv*Dv}
    tensor_t y_gated;  // {1, Hv*Dv}
    tensor_t out;      // {1, hidden}

    // Cached shape signature; on mismatch (e.g., different model loaded on the
    // same thread, or prefill drops into this path), reallocate.
    int Hk = -1, Dk = -1, Hv = -1, Dv = -1;
    size_t hidden = 0;
    zedinferDeviceType_t device_type = ZEDINFER_DEVICE_CPU;
    int device_id = -1;
    zedinferDataType_t dtype = ZEDINFER_DTYPE_F32;
};

static thread_local LinearAttnDecodeScratch s_la_scratch;

// Same idea for the 10 Qwen3.5 full-attention layers: 6 fresh Tensor::create
// calls per decode token (q_raw, gate, k_raw, v, q_normed, k_normed, out) all
// hit the populated pool. attn itself is allocated inside ctx.attend (paged
// attention manages its own buffer), so we don't capture it here.
struct FullAttnDecodeScratch {
    tensor_t q_raw;    // {1, q_dim}
    tensor_t gate;     // {1, q_dim}
    tensor_t k_raw;    // {1, kv_dim}
    tensor_t v;        // {1, kv_dim}
    tensor_t q_normed; // {1, q_dim}
    tensor_t k_normed; // {1, kv_dim}
    tensor_t attn;     // {1, Hq, Dh} — pre-alloc passed into ctx.attend
    tensor_t out;      // {1, hidden}

    size_t Hq = 0, Hkv = 0, Dh = 0, hidden = 0;
    zedinferDeviceType_t device_type = ZEDINFER_DEVICE_CPU;
    int device_id = -1;
    zedinferDataType_t dtype = ZEDINFER_DTYPE_F32;
};

static thread_local FullAttnDecodeScratch s_fa_scratch;

// Max batch we pre-size the per-layer scratch buffers for. N=1 covers normal
// decode; N=2 covers Qwen3.5 MTP speculative-decode 2-token verify. Larger N
// (prefill) falls back to per-call alloc.
static constexpr size_t kMaxSmallDecodeBatch = 2;

static void ensure_fa_scratch(const ExecutorConfig& exec, size_t Hq, size_t Hkv, size_t Dh, size_t hidden) {
    auto& s = s_fa_scratch;
    if (s.Hq == Hq && s.Hkv == Hkv && s.Dh == Dh && s.hidden == hidden && s.device_type == exec.device_type
        && s.device_id == exec.device_id && s.dtype == exec.data_type && s.q_raw) {
        return;
    }
    const size_t q_dim = Hq * Dh;
    const size_t kv_dim = Hkv * Dh;
    auto mk = [&](std::vector<size_t> shape) {
        return Tensor::create(std::move(shape), exec.data_type, exec.device_type, exec.device_id);
    };
    // Buffers are sized at kMaxSmallDecodeBatch (=2) along the batch dim so the
    // same scratch backs both N=1 decode and N=2 MTP spec verify. The use sites
    // slice the first N rows; slice(0, 0, 1) is a zero-copy contiguous view.
    const size_t B = kMaxSmallDecodeBatch;
    s.q_raw = mk({B, q_dim});
    s.gate = mk({B, q_dim});
    s.k_raw = mk({B, kv_dim});
    s.v = mk({B, kv_dim});
    s.q_normed = mk({B, q_dim});
    s.k_normed = mk({B, kv_dim});
    s.attn = mk({B, Hq, Dh});
    s.out = mk({B, hidden});
    s.Hq = Hq;
    s.Hkv = Hkv;
    s.Dh = Dh;
    s.hidden = hidden;
    s.device_type = exec.device_type;
    s.device_id = exec.device_id;
    s.dtype = exec.data_type;
}

static void ensure_la_scratch(const ExecutorConfig& exec, int Hk, int Dk, int Hv, int Dv, size_t hidden) {
    auto& s = s_la_scratch;
    if (s.Hk == Hk && s.Dk == Dk && s.Hv == Hv && s.Dv == Dv && s.hidden == hidden && s.device_type == exec.device_type
        && s.device_id == exec.device_id && s.dtype == exec.data_type && s.qkv) {
        return;
    }
    const size_t Hk_Dk = static_cast<size_t>(Hk) * static_cast<size_t>(Dk);
    const size_t Hv_Dv = static_cast<size_t>(Hv) * static_cast<size_t>(Dv);
    const size_t qkv_dim = 2 * Hk_Dk + Hv_Dv;
    auto mk = [&](std::vector<size_t> shape) {
        return Tensor::create(std::move(shape), exec.data_type, exec.device_type, exec.device_id);
    };
    const size_t B = kMaxSmallDecodeBatch;
    s.qkv = mk({B, qkv_dim});
    s.z = mk({B, Hv_Dv});
    s.a = mk({B, static_cast<size_t>(Hv)});
    s.b = mk({B, static_cast<size_t>(Hv)});
    s.qkv_conv = mk({B, qkv_dim});
    s.q_ssm = mk({B, Hk_Dk});
    s.k_ssm = mk({B, Hk_Dk});
    s.v_ssm = mk({B, Hv_Dv});
    s.y = mk({B, Hv_Dv});
    s.y_normed = mk({B, Hv_Dv});
    s.y_gated = mk({B, Hv_Dv});
    s.out = mk({B, hidden});
    s.Hk = Hk;
    s.Dk = Dk;
    s.Hv = Hv;
    s.Dv = Dv;
    s.hidden = hidden;
    s.device_type = exec.device_type;
    s.device_id = exec.device_id;
    s.dtype = exec.data_type;
}

// Helper: slice the first `N` rows of a scratch buffer pre-sized at
// [kMaxSmallDecodeBatch, ...]. Zero-copy view; safe for ops::linear since the
// row-major contiguous layout is preserved.
static inline tensor_t scratch_view(const tensor_t& buf, size_t N) {
    return (buf->shape()[0] == N) ? buf : buf->slice(0, 0, N);
}

// Hybrid forward layer-wide N=2 scratch (spec-decode verify). Mirrors the
// N=1 DecodeScratch fields used by hybrid_transformer_forward's outer loop,
// at [2, ...] dim. Without this each verify step pays 40 layers × 4 fresh
// [2, H] allocs = ~20 ms / step under ALL_GPU; this drops it to a one-time
// alloc on first use.
struct HybridN2Scratch {
    tensor_t ids;          // [2] I32
    tensor_t pos_ids;      // [2] I64
    tensor_t hidden;       // [2, H]
    tensor_t hidden_out;   // [2, H]
    tensor_t normed;       // [2, H]
    tensor_t h1;           // [2, H]
    tensor_t normed_post;  // [2, H]
    tensor_t final_normed; // [2, H]
    tensor_t logits;       // [2, V]
    tensor_t hidden_snap;  // [2, H] — MTP residual snapshot

    size_t H = 0;
    size_t V = 0;
    zedinferDeviceType_t device_type = ZEDINFER_DEVICE_CPU;
    int device_id = -1;
    zedinferDataType_t dtype = ZEDINFER_DTYPE_F32;
};
static thread_local HybridN2Scratch s_hybrid_n2;

static HybridN2Scratch& ensure_hybrid_n2(size_t H, size_t V, const ExecutorConfig& exec) {
    auto& s = s_hybrid_n2;
    if (s.hidden && s.H == H && s.V == V && s.device_type == exec.device_type && s.device_id == exec.device_id
        && s.dtype == exec.data_type) {
        return s;
    }
    auto mk = [&](std::vector<size_t> shape, zedinferDataType_t t) {
        return Tensor::create(std::move(shape), t, exec.device_type, exec.device_id);
    };
    auto mkf = [&](std::vector<size_t> shape) { return mk(shape, exec.data_type); };
    const size_t B = kMaxSmallDecodeBatch;
    s.ids = mk({B}, ZEDINFER_DTYPE_I32);
    s.pos_ids = mk({B}, ZEDINFER_DTYPE_I64);
    s.hidden = mkf({B, H});
    s.hidden_out = mkf({B, H});
    s.normed = mkf({B, H});
    s.h1 = mkf({B, H});
    s.normed_post = mkf({B, H});
    s.final_normed = mkf({B, H});
    s.logits = mkf({B, V});
    s.hidden_snap = mkf({B, H});
    s.H = H;
    s.V = V;
    s.device_type = exec.device_type;
    s.device_id = exec.device_id;
    s.dtype = exec.data_type;
    return s;
}

// Forward declarations of the per-layer-kind helpers. Each is implemented in
// its own subsequent commit (T13 = dense_mlp, T14 = full_attn, T15 = linear_attn,
// T16 = moe_mlp). Until then they throw a clearly-tagged exception so the
// hybrid path fails fast with a specific error rather than silently producing
// garbage.

static tensor_t forward_linear_attn_layer(const HybridForwardConfig& m, tensor_t h_in, size_t L, InferenceRequest& req,
                                          const ExecutorConfig& exec);

static tensor_t forward_full_attn_layer(const HybridForwardConfig& m, PagedForwardContext& ctx, tensor_t h_in,
                                        tensor_t pos_ids_thw, size_t L, const ExecutorConfig& exec);

// Build [3, N] int32 pos_ids_thw on the compute device. For text-only sequences
// the three axes collapse to (idx, idx, idx); multimodal pipelines will pass
// `req.pos_ids_thw()` here directly instead. Caller-owned tensor with one H2D
// per forward call (cheap relative to per-layer kernels).
static tensor_t build_pos_ids_thw_text_only(PagedForwardContext& ctx, const ExecutorConfig& exec) {
    tensor_t ids_unused, pos_ids;
    // prepare_inputs is idempotent here — outer loop already called it; calling
    // again just rebuilds the same buffers. We only need pos_ids (1D int64).
    ctx.prepare_inputs(ids_unused, pos_ids, exec);

    const size_t N = static_cast<size_t>(pos_ids->numel());
    // Round-trip pos_ids back to host once (small N) and broadcast into [3, N] int32.
    std::vector<int64_t> host64(N);
    auto* api = device::getRuntimeAPI(exec.device_type);
    api->memcpy_sync(host64.data(), pos_ids->data(), N * sizeof(int64_t), ZEDINFER_MEMCPY_D2H);
    std::vector<int32_t> host_thw(3 * N);
    for (size_t i = 0; i < N; ++i) {
        const int32_t p = static_cast<int32_t>(host64[i]);
        host_thw[0 * N + i] = p; // t
        host_thw[1 * N + i] = p; // h
        host_thw[2 * N + i] = p; // w
    }
    auto thw = Tensor::create({3, N}, ZEDINFER_DTYPE_I32, exec.device_type, exec.device_id);
    api->memcpy_sync(thw->data(), host_thw.data(), 3 * N * sizeof(int32_t), ZEDINFER_MEMCPY_H2D);
    return thw;
}

static tensor_t forward_dense_mlp(const HybridForwardConfig& m, tensor_t h_post, size_t L, const ExecutorConfig& exec);

static tensor_t forward_moe_mlp(const HybridForwardConfig& m, tensor_t h_post, size_t L, const ExecutorConfig& exec,
                                DecodeScratch* scratch);

tensor_t hybrid_transformer_forward(const HybridForwardConfig& m, PagedForwardContext& ctx, InferenceRequest& req,
                                    const ExecutorConfig& exec, DecodeScratch* scratch, tensor_t input_embeds,
                                    tensor_t* hidden_out) {
    const auto& cfg = m.config;
    const size_t N = static_cast<size_t>(ctx.num_tokens());
    const size_t hidden_size = cfg.hidden_size;

    auto make = [&](std::vector<size_t> shape) {
        return Tensor::create(std::move(shape), exec.data_type, exec.device_type, exec.device_id);
    };

    // Pure decode (N == 1) can reuse the engine-wide DecodeScratch hidden-size
    // buffers for every layer's intermediate state. Without this each layer
    // does four make({1, hidden}) calls against the populated MoE pool, costing
    // 4 x 40 = 160 BestFitMemoryPool round-trips per token.
    bool use_scratch = (scratch != nullptr && N == 1 && !input_embeds);
    if (use_scratch && std::getenv("ZEDINFER_DISABLE_SCRATCH") != nullptr) {
        use_scratch = false;
    }
    // For N=2 (MTP spec-decode verify) use a thread_local persistent N=2
    // scratch to avoid the same per-layer fresh-alloc storm under ALL_GPU.
    // Mutually exclusive with use_scratch (which is N=1 only).
    const bool use_n2_scratch
        = (!use_scratch && N == 2 && !input_embeds && std::getenv("ZEDINFER_DISABLE_SCRATCH") == nullptr);
    HybridN2Scratch* n2 = nullptr;
    if (use_n2_scratch) {
        n2 = &ensure_hybrid_n2(hidden_size, cfg.vocab_size, exec);
    }
    auto n2_view = [&](const tensor_t& buf) { return scratch_view(buf, N); };

    // Prepare token ids + position ids. For the hybrid path the pos_ids carry
    // (t, h, w) — but the outer loop only consumes them via PagedForwardContext;
    // the per-layer kernels read pos_ids_thw from req when they need 3D mrope.
    tensor_t ids, pos_ids;
    if (use_scratch) {
        ids = scratch->ids;
        pos_ids = scratch->pos_ids;
        ctx.prepare_inputs_into(ids, pos_ids);
    } else if (use_n2_scratch) {
        ids = n2_view(n2->ids);
        pos_ids = n2_view(n2->pos_ids);
        ctx.prepare_inputs_into(ids, pos_ids);
    } else {
        ctx.prepare_inputs(ids, pos_ids, exec);
    }

    // Layer-0 hidden state. Vision pipelines pass input_embeds (already
    // shaped [N, hidden_size] with vision-tower embeddings scattered over
    // <|image_pad|> positions); text-only path runs the embed_tokens lookup.
    tensor_t hidden;
    if (input_embeds) {
        hidden = input_embeds;
    } else {
        hidden = use_scratch ? scratch->hidden : use_n2_scratch ? n2_view(n2->hidden) : make({N, hidden_size});
        ops::embedding(hidden, ids, m.W("embed_tokens.weight"));
    }

    // Build 3D positional ids once per forward. Vision pipelines will pass
    // req.pos_ids_thw() instead; text-only sequences get the (idx, idx, idx)
    // broadcast from build_pos_ids_thw_text_only.
    tensor_t pos_ids_thw = req.pos_ids_thw();
    if (!pos_ids_thw) {
        pos_ids_thw = build_pos_ids_thw_text_only(ctx, exec);
    }

    // Layer loop with FusedAddRMSNorm.
    //
    // The Qwen3.5 transformer block is:
    //   h_in = (1+input_norm) * rms(hidden)               // [A] pre-attn norm
    //   attn_out = attention(h_in)
    //   h1 = hidden + attn_out                            // [B] residual after attn
    //   h_post = (1+post_norm) * rms(h1)                  // [C] pre-mlp norm
    //   mlp_out = mlp(h_post)
    //   hidden_next = h1 + mlp_out                        // [D] residual after mlp
    //   then [A]@next_layer or [E] final_norm at last layer
    //
    // Fusion:
    //   [B]+[C]: `fused_add_rms_norm(attn_out, hidden, post_norm)`
    //             → hidden becomes hidden + attn_out (the new residual stream)
    //             → attn_out becomes the normalized h_post (reused as `h_post`)
    //   [D]+[A]@next: `fused_add_rms_norm(mlp_out, hidden, input_norm_{L+1})`
    //             → hidden becomes hidden + mlp_out
    //             → mlp_out becomes h_in for next layer
    //   For the last layer, [D]+[E]: `fused_add_rms_norm(mlp_out, hidden, final_norm)`
    //   so the final `rms_norm` after the loop disappears entirely.
    //
    // The residual stream lives in-place in `hidden` — no ping-pong needed. The
    // scratch->h1 / scratch->normed_post / scratch->hidden_out buffers become
    // unused but are kept allocated (used by the CPU fallback if any). Only the
    // very first `input_layernorm_0` keeps a standalone `rms_norm` call since
    // there is no preceding residual add to fuse with.

    auto h_in = use_scratch ? scratch->normed : use_n2_scratch ? n2_view(n2->normed) : make({N, hidden_size});
    // [A] for L=0: standalone, no preceding add.
    ops::rms_norm(h_in, hidden, m.W(m.prefix(0) + "input_layernorm.weight"), cfg.rms_norm_eps,
                  /*add_one_to_weight=*/true);

    for (size_t L = 0; L < cfg.num_hidden_layers; ++L) {
        // Dispatch attention by layer kind. h_in is the normalized input.
        tensor_t attn_out = m.is_linear_attn_layer(L) ? forward_linear_attn_layer(m, h_in, L, req, exec)
                                                      : forward_full_attn_layer(m, ctx, h_in, pos_ids_thw, L, exec);

        // [B]+[C] fused: hidden += attn_out; attn_out := rmsnorm(hidden) * (1 + post_norm).
        // After the call attn_out aliases what used to be `h_post`.
        ops::fused_add_rms_norm(attn_out, hidden, m.W(m.prefix(L) + "post_attention_layernorm.weight"),
                                cfg.rms_norm_eps, /*add_one_to_weight=*/true);
        tensor_t h_post = attn_out;

        // Dispatch MLP by sparsity. moe_layer_forward routes to moe_decode (N=1)
        // or its small-N path (N=2 spec verify) when scratch is provided.
        tensor_t mlp_out
            = m.is_moe_layer(L) ? forward_moe_mlp(m, h_post, L, exec, scratch) : forward_dense_mlp(m, h_post, L, exec);

        // [D]+next-norm fused. For L < last layer use next layer's input_layernorm;
        // for the last layer use the model's final norm — collapsing the post-loop
        // standalone rms_norm into this fused call. After this:
        //   hidden = residual stream after this layer (input for next layer's add)
        //   mlp_out = h_in for next layer (or final_normed at the last layer)
        tensor_t next_norm_w
            = (L + 1 < cfg.num_hidden_layers) ? m.W(m.prefix(L + 1) + "input_layernorm.weight") : m.W("norm.weight");

        // MTP head wants a snapshot of the PRE-final-norm residual stream. At the
        // last layer that snapshot is precisely `hidden` *after* the fused add but
        // before the norm is applied — but the fused kernel writes the residual
        // first (the add result) and then reads it to produce the norm output. So
        // after the fused call, `hidden` is exactly the pre-final-norm residual.
        // We snapshot it after the fused call (below) when L == last layer.
        ops::fused_add_rms_norm(mlp_out, hidden, next_norm_w, cfg.rms_norm_eps,
                                /*add_one_to_weight=*/true);
        h_in = mlp_out;
    }

    // MTP head consumes the residual stream BEFORE the final norm. After the
    // loop, `hidden` holds residual+all_layers (i.e. pre-final-norm value),
    // since the last layer's fused call wrote the sum into `hidden` then
    // normalized into `mlp_out` (now aliased by `h_in`). Async D2D to a snap
    // buffer so the lm_head below can run concurrently.
    if (hidden_out != nullptr) {
        tensor_t snap = use_n2_scratch ? n2_view(n2->hidden_snap) : make({N, hidden_size});
        auto* api = device::getRuntimeAPI(exec.device_type);
        auto compute_stream = core::context().runtime().stream();
        api->memcpy_async(snap->data(), hidden->data(), N * hidden_size * utils::dsize(exec.data_type),
                          ZEDINFER_MEMCPY_D2D, compute_stream);
        *hidden_out = std::move(snap);
    }

    // `h_in` is now the final-normalized hidden (formerly `final_normed`), so
    // skip the standalone final `rms_norm` and go straight to lm_head.
    auto logits = use_scratch ? scratch->logits : use_n2_scratch ? n2_view(n2->logits) : make({N, cfg.vocab_size});
    m.dispatch_linear(logits, h_in, "lm_head", nullptr);

    ctx.finalize();
    return logits;
}

// ============================================================================
// Per-layer-kind stubs. Replaced in T13-T16. Each throws a tagged exception
// so the calling site sees exactly which sub-function is missing.
// ============================================================================

static tensor_t forward_linear_attn_layer(const HybridForwardConfig& m, tensor_t h_in, size_t L, InferenceRequest& req,
                                          const ExecutorConfig& exec) {
    if (req.ssm_slot_idx() < 0) {
        throw std::runtime_error("hybrid: forward_linear_attn_layer L=" + std::to_string(L)
                                 + " — request has no SSM slot (scheduler did not call acquire_slot)");
    }
    if (!m.ssm_pool) {
        throw std::runtime_error("hybrid: forward_linear_attn_layer L=" + std::to_string(L)
                                 + " — HybridForwardConfig::ssm_pool is null");
    }

    const size_t N = static_cast<size_t>(h_in->shape()[0]);
    const int Hv = m.linear_attn.num_v_heads;
    const int Dv = m.linear_attn.value_head_dim;
    const int Hk = m.linear_attn.num_k_heads;
    const int Dk = m.linear_attn.key_head_dim;
    const size_t Hk_Dk = static_cast<size_t>(Hk) * static_cast<size_t>(Dk);
    const size_t Hv_Dv = static_cast<size_t>(Hv) * static_cast<size_t>(Dv);
    const size_t qkv_dim = 2 * Hk_Dk + Hv_Dv;

    const auto p = m.prefix(L) + "linear_attn.";

    auto make = [&](std::vector<size_t> shape) {
        return Tensor::create(std::move(shape), exec.data_type, exec.device_type, exec.device_id);
    };

    // For N ∈ {1, 2} reuse the thread_local persistent scratch (12 buffers
    // sized at kMaxSmallDecodeBatch=2 along the batch dim). For prefill (N > 2)
    // fall back to per-call alloc. The buffers are dimensioned for N=2 so a
    // slice(0, 0, N) call returns a zero-copy contiguous view of the first N
    // rows, valid for both N=1 decode and N=2 MTP spec-decode verify.
    const bool use_la_scratch = (N <= kMaxSmallDecodeBatch);
    if (use_la_scratch) {
        ensure_la_scratch(exec, Hk, Dk, Hv, Dv, m.config.hidden_size);
    }

    auto la_view = [&](const tensor_t& buf) { return scratch_view(buf, N); };

    // 1. Four input projections (all BF16; linear_attn is excluded from quantization
    //    per Qwen3.5 dynamic rules).
    auto qkv = use_la_scratch ? la_view(s_la_scratch.qkv) : make({N, qkv_dim});
    auto z = use_la_scratch ? la_view(s_la_scratch.z) : make({N, Hv_Dv});
    auto a = use_la_scratch ? la_view(s_la_scratch.a) : make({N, static_cast<size_t>(Hv)});
    auto b = use_la_scratch ? la_view(s_la_scratch.b) : make({N, static_cast<size_t>(Hv)});
    ops::linear(qkv, h_in, m.W(p + "in_proj_qkv.weight"));
    ops::linear(z, h_in, m.W(p + "in_proj_z.weight"));
    ops::linear(a, h_in, m.W(p + "in_proj_a.weight"));
    ops::linear(b, h_in, m.W(p + "in_proj_b.weight"));

    // 2-4. Stateful core: depthwise causal conv1d (kernel=4, folds in silu) ->
    //       split into q/k/v -> L2-normalize q/k (HF use_qk_l2norm_in_kernel=True,
    //       scale q by 1/sqrt(Dk)) -> GatedDeltaNet delta-rule recurrence
    //       (in-place SSM state update + per-token readout y). Factored into a
    //       per-(row-range, slot) helper so MTP spec-decode verify can route the
    //       draft token's recurrence to the pool's temp slot, keeping the real
    //       slot at the post-last_token state (free reject, no rollback).
    auto qkv_conv = use_la_scratch ? la_view(s_la_scratch.qkv_conv) : make({N, qkv_dim});
    auto q_ssm = use_la_scratch ? la_view(s_la_scratch.q_ssm) : make({N, Hk_Dk});
    auto k_ssm = use_la_scratch ? la_view(s_la_scratch.k_ssm) : make({N, Hk_Dk});
    auto v_ssm = use_la_scratch ? la_view(s_la_scratch.v_ssm) : make({N, Hv_Dv});
    auto y = use_la_scratch ? la_view(s_la_scratch.y) : make({N, Hv_Dv});
    const size_t elt = utils::dsize(exec.data_type);
    const float qk_scale_q = 1.0f / std::sqrt(static_cast<float>(Dk));
    tensor_t conv_w = m.W(p + "conv1d.weight");
    tensor_t A_log_w = m.W(p + "A_log");
    tensor_t dt_bias_w = m.W(p + "dt_bias");
    const int layer_idx = m.linear_layer_index(L);

    auto run_stateful = [&](size_t r0, size_t r1, int slot) {
        const size_t n = r1 - r0;
        auto qkv_c = qkv_conv->slice(0, r0, r1);
        ops::mamba::causal_conv1d(qkv_c, qkv->slice(0, r0, r1), conv_w, m.ssm_pool->view(), slot, layer_idx);
        auto q_s = q_ssm->slice(0, r0, r1);
        auto k_s = k_ssm->slice(0, r0, r1);
        auto v_s = v_ssm->slice(0, r0, r1);
        ops::mamba::copy_strided_rows(q_s, qkv_c, 0, Hk_Dk, qkv_dim, n, elt);
        ops::mamba::copy_strided_rows(k_s, qkv_c, Hk_Dk, Hk_Dk, qkv_dim, n, elt);
        ops::mamba::copy_strided_rows(v_s, qkv_c, 2 * Hk_Dk, Hv_Dv, qkv_dim, n, elt);
        ops::mamba::qk_l2norm_inplace(q_s, Hk, Dk, qk_scale_q, 1e-6f);
        ops::mamba::qk_l2norm_inplace(k_s, Hk, Dk, 1.0f, 1e-6f);
        ops::mamba::GDNParams gp;
        gp.state_view = m.ssm_pool->view();
        gp.slot_idx = slot;
        gp.layer_idx = layer_idx;
        gp.q = q_s;
        gp.k = k_s;
        gp.v = v_s;
        gp.b = b->slice(0, r0, r1);
        gp.a = a->slice(0, r0, r1);
        gp.A_log = A_log_w;
        gp.dt_bias = dt_bias_w;
        gp.out = y->slice(0, r0, r1);
        gp.num_tokens = static_cast<int>(n);
        ops::mamba::gdn(gp);
    };

    if (req.mtp_spec_verify_active && N == 2) {
        // MTP spec-decode verify. Commit token 0 (last_token) into the request's
        // real slot -> post-last_token state. Seed the pool's temp slot with that
        // state and apply token 1 (draft) there -> post-draft state. The real slot
        // is left at the post-last_token state, so a reject needs no rollback and
        // an accept just promotes temp->real (Scheduler::process_results).
        const int real_slot = req.ssm_slot_idx();
        const int temp_slot = m.ssm_pool->spec_temp_slot();
        run_stateful(0, 1, real_slot);
        m.ssm_pool->copy_layer_state(temp_slot, real_slot, layer_idx);
        run_stateful(1, 2, temp_slot);
    } else {
        run_stateful(0, N, req.ssm_slot_idx());
    }

    // 6. RMSNorm on per-V-head value_head_dim axis. Qwen3.5 stores the norm
    //    weight as fp32 (`mamba_ssm_dtype=float32`) on disk but we pre-cast it
    //    to bf16 once at model load (see fixup_qwen3_5_linear_attn_norm_weights
    //    in qwen3_5.cpp) so the rms_norm same-dtype check passes without any
    //    per-call D2H+H2D.
    tensor_t norm_w = m.W(p + "norm.weight");

    auto y_normed = use_la_scratch ? la_view(s_la_scratch.y_normed) : make({N, Hv_Dv});
    ops::rms_norm(y_normed->view({N * static_cast<size_t>(Hv), static_cast<size_t>(Dv)}),
                  y->view({N * static_cast<size_t>(Hv), static_cast<size_t>(Dv)}), norm_w, m.config.rms_norm_eps);

    // 7. GDN output gating: y_normed = silu(z) * y_normed.
    //    Performed externally because the kernel only emits the delta-rule
    //    readout; Qwen3.5 applies silu(z) * o_norm before out_proj.
    {
        auto y_gated = use_la_scratch ? la_view(s_la_scratch.y_gated) : make({N, Hv_Dv});
        ops::silu_mul(y_gated, z, y_normed);
        y_normed = y_gated;
    }

    // 8. Output projection [Hv_Dv → hidden_size].
    auto out = use_la_scratch ? la_view(s_la_scratch.out) : make({N, m.config.hidden_size});
    ops::linear(out, y_normed, m.W(p + "out_proj.weight"));
    return out;
}

static tensor_t forward_full_attn_layer(const HybridForwardConfig& m, PagedForwardContext& ctx, tensor_t h_in,
                                        tensor_t pos_ids_thw, size_t L, const ExecutorConfig& exec) {
    const size_t N = static_cast<size_t>(h_in->shape()[0]);
    const size_t Hq = m.config.num_attention_heads;
    const size_t Hkv = m.config.num_key_value_heads;
    const size_t Dh = m.config.head_dim > 0 ? m.config.head_dim : (m.config.hidden_size / Hq);
    const size_t q_dim = Hq * Dh;
    const size_t kv_dim = Hkv * Dh;
    const auto p = m.prefix(L) + "self_attn.";

    auto make = [&](std::vector<size_t> shape) {
        return Tensor::create(std::move(shape), exec.data_type, exec.device_type, exec.device_id);
    };

    const bool use_fa_scratch = (N <= kMaxSmallDecodeBatch);
    if (use_fa_scratch) {
        ensure_fa_scratch(exec, Hq, Hkv, Dh, m.config.hidden_size);
    }
    auto fa_view = [&](const tensor_t& buf) { return scratch_view(buf, N); };

    // q_proj has doubled output [2*q_dim, hidden_size] — first q_dim rows are
    // the q projection, second q_dim rows are the output gate. Slice along
    // dim 0 (outer-most) is contiguous, so we can call linear directly twice
    // instead of one combined GEMM + materializing splits.
    auto q_proj_w = m.W(p + "q_proj.weight");           // [2*q_dim, hidden]
    auto w_q = q_proj_w->slice(0, 0, q_dim);            // [q_dim, hidden]
    auto w_gate = q_proj_w->slice(0, q_dim, 2 * q_dim); // [q_dim, hidden]

    auto q_raw = use_fa_scratch ? fa_view(s_fa_scratch.q_raw) : make({N, q_dim});
    auto gate = use_fa_scratch ? fa_view(s_fa_scratch.gate) : make({N, q_dim});
    ops::linear(q_raw, h_in, w_q);
    ops::linear(gate, h_in, w_gate);

    auto k_raw = use_fa_scratch ? fa_view(s_fa_scratch.k_raw) : make({N, kv_dim});
    auto v = use_fa_scratch ? fa_view(s_fa_scratch.v) : make({N, kv_dim});
    ops::linear(k_raw, h_in, m.W(p + "k_proj.weight"));
    ops::linear(v, h_in, m.W(p + "v_proj.weight"));

    // Per-head RMSNorm on q / k (Qwen3.5 uses Qwen3_5MoeRMSNorm with (1+w)).
    auto q_normed = use_fa_scratch ? fa_view(s_fa_scratch.q_normed) : make({N, q_dim});
    auto k_normed = use_fa_scratch ? fa_view(s_fa_scratch.k_normed) : make({N, kv_dim});
    ops::rms_norm(q_normed->view({N * Hq, Dh}), q_raw->view({N * Hq, Dh}), m.W(p + "q_norm.weight"),
                  m.config.rms_norm_eps,
                  /*add_one_to_weight=*/true);
    ops::rms_norm(k_normed->view({N * Hkv, Dh}), k_raw->view({N * Hkv, Dh}), m.W(p + "k_norm.weight"),
                  m.config.rms_norm_eps,
                  /*add_one_to_weight=*/true);

    // 3D MRoPE with partial_rotary_factor (Qwen3.5 = 0.25 → first 64 of 256
    // head_dim rotated, rest pass-through). Rotation is in-place on the
    // per-head view, so reshape to [N, H, Dh] first.
    auto q_view = q_normed->view({N, Hq, Dh});
    auto k_view = k_normed->view({N, Hkv, Dh});
    ops::mrope_3d(q_view, pos_ids_thw, m.mrope);
    ops::mrope_3d(k_view, pos_ids_thw, m.mrope);

    // Paged KV cache uses LOGICAL kv layer index: only full-attention layers
    // contribute to the pool (linear-attention layers hold SSM state instead).
    const int kv_idx = m.full_layer_index(L);
    ctx.write_kv(kv_idx, k_normed, v);

    // Paged attention. attend allocates and returns the attn output tensor;
    // for N=1 decode reuse the persistent attn buffer (shape [1, Hq, Dh])
    // so the per-layer alloc doesn't hit the populated pool. For N=2 we pass
    // a [2, Hq, Dh] view of the same scratch buffer.
    const float scale = 1.0f / std::sqrt(static_cast<float>(Dh));
    auto attn
        = ctx.attend(kv_idx, q_normed, scale, exec, Hq, Hkv, Dh, use_fa_scratch ? fa_view(s_fa_scratch.attn) : nullptr);

    // Attention output gate: attn := attn * sigmoid(gate), in place.
    // Both tensors are shape [N, Hq, Dh] — flatten matches because attn comes
    // back from ctx.attend already shaped for the o_proj input.
    ops::attn_output_gate(attn, gate);

    // o_proj: [q_dim → hidden]. Input width is q_dim (NOT doubled — the gate
    // was consumed in attn_output_gate). attn from ctx.attend has shape
    // [N, Hq, Dh]; view it as [N, q_dim] so ops::linear reads K=q_dim from
    // in->dim(1) rather than the (3-D) middle dimension Hq, which would
    // collapse the o_proj GEMM to a 16-channel reduction and produce a
    // ~100x attenuated residual that drifted from HF after a single layer.
    auto out = use_fa_scratch ? fa_view(s_fa_scratch.out) : make({N, m.config.hidden_size});
    ops::linear(out, attn->view({N, q_dim}), m.W(p + "o_proj.weight"));
    return out;
}

static tensor_t forward_dense_mlp(const HybridForwardConfig& m, tensor_t h_post, size_t L, const ExecutorConfig& exec) {
    const size_t N = static_cast<size_t>(h_post->shape()[0]);
    const size_t hidden = m.config.hidden_size;
    const size_t inter = m.config.intermediate_size;
    const auto p = m.prefix(L);

    auto make = [&](std::vector<size_t> shape) {
        return Tensor::create(std::move(shape), exec.data_type, exec.device_type, exec.device_id);
    };

    // gate / up projections — Qwen3.5 dense MLP is GPTQ Int4 (the loader
    // already remapped *.qweight → *.weight_packed, so dispatch_linear
    // routes to ops::linear_quantized automatically).
    auto gate = make({N, inter});
    auto up = make({N, inter});
    m.dispatch_linear(gate, h_post, p + "mlp.gate_proj", nullptr);
    m.dispatch_linear(up, h_post, p + "mlp.up_proj", nullptr);

    // SwiGLU activation
    auto act = make({N, inter});
    ops::swiglu(act, gate, up);

    // down projection back to hidden
    auto down = make({N, hidden});
    m.dispatch_linear(down, act, p + "mlp.down_proj", nullptr);
    return down;
}

static tensor_t forward_moe_mlp(const HybridForwardConfig& m, tensor_t h_post, size_t L, const ExecutorConfig& exec,
                                DecodeScratch* scratch) {
    // 35B-A3B path: reuse v0.2.0 moe_layer_forward verbatim. HybridForwardConfig
    // inherits ModelForwardConfig, so all the fields moe_layer_forward needs
    // (num_experts, num_experts_per_tok, expert_pool, router_weights,
    // expert_quant_*, has_shared_expert) are populated by Qwen3_5MoeModel's
    // hybrid_forward_config(); no MoE-specific code lives here.
    const size_t N = static_cast<size_t>(h_post->shape()[0]);
    const size_t hidden = m.config.hidden_size;
    // For N=1 decode reuse scratch->down as the per-layer output buffer (it's
    // the dense/MoE MLP result slot already pre-allocated at engine init).
    // moe_layer_forward writes the full output (routed sum + optional shared
    // expert) into this buffer; the caller's ops::add reads it immediately
    // before the next layer's call reuses the same slot.
    tensor_t out = (scratch != nullptr && N == 1)
                     ? scratch->down
                     : Tensor::create({N, hidden}, exec.data_type, exec.device_type, exec.device_id);
    moe_layer_forward(m, out, h_post, static_cast<int>(L), exec, scratch);
    return out;
}

} // namespace zedinfer::model
