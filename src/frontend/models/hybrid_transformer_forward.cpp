#include "frontend/models/hybrid_transformer_forward.hpp"

#include "backend/device/runtime_api.hpp"
#include "backend/ops/attn_output_gate/attn_output_gate.hpp"
#include "backend/ops/mamba/causal_conv1d.hpp"
#include "backend/ops/mamba/gdn.hpp"
#include "backend/ops/mamba/qk_l2norm.hpp"
#include "backend/ops/mamba/ssu.hpp"
#include "backend/ops/mrope/mrope_3d.hpp"
#include "backend/ops/ops.hpp"
#include "utils/types.hpp"
#include "frontend/models/decode_scratch.hpp"
#include "frontend/models/moe_forward.hpp"
#include "frontend/models/paged_forward_context.hpp"
#include "zedinfer/activation.hpp"
#include "zedinfer/request.hpp"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace zedinfer::model {

// Forward declarations of the per-layer-kind helpers. Each is implemented in
// its own subsequent commit (T13 = dense_mlp, T14 = full_attn, T15 = linear_attn,
// T16 = moe_mlp). Until then they throw a clearly-tagged exception so the
// hybrid path fails fast with a specific error rather than silently producing
// garbage.

static tensor_t forward_linear_attn_layer(const HybridForwardConfig& m,
                                            tensor_t h_in, size_t L,
                                            InferenceRequest& req,
                                            const ExecutorConfig& exec);

static tensor_t forward_full_attn_layer(const HybridForwardConfig& m,
                                          PagedForwardContext& ctx,
                                          tensor_t h_in, tensor_t pos_ids_thw,
                                          size_t L,
                                          const ExecutorConfig& exec);

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

static tensor_t forward_dense_mlp(const HybridForwardConfig& m,
                                    tensor_t h_post, size_t L,
                                    const ExecutorConfig& exec);

static tensor_t forward_moe_mlp(const HybridForwardConfig& m,
                                  tensor_t h_post, size_t L,
                                  const ExecutorConfig& exec,
                                  DecodeScratch* scratch);

tensor_t hybrid_transformer_forward(const HybridForwardConfig& m,
                                     PagedForwardContext& ctx,
                                     InferenceRequest& req,
                                     const ExecutorConfig& exec,
                                     DecodeScratch* /*scratch*/,
                                     tensor_t input_embeds) {
    const auto& cfg = m.config;
    const size_t N = static_cast<size_t>(ctx.num_tokens());
    const size_t hidden_size = cfg.hidden_size;

    auto make = [&](std::vector<size_t> shape) {
        return Tensor::create(std::move(shape), exec.data_type, exec.device_type, exec.device_id);
    };

    // Prepare token ids + position ids. For the hybrid path the pos_ids carry
    // (t, h, w) — but the outer loop only consumes them via PagedForwardContext;
    // the per-layer kernels read pos_ids_thw from req when they need 3D mrope.
    tensor_t ids, pos_ids;
    ctx.prepare_inputs(ids, pos_ids, exec);

    // Layer-0 hidden state. Vision pipelines pass input_embeds (already
    // shaped [N, hidden_size] with vision-tower embeddings scattered over
    // <|image_pad|> positions); text-only path runs the embed_tokens lookup.
    tensor_t hidden;
    if (input_embeds) {
        hidden = input_embeds;
    } else {
        hidden = make({N, hidden_size});
        ops::embedding(hidden, ids, m.W("embed_tokens.weight"));
    }

    // Build 3D positional ids once per forward. Vision pipelines will pass
    // req.pos_ids_thw() instead; text-only sequences get the (idx, idx, idx)
    // broadcast from build_pos_ids_thw_text_only.
    tensor_t pos_ids_thw = req.pos_ids_thw();
    if (!pos_ids_thw) {
        pos_ids_thw = build_pos_ids_thw_text_only(ctx, exec);
    }

    for (size_t L = 0; L < cfg.num_hidden_layers; ++L) {
        const auto p = m.prefix(L);

        // Pre-attention norm
        auto h_in = make({N, hidden_size});
        ops::rms_norm(h_in, hidden, m.W(p + "input_layernorm.weight"), cfg.rms_norm_eps);

        // Dispatch attention by layer kind
        tensor_t attn_out = m.is_linear_attn_layer(L)
                              ? forward_linear_attn_layer(m, h_in, L, req, exec)
                              : forward_full_attn_layer(m, ctx, h_in, pos_ids_thw, L, exec);

        // Residual after attention
        auto h1 = make({N, hidden_size});
        ops::add(h1, hidden, attn_out);

        // Post-attention norm
        auto h_post = make({N, hidden_size});
        ops::rms_norm(h_post, h1, m.W(p + "post_attention_layernorm.weight"), cfg.rms_norm_eps);

        // Dispatch MLP by sparsity
        tensor_t mlp_out = m.is_moe_layer(L)
                              ? forward_moe_mlp(m, h_post, L, exec, nullptr)
                              : forward_dense_mlp(m, h_post, L, exec);

        // Residual after MLP — overwrite hidden for the next layer
        hidden = make({N, hidden_size});
        ops::add(hidden, h1, mlp_out);
    }

    // Final norm + lm_head projection
    auto final_normed = make({N, hidden_size});
    ops::rms_norm(final_normed, hidden, m.W("norm.weight"), cfg.rms_norm_eps);

    auto logits = make({N, cfg.vocab_size});
    m.dispatch_linear(logits, final_normed, "lm_head", nullptr);

    ctx.finalize();
    return logits;
}

// ============================================================================
// Per-layer-kind stubs. Replaced in T13-T16. Each throws a tagged exception
// so the calling site sees exactly which sub-function is missing.
// ============================================================================

static tensor_t forward_linear_attn_layer(const HybridForwardConfig& m, tensor_t h_in,
                                            size_t L, InferenceRequest& req,
                                            const ExecutorConfig& exec) {
    if (req.ssm_slot_idx() < 0) {
        throw std::runtime_error("hybrid: forward_linear_attn_layer L=" + std::to_string(L)
                                 + " — request has no SSM slot (scheduler did not call acquire_slot)");
    }
    if (!m.ssm_pool) {
        throw std::runtime_error("hybrid: forward_linear_attn_layer L=" + std::to_string(L)
                                 + " — HybridForwardConfig::ssm_pool is null");
    }

    const size_t N    = static_cast<size_t>(h_in->shape()[0]);
    const int    Hv   = m.linear_attn.num_v_heads;
    const int    Dv   = m.linear_attn.value_head_dim;
    const int    Hk   = m.linear_attn.num_k_heads;
    const int    Dk   = m.linear_attn.key_head_dim;
    const size_t Hk_Dk = static_cast<size_t>(Hk) * static_cast<size_t>(Dk);
    const size_t Hv_Dv = static_cast<size_t>(Hv) * static_cast<size_t>(Dv);
    const size_t qkv_dim = 2 * Hk_Dk + Hv_Dv;

    const auto p = m.prefix(L) + "linear_attn.";

    auto make = [&](std::vector<size_t> shape) {
        return Tensor::create(std::move(shape), exec.data_type, exec.device_type, exec.device_id);
    };

    // 1. Four input projections (all BF16; linear_attn is excluded from quantization
    //    per Qwen3.5 dynamic rules).
    auto qkv = make({N, qkv_dim});
    auto z   = make({N, Hv_Dv});
    auto a   = make({N, static_cast<size_t>(Hv)});
    auto b   = make({N, static_cast<size_t>(Hv)});
    ops::linear(qkv, h_in, m.W(p + "in_proj_qkv.weight"));
    ops::linear(z,   h_in, m.W(p + "in_proj_z.weight"));
    ops::linear(a,   h_in, m.W(p + "in_proj_a.weight"));
    ops::linear(b,   h_in, m.W(p + "in_proj_b.weight"));

    // 2. Depthwise causal conv1d (kernel=4) with conv state from SSMStatePool.
    //    Kernel folds silu into its output, so qkv_conv = silu(conv1d(qkv)).
    auto qkv_conv = make({N, qkv_dim});
    ops::mamba::causal_conv1d(qkv_conv, qkv, m.W(p + "conv1d.weight"),
                                m.ssm_pool->view(), req.ssm_slot_idx(),
                                m.linear_layer_index(L));

    // 3. Split qkv_conv [N, qkv_dim] into contiguous q [N, Hk_Dk], k [N, Hk_Dk],
    //    v [N, Hv_Dv] for the SSU kernel. Strided copy lives in ops-nvidia.
    auto q_ssm = make({N, Hk_Dk});
    auto k_ssm = make({N, Hk_Dk});
    auto v_ssm = make({N, Hv_Dv});
    const size_t elt = utils::dsize(exec.data_type);
    ops::mamba::copy_strided_rows(q_ssm, qkv_conv, 0,         Hk_Dk, qkv_dim, N, elt);
    ops::mamba::copy_strided_rows(k_ssm, qkv_conv, Hk_Dk,     Hk_Dk, qkv_dim, N, elt);
    ops::mamba::copy_strided_rows(v_ssm, qkv_conv, 2 * Hk_Dk, Hv_Dv, qkv_dim, N, elt);

    // 3b. L2-normalize q and k per (token, k-head) row; scale q by 1/sqrt(Dk).
    //     HF Qwen3_5MoeGatedDeltaNet.forward calls recurrent_gated_delta_rule
    //     with use_qk_l2norm_in_kernel=True (see modeling_qwen3_5_moe.py:528,539),
    //     which inside torch_recurrent_gated_delta_rule applies l2norm(q, k,
    //     eps=1e-6) and then scale = 1/sqrt(head_k_dim) to q. Our GDN kernel
    //     consumes pre-normalized inputs (matches the fixture contract).
    const float qk_scale_q = 1.0f / std::sqrt(static_cast<float>(Dk));
    ops::mamba::qk_l2norm_inplace(q_ssm, Hk, Dk, qk_scale_q, 1e-6f);
    ops::mamba::qk_l2norm_inplace(k_ssm, Hk, Dk, 1.0f,       1e-6f);

    // 4. GDN: in-place update of SSMStatePool slot's state buffer + write out y.
    //    Per P3 retrospective: Qwen3.5's linear-attn is GatedDeltaNet, not
    //    Mamba2 SSM. The kernel applies sigmoid(b), softplus(a+dt_bias), and
    //    the delta-rule recurrence S = decay * S + beta * outer(v - S k, k).
    //    The silu(z) output gate is applied separately below (Step 7), not
    //    inside the kernel.
    auto y = make({N, Hv_Dv});
    ops::mamba::GDNParams gp;
    gp.state_view = m.ssm_pool->view();
    gp.slot_idx   = req.ssm_slot_idx();
    gp.layer_idx  = m.linear_layer_index(L);
    gp.q = q_ssm; gp.k = k_ssm; gp.v = v_ssm;
    gp.b = b;     gp.a = a;
    gp.A_log   = m.W(p + "A_log");
    gp.dt_bias = m.W(p + "dt_bias");
    gp.out = y;
    gp.num_tokens = static_cast<int>(N);
    ops::mamba::gdn(gp);

    // 6. RMSNorm on per-V-head value_head_dim axis. Qwen3.5 stores the norm
    //    weight as fp32 (`mamba_ssm_dtype=float32`) on disk but we pre-cast it
    //    to bf16 once at model load (see fixup_qwen3_5_linear_attn_norm_weights
    //    in qwen3_5.cpp) so the rms_norm same-dtype check passes without any
    //    per-call D2H+H2D.
    tensor_t norm_w = m.W(p + "norm.weight");

    auto y_normed = make({N, Hv_Dv});
    ops::rms_norm(y_normed->view({N * static_cast<size_t>(Hv), static_cast<size_t>(Dv)}),
                  y->view({N * static_cast<size_t>(Hv), static_cast<size_t>(Dv)}),
                  norm_w, m.config.rms_norm_eps);

    // 7. GDN output gating: y_normed = silu(z) * y_normed.
    //    Performed externally because the kernel only emits the delta-rule
    //    readout; Qwen3.5 applies silu(z) * o_norm before out_proj.
    {
        auto y_gated = make({N, Hv_Dv});
        ops::silu_mul(y_gated, z, y_normed);
        y_normed = y_gated;
    }

    // 8. Output projection [Hv_Dv → hidden_size].
    auto out = make({N, m.config.hidden_size});
    ops::linear(out, y_normed, m.W(p + "out_proj.weight"));
    return out;
}

static tensor_t forward_full_attn_layer(const HybridForwardConfig& m,
                                          PagedForwardContext& ctx,
                                          tensor_t h_in, tensor_t pos_ids_thw,
                                          size_t L,
                                          const ExecutorConfig& exec) {
    const size_t N = static_cast<size_t>(h_in->shape()[0]);
    const size_t Hq  = m.config.num_attention_heads;
    const size_t Hkv = m.config.num_key_value_heads;
    const size_t Dh  = m.config.head_dim > 0 ? m.config.head_dim : (m.config.hidden_size / Hq);
    const size_t q_dim  = Hq  * Dh;
    const size_t kv_dim = Hkv * Dh;
    const auto p = m.prefix(L) + "self_attn.";

    auto make = [&](std::vector<size_t> shape) {
        return Tensor::create(std::move(shape), exec.data_type, exec.device_type, exec.device_id);
    };

    // q_proj has doubled output [2*q_dim, hidden_size] — first q_dim rows are
    // the q projection, second q_dim rows are the output gate. Slice along
    // dim 0 (outer-most) is contiguous, so we can call linear directly twice
    // instead of one combined GEMM + materializing splits.
    auto q_proj_w = m.W(p + "q_proj.weight");           // [2*q_dim, hidden]
    auto w_q    = q_proj_w->slice(0, 0, q_dim);          // [q_dim, hidden]
    auto w_gate = q_proj_w->slice(0, q_dim, 2 * q_dim);  // [q_dim, hidden]

    auto q_raw = make({N, q_dim});
    auto gate  = make({N, q_dim});
    ops::linear(q_raw, h_in, w_q);
    ops::linear(gate,  h_in, w_gate);

    auto k_raw = make({N, kv_dim});
    auto v     = make({N, kv_dim});
    ops::linear(k_raw, h_in, m.W(p + "k_proj.weight"));
    ops::linear(v,     h_in, m.W(p + "v_proj.weight"));

    // Per-head RMSNorm on q / k (Qwen3 / Qwen3.5 family).
    auto q_normed = make({N, q_dim});
    auto k_normed = make({N, kv_dim});
    ops::rms_norm(q_normed->view({N * Hq,  Dh}),
                  q_raw->view({N * Hq,  Dh}),
                  m.W(p + "q_norm.weight"), m.config.rms_norm_eps);
    ops::rms_norm(k_normed->view({N * Hkv, Dh}),
                  k_raw->view({N * Hkv, Dh}),
                  m.W(p + "k_norm.weight"), m.config.rms_norm_eps);

    // 3D MRoPE with partial_rotary_factor (Qwen3.5 = 0.25 → first 64 of 256
    // head_dim rotated, rest pass-through). Rotation is in-place on the
    // per-head view, so reshape to [N, H, Dh] first.
    auto q_view = q_normed->view({N, Hq,  Dh});
    auto k_view = k_normed->view({N, Hkv, Dh});
    ops::mrope_3d(q_view, pos_ids_thw, m.mrope);
    ops::mrope_3d(k_view, pos_ids_thw, m.mrope);

    // Paged KV cache uses LOGICAL kv layer index: only full-attention layers
    // contribute to the pool (linear-attention layers hold SSM state instead).
    const int kv_idx = m.full_layer_index(L);
    ctx.write_kv(kv_idx, k_normed, v);

    // Paged attention. attend allocates and returns the attn output tensor.
    const float scale = 1.0f / std::sqrt(static_cast<float>(Dh));
    auto attn = ctx.attend(kv_idx, q_normed, scale, exec, Hq, Hkv, Dh, nullptr);

    // Attention output gate: attn := attn * sigmoid(gate), in place.
    // Both tensors are shape [N, Hq, Dh] — flatten matches because attn comes
    // back from ctx.attend already shaped for the o_proj input.
    ops::attn_output_gate(attn, gate);

    // o_proj: [q_dim → hidden]. Input width is q_dim (NOT doubled — the gate
    // was consumed in attn_output_gate). attn from ctx.attend already has
    // the right layout for o_proj input.
    auto out = make({N, m.config.hidden_size});
    ops::linear(out, attn, m.W(p + "o_proj.weight"));
    return out;
}

static tensor_t forward_dense_mlp(const HybridForwardConfig& m, tensor_t h_post, size_t L,
                                    const ExecutorConfig& exec) {
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

static tensor_t forward_moe_mlp(const HybridForwardConfig& m, tensor_t h_post, size_t L,
                                  const ExecutorConfig& exec, DecodeScratch* scratch) {
    // 35B-A3B path: reuse v0.2.0 moe_layer_forward verbatim. HybridForwardConfig
    // inherits ModelForwardConfig, so all the fields moe_layer_forward needs
    // (num_experts, num_experts_per_tok, expert_pool, router_weights,
    // expert_quant_*, has_shared_expert) are populated by Qwen3_5MoeModel's
    // hybrid_forward_config(); no MoE-specific code lives here.
    const size_t N = static_cast<size_t>(h_post->shape()[0]);
    const size_t hidden = m.config.hidden_size;
    auto out = Tensor::create({N, hidden}, exec.data_type, exec.device_type, exec.device_id);
    moe_layer_forward(m, out, h_post, static_cast<int>(L), exec, scratch);
    return out;
}

} // namespace zedinfer::model
