#include "frontend/models/hybrid_transformer_forward.hpp"

#include "backend/ops/ops.hpp"
#include "frontend/models/decode_scratch.hpp"
#include "frontend/models/paged_forward_context.hpp"
#include "zedinfer/activation.hpp"
#include "zedinfer/request.hpp"

#include <cmath>
#include <stdexcept>
#include <string>

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
                                          tensor_t h_in, size_t L,
                                          const ExecutorConfig& exec);

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

    for (size_t L = 0; L < cfg.num_hidden_layers; ++L) {
        const auto p = m.prefix(L);

        // Pre-attention norm
        auto h_in = make({N, hidden_size});
        ops::rms_norm(h_in, hidden, m.W(p + "input_layernorm.weight"), cfg.rms_norm_eps);

        // Dispatch attention by layer kind
        tensor_t attn_out = m.is_linear_attn_layer(L)
                              ? forward_linear_attn_layer(m, h_in, L, req, exec)
                              : forward_full_attn_layer(m, ctx, h_in, L, exec);

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

static tensor_t forward_linear_attn_layer(const HybridForwardConfig& /*m*/, tensor_t /*h_in*/, size_t L,
                                            InferenceRequest& /*req*/, const ExecutorConfig& /*exec*/) {
    throw std::runtime_error("hybrid: forward_linear_attn_layer L=" + std::to_string(L)
                             + " not yet impl (P2-T15)");
}

static tensor_t forward_full_attn_layer(const HybridForwardConfig& /*m*/, PagedForwardContext& /*ctx*/,
                                          tensor_t /*h_in*/, size_t L, const ExecutorConfig& /*exec*/) {
    throw std::runtime_error("hybrid: forward_full_attn_layer L=" + std::to_string(L)
                             + " not yet impl (P2-T14)");
}

static tensor_t forward_dense_mlp(const HybridForwardConfig& /*m*/, tensor_t /*h_post*/, size_t L,
                                    const ExecutorConfig& /*exec*/) {
    throw std::runtime_error("hybrid: forward_dense_mlp L=" + std::to_string(L)
                             + " not yet impl (P2-T13)");
}

static tensor_t forward_moe_mlp(const HybridForwardConfig& /*m*/, tensor_t /*h_post*/, size_t L,
                                  const ExecutorConfig& /*exec*/, DecodeScratch* /*scratch*/) {
    throw std::runtime_error("hybrid: forward_moe_mlp L=" + std::to_string(L)
                             + " not yet impl (P2-T16)");
}

} // namespace zedinfer::model
