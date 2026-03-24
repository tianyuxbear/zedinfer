#include "frontend/models/forward_config.hpp"
#include "frontend/models/forward_context.hpp"
#include "frontend/models/decode_scratch.hpp"
#include "backend/ops/ops.hpp"

#include <cmath>
#include <utility>

namespace zedinfer::model {

/**
 * Shared transformer forward loop.
 * When scratch != nullptr and N == 1, uses pre-allocated decode buffers
 * (zero Tensor::create per step). For prefill (N > 1), falls back to
 * dynamic allocation via Tensor::create.
 */
tensor_t transformer_forward(
    const ModelForwardConfig &model,
    ForwardContext &ctx,
    const ExecutorConfig &exec_config,
    DecodeScratch *scratch) {

    const auto &cfg = model.config;
    const size_t N = static_cast<size_t>(ctx.num_tokens());
    const size_t hidden_size = cfg.hidden_size;
    const size_t nhead = cfg.num_attention_heads;
    const size_t nkvhead = cfg.num_key_value_heads;
    const size_t head_dim = hidden_size / nhead;
    const size_t kv_dim = nkvhead * head_dim;
    const size_t inter = cfg.intermediate_size;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    const bool use_scratch = (scratch != nullptr && N == 1);

    auto make = [&](std::vector<size_t> shape) {
        return Tensor::create(shape, exec_config.data_type,
                              exec_config.device_type, exec_config.device_id);
    };

    // Prepare inputs
    tensor_t ids, pos_ids;
    if (use_scratch) {
        ids = scratch->ids;
        pos_ids = scratch->pos_ids;
        ctx.prepare_inputs_into(ids, pos_ids);
    } else {
        ctx.prepare_inputs(ids, pos_ids, exec_config);
    }

    // Embedding
    auto hidden = use_scratch ? scratch->hidden : make({N, hidden_size});
    ops::embedding(hidden, ids, model.W("embed_tokens.weight"));

    // Transformer layers
    for (size_t L = 0; L < cfg.num_hidden_layers; ++L) {
        auto p = model.prefix(L);

        // Input norm
        auto normed = use_scratch ? scratch->normed : make({N, hidden_size});
        ops::rms_norm(normed, hidden, model.W(p + "input_layernorm.weight"), cfg.rms_norm_eps);

        // Q/K/V projections
        auto q = use_scratch ? scratch->q : make({N, hidden_size});
        ops::linear(q, normed, model.W(p + "self_attn.q_proj.weight"), model.q_bias(p));

        auto k = use_scratch ? scratch->k : make({N, kv_dim});
        ops::linear(k, normed, model.W(p + "self_attn.k_proj.weight"), model.k_bias(p));

        auto v = use_scratch ? scratch->v : make({N, kv_dim});
        ops::linear(v, normed, model.W(p + "self_attn.v_proj.weight"), model.v_bias(p));

        // Optional per-head Q/K norm (Qwen3)
        tensor_t q_for_rope, k_for_rope;
        if (model.has_qk_norm) {
            auto q_normed = use_scratch ? scratch->q_normed : make({N * nhead, head_dim});
            ops::rms_norm(q_normed, q->view({N * nhead, head_dim}),
                          model.W(p + "self_attn.q_norm.weight"), cfg.rms_norm_eps);

            auto k_normed = use_scratch ? scratch->k_normed : make({N * nkvhead, head_dim});
            ops::rms_norm(k_normed, k->view({N * nkvhead, head_dim}),
                          model.W(p + "self_attn.k_norm.weight"), cfg.rms_norm_eps);

            q_for_rope = q_normed->view({N, nhead, head_dim});
            k_for_rope = k_normed->view({N, nkvhead, head_dim});
        } else {
            q_for_rope = q->view({N, nhead, head_dim});
            k_for_rope = k->view({N, nkvhead, head_dim});
        }

        // RoPE
        auto q_rope = use_scratch ? scratch->q_rope : make({N, nhead, head_dim});
        ops::rope(q_rope, q_for_rope, pos_ids, cfg.rope_theta);

        auto k_rope = use_scratch ? scratch->k_rope : make({N, nkvhead, head_dim});
        ops::rope(k_rope, k_for_rope, pos_ids, cfg.rope_theta);

        // KV write (context-specific)
        ctx.write_kv(L, k_rope, v);

        // Attention (context-specific)
        tensor_t attn_pre = use_scratch ? scratch->attn_out : nullptr;
        auto attn = ctx.attend(L, q_rope, scale, exec_config, nhead, nkvhead, head_dim, attn_pre);

        // O projection + residual
        auto o = use_scratch ? scratch->o : make({N, hidden_size});
        ops::linear(o, attn->view({N, hidden_size}),
                    model.W(p + "self_attn.o_proj.weight"), nullptr);

        auto h1 = use_scratch ? scratch->h1 : make({N, hidden_size});
        ops::add(h1, hidden, o);

        // MLP: norm -> gate/up -> swiglu -> down -> residual
        auto normed_post = use_scratch ? scratch->normed_post : make({N, hidden_size});
        ops::rms_norm(normed_post, h1, model.W(p + "post_attention_layernorm.weight"), cfg.rms_norm_eps);

        auto gate = use_scratch ? scratch->gate : make({N, inter});
        auto up = use_scratch ? scratch->up : make({N, inter});
        ops::linear(gate, normed_post, model.W(p + "mlp.gate_proj.weight"), nullptr);
        ops::linear(up, normed_post, model.W(p + "mlp.up_proj.weight"), nullptr);

        auto act = use_scratch ? scratch->act : make({N, inter});
        ops::swiglu(act, gate, up);

        auto down = use_scratch ? scratch->down : make({N, hidden_size});
        ops::linear(down, act, model.W(p + "mlp.down_proj.weight"), nullptr);

        if (use_scratch) {
            // Ping-pong: write result to hidden_out, then swap for next layer
            ops::add(scratch->hidden_out, h1, down);
            std::swap(scratch->hidden, scratch->hidden_out);
            hidden = scratch->hidden;
        } else {
            hidden = make({N, hidden_size});
            ops::add(hidden, h1, down);
        }
    }

    // Output head
    auto final_normed = use_scratch ? scratch->final_normed : make({N, hidden_size});
    ops::rms_norm(final_normed, hidden, model.W("norm.weight"), cfg.rms_norm_eps);

    auto logits = use_scratch ? scratch->logits : make({N, cfg.vocab_size});
    ops::linear(logits, final_normed, model.W("lm_head.weight"), nullptr);

    ctx.finalize();
    return logits;
}

} // namespace zedinfer::model
