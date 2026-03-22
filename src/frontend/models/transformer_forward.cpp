#include "frontend/models/forward_config.hpp"
#include "frontend/models/forward_context.hpp"
#include "backend/ops/ops.hpp"

#include <cmath>

namespace zedinfer::model {

/**
 * Shared transformer forward loop.
 * Model differences are captured in ModelForwardConfig (bias, Q/K norm).
 * Execution mode differences are captured in ForwardContext (KV write, attention).
 */
tensor_t transformer_forward(
    const ModelForwardConfig &model,
    ForwardContext &ctx,
    const ExecutorConfig &exec_config) {

    const auto &cfg = model.config;
    const size_t N = static_cast<size_t>(ctx.num_tokens());
    const size_t hidden_size = cfg.hidden_size;
    const size_t nhead = cfg.num_attention_heads;
    const size_t nkvhead = cfg.num_key_value_heads;
    const size_t head_dim = hidden_size / nhead;
    const size_t kv_dim = nkvhead * head_dim;
    const size_t inter = cfg.intermediate_size;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    auto make = [&](std::vector<size_t> shape) {
        return Tensor::create(shape, exec_config.data_type,
                              exec_config.device_type, exec_config.device_id);
    };

    // Prepare inputs
    tensor_t ids, pos_ids;
    ctx.prepare_inputs(ids, pos_ids, exec_config);

    // Embedding
    auto hidden = make({N, hidden_size});
    ops::embedding(hidden, ids, model.W("embed_tokens.weight"));

    // Transformer layers
    for (size_t L = 0; L < cfg.num_hidden_layers; ++L) {
        auto p = model.prefix(L);

        // Input norm
        auto normed = make({N, hidden_size});
        ops::rms_norm(normed, hidden, model.W(p + "input_layernorm.weight"), cfg.rms_norm_eps);

        // Q/K/V projections
        auto q = make({N, hidden_size});
        ops::linear(q, normed, model.W(p + "self_attn.q_proj.weight"), model.q_bias(p));

        auto k = make({N, kv_dim});
        ops::linear(k, normed, model.W(p + "self_attn.k_proj.weight"), model.k_bias(p));

        auto v = make({N, kv_dim});
        ops::linear(v, normed, model.W(p + "self_attn.v_proj.weight"), model.v_bias(p));

        // Optional per-head Q/K norm (Qwen3)
        tensor_t q_for_rope, k_for_rope;
        if (model.has_qk_norm) {
            auto q_normed = make({N * nhead, head_dim});
            ops::rms_norm(q_normed, q->view({N * nhead, head_dim}),
                          model.W(p + "self_attn.q_norm.weight"), cfg.rms_norm_eps);

            auto k_normed = make({N * nkvhead, head_dim});
            ops::rms_norm(k_normed, k->view({N * nkvhead, head_dim}),
                          model.W(p + "self_attn.k_norm.weight"), cfg.rms_norm_eps);

            q_for_rope = q_normed->view({N, nhead, head_dim});
            k_for_rope = k_normed->view({N, nkvhead, head_dim});
        } else {
            q_for_rope = q->view({N, nhead, head_dim});
            k_for_rope = k->view({N, nkvhead, head_dim});
        }

        // RoPE
        auto q_rope = make({N, nhead, head_dim});
        ops::rope(q_rope, q_for_rope, pos_ids, cfg.rope_theta);

        auto k_rope = make({N, nkvhead, head_dim});
        ops::rope(k_rope, k_for_rope, pos_ids, cfg.rope_theta);

        // KV write (context-specific)
        ctx.write_kv(L, k_rope, v);

        // Attention (context-specific)
        auto attn = ctx.attend(L, q_rope, scale, exec_config, nhead, nkvhead, head_dim);

        // O projection + residual
        auto o = make({N, hidden_size});
        ops::linear(o, attn->view({N, hidden_size}),
                    model.W(p + "self_attn.o_proj.weight"), nullptr);

        auto h1 = make({N, hidden_size});
        ops::add(h1, hidden, o);

        // MLP: norm -> gate/up -> swiglu -> down -> residual
        normed = make({N, hidden_size});
        ops::rms_norm(normed, h1, model.W(p + "post_attention_layernorm.weight"), cfg.rms_norm_eps);

        auto gate = make({N, inter});
        auto up = make({N, inter});
        ops::linear(gate, normed, model.W(p + "mlp.gate_proj.weight"), nullptr);
        ops::linear(up, normed, model.W(p + "mlp.up_proj.weight"), nullptr);

        auto act = make({N, inter});
        ops::swiglu(act, gate, up);

        auto down = make({N, hidden_size});
        ops::linear(down, act, model.W(p + "mlp.down_proj.weight"), nullptr);

        hidden = make({N, hidden_size});
        ops::add(hidden, h1, down);
    }

    // Output head
    auto final_normed = make({N, hidden_size});
    ops::rms_norm(final_normed, hidden, model.W("norm.weight"), cfg.rms_norm_eps);

    auto logits = make({N, cfg.vocab_size});
    ops::linear(logits, final_normed, model.W("lm_head.weight"), nullptr);

    ctx.finalize();
    return logits;
}

} // namespace zedinfer::model
