#include "backend/ops/ops.hpp"
#include "frontend/models/decode_scratch.hpp"
#include "frontend/models/forward_config.hpp"
#include "frontend/models/moe_forward.hpp"
#include "frontend/models/paged_forward_context.hpp"

#include <cmath>
#include <utility>

namespace zedinfer::model {

/**
 * Shared transformer forward loop.
 * When scratch != nullptr and N == 1, uses pre-allocated decode buffers
 * (zero Tensor::create per step). For prefill (N > 1), falls back to
 * dynamic allocation via Tensor::create.
 */
tensor_t transformer_forward(const ModelForwardConfig& model, PagedForwardContext& ctx,
                             const ExecutorConfig& exec_config, DecodeScratch* scratch,
                             tensor_t input_embeds) {
    const auto& cfg = model.config;
    const size_t N = static_cast<size_t>(ctx.num_tokens());
    const size_t hidden_size = cfg.hidden_size;
    const size_t nhead = cfg.num_attention_heads;
    const size_t nkvhead = cfg.num_key_value_heads;
    // Use head_dim from config (may differ from hidden_size/nhead, e.g. Qwen3-30B-A3B has
    // head_dim=128 with hidden_size=2048 and 32 heads, so q_dim = 32*128 = 4096 != hidden_size)
    const size_t head_dim = cfg.head_dim > 0 ? cfg.head_dim : hidden_size / nhead;
    const size_t q_dim = nhead * head_dim;
    const size_t kv_dim = nkvhead * head_dim;
    const size_t inter = cfg.intermediate_size;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    const bool use_scratch = (scratch != nullptr && N == 1);

    auto make = [&](std::vector<size_t> shape) {
        return Tensor::create(shape, exec_config.data_type, exec_config.device_type, exec_config.device_id);
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

    // Embedding (or external embeds for multimodal vision pipeline).
    // When input_embeds is provided the caller has already shaped it as
    // [N, hidden_size] with vision-tower output scattered into the right
    // token positions; we adopt it as the layer-0 hidden state directly.
    // Vision pipelines always run prefill (N > 1), so scratch is unused here.
    tensor_t hidden;
    if (input_embeds) {
        hidden = input_embeds;
    } else {
        hidden = use_scratch ? scratch->hidden : make({N, hidden_size});
        ops::embedding(hidden, ids, model.W("embed_tokens.weight"));
    }

    // Transformer layers
    for (size_t L = 0; L < cfg.num_hidden_layers; ++L) {
        auto p = model.prefix(L);

        // Input norm
        auto normed = use_scratch ? scratch->normed : make({N, hidden_size});
        ops::rms_norm(normed, hidden, model.W(p + "input_layernorm.weight"), cfg.rms_norm_eps);

        // Q/K/V projections (q_dim may differ from hidden_size when head_dim != hidden_size/nhead)
        auto q = use_scratch ? scratch->q : make({N, q_dim});
        model.dispatch_linear(q, normed, p + "self_attn.q_proj", model.q_bias(p));

        auto k = use_scratch ? scratch->k : make({N, kv_dim});
        model.dispatch_linear(k, normed, p + "self_attn.k_proj", model.k_bias(p));

        auto v = use_scratch ? scratch->v : make({N, kv_dim});
        model.dispatch_linear(v, normed, p + "self_attn.v_proj", model.v_bias(p));

        // Optional per-head Q/K norm (Qwen3)
        tensor_t q_for_rope, k_for_rope;
        if (model.has_qk_norm) {
            auto q_normed = use_scratch ? scratch->q_normed : make({N * nhead, head_dim});
            ops::rms_norm(q_normed, q->view({N * nhead, head_dim}), model.W(p + "self_attn.q_norm.weight"),
                          cfg.rms_norm_eps);

            auto k_normed = use_scratch ? scratch->k_normed : make({N * nkvhead, head_dim});
            ops::rms_norm(k_normed, k->view({N * nkvhead, head_dim}), model.W(p + "self_attn.k_norm.weight"),
                          cfg.rms_norm_eps);

            q_for_rope = q_normed->view({N, nhead, head_dim});
            k_for_rope = k_normed->view({N, nkvhead, head_dim});
        } else {
            q_for_rope = q->view({N, nhead, head_dim});
            k_for_rope = k->view({N, nkvhead, head_dim});
        }

        // RoPE
        auto q_rope = use_scratch ? scratch->q_rope : make({N, nhead, head_dim});
        auto k_rope = use_scratch ? scratch->k_rope : make({N, nkvhead, head_dim});
        ops::rope_qk(q_rope, k_rope, q_for_rope, k_for_rope, pos_ids, cfg.rope_theta);

        // KV write (context-specific)
        ctx.write_kv(L, k_rope, v);

        // Attention (context-specific)
        tensor_t attn_pre = use_scratch ? scratch->attn_out : nullptr;
        auto attn = ctx.attend(L, q_rope, scale, exec_config, nhead, nkvhead, head_dim, attn_pre);

        // O projection: input is [N, q_dim], output is [N, hidden_size]
        auto o = use_scratch ? scratch->o : make({N, hidden_size});
        model.dispatch_linear(o, attn->view({N, q_dim}), p + "self_attn.o_proj", nullptr);

        auto h1 = use_scratch ? scratch->h1 : make({N, hidden_size});
        ops::add(h1, hidden, o);

        // MLP / MoE: norm -> [dense MLP or MoE dispatch] -> residual
        auto normed_post = use_scratch ? scratch->normed_post : make({N, hidden_size});
        ops::rms_norm(normed_post, h1, model.W(p + "post_attention_layernorm.weight"), cfg.rms_norm_eps);

        tensor_t down;
        if (model.is_moe_layer(L)) {
            // MoE layer: router + expert dispatch + shared expert
            down = use_scratch ? scratch->down : make({N, hidden_size});
            moe_layer_forward(model, down, normed_post, static_cast<int>(L), exec_config, scratch);
        } else {
            // Dense MLP: gate/up -> swiglu -> down
            auto gate = use_scratch ? scratch->gate : make({N, inter});
            auto up = use_scratch ? scratch->up : make({N, inter});
            model.dispatch_linear(gate, normed_post, p + "mlp.gate_proj", nullptr);
            model.dispatch_linear(up, normed_post, p + "mlp.up_proj", nullptr);

            auto act = use_scratch ? scratch->act : make({N, inter});
            ops::swiglu(act, gate, up);

            down = use_scratch ? scratch->down : make({N, hidden_size});
            model.dispatch_linear(down, act, p + "mlp.down_proj", nullptr);
        }

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
    model.dispatch_linear(logits, final_normed, "lm_head", nullptr);

    ctx.finalize();
    return logits;
}

} // namespace zedinfer::model
