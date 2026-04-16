#include "frontend/models/decode_scratch.hpp"
#include "frontend/models/forward_config.hpp"
#include "frontend/models/qwen3_moe.hpp"

namespace zedinfer::model {

std::unique_ptr<DecodeScratch> DecodeScratch::create(const ModelConfig& cfg, bool has_qk_norm,
                                                     const ExecutorConfig& exec_config,
                                                     const ModelForwardConfig* fwd_cfg) {
    auto s = std::make_unique<DecodeScratch>();
    auto dt = exec_config.data_type;
    auto dev = exec_config.device_type;
    auto did = exec_config.device_id;

    size_t H = cfg.hidden_size;
    size_t nhead = cfg.num_attention_heads;
    size_t nkvhead = cfg.num_key_value_heads;
    // Use explicit head_dim from config if available (Qwen3-MoE: head_dim != H/nhead)
    size_t head_dim = cfg.head_dim > 0 ? cfg.head_dim : H / nhead;
    size_t q_dim = nhead * head_dim;
    size_t kv_dim = nkvhead * head_dim;
    size_t inter = cfg.intermediate_size;
    size_t V = cfg.vocab_size;

    auto mk = [&](std::vector<size_t> shape, zedinferDataType_t t) { return Tensor::create(shape, t, dev, did); };
    auto mkf = [&](std::vector<size_t> shape) { return mk(shape, dt); };

    // Inputs
    s->ids = mk({1}, ZEDINFER_DTYPE_I32);
    s->pos_ids = mk({1}, ZEDINFER_DTYPE_I64);

    // Hidden state ping-pong
    s->hidden = mkf({1, H});
    s->hidden_out = mkf({1, H});

    // Per-layer
    s->normed = mkf({1, H});
    s->q = mkf({1, q_dim});
    s->k = mkf({1, kv_dim});
    s->v = mkf({1, kv_dim});
    if (has_qk_norm) {
        s->q_normed = mkf({nhead, head_dim});
        s->k_normed = mkf({nkvhead, head_dim});
    }
    s->q_rope = mkf({1, nhead, head_dim});
    s->k_rope = mkf({1, nkvhead, head_dim});
    s->attn_out = mkf({1, nhead, head_dim});
    s->o = mkf({1, H});
    s->h1 = mkf({1, H});
    s->normed_post = mkf({1, H});

    // MoE scratch buffers (only for MoE models, dense MLP buffers still used for non-MoE)
    const auto* moe_cfg = dynamic_cast<const Qwen3MoEConfig*>(&cfg);
    if (moe_cfg && fwd_cfg && fwd_cfg->is_moe) {
        // Use sizes from forward_config (detected from actual weight shapes)
        size_t moe_inter = fwd_cfg->moe_intermediate_size;
        size_t shared_inter = fwd_cfg->shared_expert_intermediate_size;

        // Router logits must be in model dtype for ops::linear compatibility.
        // Conversion to F32 for top-k is done in moe_layer_forward.
        s->router_logits = mkf({1, moe_cfg->num_experts});
        s->moe_output = mkf({1, H});
        s->expert_gate = mkf({1, moe_inter});
        s->expert_up = mkf({1, moe_inter});
        s->expert_act = mkf({1, moe_inter});
        s->expert_down = mkf({1, H});
        s->shared_gate = mkf({1, shared_inter});
        s->shared_up = mkf({1, shared_inter});
        s->shared_act = mkf({1, shared_inter});
        s->shared_down = mkf({1, H});

        // For MoE, 'down' is used as the final MoE output in transformer_forward
        s->down = mkf({1, H});
    } else {
        s->gate = mkf({1, inter});
        s->up = mkf({1, inter});
        s->act = mkf({1, inter});
        s->down = mkf({1, H});
    }

    // Output head
    s->final_normed = mkf({1, H});
    s->logits = mkf({1, V});

    return s;
}

} // namespace zedinfer::model
