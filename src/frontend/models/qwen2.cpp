#include "frontend/models/qwen2.hpp"
#include "backend/kvcache/base.hpp"
#include "backend/ops/ops.hpp"
#include "zedinfer/activation.hpp"

#include <cmath>
#include <string>

namespace zedinfer::model {

std::string Qwen2Model::get_embedding_weight_name() const {
    return "embed_tokens.weight";
}

std::string Qwen2Model::get_output_norm_weight_name() const {
    return "norm.weight";
}

std::string Qwen2Model::get_output_weight_name() const {
    return "lm_head.weight";
}

std::vector<std::string> Qwen2Model::get_layer_weight_names(int layer_idx) const {
    std::string prefix = "layers." + std::to_string(layer_idx) + ".";

    return {
        prefix + "self_attn.q_proj.weight",
        prefix + "self_attn.q_proj.bias",
        prefix + "self_attn.k_proj.weight",
        prefix + "self_attn.k_proj.bias",
        prefix + "self_attn.v_proj.weight",
        prefix + "self_attn.v_proj.bias",
        prefix + "self_attn.o_proj.weight",
        prefix + "mlp.gate_proj.weight",
        prefix + "mlp.up_proj.weight",
        prefix + "mlp.down_proj.weight",
        prefix + "input_layernorm.weight",
        prefix + "post_attention_layernorm.weight"};
}

size_t Qwen2Model::calculate_num_parameters() const {
    size_t total = 0;
    for (const auto &[name, tensor] : weights_->get_all_weights()) {
        total += tensor->numel();
    }
    return total;
}

tensor_t Qwen2Model::forward(
    const std::vector<int> &input_ids,
    int past_len,
    kvcache::KVCache &kvcache,
    const ExecutorConfig &exec_config) {

    const auto &cfg = config_;
    const size_t sl = input_ids.size();
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
    auto make_typed = [&](std::vector<size_t> shape, zedinferDataType_t dtype) {
        return Tensor::create(shape, dtype,
                              exec_config.device_type, exec_config.device_id);
    };
    auto W = [&](const std::string &name) { return weights_->get_tensor(name); };

    // Prepare inputs
    auto ids = make_typed({sl}, ZEDINFER_DTYPE_I32);
    ids->load(input_ids.data());

    std::vector<int64_t> pos(sl);
    for (size_t i = 0; i < sl; ++i) pos[i] = past_len + static_cast<int64_t>(i);
    auto pos_ids = make_typed({sl}, ZEDINFER_DTYPE_I64);
    pos_ids->load(pos.data());

    // Embedding
    auto hidden = make({sl, hidden_size});
    ops::embedding(hidden, ids, W("embed_tokens.weight"));

    // Transformer layers
    for (size_t L = 0; L < cfg.num_hidden_layers; ++L) {
        std::string p = "layers." + std::to_string(L) + ".";

        // Attention block
        auto normed = make({sl, hidden_size});
        ops::rms_norm(normed, hidden, W(p + "input_layernorm.weight"), cfg.rms_norm_eps);

        auto q = make({sl, hidden_size});
        ops::linear(q, normed, W(p + "self_attn.q_proj.weight"), W(p + "self_attn.q_proj.bias"));

        auto k_tmp = make({sl, kv_dim});
        ops::linear(k_tmp, normed, W(p + "self_attn.k_proj.weight"), W(p + "self_attn.k_proj.bias"));

        auto v_slot = kvcache.get_v_cache_slice(L, past_len, sl);
        ops::linear(v_slot->view({sl, kv_dim}), normed, W(p + "self_attn.v_proj.weight"), W(p + "self_attn.v_proj.bias"));

        auto q_rope = make({sl, nhead, head_dim});
        ops::rope(q_rope, q->view({sl, nhead, head_dim}), pos_ids, cfg.rope_theta);

        auto k_slot = kvcache.get_k_cache_slice(L, past_len, sl);
        ops::rope(k_slot->view({sl, nkvhead, head_dim}), k_tmp->view({sl, nkvhead, head_dim}), pos_ids, cfg.rope_theta);

        auto attn = make({sl, nhead, head_dim});
        if (sl == 1 && kvcache.is_paged()) {
            // Decode writes K/V directly into block memory (zero-copy).
            // Paged attention reads from blocks — no scatter needed.
            ops::paged_attention_decode(
                attn->view({nhead, head_dim}), q_rope->view({nhead, head_dim}),
                kvcache.k_pool_data(),
                kvcache.k_block_ids(L), kvcache.v_block_ids(L),
                past_len + 1, // seq_len in KV cache (past + current token)
                scale, exec_config.data_type, exec_config.device_type, exec_config.device_id,
                nhead, nkvhead, head_dim, kvcache.block_size());
        } else if (kvcache.is_paged()) {
            // Paged prefill: scatter this layer's write buffer to blocks,
            // then read K/V directly from blocks (no gather copy).
            kvcache.scatter_layer_to_blocks(L);
            ops::paged_attention_prefill(
                attn, q_rope,
                kvcache.k_pool_data(),
                kvcache.k_block_ids(L), kvcache.v_block_ids(L),
                sl, past_len,
                scale, exec_config.data_type, exec_config.device_type, exec_config.device_id,
                nhead, nkvhead, head_dim, kvcache.block_size());
        } else {
            // Non-paged: contiguous KV + standard attention
            ops::self_attention(attn, q_rope, kvcache.get_k_cache_slice(L, past_len + sl), kvcache.get_v_cache_slice(L, past_len + sl), scale);
        }

        auto o = make({sl, hidden_size});
        ops::linear(o, attn->view({sl, hidden_size}), W(p + "self_attn.o_proj.weight"), nullptr);

        auto h1 = make({sl, hidden_size});
        ops::add(h1, hidden, o);

        // MLP block
        normed = make({sl, hidden_size});
        ops::rms_norm(normed, h1, W(p + "post_attention_layernorm.weight"), cfg.rms_norm_eps);

        auto gate = make({sl, inter});
        auto up = make({sl, inter});
        ops::linear(gate, normed, W(p + "mlp.gate_proj.weight"), nullptr);
        ops::linear(up, normed, W(p + "mlp.up_proj.weight"), nullptr);

        auto act = make({sl, inter});
        ops::swiglu(act, gate, up);

        auto down = make({sl, hidden_size});
        ops::linear(down, act, W(p + "mlp.down_proj.weight"), nullptr);

        hidden = make({sl, hidden_size});
        ops::add(hidden, h1, down);
    }

    // Output head
    auto normed = make({sl, hidden_size});
    ops::rms_norm(normed, hidden, W("norm.weight"), cfg.rms_norm_eps);

    auto logits = make({sl, cfg.vocab_size});
    ops::linear(logits, normed, W("lm_head.weight"), nullptr);

    kvcache.update_seq_len(sl);
    return logits;
}

} // namespace zedinfer::model
