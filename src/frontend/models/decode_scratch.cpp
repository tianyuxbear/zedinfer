#include "frontend/models/decode_scratch.hpp"

namespace zedinfer::model {

std::unique_ptr<DecodeScratch> DecodeScratch::create(
    const ModelConfig &cfg,
    bool has_qk_norm,
    const ExecutorConfig &exec_config) {

    auto s = std::make_unique<DecodeScratch>();
    auto dt = exec_config.data_type;
    auto dev = exec_config.device_type;
    auto did = exec_config.device_id;

    size_t H = cfg.hidden_size;
    size_t nhead = cfg.num_attention_heads;
    size_t nkvhead = cfg.num_key_value_heads;
    size_t head_dim = H / nhead;
    size_t kv_dim = nkvhead * head_dim;
    size_t inter = cfg.intermediate_size;
    size_t V = cfg.vocab_size;

    auto mk = [&](std::vector<size_t> shape, zedinferDataType_t t) {
        return Tensor::create(shape, t, dev, did);
    };
    auto mkf = [&](std::vector<size_t> shape) { return mk(shape, dt); };

    // Inputs
    s->ids = mk({1}, ZEDINFER_DTYPE_I32);
    s->pos_ids = mk({1}, ZEDINFER_DTYPE_I64);

    // Hidden state ping-pong
    s->hidden = mkf({1, H});
    s->hidden_out = mkf({1, H});

    // Per-layer
    s->normed = mkf({1, H});
    s->q = mkf({1, H});
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
    s->gate = mkf({1, inter});
    s->up = mkf({1, inter});
    s->act = mkf({1, inter});
    s->down = mkf({1, H});

    // Output head
    s->final_normed = mkf({1, H});
    s->logits = mkf({1, V});

    return s;
}

} // namespace zedinfer::model
