#include "frontend/models/forward_context.hpp"
#include "backend/core/context/context.hpp"
#include "backend/ops/ops.hpp"
#include "utils/types.hpp"

#include <cstring>

namespace zedinfer::model {

ContiguousForwardContext::ContiguousForwardContext(
    const std::vector<int> &input_ids, int past_len,
    kvcache::DynamicKVCache &kvcache)
    : input_ids_(input_ids), past_len_(past_len),
      sl_(static_cast<int>(input_ids.size())), kvcache_(kvcache) {}

void ContiguousForwardContext::prepare_inputs(
    tensor_t &ids, tensor_t &pos_ids, const ExecutorConfig &exec_config) {

    ids = Tensor::create({static_cast<size_t>(sl_)}, ZEDINFER_DTYPE_I32,
                         exec_config.device_type, exec_config.device_id);
    ids->load(input_ids_.data());

    std::vector<int64_t> pos(sl_);
    for (int i = 0; i < sl_; ++i) pos[i] = past_len_ + static_cast<int64_t>(i);
    pos_ids = Tensor::create({static_cast<size_t>(sl_)}, ZEDINFER_DTYPE_I64,
                              exec_config.device_type, exec_config.device_id);
    pos_ids->load(pos.data());
}

void ContiguousForwardContext::write_kv(int layer, tensor_t k, tensor_t v) {
    auto &cfg = kvcache_.kv_config();
    size_t kv_dim = cfg.num_kv_heads * cfg.head_dim;
    size_t dtype_size = utils::dsize(cfg.dtype);

    auto v_slot = kvcache_.get_v_cache_slice(layer, past_len_, sl_);
    size_t v_bytes = sl_ * kv_dim * dtype_size;
    if (cfg.device_type == ZEDINFER_DEVICE_CPU) {
        std::memcpy(v_slot->data(), v->data(), v_bytes);
    } else {
        core::context().setDevice(cfg.device_type, cfg.device_id);
        core::context().runtime().api()->memcpy_sync(
            v_slot->data(), v->data(), v_bytes, ZEDINFER_MEMCPY_D2D);
    }

    auto k_slot = kvcache_.get_k_cache_slice(layer, past_len_, sl_);
    size_t k_bytes = sl_ * cfg.num_kv_heads * cfg.head_dim * dtype_size;
    if (cfg.device_type == ZEDINFER_DEVICE_CPU) {
        std::memcpy(k_slot->data(), k->data(), k_bytes);
    } else {
        core::context().setDevice(cfg.device_type, cfg.device_id);
        core::context().runtime().api()->memcpy_sync(
            k_slot->data(), k->data(), k_bytes, ZEDINFER_MEMCPY_D2D);
    }
}

tensor_t ContiguousForwardContext::attend(
    int layer, tensor_t q_rope, float scale,
    const ExecutorConfig &exec_config,
    size_t nhead, size_t nkvhead, size_t head_dim) {

    auto &cfg = kvcache_.kv_config();
    auto attn = Tensor::create({static_cast<size_t>(sl_), nhead, head_dim},
                                exec_config.data_type,
                                exec_config.device_type, exec_config.device_id);

    ops::AttentionConfig attn_cfg{
        static_cast<int>(nhead), static_cast<int>(nkvhead), static_cast<int>(head_dim),
        scale, 0, cfg.dtype, cfg.device_type, cfg.device_id};

    ops::AttentionParams params{attn_cfg};
    params.out = attn;
    params.q = q_rope;
    params.k_contiguous = kvcache_.get_k_cache_slice(layer, past_len_ + sl_);
    params.v_contiguous = kvcache_.get_v_cache_slice(layer, past_len_ + sl_);

    ops::attention(params);
    return attn;
}

void ContiguousForwardContext::finalize() {
    kvcache_.update_seq_len(sl_);
}

} // namespace zedinfer::model
