#include "frontend/models/forward_context.hpp"
#include "backend/core/context/context.hpp"
#include "backend/ops/ops.hpp"
#include "utils/types.hpp"

#include <cstring>

namespace zedinfer::model {

// ============================================================================
// ContiguousForwardContext — for warmup/profile with DynamicKVCache
// ============================================================================

ContiguousForwardContext::ContiguousForwardContext(
    const std::vector<int> &input_ids, int past_len,
    kvcache::KVCache &kvcache)
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
    size_t nkvhead = kvcache_.config().num_kv_heads;
    size_t head_dim_val = kvcache_.config().head_dim;
    size_t kv_dim = nkvhead * head_dim_val;
    size_t dtype_size = utils::dsize(kvcache_.config().dtype);

    // V: copy to contiguous KV cache slot
    auto v_slot = kvcache_.get_v_cache_slice(layer, past_len_, sl_);
    size_t v_bytes = sl_ * kv_dim * dtype_size;
    if (kvcache_.config().device_type == ZEDINFER_DEVICE_CPU) {
        std::memcpy(v_slot->data(), v->data(), v_bytes);
    } else {
        core::context().setDevice(kvcache_.config().device_type, kvcache_.config().device_id);
        core::context().runtime().api()->memcpy_sync(
            v_slot->data(), v->data(), v_bytes, ZEDINFER_MEMCPY_D2D);
    }

    // K: copy to contiguous KV cache slot
    auto k_slot = kvcache_.get_k_cache_slice(layer, past_len_, sl_);
    size_t k_bytes = sl_ * nkvhead * head_dim_val * dtype_size;
    if (kvcache_.config().device_type == ZEDINFER_DEVICE_CPU) {
        std::memcpy(k_slot->data(), k->data(), k_bytes);
    } else {
        core::context().setDevice(kvcache_.config().device_type, kvcache_.config().device_id);
        core::context().runtime().api()->memcpy_sync(
            k_slot->data(), k->data(), k_bytes, ZEDINFER_MEMCPY_D2D);
    }
}

tensor_t ContiguousForwardContext::attend(
    int layer, tensor_t q_rope, float scale,
    const ExecutorConfig &exec_config,
    size_t nhead, size_t /*nkvhead*/, size_t head_dim) {

    auto attn = Tensor::create({static_cast<size_t>(sl_), nhead, head_dim},
                                exec_config.data_type,
                                exec_config.device_type, exec_config.device_id);

    ops::self_attention(attn, q_rope,
        kvcache_.get_k_cache_slice(layer, past_len_ + sl_),
        kvcache_.get_v_cache_slice(layer, past_len_ + sl_), scale);

    return attn;
}

void ContiguousForwardContext::finalize() {
    kvcache_.update_seq_len(sl_);
}

} // namespace zedinfer::model
