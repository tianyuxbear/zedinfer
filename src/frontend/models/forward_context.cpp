#include "frontend/models/forward_context.hpp"
#include "backend/core/context/context.hpp"
#include "backend/ops/ops.hpp"
#include "utils/types.hpp"

#include <cstring>

namespace zedinfer::model {

// ============================================================================
// SingleForwardContext
// ============================================================================

SingleForwardContext::SingleForwardContext(
    const std::vector<int> &input_ids, int past_len,
    kvcache::KVCache &kvcache)
    : input_ids_(input_ids), past_len_(past_len),
      sl_(static_cast<int>(input_ids.size())), kvcache_(kvcache) {}

void SingleForwardContext::prepare_inputs(
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

void SingleForwardContext::write_kv(int layer, tensor_t k, tensor_t v) {
    size_t nkvhead = kvcache_.config().num_kv_heads;
    size_t head_dim_val = kvcache_.config().head_dim;
    size_t kv_dim = nkvhead * head_dim_val;

    // V: write via KVCache interface (direct-to-block for decode, write buffer for prefill)
    auto v_slot = kvcache_.get_v_cache_slice(layer, past_len_, sl_);
    // v is [sl, kv_dim] from linear output — need to copy to v_slot
    // Actually, v was already written by linear into v_slot in the old code.
    // In the new design, linear writes to a separate v tensor, and we copy here.
    // For efficiency, we need v_slot to be the linear output target.
    // BUT the shared loop does: ops::linear(v, normed, W_v, bias_v)
    // Then calls write_kv(layer, k_rope, v).
    // So v is a separate tensor. We need to copy v → v_slot.

    // Copy v to KV cache slot
    if (kvcache_.config().device_type == ZEDINFER_DEVICE_CPU) {
        std::memcpy(v_slot->data(), v->data(),
                    sl_ * kv_dim * utils::dsize(kvcache_.config().dtype));
    } else {
        core::context().setDevice(kvcache_.config().device_type, kvcache_.config().device_id);
        core::context().runtime().api()->memcpy_sync(
            v_slot->data(), v->data(),
            sl_ * kv_dim * utils::dsize(kvcache_.config().dtype),
            ZEDINFER_MEMCPY_D2D);
    }

    // K: write via KVCache interface (k is already after RoPE)
    auto k_slot = kvcache_.get_k_cache_slice(layer, past_len_, sl_);
    if (kvcache_.config().device_type == ZEDINFER_DEVICE_CPU) {
        std::memcpy(k_slot->data(), k->data(),
                    sl_ * nkvhead * head_dim_val * utils::dsize(kvcache_.config().dtype));
    } else {
        core::context().setDevice(kvcache_.config().device_type, kvcache_.config().device_id);
        core::context().runtime().api()->memcpy_sync(
            k_slot->data(), k->data(),
            sl_ * nkvhead * head_dim_val * utils::dsize(kvcache_.config().dtype),
            ZEDINFER_MEMCPY_D2D);
    }
}

tensor_t SingleForwardContext::attend(
    int layer, tensor_t q_rope, float scale,
    const ExecutorConfig &exec_config,
    size_t nhead, size_t nkvhead, size_t head_dim) {

    auto make = [&](std::vector<size_t> shape) {
        return Tensor::create(shape, exec_config.data_type,
                              exec_config.device_type, exec_config.device_id);
    };

    auto attn = make({static_cast<size_t>(sl_), nhead, head_dim});

    if (sl_ == 1 && kvcache_.is_paged()) {
        ops::paged_attention_decode(
            attn->view({nhead, head_dim}), q_rope->view({nhead, head_dim}),
            kvcache_.k_pool_data(),
            kvcache_.k_block_ids(layer), kvcache_.v_block_ids(layer),
            past_len_ + 1,
            scale, exec_config.data_type, exec_config.device_type, exec_config.device_id,
            nhead, nkvhead, head_dim, kvcache_.block_size());
    } else if (kvcache_.is_paged()) {
        kvcache_.scatter_layer_to_blocks(layer);
        ops::paged_attention_prefill(
            attn, q_rope,
            kvcache_.k_pool_data(),
            kvcache_.k_block_ids(layer), kvcache_.v_block_ids(layer),
            sl_, past_len_,
            scale, exec_config.data_type, exec_config.device_type, exec_config.device_id,
            nhead, nkvhead, head_dim, kvcache_.block_size());
    } else {
        ops::self_attention(attn, q_rope,
            kvcache_.get_k_cache_slice(layer, past_len_ + sl_),
            kvcache_.get_v_cache_slice(layer, past_len_ + sl_), scale);
    }

    return attn;
}

void SingleForwardContext::finalize() {
    kvcache_.update_seq_len(sl_);
}

// ============================================================================
// BatchedForwardContext
// ============================================================================

static void scatter_token_to_block(
    const std::byte *src, size_t token_bytes,
    kvcache::SequenceBlockTable &table, int layer, int global_pos,
    bool is_k, kvcache::BlockPool &pool) {

    int bs = pool.config().block_size;
    int block_idx = global_pos / bs;
    int offset = global_pos % bs;
    auto &blocks = is_k ? table.k_blocks[layer] : table.v_blocks[layer];

    void *dst = static_cast<std::byte *>(pool.block_data(blocks[block_idx]))
                + offset * token_bytes;

    if (pool.device_type() == ZEDINFER_DEVICE_CPU) {
        std::memcpy(dst, src, token_bytes);
    } else {
        core::context().setDevice(pool.device_type(), pool.device_id());
        core::context().runtime().api()->memcpy_sync(
            dst, src, token_bytes, ZEDINFER_MEMCPY_D2D);
    }
}

BatchedForwardContext::BatchedForwardContext(
    const BatchContext &batch, kvcache::BlockAllocator &allocator,
    const ExecutorConfig & /*exec_config*/)
    : batch_(batch), allocator_(allocator), total_(batch.total_tokens()) {}

void BatchedForwardContext::prepare_inputs(
    tensor_t &ids, tensor_t &pos_ids, const ExecutorConfig &exec_config) {

    ids = Tensor::create({static_cast<size_t>(total_)}, ZEDINFER_DTYPE_I32,
                         exec_config.device_type, exec_config.device_id);
    ids->load(batch_.token_ids.data());

    pos_ids = Tensor::create({static_cast<size_t>(total_)}, ZEDINFER_DTYPE_I64,
                              exec_config.device_type, exec_config.device_id);
    pos_ids->load(batch_.position_ids.data());
}

void BatchedForwardContext::write_kv(int layer, tensor_t k, tensor_t v) {
    auto &pool = allocator_.pool();
    size_t token_bytes = pool.config().token_bytes();

    for (const auto &slot : batch_.slots) {
        auto *k_src = static_cast<const std::byte *>(k->data())
                      + slot.token_offset * token_bytes;
        auto *v_src = static_cast<const std::byte *>(v->data())
                      + slot.token_offset * token_bytes;

        for (int t = 0; t < slot.num_tokens; ++t) {
            int global_pos = slot.past_len + t;
            scatter_token_to_block(k_src + t * token_bytes, token_bytes,
                                   slot.request->block_table, layer, global_pos,
                                   true, pool);
            scatter_token_to_block(v_src + t * token_bytes, token_bytes,
                                   slot.request->block_table, layer, global_pos,
                                   false, pool);
        }
    }
}

tensor_t BatchedForwardContext::attend(
    int layer, tensor_t q_rope, float scale,
    const ExecutorConfig &exec_config,
    size_t nhead, size_t nkvhead, size_t head_dim) {

    auto make = [&](std::vector<size_t> shape) {
        return Tensor::create(shape, exec_config.data_type,
                              exec_config.device_type, exec_config.device_id);
    };
    auto make_typed = [&](std::vector<size_t> shape, zedinferDataType_t dtype) {
        return Tensor::create(shape, dtype,
                              exec_config.device_type, exec_config.device_id);
    };

    auto &pool = allocator_.pool();
    int block_size = pool.config().block_size;
    auto attn = make({static_cast<size_t>(total_), nhead, head_dim});

    int num_decode = batch_.num_decode_slots();

    // Batched decode attention
    if (num_decode > 0) {
        int max_blocks = 0;
        std::vector<InferenceRequest *> decode_reqs;
        for (const auto &slot : batch_.slots) {
            if (!slot.is_prefill) {
                int nb = static_cast<int>(slot.request->block_table.k_blocks[layer].size());
                max_blocks = std::max(max_blocks, nb);
                decode_reqs.push_back(slot.request);
            }
        }

        std::vector<int> k_bt(num_decode * max_blocks, 0);
        std::vector<int> v_bt(num_decode * max_blocks, 0);
        std::vector<int> sl(num_decode);

        for (int r = 0; r < num_decode; ++r) {
            sl[r] = decode_reqs[r]->block_table.seq_len + 1;
            auto &kb = decode_reqs[r]->block_table.k_blocks[layer];
            auto &vb = decode_reqs[r]->block_table.v_blocks[layer];
            for (size_t b = 0; b < kb.size(); ++b) {
                k_bt[r * max_blocks + b] = kb[b];
                v_bt[r * max_blocks + b] = vb[b];
            }
        }

        auto k_bt_t = make_typed({static_cast<size_t>(num_decode * max_blocks)}, ZEDINFER_DTYPE_I32);
        k_bt_t->load(k_bt.data());
        auto v_bt_t = make_typed({static_cast<size_t>(num_decode * max_blocks)}, ZEDINFER_DTYPE_I32);
        v_bt_t->load(v_bt.data());
        auto sl_t = make_typed({static_cast<size_t>(num_decode)}, ZEDINFER_DTYPE_I32);
        sl_t->load(sl.data());

        auto decode_q = q_rope->slice(0, batch_.decode_token_offset,
                                       batch_.decode_token_offset + num_decode);
        auto decode_out = attn->slice(0, batch_.decode_token_offset,
                                       batch_.decode_token_offset + num_decode);

        ops::paged_attention_decode_batched(
            decode_out, decode_q, pool.block_data(0),
            reinterpret_cast<const int *>(k_bt_t->data()),
            reinterpret_cast<const int *>(v_bt_t->data()),
            reinterpret_cast<const int *>(sl_t->data()),
            num_decode, max_blocks, scale, exec_config.data_type,
            exec_config.device_type, exec_config.device_id,
            nhead, nkvhead, head_dim, block_size);
    }

    // Prefill attention (per-request sequential)
    int prefill_offset = num_decode;
    for (const auto &slot : batch_.slots) {
        if (!slot.is_prefill) continue;
        int chunk = slot.num_tokens;

        auto pf_q = q_rope->slice(0, prefill_offset, prefill_offset + chunk);
        auto pf_out = attn->slice(0, prefill_offset, prefill_offset + chunk);

        ops::paged_attention_prefill(
            pf_out, pf_q, pool.block_data(0),
            slot.request->block_table.k_blocks[layer].data(),
            slot.request->block_table.v_blocks[layer].data(),
            chunk, slot.past_len, scale, exec_config.data_type,
            exec_config.device_type, exec_config.device_id,
            nhead, nkvhead, head_dim, block_size);

        prefill_offset += chunk;
    }

    return attn;
}

void BatchedForwardContext::finalize() {
    // No-op: scheduler updates block_table.seq_len in process_results()
}

} // namespace zedinfer::model
