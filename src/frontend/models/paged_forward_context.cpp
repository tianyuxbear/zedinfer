#include "frontend/models/paged_forward_context.hpp"
#include "backend/core/context/context.hpp"
#include "backend/ops/ops.hpp"
#include "utils/types.hpp"

#include <algorithm>
#include <cstring>

namespace zedinfer::model {

// ============================================================================
// Constructors
// ============================================================================

PagedForwardContext::PagedForwardContext(
    const std::vector<int> &input_ids, int past_len,
    kvcache::SequenceBlockTable &block_table,
    kvcache::BlockPool &pool)
    : pool_(pool), total_tokens_(static_cast<int>(input_ids.size())) {

    token_ids_ = input_ids;
    position_ids_.resize(total_tokens_);
    for (int i = 0; i < total_tokens_; ++i)
        position_ids_[i] = past_len + static_cast<int64_t>(i);

    bool is_decode = (total_tokens_ == 1 && past_len > 0);
    slots_.push_back({&block_table, 0, total_tokens_, past_len, is_decode});
}

PagedForwardContext::PagedForwardContext(
    const BatchContext &batch,
    kvcache::BlockAllocator &allocator)
    : pool_(allocator.pool()), total_tokens_(batch.total_tokens()) {

    token_ids_ = batch.token_ids;
    position_ids_ = batch.position_ids;

    for (const auto &s : batch.slots) {
        bool is_decode = !s.is_prefill;
        slots_.push_back({&s.request->block_table, s.token_offset,
                          s.num_tokens, s.past_len, is_decode});
    }
}

// ============================================================================
// Input Preparation
// ============================================================================

void PagedForwardContext::prepare_inputs(
    tensor_t &ids, tensor_t &pos_ids, const ExecutorConfig &exec_config) {

    ids = Tensor::create({static_cast<size_t>(total_tokens_)}, ZEDINFER_DTYPE_I32,
                         exec_config.device_type, exec_config.device_id);
    ids->load(token_ids_.data());

    pos_ids = Tensor::create({static_cast<size_t>(total_tokens_)}, ZEDINFER_DTYPE_I64,
                              exec_config.device_type, exec_config.device_id);
    pos_ids->load(position_ids_.data());
}

// ============================================================================
// KV Write: scatter tokens to blocks per slot
// ============================================================================

void PagedForwardContext::copy_to_block(const void *src, size_t bytes, void *dst) {
    if (pool_.device_type() == ZEDINFER_DEVICE_CPU) {
        std::memcpy(dst, src, bytes);
    } else {
        core::context().setDevice(pool_.device_type(), pool_.device_id());
        core::context().runtime().api()->memcpy_sync(
            dst, src, bytes, ZEDINFER_MEMCPY_D2D);
    }
}

void PagedForwardContext::scatter_slot_kv(
    const Slot &slot, int layer, tensor_t k, tensor_t v) {

    const int bs = pool_.config().block_size;
    const size_t token_bytes = pool_.config().token_bytes();

    auto *k_src = static_cast<const std::byte *>(k->data()) + slot.token_offset * token_bytes;
    auto *v_src = static_cast<const std::byte *>(v->data()) + slot.token_offset * token_bytes;

    auto &k_blocks = slot.block_table->k_blocks[layer];
    auto &v_blocks = slot.block_table->v_blocks[layer];

    for (int t = 0; t < slot.num_tokens; ++t) {
        int global_pos = slot.past_len + t;
        int block_idx = global_pos / bs;
        int offset = global_pos % bs;

        void *k_dst = static_cast<std::byte *>(pool_.block_data(k_blocks[block_idx]))
                      + offset * token_bytes;
        copy_to_block(k_src + t * token_bytes, token_bytes, k_dst);

        void *v_dst = static_cast<std::byte *>(pool_.block_data(v_blocks[block_idx]))
                      + offset * token_bytes;
        copy_to_block(v_src + t * token_bytes, token_bytes, v_dst);
    }
}

void PagedForwardContext::write_kv(int layer, tensor_t k, tensor_t v) {
    // For decode slots (1 token), the shared transformer_forward already
    // allocated k/v as regular tensors. We scatter them to blocks.
    // For prefill slots (N tokens), same — scatter all tokens.
    for (const auto &slot : slots_) {
        scatter_slot_kv(slot, layer, k, v);
    }
}

// ============================================================================
// Attention: paged decode (single or batched) + paged prefill
// ============================================================================

tensor_t PagedForwardContext::attend(
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

    ops::AttentionConfig attn_cfg{
        static_cast<int>(nhead), static_cast<int>(nkvhead), static_cast<int>(head_dim),
        scale, pool_.config().block_size,
        exec_config.data_type, exec_config.device_type, exec_config.device_id};

    auto attn = make({static_cast<size_t>(total_tokens_), nhead, head_dim});

    // Count decode and prefill slots
    int num_decode = 0;
    int decode_start = -1;
    for (size_t i = 0; i < slots_.size(); ++i) {
        if (slots_[i].is_decode) {
            if (decode_start < 0) decode_start = slots_[i].token_offset;
            num_decode++;
        }
    }

    // Decode attention
    if (num_decode > 0) {
        int max_blocks = 0;
        std::vector<kvcache::SequenceBlockTable *> decode_tables;
        for (const auto &slot : slots_) {
            if (!slot.is_decode) continue;
            int nb = static_cast<int>(slot.block_table->k_blocks[layer].size());
            max_blocks = std::max(max_blocks, nb);
            decode_tables.push_back(slot.block_table);
        }

        std::vector<int> k_bt(num_decode * max_blocks, 0);
        std::vector<int> v_bt(num_decode * max_blocks, 0);
        std::vector<int> sl(num_decode);

        for (int r = 0; r < num_decode; ++r) {
            sl[r] = decode_tables[r]->seq_len + 1;
            auto &kb = decode_tables[r]->k_blocks[layer];
            auto &vb = decode_tables[r]->v_blocks[layer];
            for (size_t b = 0; b < kb.size(); ++b) {
                k_bt[r * max_blocks + b] = kb[b];
                v_bt[r * max_blocks + b] = vb[b];
            }
        }

        if (num_decode == 1) {
            auto decode_q = q_rope->slice(0, decode_start, decode_start + 1);
            auto decode_out = attn->slice(0, decode_start, decode_start + 1);

            ops::AttentionParams params{attn_cfg};
            params.out = decode_out->view({nhead, head_dim});
            params.q = decode_q->view({nhead, head_dim});
            params.pool_base = pool_.block_data(0);
            params.k_block_table = decode_tables[0]->k_blocks[layer].data();
            params.v_block_table = decode_tables[0]->v_blocks[layer].data();
            params.seq_len = sl[0];
            params.seqlen_q = 1;
            ops::attention(params);
        } else {
            auto k_bt_t = make_typed({static_cast<size_t>(num_decode * max_blocks)}, ZEDINFER_DTYPE_I32);
            k_bt_t->load(k_bt.data());
            auto v_bt_t = make_typed({static_cast<size_t>(num_decode * max_blocks)}, ZEDINFER_DTYPE_I32);
            v_bt_t->load(v_bt.data());
            auto sl_t = make_typed({static_cast<size_t>(num_decode)}, ZEDINFER_DTYPE_I32);
            sl_t->load(sl.data());

            auto decode_q = q_rope->slice(0, decode_start, decode_start + num_decode);
            auto decode_out = attn->slice(0, decode_start, decode_start + num_decode);

            ops::AttentionParams params{attn_cfg};
            params.out = decode_out;
            params.q = decode_q;
            params.pool_base = pool_.block_data(0);
            params.batched_k_block_tables = reinterpret_cast<const int *>(k_bt_t->data());
            params.batched_v_block_tables = reinterpret_cast<const int *>(v_bt_t->data());
            params.batched_seq_lens = reinterpret_cast<const int *>(sl_t->data());
            params.num_requests = num_decode;
            params.max_blocks_per_seq = max_blocks;
            ops::attention(params);
        }
    }

    // Prefill attention (per-request sequential)
    for (const auto &slot : slots_) {
        if (slot.is_decode) continue;

        auto pf_q = q_rope->slice(0, slot.token_offset, slot.token_offset + slot.num_tokens);
        auto pf_out = attn->slice(0, slot.token_offset, slot.token_offset + slot.num_tokens);

        ops::AttentionParams params{attn_cfg};
        params.out = pf_out;
        params.q = pf_q;
        params.pool_base = pool_.block_data(0);
        params.k_block_table = slot.block_table->k_blocks[layer].data();
        params.v_block_table = slot.block_table->v_blocks[layer].data();
        params.seqlen_q = slot.num_tokens;
        params.past_len = slot.past_len;
        ops::attention(params);
    }

    return attn;
}

// ============================================================================
// Finalize
// ============================================================================

void PagedForwardContext::finalize() {
    // Update block_table.seq_len for single-request mode.
    // In batch mode, scheduler handles this in process_results().
    if (slots_.size() == 1 && !slots_[0].is_decode) {
        // Single prefill: update seq_len
        slots_[0].block_table->seq_len += slots_[0].num_tokens;
    } else if (slots_.size() == 1 && slots_[0].is_decode) {
        // Single decode: update seq_len
        slots_[0].block_table->seq_len += 1;
    }
    // Multi-slot (batch): scheduler updates in process_results()
}

} // namespace zedinfer::model
