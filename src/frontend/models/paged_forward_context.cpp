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
        slots_.push_back({&s.request->active_block_table(), s.token_offset,
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

void PagedForwardContext::prepare_inputs_into(tensor_t ids, tensor_t pos_ids) {
    ids->load(token_ids_.data());
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
    const int bs = pool_.config().block_size;
    const size_t token_bytes = pool_.config().token_bytes();

    if (pool_.device_type() == ZEDINFER_DEVICE_CPU) {
        // CPU: per-token memcpy (already fast, no launch overhead)
        for (const auto &slot : slots_) {
            scatter_slot_kv(slot, layer, k, v);
        }
        return;
    }

    // GPU: batch all scatter operations into arrays, then use cudaMemcpyAsync
    // to avoid per-token cudaMemcpy launch overhead.
    // Collect (src, dst, size) triplets for all tokens across all slots.
    struct ScatterOp { const void *src; void *dst; };
    std::vector<ScatterOp> ops;
    ops.reserve(total_tokens_ * 2); // K + V

    for (const auto &slot : slots_) {
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
            void *v_dst = static_cast<std::byte *>(pool_.block_data(v_blocks[block_idx]))
                          + offset * token_bytes;

            ops.push_back({k_src + t * token_bytes, k_dst});
            ops.push_back({v_src + t * token_bytes, v_dst});
        }
    }

    // Execute all copies using async memcpy on the default stream
    core::context().setDevice(pool_.device_type(), pool_.device_id());
    auto *api = core::context().runtime().api();
    for (const auto &op : ops) {
        api->memcpy_async(op.dst, op.src, token_bytes, ZEDINFER_MEMCPY_D2D, nullptr);
    }
    // No explicit sync needed — subsequent CUDA kernels (attention) on the same
    // stream will wait for the async copies to complete.
}

// ============================================================================
// Attention: paged decode (single or batched) + paged prefill
// ============================================================================

void PagedForwardContext::build_decode_cache(const ExecutorConfig &exec_config) {
    if (decode_cache_built_) return;

    // Count decode slots
    cached_num_decode_ = 0;
    cached_decode_start_ = -1;
    std::vector<kvcache::SequenceBlockTable *> decode_tables;

    for (const auto &slot : slots_) {
        if (!slot.is_decode) continue;
        if (cached_decode_start_ < 0) cached_decode_start_ = slot.token_offset;
        cached_num_decode_++;
        decode_tables.push_back(slot.block_table);
    }

    if (cached_num_decode_ <= 1 || decode_tables.empty()) {
        decode_cache_built_ = true;
        return;
    }

    // Find max blocks across all layers and all requests
    int num_layers = decode_tables[0]->num_layers;
    cached_max_blocks_ = 0;
    for (auto *dt : decode_tables) {
        for (int L = 0; L < num_layers; ++L) {
            cached_max_blocks_ = std::max(cached_max_blocks_,
                static_cast<int>(dt->k_blocks[L].size()));
        }
    }

    // Build seq_lens GPU tensor (same for all layers)
    std::vector<int> sl(cached_num_decode_);
    for (int r = 0; r < cached_num_decode_; ++r) {
        sl[r] = decode_tables[r]->seq_len + 1;
    }
    seq_lens_gpu_ = Tensor::create({static_cast<size_t>(cached_num_decode_)},
                                    ZEDINFER_DTYPE_I32,
                                    exec_config.device_type, exec_config.device_id);
    seq_lens_gpu_->load(sl.data());

    // Build per-layer block table GPU tensors
    decode_layer_cache_.resize(num_layers);
    size_t bt_size = static_cast<size_t>(cached_num_decode_) * cached_max_blocks_;

    for (int L = 0; L < num_layers; ++L) {
        std::vector<int> k_bt(bt_size, 0);
        std::vector<int> v_bt(bt_size, 0);

        for (int r = 0; r < cached_num_decode_; ++r) {
            auto &kb = decode_tables[r]->k_blocks[L];
            auto &vb = decode_tables[r]->v_blocks[L];
            for (size_t b = 0; b < kb.size(); ++b) {
                k_bt[r * cached_max_blocks_ + b] = kb[b];
                v_bt[r * cached_max_blocks_ + b] = vb[b];
            }
        }

        auto k_gpu = Tensor::create({bt_size}, ZEDINFER_DTYPE_I32,
                                     exec_config.device_type, exec_config.device_id);
        k_gpu->load(k_bt.data());
        auto v_gpu = Tensor::create({bt_size}, ZEDINFER_DTYPE_I32,
                                     exec_config.device_type, exec_config.device_id);
        v_gpu->load(v_bt.data());

        decode_layer_cache_[L] = {std::move(k_gpu), std::move(v_gpu)};
    }

    decode_cache_built_ = true;
}

tensor_t PagedForwardContext::attend(
    int layer, tensor_t q_rope, float scale,
    const ExecutorConfig &exec_config,
    size_t nhead, size_t nkvhead, size_t head_dim,
    tensor_t pre_alloc_out) {

    ops::AttentionConfig attn_cfg{
        static_cast<int>(nhead), static_cast<int>(nkvhead), static_cast<int>(head_dim),
        scale, pool_.config().block_size,
        exec_config.data_type, exec_config.device_type, exec_config.device_id};

    auto attn = pre_alloc_out ? pre_alloc_out
                : Tensor::create({static_cast<size_t>(total_tokens_), nhead, head_dim},
                                  exec_config.data_type, exec_config.device_type,
                                  exec_config.device_id);

    // Build decode cache on first layer (uploads all layers' block tables at once)
    build_decode_cache(exec_config);

    // Decode attention
    if (cached_num_decode_ > 0) {
        if (cached_num_decode_ == 1) {
            // Single decode: use non-batched kernel (no GPU block table needed)
            auto decode_q = q_rope->slice(0, cached_decode_start_, cached_decode_start_ + 1);
            auto decode_out = attn->slice(0, cached_decode_start_, cached_decode_start_ + 1);

            auto *dt = slots_[0].block_table; // first slot is decode for single request
            for (const auto &slot : slots_) {
                if (slot.is_decode) { dt = slot.block_table; break; }
            }

            ops::AttentionParams params{attn_cfg};
            params.out = decode_out->view({nhead, head_dim});
            params.q = decode_q->view({nhead, head_dim});
            params.pool_base = pool_.block_data(0);
            params.k_block_table = dt->k_blocks[layer].data();
            params.v_block_table = dt->v_blocks[layer].data();
            params.seq_len = dt->seq_len + 1;
            params.seqlen_q = 1;
            ops::attention(params);
        } else {
            // Multi decode: use cached GPU block tables
            auto decode_q = q_rope->slice(0, cached_decode_start_,
                                           cached_decode_start_ + cached_num_decode_);
            auto decode_out = attn->slice(0, cached_decode_start_,
                                           cached_decode_start_ + cached_num_decode_);

            ops::AttentionParams params{attn_cfg};
            params.out = decode_out;
            params.q = decode_q;
            params.pool_base = pool_.block_data(0);
            params.batched_k_block_tables = reinterpret_cast<const int *>(
                decode_layer_cache_[layer].k_bt_gpu->data());
            params.batched_v_block_tables = reinterpret_cast<const int *>(
                decode_layer_cache_[layer].v_bt_gpu->data());
            params.batched_seq_lens = reinterpret_cast<const int *>(seq_lens_gpu_->data());
            params.num_requests = cached_num_decode_;
            params.max_blocks_per_seq = cached_max_blocks_;
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
    // No-op: seq_len is updated by Scheduler::process_results() for all paths.
}

} // namespace zedinfer::model
