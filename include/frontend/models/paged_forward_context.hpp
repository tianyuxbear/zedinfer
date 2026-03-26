#pragma once

#include "backend/kvcache/block_pool.hpp"
#include "backend/ops/attention_params.hpp"
#include "backend/tensor/tensor.hpp"
#include "zedinfer/activation.hpp"
#include "zedinfer/batch_context.hpp"

#include <vector>

namespace zedinfer::model {

/**
 * Paged forward context for single-request and batched execution.
 * Directly used by transformer_forward (no virtual dispatch).
 *
 * Handles:
 *   - Token scatter to blocks (write_kv)
 *   - Paged attention dispatch (attend) — decode and prefill
 *   - Direct-to-block decode writes (zero-copy via is_mmap)
 */
class PagedForwardContext {
public:
    // Single-request mode (session-based, run_one)
    PagedForwardContext(const std::vector<int>& input_ids, int past_len, kvcache::SequenceBlockTable& block_table,
                        kvcache::BlockPool& pool);

    // Batch mode (continuous batching, step/run_loop)
    PagedForwardContext(const BatchContext& batch, kvcache::BlockAllocator& allocator);

    int num_tokens() const { return total_tokens_; }
    void prepare_inputs(tensor_t& ids, tensor_t& pos_ids, const ExecutorConfig& exec_config);
    void prepare_inputs_into(tensor_t ids, tensor_t pos_ids);
    void write_kv(int layer, tensor_t k, tensor_t v);
    tensor_t attend(int layer, tensor_t q_rope, float scale, const ExecutorConfig& exec_config, size_t nhead,
                    size_t nkvhead, size_t head_dim, tensor_t pre_alloc_out = nullptr);
    void finalize();

private:
    struct Slot {
        kvcache::SequenceBlockTable* block_table;
        int token_offset; // start in flattened token_ids
        int num_tokens;   // tokens in this slot
        int past_len;     // tokens already in KV cache
        bool is_decode;   // true if num_tokens == 1 and past_len > 0
    };

    kvcache::BlockPool& pool_;
    int total_tokens_;
    std::vector<Slot> slots_;

    // Input data (single-request mode stores locally, batch mode references BatchContext)
    std::vector<int> token_ids_;
    std::vector<int64_t> position_ids_;

    // Attention sub-dispatchers
    void attend_decode_single(int layer, tensor_t q_rope, tensor_t attn, const ops::AttentionConfig& cfg, size_t nhead,
                              size_t head_dim);
    void attend_decode_batched(int layer, tensor_t q_rope, tensor_t attn, const ops::AttentionConfig& cfg);
    void attend_prefill(int layer, tensor_t q_rope, tensor_t attn, const ops::AttentionConfig& cfg);

    // Helpers
    void scatter_slot_kv(const Slot& slot, int layer, tensor_t k, tensor_t v);
    void copy_to_block(const void* src, size_t bytes, void* dst);
    void build_decode_cache(const ExecutorConfig& exec_config);

    // Cached GPU block tables for batched decode (built once, reused across layers)
    // Block tables are identical across layers since K/V block IDs are per-layer
    // but the physical layout [num_reqs, max_blocks] doesn't change.
    // However, the actual block IDs DO differ per layer, so we cache the
    // host-side vectors and per-layer GPU tensors on first use.
    struct DecodeCacheEntry {
        tensor_t k_bt_gpu;
        tensor_t v_bt_gpu;
    };
    tensor_t seq_lens_gpu_;
    std::vector<DecodeCacheEntry> decode_layer_cache_;
    int cached_num_decode_ = 0;
    int cached_max_blocks_ = 0;
    int cached_decode_start_ = -1;
    bool decode_cache_built_ = false;
};

} // namespace zedinfer::model
