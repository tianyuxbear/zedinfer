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
    // Layer index semantics:
    //   - For dense/MoE models (Qwen2 / Qwen3 / Qwen3-MoE) the parameter is just
    //     the raw decoder layer index L ∈ [0, num_hidden_layers).
    //   - For hybrid models (Qwen3.5) it is the **logical KV layer index** —
    //     i.e. HybridForwardConfig::full_layer_index(L) ∈ [0, num_kv_layers),
    //     because linear-attention layers do not contribute KV blocks. The KV
    //     pool was sized with `num_kv_layers = count(layer_types=="full_attention")`
    //     in init_block_pool, so block_table->num_layers already matches.
    //   The PagedForwardContext itself is agnostic; the caller chooses the mapping.
    void write_kv(int kv_layer_idx, tensor_t k, tensor_t v);
    tensor_t attend(int kv_layer_idx, tensor_t q_rope, float scale, const ExecutorConfig& exec_config, size_t nhead,
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
    void build_flashinfer_decode_cache(const ops::AttentionConfig& cfg);
    void build_flashinfer_prefill_cache(const ops::AttentionConfig& cfg);
    void build_flashinfer_kv_write_cache();

    // Cached GPU page tables for batched decode (built once, reused across layers).
    // Physical layout [num_reqs, max_pages] is shared across layers, but the
    // actual page ids still differ per layer, so we cache per-layer tensors.
    struct DecodeCacheEntry {
        tensor_t page_bt_gpu;
    };
    struct FlashInferLayerCacheEntry {
        tensor_t kv_page_indices_gpu;
    };
    tensor_t seq_lens_gpu_;
    std::vector<DecodeCacheEntry> decode_layer_cache_;
    std::vector<FlashInferLayerCacheEntry> flashinfer_decode_layer_cache_;
    std::vector<FlashInferLayerCacheEntry> flashinfer_kv_write_layer_cache_;
    tensor_t flashinfer_decode_kv_indptr_gpu_;
    tensor_t flashinfer_decode_kv_last_page_len_gpu_;
    tensor_t flashinfer_decode_qo_indptr_gpu_;
    tensor_t flashinfer_decode_descriptor_gpu_;
    const ops::FlashInferDecodePlan* flashinfer_decode_plan_ = nullptr;
    std::vector<int> flashinfer_decode_kv_indptr_host_;
    std::vector<int> flashinfer_decode_kv_last_page_len_host_;
    std::vector<int> flashinfer_decode_qo_indptr_host_;
    bool flashinfer_decode_cache_built_ = false;
    bool flashinfer_decode_uses_prefill_kernel_ = false;
    std::vector<const Slot*> flashinfer_prefill_slots_;
    tensor_t flashinfer_prefill_qo_indptr_gpu_;
    tensor_t flashinfer_prefill_kv_indptr_gpu_;
    tensor_t flashinfer_prefill_kv_last_page_len_gpu_;
    std::vector<int> flashinfer_prefill_qo_indptr_host_;
    std::vector<int> flashinfer_prefill_kv_indptr_host_;
    std::vector<int> flashinfer_prefill_kv_last_page_len_host_;
    tensor_t flashinfer_kv_write_kv_indptr_gpu_;
    tensor_t flashinfer_kv_write_kv_last_page_len_gpu_;
    tensor_t flashinfer_kv_write_batch_indices_gpu_;
    tensor_t flashinfer_kv_write_positions_gpu_;
    std::vector<int> flashinfer_kv_write_kv_indptr_host_;
    std::vector<int> flashinfer_kv_write_kv_last_page_len_host_;
    int flashinfer_kv_write_batch_size_ = 0;
    bool flashinfer_kv_write_decode_only_ = false;
    bool flashinfer_kv_write_cache_built_ = false;
    int flashinfer_prefill_start_ = -1;
    int flashinfer_prefill_total_tokens_ = 0;
    bool flashinfer_prefill_cache_built_ = false;
    int cached_num_decode_ = 0;
    int cached_max_blocks_ = 0;
    int cached_decode_start_ = -1;
    bool decode_cache_built_ = false;
};

} // namespace zedinfer::model
