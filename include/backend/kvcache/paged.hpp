#pragma once

#include "backend/kvcache/base.hpp"
#include "backend/kvcache/block_pool.hpp"

#include <vector>

namespace zedinfer::kvcache {

/**
 * Paged KV cache backed by a shared BlockPool.
 *
 * Implements the KVCache interface so the model forward code is unchanged.
 * New K/V tokens are written into a small contiguous write buffer, then
 * scattered into blocks when update_seq_len() is called.
 * For attention reads, blocks are gathered into contiguous temporary buffers.
 *
 * PR-8 adds paged attention kernels that read blocks directly,
 * eliminating the gather copy.
 */
class PagedKVCache : public KVCache {
public:
    PagedKVCache(const KVCacheConfig &config, BlockAllocator &allocator,
                 int initial_estimated_tokens = 256);
    ~PagedKVCache() override;

    // KVCache interface
    int allocated_capacity() const override;
    tensor_t get_k_cache(int layer_idx) override;
    tensor_t get_v_cache(int layer_idx) override;

    // Write slot: returns a view into the write buffer.
    tensor_t get_k_cache_slice(int layer_idx, int past_len, int seq_len) override;
    tensor_t get_v_cache_slice(int layer_idx, int past_len, int seq_len) override;

    // Read full history: gathers blocks into contiguous buffer.
    tensor_t get_k_cache_slice(int layer_idx, int total_len) override;
    tensor_t get_v_cache_slice(int layer_idx, int total_len) override;

    // Override: scatter write buffer into blocks after model writes.
    void update_seq_len(int new_tokens) override;

    void reset() override;
    size_t memory_usage() const override;
    float utilization() const override;

    // Paged attention interface
    bool is_paged() const override { return true; }
    void *k_pool_data() const override;
    void *v_pool_data() const override;
    const int *k_block_ids(int layer_idx) const override;
    const int *v_block_ids(int layer_idx) const override;
    int block_size() const override;
    void scatter_layer_to_blocks(int layer_idx) override;

    // Access block table
    const SequenceBlockTable &block_table() const { return block_table_; }

private:
    BlockAllocator &allocator_;
    SequenceBlockTable block_table_;

    // Write buffers: one per layer, K and V
    // [max_write_tokens, num_kv_heads, head_dim]
    std::vector<tensor_t> write_k_bufs_;
    std::vector<tensor_t> write_v_bufs_;
    int write_buf_capacity_ = 0;

    // Gather buffers: reused across attention reads
    std::vector<tensor_t> gather_k_bufs_;
    std::vector<tensor_t> gather_v_bufs_;
    int gather_capacity_ = 0;

    // Pending write tracking
    int pending_write_past_len_ = -1;
    int pending_write_seq_len_ = 0;
    bool direct_write_to_blocks_ = false;  // true when decode writes directly to block memory
    int scattered_layers_ = 0;             // count of layers already scattered (prefill path)

    // Helpers
    void ensure_write_capacity(int seq_len);
    void ensure_gather_capacity(int total_len);
    void ensure_blocks_for_tokens(int total_tokens);
    void scatter_to_blocks(int past_len, int new_tokens);
    tensor_t gather_contiguous(int layer_idx, int total_len, bool is_k);
    void copy_region(void *dst, const void *src, size_t bytes);
};

} // namespace zedinfer::kvcache
