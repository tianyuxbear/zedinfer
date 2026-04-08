#pragma once

#include "backend/tensor/tensor.hpp"
#include "zedinfer.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace zedinfer::kvcache {

/**
 * Configuration for KV cache pages.
 */
struct BlockConfig {
    int block_size = 16; // tokens per block
    int num_kv_heads;
    int head_dim;
    zedinferDataType_t dtype = ZEDINFER_DTYPE_BF16;

    // Bytes for one K page or one V page.
    size_t block_bytes() const;
    size_t token_bytes() const;
};

/**
 * Per-sequence page table mapping logical pages to shared physical page IDs.
 * The same page ID indexes both the K pool and the V pool.
 */
struct SequenceBlockTable {
    // page_ids[layer] = list of physical page IDs for that layer
    std::vector<std::vector<int>> pages; // [num_layers][pages_per_layer]
    int seq_len = 0;
    int num_layers = 0;

    // Runtime-only GPU cache for full per-layer page tables used by FlashInfer.
    // The cache is invalidated whenever the logical page table changes.
    std::vector<tensor_t> flashinfer_page_tables_gpu;
    tensor_t flashinfer_single_decode_kv_indptr_gpu;
    tensor_t flashinfer_single_decode_kv_last_page_len_gpu;
    tensor_t flashinfer_single_decode_qo_indptr_gpu;
    tensor_t flashinfer_single_decode_descriptor_gpu;
    zedinferDeviceType_t flashinfer_cache_device_type = ZEDINFER_DEVICE_CPU;
    int flashinfer_cache_device_id = -1;
    size_t flashinfer_cache_pages_per_layer = 0;

    SequenceBlockTable() = default;
    SequenceBlockTable(const SequenceBlockTable& other)
        : pages(other.pages), seq_len(other.seq_len), num_layers(other.num_layers) {}
    SequenceBlockTable& operator=(const SequenceBlockTable& other) {
        if (this != &other) {
            pages = other.pages;
            seq_len = other.seq_len;
            num_layers = other.num_layers;
            clear_runtime_caches();
        }
        return *this;
    }
    SequenceBlockTable(SequenceBlockTable&&) noexcept = default;
    SequenceBlockTable& operator=(SequenceBlockTable&&) noexcept = default;

    void clear_runtime_caches() {
        flashinfer_page_tables_gpu.clear();
        flashinfer_single_decode_kv_indptr_gpu.reset();
        flashinfer_single_decode_kv_last_page_len_gpu.reset();
        flashinfer_single_decode_qo_indptr_gpu.reset();
        flashinfer_single_decode_descriptor_gpu.reset();
        flashinfer_cache_device_type = ZEDINFER_DEVICE_CPU;
        flashinfer_cache_device_id = -1;
        flashinfer_cache_pages_per_layer = 0;
    }
};

/**
 * Per-page metadata for reference counting and prefix caching.
 */
struct BlockMeta {
    int ref_count = 0;         // 0 = free, 1 = exclusive, >1 = shared
    uint64_t content_hash = 0; // chain hash of token IDs (0 = unset)
    uint64_t last_access = 0;  // monotonic counter for LRU eviction
    bool immutable = false;    // true when block is full and finalized
};

/**
 * Pool of fixed-size KV cache pages backed by paired K/V allocations.
 * Supports reference counting for prefix caching.
 */
class BlockPool {
public:
    BlockPool(BlockConfig config, int num_blocks, zedinferDeviceType_t device_type, int device_id);
    ~BlockPool();

    // Block lifecycle
    int allocate();          // find free block (ref_count==0, hash==0), set ref_count=1
    void free(int block_id); // hard free: reset ref_count, hash, immutable

    // Prefix caching operations
    void share(int block_id);   // increment ref_count
    void release(int block_id); // decrement ref_count; keep hash if evictable
    int ref_count(int block_id) const;
    void set_content_hash(int block_id, uint64_t hash);
    uint64_t content_hash(int block_id) const;
    void set_immutable(int block_id, bool immutable);
    bool is_immutable(int block_id) const;
    void touch(int block_id); // update LRU counter

    // Eviction
    int evict_one(); // free LRU block with ref_count==0 and hash!=0
    int evictable_count() const;

    // Access
    void* k_block_data(int block_id) const;
    void* v_block_data(int block_id) const;
    void* k_pool_base() const { return k_pool_memory_; }
    void* v_pool_base() const { return v_pool_memory_; }
    const BlockConfig& config() const { return config_; }
    zedinferDeviceType_t device_type() const { return device_type_; }
    int device_id() const { return device_id_; }

    // Stats
    int total_blocks() const { return num_blocks_; }
    int free_blocks() const;      // ref_count==0 && hash==0
    int used_blocks() const;
    int available_blocks() const; // free + evictable
    size_t memory_usage() const;

    // Non-copyable
    BlockPool(const BlockPool&) = delete;
    BlockPool& operator=(const BlockPool&) = delete;

private:
    BlockConfig config_;
    zedinferDeviceType_t device_type_;
    int device_id_;
    int num_blocks_;
    size_t block_bytes_;

    void* k_pool_memory_ = nullptr;
    void* v_pool_memory_ = nullptr;
    std::vector<BlockMeta> block_meta_;
    uint64_t access_counter_ = 0;
    int next_free_ = 0;       // hint for allocation scan
    int free_count_ = 0;      // blocks with ref_count==0 && hash==0
    int evictable_count_ = 0; // blocks with ref_count==0 && hash!=0
};

/**
 * Manages block allocation for sequences.
 * Wraps BlockPool with per-sequence/per-layer logic.
 */
class BlockAllocator {
public:
    BlockAllocator(BlockPool& pool, int num_layers);

    // Allocate blocks for a new sequence with estimated token count.
    SequenceBlockTable allocate_sequence(int estimated_tokens);

    // Allocate one additional page for a layer when current pages are full.
    int extend_sequence(SequenceBlockTable& table, int layer);

    // Ensure the block table has enough blocks for needed_len tokens across all layers.
    void ensure_blocks(SequenceBlockTable& table, int needed_len);

    // Free all blocks held by a sequence (hard free, clears hash).
    void free_sequence(SequenceBlockTable& table);

    // Release all blocks (decrements ref_count, keeps hash for caching).
    void release_sequence(SequenceBlockTable& table);

    int available_blocks() const;
    int block_size() const;
    int num_layers() const { return num_layers_; }
    BlockPool& pool() { return pool_; }
    const BlockPool& pool() const { return pool_; }

private:
    BlockPool& pool_;
    int num_layers_;
};

} // namespace zedinfer::kvcache
