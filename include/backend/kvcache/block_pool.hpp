#pragma once

#include "backend/ops/attention_params.hpp"
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
/**
 * Runtime-only FlashInfer plan / metadata cache for one sequence.
 *
 * This is backend-specific state (FlashInfer decode plans + the GPU index
 * buffers and the parameter keys used to detect when a cached plan is stale).
 * It is rebuilt on demand from the logical page table and is keyed to a
 * specific table's page layout and device, so it must never be shared across
 * tables — `SequenceBlockTable`'s copy leaves it default. Grouping it here keeps
 * the logical page table decoupled from the attention backend, lets the whole
 * cache reset in a single assignment, and means adding a cache field no longer
 * requires hand-editing a 20-line reset routine or the table's copy semantics.
 */
struct FlashInferSeqCache {
    // Full per-layer page tables on GPU (one tensor per layer).
    std::vector<tensor_t> page_tables_gpu;

    // Single-decode CSR/index buffers on GPU.
    tensor_t single_decode_kv_indptr_gpu;
    tensor_t single_decode_kv_last_page_len_gpu;
    tensor_t single_decode_qo_indptr_gpu;
    tensor_t single_decode_descriptor_gpu;

    // Cached single-decode plan + the parameter keys that determine its validity.
    ops::FlashInferDecodePlan single_decode_plan;
    bool single_decode_plan_ready = false;
    int single_decode_plan_total_pages = -1;
    int single_decode_plan_nhead = 0;
    int single_decode_plan_nkvhead = 0;
    int single_decode_plan_head_dim = 0;
    int single_decode_plan_block_size = 0;
    zedinferDataType_t single_decode_plan_dtype = ZEDINFER_DTYPE_BF16;
    zedinferDeviceType_t single_decode_plan_device_type = ZEDINFER_DEVICE_CPU;
    int single_decode_plan_device_id = -1;
    int single_decode_plan_fastpath_probe_pages = -1;
    bool single_decode_plan_disable_fastpath = false;

    // Device / layout the page-table cache above was built for.
    zedinferDeviceType_t cache_device_type = ZEDINFER_DEVICE_CPU;
    int cache_device_id = -1;
    size_t cache_pages_per_layer = 0;
};

struct SequenceBlockTable {
    // page_ids[layer] = list of physical page IDs for that layer
    std::vector<std::vector<int>> pages; // [num_layers][pages_per_layer]
    int seq_len = 0;
    int num_layers = 0;

    // Backend attention cache: rebuilt on demand, never carried across a copy.
    FlashInferSeqCache fi;

    SequenceBlockTable() = default;
    // Copy carries only the logical page table; the FlashInfer cache is left
    // default (it is table-specific GPU state, rebuilt lazily on first use).
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

    // Invalidate the backend cache (e.g. after the logical page table changes).
    // FlashInferDecodePlan and the tensor_t buffers are RAII, so default-assigning
    // releases their resources — equivalent to the previous per-field reset.
    void clear_runtime_caches() { fi = FlashInferSeqCache{}; }
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
