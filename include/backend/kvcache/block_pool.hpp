#pragma once

#include "zedinfer.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace zedinfer::kvcache {

/**
 * Configuration for KV cache blocks.
 */
struct BlockConfig {
    int block_size = 16; // tokens per block
    int num_kv_heads;
    int head_dim;
    zedinferDataType_t dtype = ZEDINFER_DTYPE_BF16;

    size_t block_bytes() const;
    size_t token_bytes() const;
};

/**
 * Per-sequence block table mapping logical blocks to physical block IDs.
 */
struct SequenceBlockTable {
    // block_ids[layer] = list of physical block IDs for that layer
    std::vector<std::vector<int>> k_blocks; // [num_layers][blocks_per_layer]
    std::vector<std::vector<int>> v_blocks; // [num_layers][blocks_per_layer]
    int seq_len = 0;
    int num_layers = 0;
};

/**
 * Per-block metadata for reference counting and prefix caching.
 */
struct BlockMeta {
    int ref_count = 0;          // 0 = free, 1 = exclusive, >1 = shared
    uint64_t content_hash = 0;  // chain hash of token IDs (0 = unset)
    uint64_t last_access = 0;   // monotonic counter for LRU eviction
    bool immutable = false;     // true when block is full and finalized
};

/**
 * Pool of fixed-size KV cache blocks backed by a single contiguous allocation.
 * Supports reference counting for prefix caching.
 */
class BlockPool {
public:
    BlockPool(BlockConfig config, int num_blocks,
              zedinferDeviceType_t device_type, int device_id);
    ~BlockPool();

    // Block lifecycle
    int allocate();          // find free block (ref_count==0, hash==0), set ref_count=1
    void free(int block_id); // hard free: reset ref_count, hash, immutable

    // Prefix caching operations
    void share(int block_id);               // increment ref_count
    void release(int block_id);             // decrement ref_count; keep hash if evictable
    int ref_count(int block_id) const;
    void set_content_hash(int block_id, uint64_t hash);
    uint64_t content_hash(int block_id) const;
    void set_immutable(int block_id, bool immutable);
    bool is_immutable(int block_id) const;
    void touch(int block_id);               // update LRU counter

    // Eviction
    int evict_one();                         // free LRU block with ref_count==0 and hash!=0
    int evictable_count() const;

    // Access
    void *block_data(int block_id) const;
    const BlockConfig &config() const { return config_; }
    zedinferDeviceType_t device_type() const { return device_type_; }
    int device_id() const { return device_id_; }

    // Stats
    int total_blocks() const { return num_blocks_; }
    int free_blocks() const;                 // ref_count==0 && hash==0
    int used_blocks() const;
    int available_blocks() const;            // free + evictable
    size_t memory_usage() const;

    // Non-copyable
    BlockPool(const BlockPool &) = delete;
    BlockPool &operator=(const BlockPool &) = delete;

private:
    BlockConfig config_;
    zedinferDeviceType_t device_type_;
    int device_id_;
    int num_blocks_;
    size_t block_bytes_;

    void *pool_memory_ = nullptr;
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
    BlockAllocator(BlockPool &pool, int num_layers);

    // Allocate blocks for a new sequence with estimated token count.
    SequenceBlockTable allocate_sequence(int estimated_tokens);

    // Allocate one additional block for a layer+type when current blocks are full.
    int extend_sequence(SequenceBlockTable &table, int layer, bool is_k);

    // Ensure the block table has enough blocks for needed_len tokens across all layers.
    void ensure_blocks(SequenceBlockTable &table, int needed_len);

    // Free all blocks held by a sequence (hard free, clears hash).
    void free_sequence(SequenceBlockTable &table);

    // Release all blocks (decrements ref_count, keeps hash for caching).
    void release_sequence(SequenceBlockTable &table);

    int available_blocks() const;
    int block_size() const;
    int num_layers() const { return num_layers_; }
    BlockPool &pool() { return pool_; }
    const BlockPool &pool() const { return pool_; }

private:
    BlockPool &pool_;
    int num_layers_;
};

} // namespace zedinfer::kvcache
