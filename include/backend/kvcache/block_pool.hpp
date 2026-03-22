#pragma once

#include "zedinfer.h"

#include <cstddef>
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
 * Pool of fixed-size KV cache blocks backed by a single contiguous allocation.
 * Block assignment is tracked with a bitmap.
 */
class BlockPool {
public:
    BlockPool(BlockConfig config, int num_blocks,
              zedinferDeviceType_t device_type, int device_id);
    ~BlockPool();

    // Block lifecycle
    int allocate();          // returns block_id [0..num_blocks-1], -1 if full
    void free(int block_id);

    // Access
    void *block_data(int block_id) const;
    const BlockConfig &config() const { return config_; }
    zedinferDeviceType_t device_type() const { return device_type_; }
    int device_id() const { return device_id_; }

    // Stats
    int total_blocks() const { return num_blocks_; }
    int free_blocks() const { return free_count_; }
    int used_blocks() const { return num_blocks_ - free_count_; }
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
    std::vector<bool> allocated_;
    int free_count_;
    int next_free_ = 0; // hint for allocation scan
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

    // Free all blocks held by a sequence.
    void free_sequence(SequenceBlockTable &table);

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
