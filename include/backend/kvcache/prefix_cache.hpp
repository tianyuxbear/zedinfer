#pragma once

#include "backend/kvcache/block_pool.hpp"

#include <cstdint>
#include <unordered_map>
#include <vector>

namespace zedinfer::kvcache {

/**
 * Prefix cache for sharing KV blocks across requests with identical token prefixes.
 *
 * Uses chain hashing: each block's hash incorporates its parent block's hash,
 * so the same tokens at different positions in a sequence will NOT match.
 *
 * One hash entry covers all layers — if tokens match, KV values at every layer
 * are identical (given the same model weights).
 */
class PrefixCache {
public:
    explicit PrefixCache(BlockPool &pool, int num_layers);

    /**
     * Match a prompt's prefix against cached blocks.
     * Returns the number of tokens matched (always a multiple of block_size).
     * Fills matched_table with shared block IDs for the matched prefix.
     * Calls pool.share() and pool.touch() on matched blocks.
     */
    int match_prefix(const std::vector<int> &token_ids,
                     int block_size,
                     SequenceBlockTable &matched_table);

    /**
     * Insert completed full blocks from a request into the cache.
     * Only full blocks (block_size tokens) are cached; partial blocks are skipped.
     * Sets content_hash and immutable on cached blocks.
     */
    void insert_blocks(const std::vector<int> &token_ids,
                       int block_size,
                       const SequenceBlockTable &table);

    /**
     * Remove a cache entry (used during eviction).
     */
    void remove(uint64_t content_hash);

    // Stats
    size_t cache_size() const { return cache_.size(); }
    size_t hit_count() const { return hits_; }
    size_t miss_count() const { return misses_; }

    /**
     * Compute chain hash for a block of tokens.
     */
    static uint64_t compute_block_hash(const std::vector<int> &tokens,
                                       int start, int end,
                                       uint64_t parent_hash);

private:
    struct CacheEntry {
        std::vector<int> k_block_ids; // [num_layers]
        std::vector<int> v_block_ids; // [num_layers]
    };

    BlockPool &pool_;
    int num_layers_;
    std::unordered_map<uint64_t, CacheEntry> cache_;

    mutable size_t hits_ = 0;
    mutable size_t misses_ = 0;
};

} // namespace zedinfer::kvcache
