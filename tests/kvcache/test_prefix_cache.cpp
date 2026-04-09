#include "backend/kvcache/block_pool.hpp"
#include "backend/kvcache/prefix_cache.hpp"

#include <gtest/gtest.h>

using namespace zedinfer::kvcache;

class PrefixCacheTest : public ::testing::Test {
protected:
    void SetUp() override {
        BlockConfig cfg;
        cfg.block_size = 4; // small for testing
        cfg.num_kv_heads = 2;
        cfg.head_dim = 8;
        cfg.dtype = ZEDINFER_DTYPE_F32;

        pool_ = std::make_unique<BlockPool>(cfg, 256, ZEDINFER_DEVICE_CPU, 0);
        allocator_ = std::make_unique<BlockAllocator>(*pool_, num_layers_);
        cache_ = std::make_unique<PrefixCache>(*pool_, num_layers_);
    }

    static constexpr int num_layers_ = 2;
    std::unique_ptr<BlockPool> pool_;
    std::unique_ptr<BlockAllocator> allocator_;
    std::unique_ptr<PrefixCache> cache_;
};

// ============================================================================
// Hash computation
// ============================================================================

TEST_F(PrefixCacheTest, HashDeterministic) {
    std::vector<int> tokens = {10, 20, 30, 40};
    auto h1 = PrefixCache::compute_block_hash(tokens, 0, 4, 0);
    auto h2 = PrefixCache::compute_block_hash(tokens, 0, 4, 0);
    EXPECT_EQ(h1, h2);
}

TEST_F(PrefixCacheTest, HashDifferentTokens) {
    std::vector<int> t1 = {10, 20, 30, 40};
    std::vector<int> t2 = {10, 20, 30, 41};
    auto h1 = PrefixCache::compute_block_hash(t1, 0, 4, 0);
    auto h2 = PrefixCache::compute_block_hash(t2, 0, 4, 0);
    EXPECT_NE(h1, h2);
}

TEST_F(PrefixCacheTest, HashChainDiffers) {
    // Same tokens but different parent hash → different result
    std::vector<int> tokens = {10, 20, 30, 40};
    auto h1 = PrefixCache::compute_block_hash(tokens, 0, 4, 0);
    auto h2 = PrefixCache::compute_block_hash(tokens, 0, 4, 12345);
    EXPECT_NE(h1, h2);
}

// ============================================================================
// Insert and match
// ============================================================================

TEST_F(PrefixCacheTest, InsertAndMatchFullPrefix) {
    // Simulate: request with 8 tokens (2 full blocks of 4)
    std::vector<int> tokens = {1, 2, 3, 4, 5, 6, 7, 8};
    auto table = allocator_->allocate_sequence(8);

    // Insert blocks into cache
    cache_->insert_blocks(tokens, 4, table);
    EXPECT_EQ(cache_->cache_size(), 2); // 2 full blocks cached

    // New request with same prefix should match
    SequenceBlockTable matched;
    int matched_tokens = cache_->match_prefix(tokens, 4, matched);
    EXPECT_EQ(matched_tokens, 8);
    EXPECT_EQ(matched.num_layers, num_layers_);
    EXPECT_EQ(matched.pages[0].size(), 2u); // 2 pages matched per layer
    EXPECT_EQ(matched.seq_len, 8);
}

TEST_F(PrefixCacheTest, PartialPrefixMatch) {
    // Insert 8-token prefix
    std::vector<int> prefix = {1, 2, 3, 4, 5, 6, 7, 8};
    auto table = allocator_->allocate_sequence(8);
    cache_->insert_blocks(prefix, 4, table);

    // Request with same first block but different second block
    std::vector<int> new_tokens = {1, 2, 3, 4, 99, 98, 97, 96};
    SequenceBlockTable matched;
    int matched_tokens = cache_->match_prefix(new_tokens, 4, matched);

    // Only first block matches (chain hash breaks at block 1)
    EXPECT_EQ(matched_tokens, 4);
    EXPECT_EQ(matched.pages[0].size(), 1u);
}

TEST_F(PrefixCacheTest, NoMatch) {
    std::vector<int> tokens = {1, 2, 3, 4};
    auto table = allocator_->allocate_sequence(4);
    cache_->insert_blocks(tokens, 4, table);

    // Completely different tokens
    std::vector<int> other = {99, 98, 97, 96};
    SequenceBlockTable matched;
    int matched_tokens = cache_->match_prefix(other, 4, matched);
    EXPECT_EQ(matched_tokens, 0);
}

TEST_F(PrefixCacheTest, PartialBlockNotCached) {
    // 6 tokens: 1 full block (4 tokens) + 1 partial block (2 tokens)
    std::vector<int> tokens = {1, 2, 3, 4, 5, 6};
    auto table = allocator_->allocate_sequence(6);
    cache_->insert_blocks(tokens, 4, table);

    // Only 1 full block should be cached
    EXPECT_EQ(cache_->cache_size(), 1);
}

// ============================================================================
// Ref counting integration
// ============================================================================

TEST_F(PrefixCacheTest, MatchIncrementsRefCount) {
    std::vector<int> tokens = {1, 2, 3, 4};
    auto table = allocator_->allocate_sequence(4);
    int page = table.pages[0][0];

    cache_->insert_blocks(tokens, 4, table);

    // Release original owner's reference
    allocator_->release_sequence(table);

    // Page should be evictable (ref_count==0, hash set)
    EXPECT_EQ(pool_->ref_count(page), 0);

    // Match should share (ref_count → 1)
    SequenceBlockTable matched;
    cache_->match_prefix(tokens, 4, matched);
    EXPECT_EQ(pool_->ref_count(page), 1);
}

// ============================================================================
// Stats
// ============================================================================

TEST_F(PrefixCacheTest, HitMissStats) {
    std::vector<int> tokens = {1, 2, 3, 4, 5, 6, 7, 8};
    auto table = allocator_->allocate_sequence(8);
    cache_->insert_blocks(tokens, 4, table);

    // Full match: 2 hits
    SequenceBlockTable m1;
    cache_->match_prefix(tokens, 4, m1);
    EXPECT_EQ(cache_->hit_count(), 2u);
    EXPECT_EQ(cache_->miss_count(), 0u);

    // Partial match: 1 hit + 1 miss
    std::vector<int> partial = {1, 2, 3, 4, 99, 99, 99, 99};
    SequenceBlockTable m2;
    cache_->match_prefix(partial, 4, m2);
    EXPECT_EQ(cache_->hit_count(), 3u);
    EXPECT_EQ(cache_->miss_count(), 1u);
}
