#include "backend/kvcache/block_pool.hpp"

#include <gtest/gtest.h>

using namespace zedinfer::kvcache;

class BlockPoolTest : public ::testing::Test {
protected:
    void SetUp() override {
        BlockConfig cfg;
        cfg.block_size = 4;
        cfg.num_kv_heads = 2;
        cfg.head_dim = 8;
        cfg.dtype = ZEDINFER_DTYPE_F32;
        // Use CPU for testing (no GPU needed)
        pool_ = std::make_unique<BlockPool>(cfg, 16, ZEDINFER_DEVICE_CPU, 0);
    }

    std::unique_ptr<BlockPool> pool_;
};

// ============================================================================
// Basic allocation
// ============================================================================

TEST_F(BlockPoolTest, AllocateAndFree) {
    EXPECT_EQ(pool_->free_blocks(), 16);
    EXPECT_EQ(pool_->used_blocks(), 0);

    int b0 = pool_->allocate();
    EXPECT_GE(b0, 0);
    EXPECT_EQ(pool_->free_blocks(), 15);
    EXPECT_EQ(pool_->used_blocks(), 1);

    pool_->free(b0);
    EXPECT_EQ(pool_->free_blocks(), 16);
    EXPECT_EQ(pool_->used_blocks(), 0);
}

TEST_F(BlockPoolTest, AllocateAll) {
    for (int i = 0; i < 16; ++i) {
        EXPECT_GE(pool_->allocate(), 0);
    }
    EXPECT_EQ(pool_->free_blocks(), 0);
    EXPECT_EQ(pool_->used_blocks(), 16);
    EXPECT_EQ(pool_->allocate(), -1); // pool full
}

// ============================================================================
// Reference counting
// ============================================================================

TEST_F(BlockPoolTest, ShareAndRelease) {
    int b = pool_->allocate();
    EXPECT_EQ(pool_->ref_count(b), 1);
    EXPECT_EQ(pool_->used_blocks(), 1);

    pool_->share(b);
    EXPECT_EQ(pool_->ref_count(b), 2);
    EXPECT_EQ(pool_->used_blocks(), 1); // still 1 used block

    pool_->release(b);
    EXPECT_EQ(pool_->ref_count(b), 1);
    EXPECT_EQ(pool_->used_blocks(), 1);

    pool_->release(b);
    EXPECT_EQ(pool_->ref_count(b), 0);
    // No hash set, so it becomes free (not evictable)
    EXPECT_EQ(pool_->free_blocks(), 16);
    EXPECT_EQ(pool_->used_blocks(), 0);
}

// ============================================================================
// Evictable state (ref_count==0, hash!=0)
// ============================================================================

TEST_F(BlockPoolTest, EvictableState) {
    int b = pool_->allocate();
    pool_->set_content_hash(b, 0x12345);
    pool_->set_immutable(b, true);

    EXPECT_EQ(pool_->free_blocks(), 15);
    EXPECT_EQ(pool_->evictable_count(), 0);  // ref_count > 0, not evictable yet

    pool_->release(b);
    // Now ref_count==0 and hash!=0 → evictable
    EXPECT_EQ(pool_->free_blocks(), 15);
    EXPECT_EQ(pool_->evictable_count(), 1);
    EXPECT_EQ(pool_->available_blocks(), 16); // free + evictable
}

TEST_F(BlockPoolTest, EvictOne) {
    int b = pool_->allocate();
    pool_->set_content_hash(b, 0xABCD);
    pool_->touch(b);
    pool_->release(b);

    EXPECT_EQ(pool_->evictable_count(), 1);

    int evicted = pool_->evict_one();
    EXPECT_EQ(evicted, b);
    EXPECT_EQ(pool_->evictable_count(), 0);
    EXPECT_EQ(pool_->free_blocks(), 16);
}

TEST_F(BlockPoolTest, EvictLRU) {
    int b0 = pool_->allocate();
    int b1 = pool_->allocate();

    pool_->set_content_hash(b0, 0x111);
    pool_->touch(b0);
    pool_->set_content_hash(b1, 0x222);
    pool_->touch(b1); // b1 is more recent

    pool_->release(b0);
    pool_->release(b1);

    // Should evict b0 (older last_access)
    int evicted = pool_->evict_one();
    EXPECT_EQ(evicted, b0);
}

TEST_F(BlockPoolTest, AllocateTriggersEviction) {
    // Fill the pool
    std::vector<int> blocks;
    for (int i = 0; i < 16; ++i) blocks.push_back(pool_->allocate());

    // Make one evictable
    pool_->set_content_hash(blocks[5], 0xFFF);
    pool_->release(blocks[5]);

    EXPECT_EQ(pool_->free_blocks(), 0);
    EXPECT_EQ(pool_->evictable_count(), 1);

    // Allocate should evict blocks[5]
    int b = pool_->allocate();
    EXPECT_GE(b, 0);
    EXPECT_EQ(pool_->evictable_count(), 0);
}

// ============================================================================
// Share from evictable state (prefix cache reuse)
// ============================================================================

TEST_F(BlockPoolTest, ShareFromEvictable) {
    int b = pool_->allocate();
    pool_->set_content_hash(b, 0x999);
    pool_->release(b);
    EXPECT_EQ(pool_->evictable_count(), 1);
    EXPECT_EQ(pool_->used_blocks(), 0);

    // Share reactivates: evictable → used
    pool_->share(b);
    EXPECT_EQ(pool_->evictable_count(), 0);
    EXPECT_EQ(pool_->used_blocks(), 1);
    EXPECT_EQ(pool_->ref_count(b), 1);
}

// ============================================================================
// Counter consistency
// ============================================================================

TEST_F(BlockPoolTest, CounterConsistency) {
    // Throughout various operations, free + evictable + used == total
    auto check = [&]() {
        EXPECT_EQ(pool_->free_blocks() + pool_->evictable_count() + pool_->used_blocks(),
                  pool_->total_blocks());
    };

    check();

    int b0 = pool_->allocate(); check();
    int b1 = pool_->allocate(); check();

    pool_->share(b0); check();
    pool_->set_content_hash(b1, 0x123); check();

    pool_->release(b0); check();
    pool_->release(b0); check(); // ref_count 0, no hash → free

    pool_->release(b1); check(); // ref_count 0, has hash → evictable

    pool_->evict_one(); check();
}
