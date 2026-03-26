#include "backend/core/memory/memory_pool.hpp"

#include <chrono>
#include <cstdlib>
#include <gtest/gtest.h>
#include <vector>

using namespace zedinfer::core::memory;

// Test Fixture for memory pool tests
class MemoryPoolTest : public ::testing::Test {
protected:
    void SetUp() override {
        config_.alignment = 64;
        config_.min_split_size = 4096;
        config_.initial_block_size = 1 * 1024 * 1024; // 1 MB
        config_.max_pool_size = 100 * 1024 * 1024;    // 100 MB
        config_.fragmentation_threshold = 0.3f;
        config_.preallocate_sizes = {};

        pool_ = std::make_unique<BestFitMemoryPool>([](size_t size) { return std::malloc(size); },
                                                    [](void* ptr) { std::free(ptr); }, config_);
    }

    void TearDown() override { pool_.reset(); }

    bool isAligned(void* ptr, size_t alignment) { return (reinterpret_cast<uintptr_t>(ptr) % alignment) == 0; }

    MemoryPoolConfig config_;
    std::unique_ptr<BestFitMemoryPool> pool_;
};

// Test 1: Address Alignment
TEST_F(MemoryPoolTest, AddressAlignment) {
    std::vector<std::byte*> ptrs;
    size_t sizes[] = {16, 128, 1024, 4096, 65536};

    for (auto size : sizes) {
        auto ptr = pool_->allocate(size);
        ASSERT_NE(ptr, nullptr) << "Allocation of size " << size << " failed";
        EXPECT_TRUE(isAligned(ptr, config_.alignment))
            << "Pointer " << ptr << " (size " << size << ") is not " << config_.alignment << "-byte aligned";
        ptrs.push_back(ptr);
    }

    // Cleanup
    for (auto ptr : ptrs) { pool_->deallocate(ptr); }
}

TEST_F(MemoryPoolTest, AllAllocationsAreAligned) {
    const int NUM_ALLOCS = 100;
    std::vector<std::byte*> ptrs;

    // Test various sizes
    for (int i = 0; i < NUM_ALLOCS; ++i) {
        size_t size = 64 + (i * 137) % 10000; // Varying sizes
        auto ptr = pool_->allocate(size);
        ASSERT_NE(ptr, nullptr);
        EXPECT_TRUE(isAligned(ptr, config_.alignment));
        ptrs.push_back(ptr);
    }

    // Cleanup
    for (auto ptr : ptrs) { pool_->deallocate(ptr); }
}

// Test 2: Minimum Split Size
TEST_F(MemoryPoolTest, MinimumSplitSize) {
    auto stats_before = pool_->getDetailedFragmentation();
    ASSERT_EQ(stats_before.free_block_count, 0);

    // Allocate a block where remaining < min_split_size (should NOT split)
    auto ptr1 = pool_->allocate(1024 * 1024 - 2048); // ~2KB remaining < 4KB
    auto stats_after1 = pool_->getDetailedFragmentation();
    ASSERT_EQ(stats_after1.free_block_count, 0);
    pool_->deallocate(ptr1);

    // Allocate a block where remaining > min_split_size (SHOULD split)
    auto ptr2 = pool_->allocate(512 * 1024); // ~512KB remaining > 4KB
    auto stats_after2 = pool_->getDetailedFragmentation();
    ASSERT_EQ(stats_after2.free_block_count, 1);

    pool_->deallocate(ptr2);

    // When there's enough space to split, we should see more free blocks
    EXPECT_GT(stats_after2.free_block_count, stats_after1.free_block_count)
        << "min_split_size should prevent splitting when remaining is too small";
}

TEST_F(MemoryPoolTest, MinSplitSizePreventsSmallFragments) {
    // Allocate and deallocate to create a known state
    std::vector<std::byte*> ptrs;
    for (int i = 0; i < 10; ++i) {
        ptrs.push_back(pool_->allocate(1023 * 1024)); // 100KB each
    }
    auto stats = pool_->getDetailedFragmentation();
    ASSERT_EQ(stats.free_block_count, 0);

    for (auto ptr : ptrs) { pool_->deallocate(ptr); }

    stats = pool_->getDetailedFragmentation();
    ASSERT_EQ(stats.free_block_count, 10);

    // After coalescing, we shouldn't have tiny fragments
    if (stats.free_block_count > 0) {
        size_t avg_free_block_size = stats.total_free_bytes / stats.free_block_count;
        EXPECT_GT(avg_free_block_size, config_.initial_block_size)
            << "Average free block size should be >= initial_block_size";
    }
}

// Test 3: Fragmentation Calculation
TEST_F(MemoryPoolTest, FragmentationCalculation) {
    // Scenario 1: Empty pool (should have low/zero fragmentation)
    auto stats1 = pool_->getDetailedFragmentation();
    EXPECT_EQ(stats1.fragmentation_ratio, 0.0f) << "Empty pool should have zero fragmentation";

    // Scenario 2: Create checkerboard fragmentation
    std::vector<std::byte*> ptrs;
    for (int i = 0; i < 20; ++i) {
        ptrs.push_back(pool_->allocate(256 * 1024)); // 256KB each
    }

    auto stats2 = pool_->getDetailedFragmentation();
    ASSERT_EQ(stats2.free_block_count, 0);

    // Deallocate odd positions to create fragmentation
    for (size_t i = 1; i < ptrs.size(); i += 2) {
        pool_->deallocate(ptrs[i]);
        ptrs[i] = nullptr;
    }

    auto stats3 = pool_->getDetailedFragmentation();
    ASSERT_EQ(stats3.free_block_count, 10);
    ASSERT_GE(stats3.largest_free_block, 256 * 1024);
    EXPECT_GT(stats3.fragmentation_ratio, 0.8f) << "Checkerboard pattern should create significant fragmentation";

    // Cleanup
    for (auto ptr : ptrs) {
        if (ptr) {
            pool_->deallocate(ptr);
        }
    }
    auto stats4 = pool_->getDetailedFragmentation();
    ASSERT_EQ(stats4.free_block_count, 10);

    // Trigger coalescing
    for (int i = 0; i < 128; ++i) {
        auto p = pool_->allocate(4096);
        pool_->deallocate(p);
    }

    auto stats5 = pool_->getDetailedFragmentation();
    ASSERT_EQ(stats5.free_block_count, 5);
}

TEST_F(MemoryPoolTest, FragmentationStatsAccuracy) {
    auto stats = pool_->getDetailedFragmentation();

    // Basic sanity checks
    EXPECT_GE(stats.fragmentation_ratio, 0.0f);
    EXPECT_LE(stats.fragmentation_ratio, 1.0f);
    EXPECT_GE(stats.memory_utilization, 0.0f);
    EXPECT_LE(stats.memory_utilization, 1.0f);

    // Allocate some memory
    auto ptr = pool_->allocate(1024 * 1024);
    stats = pool_->getDetailedFragmentation();

    EXPECT_GT(stats.memory_utilization, 0.0f) << "After allocation, utilization should be > 0";
    EXPECT_EQ(pool_->getTotalUsed(), stats.memory_utilization * pool_->getTotalAllocated())
        << "Utilization calculation should match";

    pool_->deallocate(ptr);
}

TEST_F(MemoryPoolTest, LargestFreeBlockTracking) {
    // Allocate and create a specific fragmentation pattern
    auto ptr1 = pool_->allocate(1 * 1024 * 1024); // 1MB
    auto ptr2 = pool_->allocate(2 * 1024 * 1024); // 2MB
    auto ptr3 = pool_->allocate(3 * 1024 * 1024); // 3MB

    pool_->deallocate(ptr2);                      // Free the 2MB block

    auto stats = pool_->getDetailedFragmentation();

    // The largest free block should be tracked correctly
    EXPECT_GE(stats.largest_free_block, 2 * 1024 * 1024);

    pool_->deallocate(ptr1);
    pool_->deallocate(ptr3);
}

// Test 4: Block Coalescing
TEST_F(MemoryPoolTest, BlockCoalescing) {
    // 1. Create a 10MB contiguous region
    auto base = pool_->allocate(10 * 1024 * 1024);
    pool_->deallocate(base);

    // 2. Allocate 4 x 2MB blocks
    auto ptr1 = pool_->allocate(2 * 1024 * 1024);
    auto ptr2 = pool_->allocate(2 * 1024 * 1024);
    auto ptr3 = pool_->allocate(2 * 1024 * 1024);
    auto ptr4 = pool_->allocate(2 * 1024 * 1024);

    // 3. Free all
    pool_->deallocate(ptr1);
    pool_->deallocate(ptr2);
    pool_->deallocate(ptr3);
    pool_->deallocate(ptr4);

    // 4. Trigger coalescing
    for (int i = 0; i < 124; ++i) {
        auto p = pool_->allocate(4096);
        pool_->deallocate(p);
    }

    // 5. Verify: can allocate 8MB (result of coalescing 4 x 2MB blocks)
    auto large = pool_->allocate(8 * 1024 * 1024);
    EXPECT_EQ(large, base);

    pool_->deallocate(large);
}

TEST_F(MemoryPoolTest, IncrementalCoalescing) {
    // Allocate 3 x 512KB blocks (smaller, easier to control)
    std::vector<std::byte*> ptrs;
    for (int i = 0; i < 3; ++i) { ptrs.push_back(pool_->allocate(512 * 1024)); }

    // Free one by one, observe incremental coalescing
    pool_->deallocate(ptrs[1]);
    auto stats1 = pool_->getDetailedFragmentation();

    pool_->deallocate(ptrs[0]);
    auto stats2 = pool_->getDetailedFragmentation();

    EXPECT_GE(stats2.largest_free_block, stats1.largest_free_block);

    auto large = pool_->allocate(1024 * 1024); // 1MB > 512KB
    EXPECT_EQ(large, ptrs[0]);

    pool_->deallocate(ptrs[2]);
    pool_->deallocate(large);
}

// Test 5: Performance Benchmark
TEST_F(MemoryPoolTest, PerformanceBenchmark) {
    const int NUM_ALLOCS = 10000;
    std::vector<std::byte*> ptrs;
    ptrs.reserve(NUM_ALLOCS);

    auto base = pool_->allocate(config_.max_pool_size);
    pool_->deallocate(base);

    size_t sizes[] = {64, 256, 1024, 4096, 16384, 65536};

    auto start = std::chrono::high_resolution_clock::now();

    // Allocation phase
    for (int i = 0; i < NUM_ALLOCS; ++i) {
        size_t size = sizes[i % 6];
        auto ptr = pool_->allocate(size);
        if (ptr) {
            ptrs.push_back(ptr);
        }
    }

    auto mid = std::chrono::high_resolution_clock::now();

    // Mixed phase: deallocate half and reallocate
    for (size_t i = 0; i < ptrs.size() / 2; ++i) { pool_->deallocate(ptrs[i]); }

    for (int i = 0; i < NUM_ALLOCS / 2; ++i) {
        size_t size = sizes[i % 6];
        auto ptr = pool_->allocate(size);
        if (ptr) {
            ptrs.push_back(ptr);
        }
    }

    auto end = std::chrono::high_resolution_clock::now();

    // Cleanup
    for (auto ptr : ptrs) {
        if (ptr) {
            pool_->deallocate(ptr);
        }
    }

    auto alloc_time = std::chrono::duration_cast<std::chrono::microseconds>(mid - start).count();
    auto mixed_time = std::chrono::duration_cast<std::chrono::microseconds>(end - mid).count();

    // Performance expectations (adjust based on your requirements)
    EXPECT_LT(alloc_time / NUM_ALLOCS, 10) << "Average allocation should be < 10 microseconds";
    EXPECT_LT(mixed_time / NUM_ALLOCS, 20) << "Average mixed operation should be < 20 microseconds";

    auto stats = pool_->getDetailedFragmentation();
    EXPECT_LT(stats.fragmentation_ratio, 0.5f) << "After heavy usage, fragmentation should be < 50%";
}

TEST_F(MemoryPoolTest, AllocationSpeed) {
    const int NUM_ALLOCS = 1000;

    auto start = std::chrono::high_resolution_clock::now();

    std::vector<std::byte*> ptrs;
    for (int i = 0; i < NUM_ALLOCS; ++i) { ptrs.push_back(pool_->allocate(4096)); }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    // Should complete quickly
    EXPECT_LT(duration, 10000) << "1000 allocations should complete in < 10ms";

    // Cleanup
    for (auto ptr : ptrs) { pool_->deallocate(ptr); }
}

// Test 6: Edge Cases
TEST_F(MemoryPoolTest, ZeroSizeAllocation) {
    auto ptr = pool_->allocate(0);
    EXPECT_EQ(ptr, nullptr) << "Zero size allocation should return nullptr";
}

TEST_F(MemoryPoolTest, VeryLargeAllocation) {
    // Try to allocate more than max_pool_size (should fail with no growth)
    MemoryPoolConfig small_config = config_;
    small_config.max_pool_size = 10 * 1024 * 1024; // 10 MB
    small_config.allow_growth = false;

    BestFitMemoryPool small_pool([](size_t size) { return std::malloc(size); }, [](void* ptr) { std::free(ptr); },
                                 small_config);

    auto ptr = small_pool.allocate(100 * 1024 * 1024); // 100 MB
    EXPECT_EQ(ptr, nullptr) << "Allocation exceeding max_pool_size should fail";
}

TEST_F(MemoryPoolTest, NullptrDeallocation) {
    // Deallocating nullptr should be safe
    EXPECT_NO_THROW(pool_->deallocate(nullptr));
}

TEST_F(MemoryPoolTest, InvalidPointerDeallocation) {
    std::byte dummy;
    // Deallocating a pointer not from the pool should be safe (ignored)
    EXPECT_NO_THROW(pool_->deallocate(&dummy));
}

// Test 7: Memory Statistics
TEST_F(MemoryPoolTest, MemoryStatistics) {
    size_t initial_allocated = pool_->getTotalAllocated();
    size_t initial_used = pool_->getTotalUsed();

    EXPECT_EQ(initial_allocated, 0u) << "Initially no memory should be allocated";
    EXPECT_EQ(initial_used, 0u) << "Initially no memory should be in use";

    auto ptr = pool_->allocate(1024 * 1024);

    EXPECT_GT(pool_->getTotalUsed(), initial_used);
    EXPECT_LE(pool_->getTotalUsed(), pool_->getTotalAllocated());

    pool_->deallocate(ptr);

    EXPECT_EQ(pool_->getTotalUsed(), initial_used) << "After deallocation, used memory should return to initial";
}

TEST_F(MemoryPoolTest, MemoryUtilization) {
    // Allocate 50% of initial block
    size_t alloc_size = config_.initial_block_size / 2;
    auto ptr = pool_->allocate(alloc_size);

    auto stats = pool_->getDetailedFragmentation();

    EXPECT_GT(stats.memory_utilization, 0.3f) << "Should have reasonable utilization";
    EXPECT_LT(stats.memory_utilization, 0.7f) << "Should not be over-utilized";

    pool_->deallocate(ptr);
}

// Test 8: Stress Test
TEST_F(MemoryPoolTest, StressTestRandomAllocations) {
    const int NUM_ITERATIONS = 1000;
    std::vector<std::byte*> ptrs;

    for (int i = 0; i < NUM_ITERATIONS; ++i) {
        // Random operation: allocate or deallocate
        if (ptrs.empty() || (rand() % 2 == 0 && ptrs.size() < 100)) {
            // Allocate
            size_t size = 64 + (rand() % 65536); // 64B to 64KB
            auto ptr = pool_->allocate(size);
            if (ptr) {
                EXPECT_TRUE(isAligned(ptr, config_.alignment));
                ptrs.push_back(ptr);
            }
        } else {
            // Deallocate
            size_t idx = rand() % ptrs.size();
            pool_->deallocate(ptrs[idx]);
            ptrs.erase(ptrs.begin() + idx);
        }
    }

    // Cleanup
    for (auto ptr : ptrs) { pool_->deallocate(ptr); }

    auto stats = pool_->getDetailedFragmentation();
    EXPECT_LT(stats.fragmentation_ratio, 0.7f) << "After stress test, fragmentation should be manageable";
}

TEST_F(MemoryPoolTest, StressTestLargeAllocations) {
    std::vector<std::byte*> ptrs;

    // Allocate many large blocks
    for (int i = 0; i < 50; ++i) {
        auto ptr = pool_->allocate(1024 * 1024); // 1MB each
        if (ptr) {
            EXPECT_TRUE(isAligned(ptr, config_.alignment));
            ptrs.push_back(ptr);
        }
    }

    EXPECT_GT(ptrs.size(), 30u) << "Should successfully allocate many large blocks";

    // Cleanup
    for (auto ptr : ptrs) { pool_->deallocate(ptr); }
}