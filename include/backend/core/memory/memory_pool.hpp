#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <unordered_map>
#include <vector>

namespace zedinfer::core::memory {

// Metadata for a memory block managed by the pool.
struct MemoryBlock {
    std::byte *ptr;      // Start of usable memory (aligned)
    size_t size;         // Usable size in bytes
    bool in_use;         // True if currently allocated
    std::byte *base_ptr; // Pointer to the start of the underlying raw block

    MemoryBlock(std::byte *ptr, size_t size, std::byte *base_ptr = nullptr);
};

// Configuration for memory pool behavior.
struct MemoryPoolConfig {
    size_t initial_block_size = 64 * 1024 * 1024;      // Initial raw block size (64 MB)
    size_t max_pool_size = 10ULL * 1024 * 1024 * 1024; // Hard limit (8 GB)
    size_t alignment = 64;                             // Allocation alignment (e.g., cache line)
    size_t min_split_size = 4 * 1024;                  // Minimum size to split a free block
    bool allow_growth = true;                          // Allow allocating new raw blocks
    float fragmentation_threshold = 0.3f;              // Defrag if fragmentation exceeds this

    // Size classes for fast best-fit lookup (sorted, terminated by SIZE_MAX)
    std::vector<size_t> size_classes = {
        256,              // Tiny
        4 * 1024,         // Small
        64 * 1024,        // Medium
        1024 * 1024,      // Large
        16 * 1024 * 1024, // XLarge
        SIZE_MAX          // Catch-all
    };

    // Block sizes to preallocate at startup for common workloads
    std::vector<size_t> preallocate_sizes = {
        1 * 1024 * 1024,
        16 * 1024 * 1024,
        64 * 1024 * 1024,
        256 * 1024 * 1024};
};

// Abstract base class for memory pools.
class MemoryPool {
public:
    virtual ~MemoryPool() = default;
    virtual std::byte *allocate(size_t size) = 0;
    virtual void deallocate(std::byte *ptr) = 0;
    virtual size_t getTotalAllocated() const = 0; // Total raw memory reserved
    virtual size_t getTotalUsed() const = 0;      // Sum of currently allocated user bytes
    virtual float getFragmentation() const = 0;   // Ratio of unusable free space
};

// Single-threaded best-fit memory pool with defragmentation and size-class optimization.
class BestFitMemoryPool : public MemoryPool {
public:
    using AllocFunc = std::function<void *(size_t)>;
    using FreeFunc = std::function<void(void *)>;

    BestFitMemoryPool(AllocFunc alloc_func,
                      FreeFunc free_func,
                      const MemoryPoolConfig &config = MemoryPoolConfig());

    ~BestFitMemoryPool() override;

    BestFitMemoryPool(const BestFitMemoryPool &) = delete;
    BestFitMemoryPool &operator=(const BestFitMemoryPool &) = delete;
    BestFitMemoryPool(BestFitMemoryPool &&) = delete;
    BestFitMemoryPool &operator=(BestFitMemoryPool &&) = delete;

    std::byte *allocate(size_t size) override;
    void deallocate(std::byte *ptr) override;

    size_t getTotalAllocated() const override;
    size_t getTotalUsed() const override;
    float getFragmentation() const override;

    struct FragmentationStats {
        float fragmentation_ratio; // Fraction of free memory that is fragmented
        size_t free_block_count;   // Number of free blocks
        size_t total_free_bytes;   // Total unused bytes
        size_t largest_free_block; // Largest contiguous free region
        float memory_utilization;  // Used / allocated
        size_t wasted_bytes;       // Free bytes that cannot satisfy any allocation
    };
    FragmentationStats getDetailedFragmentation() const;

private:
    using FreeBlockMap = std::multimap<size_t, MemoryBlock *>; // Sorted by size for best-fit

    size_t alignSize(size_t size) const;
    std::byte *alignPointer(std::byte *ptr) const;
    bool isAligned(std::byte *ptr) const;
    size_t getSizeClass(size_t size) const;
    MemoryBlock *findBestFitBlock(size_t size);
    std::byte *allocateNewBlock(size_t size);
    void coalesceBlocks();
    void tryReclaimMemory();

    void addToFreeList(MemoryBlock *block);
    void removeFromFreeList(MemoryBlock *block);
    MemoryBlock *splitBlock(MemoryBlock *block, size_t size);
    void tryCoalesceNeighbors(MemoryBlock *block);

    AllocFunc alloc_func_; // Underlying allocator (e.g., malloc or custom)
    FreeFunc free_func_;   // Corresponding deallocator
    MemoryPoolConfig config_;

    // All memory blocks (both allocated and free), owned by the pool
    std::vector<std::unique_ptr<MemoryBlock>> blocks_;

    // Maps user pointers to their MemoryBlock for O(1) deallocation
    std::unordered_map<std::byte *, MemoryBlock *> allocated_blocks_;

    // One free list per size class; each is a multimap<size, block> for best-fit search
    std::vector<FreeBlockMap> free_lists_;

    // Sorted by address to enable O(log n) neighbor lookup during coalescing
    std::map<std::byte *, MemoryBlock *> addr_to_block_;

    // Track reference counts of underlying raw blocks (for shared base deallocation)
    std::unordered_map<std::byte *, size_t> base_ref_count_;

    // Statistics and heuristics
    size_t total_allocated_ = 0; // Total bytes obtained from alloc_func_
    size_t total_used_ = 0;      // Sum of sizes of currently allocated blocks
    size_t deallocations_since_last_coalesce_ = 0;
    size_t free_block_count_ = 0;
    size_t largest_free_block_ = 0;

    static constexpr size_t COALESCE_BATCH = 128; // Deallocations before auto-coalesce
};

} // namespace zedinfer::core::memory