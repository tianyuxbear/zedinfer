#include "backend/core/memory/memory_pool.hpp"
#include "utils/check.hpp"
#include <algorithm>
#include <cstdint>

namespace neollm::core::memory {

MemoryBlock::MemoryBlock(std::byte *ptr, size_t size, std::byte *base_ptr)
    : ptr(ptr), size(size), in_use(false), base_ptr(base_ptr ? base_ptr : ptr) {}

BestFitMemoryPool::BestFitMemoryPool(
    AllocFunc alloc_func,
    FreeFunc free_func,
    const MemoryPoolConfig &config)
    : alloc_func_(std::move(alloc_func)),
      free_func_(std::move(free_func)),
      config_(config) {

    ASSERT(config_.alignment > 0 && (config_.alignment & (config_.alignment - 1)) == 0,
           "alignment must be a positive power of two");

    free_lists_.resize(config_.size_classes.size());

    // Preallocate large blocks upfront to reduce runtime allocation pressure
    // for common workloads (e.g., transformer layer buffers).
    for (size_t size : config_.preallocate_sizes) {
        allocateNewBlock(size);
    }
}

BestFitMemoryPool::~BestFitMemoryPool() {
    // All base pointers are owned exclusively; safe to free unconditionally.
    for (const auto &[base, _] : base_ref_count_) {
        free_func_(base);
    }
}

size_t BestFitMemoryPool::alignSize(size_t size) const {
    return (size + config_.alignment - 1) & ~(config_.alignment - 1);
}

std::byte *BestFitMemoryPool::alignPointer(std::byte *ptr) const {
    uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
    return reinterpret_cast<std::byte *>((addr + config_.alignment - 1) & ~(config_.alignment - 1));
}

bool BestFitMemoryPool::isAligned(std::byte *ptr) const {
    return (reinterpret_cast<uintptr_t>(ptr) & (config_.alignment - 1)) == 0;
}

size_t BestFitMemoryPool::getSizeClass(size_t size) const {
    // Size classes are sorted; linear scan is acceptable due to small class count (<10).
    for (size_t i = 0; i < config_.size_classes.size(); ++i) {
        if (size <= config_.size_classes[i]) {
            return i;
        }
    }
    return config_.size_classes.size() - 1;
}

void BestFitMemoryPool::addToFreeList(MemoryBlock *block) {
    size_t class_idx = getSizeClass(block->size);
    free_lists_[class_idx].insert({block->size, block});
    ++free_block_count_;

    // Track largest free block incrementally to avoid O(n) scans in hot paths.
    if (block->size > largest_free_block_) {
        largest_free_block_ = block->size;
    }
}

void BestFitMemoryPool::removeFromFreeList(MemoryBlock *block) {
    size_t class_idx = getSizeClass(block->size);
    auto &free_list = free_lists_[class_idx];

    // Multimap may contain multiple blocks of same size; scan to find exact pointer.
    auto range = free_list.equal_range(block->size);
    for (auto it = range.first; it != range.second; ++it) {
        if (it->second == block) {
            free_list.erase(it);
            --free_block_count_;

            // Recompute largest free block only if the removed block was the largest.
            // This is rare, so full scan is acceptable here.
            if (block->size == largest_free_block_) {
                largest_free_block_ = 0;
                for (const auto &fl : free_lists_) {
                    if (!fl.empty()) {
                        largest_free_block_ = std::max(largest_free_block_, fl.rbegin()->first);
                    }
                }
            }
            return;
        }
    }
}

MemoryBlock *BestFitMemoryPool::findBestFitBlock(size_t size) {
    // Search size classes in increasing order to find the smallest suitable block.
    size_t start_class = getSizeClass(size);
    for (size_t class_idx = start_class; class_idx < free_lists_.size(); ++class_idx) {
        auto &free_list = free_lists_[class_idx];
        if (free_list.empty()) {
            continue;
        }

        // lower_bound gives first block with size >= requested size → best-fit.
        auto it = free_list.lower_bound(size);
        if (it != free_list.end()) {
            MemoryBlock *block = it->second;
            removeFromFreeList(block);
            return block;
        }
    }

    return nullptr;
}

MemoryBlock *BestFitMemoryPool::splitBlock(MemoryBlock *block, size_t size) {
    // Avoid internal fragmentation: only split if remainder is usable.
    if (block->size <= size || (block->size - size) < config_.min_split_size) {
        return nullptr;
    }

    size_t remaining = block->size - size;
    auto new_block = std::make_unique<MemoryBlock>(block->ptr + size, remaining, block->base_ptr);
    auto new_block_ptr = new_block.get();
    blocks_.push_back(std::move(new_block));

    addToFreeList(new_block_ptr);
    addr_to_block_[new_block_ptr->ptr] = new_block_ptr;

    block->size = size;
    return new_block_ptr;
}

std::byte *BestFitMemoryPool::allocateNewBlock(size_t size) {
    size_t block_size = std::max(size, config_.initial_block_size);
    if (total_allocated_ + block_size > config_.max_pool_size) {
        if (!config_.allow_growth) {
            return nullptr;
        }
        tryReclaimMemory(); // Attempt to free unused base blocks before failing.
        if (total_allocated_ + block_size > config_.max_pool_size) {
            return nullptr;
        }
    }

    // Allocate extra space to guarantee alignment without losing usability.
    size_t alloc_size = block_size + config_.alignment;
    void *raw_ptr = alloc_func_(alloc_size);
    if (!raw_ptr) {
        return nullptr;
    }

    std::byte *aligned_ptr = alignPointer(static_cast<std::byte *>(raw_ptr));
    size_t offset = aligned_ptr - static_cast<std::byte *>(raw_ptr);
    size_t usable_size = alloc_size - offset;

    auto block = std::make_unique<MemoryBlock>(aligned_ptr, usable_size);
    block->base_ptr = static_cast<std::byte *>(raw_ptr); // Keep raw pointer for deallocation.
    auto block_ptr = block.get();
    blocks_.push_back(std::move(block));
    total_allocated_ += usable_size;

    base_ref_count_[block_ptr->base_ptr] = 0;
    addToFreeList(block_ptr);
    addr_to_block_[block_ptr->ptr] = block_ptr;

    return block_ptr->ptr;
}

void BestFitMemoryPool::tryCoalesceNeighbors(MemoryBlock *block) {
    if (block->in_use) {
        return;
    }

    // Find immediate successor in address space using sorted addr_to_block_.
    auto next_it = addr_to_block_.upper_bound(block->ptr);
    if (next_it != addr_to_block_.end()) {
        MemoryBlock *next = next_it->second;
        // Only coalesce if both are free, share the same base, and are contiguous.
        if (!next->in_use && block->base_ptr == next->base_ptr && block->ptr + block->size == next->ptr) {

            removeFromFreeList(block);
            removeFromFreeList(next);

            block->size += next->size;
            addr_to_block_.erase(next->ptr);
            blocks_.erase(
                std::remove_if(blocks_.begin(), blocks_.end(),
                               [next](const auto &b) { return b.get() == next; }),
                blocks_.end());

            addToFreeList(block);
            tryCoalesceNeighbors(block);
        }
    }
}

void BestFitMemoryPool::coalesceBlocks() {
    if (free_block_count_ == 0) {
        return;
    }

    // Full coalescing: collect all free blocks in address order and merge contiguous ones.
    // This is O(n log n) due to map traversal but only triggered under high fragmentation.
    std::vector<MemoryBlock *> free_blocks;
    for (const auto &[_, block] : addr_to_block_) {
        if (!block->in_use) {
            free_blocks.push_back(block);
        }
    }

    for (size_t i = 0; i + 1 < free_blocks.size();) {
        auto curr = free_blocks[i];
        auto next = free_blocks[i + 1];
        if (curr->base_ptr == next->base_ptr && curr->ptr + curr->size == next->ptr) {
            removeFromFreeList(curr);
            removeFromFreeList(next);

            curr->size += next->size;
            addr_to_block_.erase(next->ptr);
            blocks_.erase(
                std::remove_if(blocks_.begin(), blocks_.end(),
                               [next](const auto &b) { return b.get() == next; }),
                blocks_.end());

            addToFreeList(curr);
            free_blocks.erase(free_blocks.begin() + i + 1);
        } else {
            ++i;
        }
    }
}

std::byte *BestFitMemoryPool::allocate(size_t size) {
    if (size == 0) {
        return nullptr;
    }
    size = alignSize(size);

    MemoryBlock *block = findBestFitBlock(size);
    if (!block) {
        std::byte *ptr = allocateNewBlock(size);
        if (!ptr) {
            return nullptr;
        }
        block = addr_to_block_[ptr];
    }

    if (!block->in_use) {
        removeFromFreeList(block);
    }

    splitBlock(block, size); // May leave remainder in free list.
    block->in_use = true;
    allocated_blocks_[block->ptr] = block;
    total_used_ += block->size;
    ++base_ref_count_[block->base_ptr];

    ASSERT(isAligned(block->ptr), "Returned pointer is not aligned");
    return block->ptr;
}

void BestFitMemoryPool::deallocate(std::byte *ptr) {
    if (!ptr) {
        return;
    }

    auto it = allocated_blocks_.find(ptr);
    if (it == allocated_blocks_.end() || !it->second->in_use) {
        return;
    }

    MemoryBlock *block = it->second;
    block->in_use = false;
    total_used_ -= block->size;
    allocated_blocks_.erase(it);
    --base_ref_count_[block->base_ptr];

    addToFreeList(block);

    tryCoalesceNeighbors(block); // Opportunistic coalescing on deallocation.

    ++deallocations_since_last_coalesce_;

    // Trigger full defragmentation only when fragmentation is high and enough
    // deallocations have occurred to justify the O(n) cost.
    if (deallocations_since_last_coalesce_ >= COALESCE_BATCH && getFragmentation() > config_.fragmentation_threshold) {
        coalesceBlocks();
        deallocations_since_last_coalesce_ = 0;
    }
}

size_t BestFitMemoryPool::getTotalAllocated() const {
    return total_allocated_;
}

size_t BestFitMemoryPool::getTotalUsed() const {
    return total_used_;
}

float BestFitMemoryPool::getFragmentation() const {
    if (total_allocated_ == 0) {
        return 0.0f;
    }

    size_t total_free = total_allocated_ - total_used_;
    if (total_free == 0) {
        return 0.0f;
    }
    if (largest_free_block_ == 0) {
        return 1.0f; // All free memory is unusable.
    }

    // Fragmentation = 1 - (largest contiguous free / total free)
    // Measures how much free memory is fragmented into unusable pieces.
    return 1.0f - static_cast<float>(largest_free_block_) / static_cast<float>(total_free);
}

BestFitMemoryPool::FragmentationStats
BestFitMemoryPool::getDetailedFragmentation() const {
    FragmentationStats stats{};
    stats.total_free_bytes = total_allocated_ - total_used_;
    stats.free_block_count = free_block_count_;
    stats.largest_free_block = largest_free_block_;

    if (total_allocated_ == 0) {
        return stats;
    }

    stats.memory_utilization = static_cast<float>(total_used_) / total_allocated_;

    if (stats.total_free_bytes > 0 && largest_free_block_ > 0) {
        stats.fragmentation_ratio = 1.0f - static_cast<float>(largest_free_block_) / stats.total_free_bytes;
    } else if (stats.total_free_bytes > 0) {
        stats.fragmentation_ratio = 1.0f;
    }

    // Wasted bytes = total free - largest block → memory that cannot satisfy large requests.
    stats.wasted_bytes = (stats.free_block_count > 1) ? (stats.total_free_bytes - largest_free_block_) : 0;
    return stats;
}

void BestFitMemoryPool::tryReclaimMemory() {
    coalesceBlocks(); // Merge free blocks first to maximize reclaimable regions.

    // Identify base blocks with zero active allocations.
    std::vector<std::byte *> reclaimable_bases;
    for (const auto &[base, count] : base_ref_count_) {
        if (count == 0) {
            reclaimable_bases.push_back(base);
        }
    }

    if (reclaimable_bases.empty()) {
        return;
    }

    size_t freed_size = 0;
    for (auto base : reclaimable_bases) {
        // Purge all metadata referencing this base.
        for (auto &free_list : free_lists_) {
            for (auto it = free_list.begin(); it != free_list.end();) {
                if (it->second->base_ptr == base) {
                    --free_block_count_;
                    it = free_list.erase(it);
                } else {
                    ++it;
                }
            }
        }

        for (auto it = addr_to_block_.begin(); it != addr_to_block_.end();) {
            if (it->second->base_ptr == base) {
                it = addr_to_block_.erase(it);
            } else {
                ++it;
            }
        }

        // Free the raw memory only once (when ptr == base).
        blocks_.erase(
            std::remove_if(blocks_.begin(), blocks_.end(),
                           [this, base, &freed_size](const auto &block) {
                               if (block->base_ptr == base) {
                                   if (block->ptr == base) {
                                       free_func_(block->ptr);
                                       freed_size += block->size;
                                   }
                                   return true;
                               }
                               return false;
                           }),
            blocks_.end());

        base_ref_count_.erase(base);
    }

    total_allocated_ -= freed_size;

    // Recompute largest free block after reclamation.
    largest_free_block_ = 0;
    for (const auto &free_list : free_lists_) {
        if (!free_list.empty()) {
            largest_free_block_ = std::max(largest_free_block_, free_list.rbegin()->first);
        }
    }
}

} // namespace neollm::core::memory