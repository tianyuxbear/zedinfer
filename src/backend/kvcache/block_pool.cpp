#include "backend/kvcache/block_pool.hpp"
#include "backend/core/context/context.hpp"
#include "backend/device/runtime_api.hpp"
#include "utils/types.hpp"

#include <plog/Log.h>
#include <stdexcept>

namespace zedinfer::kvcache {

// ============================================================================
// BlockConfig
// ============================================================================

size_t BlockConfig::token_bytes() const {
    return static_cast<size_t>(num_kv_heads) * head_dim * utils::dsize(dtype);
}

size_t BlockConfig::block_bytes() const {
    return static_cast<size_t>(block_size) * token_bytes();
}

// ============================================================================
// BlockPool
// ============================================================================

BlockPool::BlockPool(BlockConfig config, int num_blocks,
                     zedinferDeviceType_t device_type, int device_id)
    : config_(config),
      device_type_(device_type),
      device_id_(device_id),
      num_blocks_(num_blocks),
      block_bytes_(config.block_bytes()),
      block_meta_(num_blocks),
      free_count_(num_blocks),
      evictable_count_(0) {

    if (num_blocks <= 0) {
        throw std::invalid_argument("[BlockPool] num_blocks must be positive");
    }

    size_t total_bytes = static_cast<size_t>(num_blocks) * block_bytes_;

    core::context().setDevice(device_type, device_id);
    auto api = device::getRuntimeAPI(device_type);
    pool_memory_ = api->malloc_device(total_bytes);

    if (!pool_memory_) {
        throw std::runtime_error("[BlockPool] Failed to allocate " +
                                 std::to_string(total_bytes / (1024 * 1024)) + " MB");
    }

    LOGI << "[BlockPool] Allocated " << num_blocks << " blocks ("
         << config.block_size << " tokens/block, "
         << total_bytes / (1024 * 1024) << " MB) on "
         << (device_type == ZEDINFER_DEVICE_CPU ? "CPU" : "GPU");
}

BlockPool::~BlockPool() {
    if (pool_memory_) {
        auto api = device::getRuntimeAPI(device_type_);
        api->free_device(pool_memory_);
        pool_memory_ = nullptr;
    }
}

int BlockPool::allocate() {
    // First try: find a truly free block (ref_count==0, hash==0)
    if (free_count_ > 0) {
        for (int i = 0; i < num_blocks_; ++i) {
            int idx = (next_free_ + i) % num_blocks_;
            auto &meta = block_meta_[idx];
            if (meta.ref_count == 0 && meta.content_hash == 0) {
                meta.ref_count = 1;
                meta.last_access = ++access_counter_;
                next_free_ = (idx + 1) % num_blocks_;
                free_count_--;
                return idx;
            }
        }
    }
    // Second try: evict an evictable block (ref_count==0, hash!=0)
    int evicted = evict_one();
    if (evicted >= 0) {
        // evict_one already moved evictable→free (evictable_count_--, free_count_++)
        block_meta_[evicted].ref_count = 1;
        block_meta_[evicted].last_access = ++access_counter_;
        free_count_--;
        return evicted;
    }
    return -1;
}

void BlockPool::free(int block_id) {
    if (block_id < 0 || block_id >= num_blocks_) {
        throw std::out_of_range("[BlockPool] Invalid block_id: " + std::to_string(block_id));
    }
    auto &meta = block_meta_[block_id];
    // Update counters based on previous state
    if (meta.ref_count > 0) {
        // was used → becoming free
    } else if (meta.content_hash != 0) {
        // was evictable → becoming free
        evictable_count_--;
    } else {
        // already free — no counter change
        return;
    }
    free_count_++;
    meta.ref_count = 0;
    meta.content_hash = 0;
    meta.last_access = 0;
    meta.immutable = false;
}

void BlockPool::share(int block_id) {
    auto &meta = block_meta_[block_id];
    if (meta.ref_count == 0) {
        // Transitioning from free/evictable → used
        if (meta.content_hash != 0) evictable_count_--;
        else free_count_--;
    }
    meta.ref_count++;
    meta.last_access = ++access_counter_;
}

void BlockPool::release(int block_id) {
    if (block_id < 0 || block_id >= num_blocks_) return;
    auto &meta = block_meta_[block_id];
    if (meta.ref_count <= 0) return;
    meta.ref_count--;
    if (meta.ref_count == 0) {
        // Transitioning used → evictable or free
        if (meta.content_hash != 0) evictable_count_++;
        else free_count_++;
    }
}

int BlockPool::ref_count(int block_id) const {
    return block_meta_[block_id].ref_count;
}

void BlockPool::set_content_hash(int block_id, uint64_t hash) {
    auto &meta = block_meta_[block_id];
    if (meta.ref_count == 0) {
        bool was_evictable = (meta.content_hash != 0);
        bool will_be_evictable = (hash != 0);
        if (was_evictable && !will_be_evictable) { evictable_count_--; free_count_++; }
        else if (!was_evictable && will_be_evictable) { free_count_--; evictable_count_++; }
    }
    meta.content_hash = hash;
}

uint64_t BlockPool::content_hash(int block_id) const {
    return block_meta_[block_id].content_hash;
}

void BlockPool::set_immutable(int block_id, bool val) {
    block_meta_[block_id].immutable = val;
}

bool BlockPool::is_immutable(int block_id) const {
    return block_meta_[block_id].immutable;
}

void BlockPool::touch(int block_id) {
    block_meta_[block_id].last_access = ++access_counter_;
}

int BlockPool::evict_one() {
    if (evictable_count_ == 0) return -1;
    int best = -1;
    uint64_t oldest = UINT64_MAX;
    for (int i = 0; i < num_blocks_; ++i) {
        auto &meta = block_meta_[i];
        if (meta.ref_count == 0 && meta.content_hash != 0 && meta.last_access < oldest) {
            oldest = meta.last_access;
            best = i;
        }
    }
    if (best >= 0) {
        block_meta_[best].content_hash = 0;
        block_meta_[best].last_access = 0;
        block_meta_[best].immutable = false;
        evictable_count_--;
        free_count_++;
    }
    return best;
}

int BlockPool::evictable_count() const {
    return evictable_count_;
}

void *BlockPool::block_data(int block_id) const {
    if (block_id < 0 || block_id >= num_blocks_) {
        throw std::out_of_range("[BlockPool] Invalid block_id: " + std::to_string(block_id));
    }
    return static_cast<std::byte *>(pool_memory_) + static_cast<size_t>(block_id) * block_bytes_;
}

int BlockPool::free_blocks() const { return free_count_; }
int BlockPool::used_blocks() const { return num_blocks_ - free_count_ - evictable_count_; }
int BlockPool::available_blocks() const { return free_count_ + evictable_count_; }

size_t BlockPool::memory_usage() const {
    return static_cast<size_t>(num_blocks_) * block_bytes_;
}

// ============================================================================
// BlockAllocator
// ============================================================================

BlockAllocator::BlockAllocator(BlockPool &pool, int num_layers)
    : pool_(pool), num_layers_(num_layers) {}

SequenceBlockTable BlockAllocator::allocate_sequence(int estimated_tokens) {
    int blocks_per_layer = (estimated_tokens + pool_.config().block_size - 1) /
                           pool_.config().block_size;

    // Total blocks needed: blocks_per_layer * num_layers * 2 (K + V)
    int total_needed = blocks_per_layer * num_layers_ * 2;
    if (total_needed > pool_.available_blocks()) {
        throw std::runtime_error(
            "[BlockAllocator] Not enough blocks: need " + std::to_string(total_needed) +
            ", available " + std::to_string(pool_.free_blocks()));
    }

    SequenceBlockTable table;
    table.num_layers = num_layers_;
    table.seq_len = 0;
    table.k_blocks.resize(num_layers_);
    table.v_blocks.resize(num_layers_);

    for (int layer = 0; layer < num_layers_; ++layer) {
        table.k_blocks[layer].reserve(blocks_per_layer);
        table.v_blocks[layer].reserve(blocks_per_layer);
        for (int b = 0; b < blocks_per_layer; ++b) {
            int kid = pool_.allocate();
            int vid = pool_.allocate();
            if (kid < 0 || vid < 0) {
                // Rollback on failure
                free_sequence(table);
                throw std::runtime_error("[BlockAllocator] Pool exhausted during allocation");
            }
            table.k_blocks[layer].push_back(kid);
            table.v_blocks[layer].push_back(vid);
        }
    }

    return table;
}

int BlockAllocator::extend_sequence(SequenceBlockTable &table, int layer, bool is_k) {
    int block_id = pool_.allocate();
    if (block_id < 0) {
        throw std::runtime_error("[BlockAllocator] Pool exhausted during extend");
    }
    if (is_k) {
        table.k_blocks[layer].push_back(block_id);
    } else {
        table.v_blocks[layer].push_back(block_id);
    }
    return block_id;
}

void BlockAllocator::ensure_blocks(SequenceBlockTable &table, int needed_len) {
    int bs = pool_.config().block_size;
    int blocks_needed = (needed_len + bs - 1) / bs;
    for (int layer = 0; layer < table.num_layers; ++layer) {
        while (static_cast<int>(table.k_blocks[layer].size()) < blocks_needed)
            extend_sequence(table, layer, true);
        while (static_cast<int>(table.v_blocks[layer].size()) < blocks_needed)
            extend_sequence(table, layer, false);
    }
}

void BlockAllocator::free_sequence(SequenceBlockTable &table) {
    for (auto &layer_blocks : table.k_blocks) {
        for (int bid : layer_blocks) pool_.free(bid);
        layer_blocks.clear();
    }
    for (auto &layer_blocks : table.v_blocks) {
        for (int bid : layer_blocks) pool_.free(bid);
        layer_blocks.clear();
    }
    table.seq_len = 0;
}

void BlockAllocator::release_sequence(SequenceBlockTable &table) {
    for (auto &layer_blocks : table.k_blocks) {
        for (int bid : layer_blocks) pool_.release(bid);
        layer_blocks.clear();
    }
    for (auto &layer_blocks : table.v_blocks) {
        for (int bid : layer_blocks) pool_.release(bid);
        layer_blocks.clear();
    }
    table.seq_len = 0;
}

int BlockAllocator::available_blocks() const {
    return pool_.available_blocks();
}

int BlockAllocator::block_size() const {
    return pool_.config().block_size;
}

} // namespace zedinfer::kvcache
