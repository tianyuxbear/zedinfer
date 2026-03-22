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
      allocated_(num_blocks, false),
      free_count_(num_blocks) {

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
    if (free_count_ == 0) return -1;

    // Scan from hint position
    for (int i = 0; i < num_blocks_; ++i) {
        int idx = (next_free_ + i) % num_blocks_;
        if (!allocated_[idx]) {
            allocated_[idx] = true;
            free_count_--;
            next_free_ = (idx + 1) % num_blocks_;
            return idx;
        }
    }
    return -1; // should not reach if free_count_ > 0
}

void BlockPool::free(int block_id) {
    if (block_id < 0 || block_id >= num_blocks_) {
        throw std::out_of_range("[BlockPool] Invalid block_id: " + std::to_string(block_id));
    }
    if (!allocated_[block_id]) {
        throw std::logic_error("[BlockPool] Double free of block " + std::to_string(block_id));
    }
    allocated_[block_id] = false;
    free_count_++;
}

void *BlockPool::block_data(int block_id) const {
    if (block_id < 0 || block_id >= num_blocks_) {
        throw std::out_of_range("[BlockPool] Invalid block_id: " + std::to_string(block_id));
    }
    return static_cast<std::byte *>(pool_memory_) + static_cast<size_t>(block_id) * block_bytes_;
}

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
    if (total_needed > pool_.free_blocks()) {
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

void BlockAllocator::free_sequence(SequenceBlockTable &table) {
    for (auto &layer_blocks : table.k_blocks) {
        for (int bid : layer_blocks) {
            pool_.free(bid);
        }
        layer_blocks.clear();
    }
    for (auto &layer_blocks : table.v_blocks) {
        for (int bid : layer_blocks) {
            pool_.free(bid);
        }
        layer_blocks.clear();
    }
    table.seq_len = 0;
}

int BlockAllocator::available_blocks() const {
    return pool_.free_blocks();
}

int BlockAllocator::block_size() const {
    return pool_.config().block_size;
}

} // namespace zedinfer::kvcache
