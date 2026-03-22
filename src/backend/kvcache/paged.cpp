#include "backend/kvcache/paged.hpp"
#include "backend/core/context/context.hpp"
#include "backend/device/runtime_api.hpp"
#include "utils/types.hpp"

#include <algorithm>
#include <cstring>
#include <plog/Log.h>
#include <stdexcept>

namespace zedinfer::kvcache {

PagedKVCache::PagedKVCache(const KVCacheConfig &config, BlockAllocator &allocator,
                           int initial_estimated_tokens)
    : KVCache(config), allocator_(allocator) {

    block_table_ = allocator_.allocate_sequence(initial_estimated_tokens);

    write_k_bufs_.resize(config_.num_layers);
    write_v_bufs_.resize(config_.num_layers);
    gather_k_bufs_.resize(config_.num_layers);
    gather_v_bufs_.resize(config_.num_layers);

    LOGI << "[PagedKVCache] Initialized with "
         << block_table_.k_blocks[0].size() << " blocks/layer, block_size="
         << allocator_.block_size();
}

PagedKVCache::~PagedKVCache() {
    allocator_.free_sequence(block_table_);
}

int PagedKVCache::allocated_capacity() const {
    if (block_table_.k_blocks.empty()) return 0;
    return static_cast<int>(block_table_.k_blocks[0].size()) * allocator_.block_size();
}

// --- Write path ---

void PagedKVCache::ensure_write_capacity(int seq_len) {
    if (seq_len <= write_buf_capacity_) return;

    int new_cap = std::max(seq_len, 512); // at least 512 for prefill
    std::vector<size_t> shape = {
        static_cast<size_t>(new_cap),
        static_cast<size_t>(config_.num_kv_heads),
        static_cast<size_t>(config_.head_dim)};

    for (int layer = 0; layer < config_.num_layers; ++layer) {
        write_k_bufs_[layer] = Tensor::create(shape, config_.dtype,
                                               config_.device_type, config_.device_id);
        write_v_bufs_[layer] = Tensor::create(shape, config_.dtype,
                                               config_.device_type, config_.device_id);
    }
    write_buf_capacity_ = new_cap;
}

tensor_t PagedKVCache::get_k_cache_slice(int layer_idx, int past_len, int seq_len) {
    validate_layer_idx(layer_idx);
    ensure_write_capacity(seq_len);

    // Track pending write for scatter in update_seq_len()
    if (pending_write_past_len_ < 0) {
        pending_write_past_len_ = past_len;
    }
    pending_write_seq_len_ = seq_len;

    return write_k_bufs_[layer_idx]->slice(0, 0, seq_len);
}

tensor_t PagedKVCache::get_v_cache_slice(int layer_idx, int past_len, int seq_len) {
    validate_layer_idx(layer_idx);
    ensure_write_capacity(seq_len);

    if (pending_write_past_len_ < 0) {
        pending_write_past_len_ = past_len;
    }
    pending_write_seq_len_ = seq_len;

    return write_v_bufs_[layer_idx]->slice(0, 0, seq_len);
}

// --- Read path (contiguous gather) ---

void PagedKVCache::ensure_gather_capacity(int total_len) {
    if (total_len <= gather_capacity_) return;

    int new_cap = std::max(total_len, 256);
    // Round up to next multiple of block_size for clean alignment
    int bs = allocator_.block_size();
    new_cap = ((new_cap + bs - 1) / bs) * bs;

    std::vector<size_t> shape = {
        static_cast<size_t>(new_cap),
        static_cast<size_t>(config_.num_kv_heads),
        static_cast<size_t>(config_.head_dim)};

    for (int layer = 0; layer < config_.num_layers; ++layer) {
        gather_k_bufs_[layer] = Tensor::create(shape, config_.dtype,
                                                config_.device_type, config_.device_id);
        gather_v_bufs_[layer] = Tensor::create(shape, config_.dtype,
                                                config_.device_type, config_.device_id);
    }
    gather_capacity_ = new_cap;
}

tensor_t PagedKVCache::gather_contiguous(int layer_idx, int total_len, bool is_k) {
    ensure_gather_capacity(total_len);

    auto &blocks = is_k ? block_table_.k_blocks[layer_idx]
                        : block_table_.v_blocks[layer_idx];
    auto &buf = is_k ? gather_k_bufs_[layer_idx] : gather_v_bufs_[layer_idx];

    const int bs = allocator_.block_size();
    const size_t token_bytes = allocator_.pool().config().token_bytes();
    const auto &pool = allocator_.pool();

    // 1. Gather committed tokens from blocks (up to current_length_)
    int committed_len = std::min(total_len, current_length_);
    int tokens_remaining = committed_len;
    size_t dst_offset = 0;

    for (size_t b = 0; b < blocks.size() && tokens_remaining > 0; ++b) {
        int tokens_in_block = std::min(bs, tokens_remaining);
        size_t copy_bytes = tokens_in_block * token_bytes;

        const void *src = pool.block_data(blocks[b]);
        void *dst = static_cast<std::byte *>(buf->data()) + dst_offset;
        copy_region(dst, src, copy_bytes);

        dst_offset += copy_bytes;
        tokens_remaining -= tokens_in_block;
    }

    // 2. Append pending write buffer tokens (not yet scattered to blocks)
    int pending_tokens = total_len - committed_len;
    if (pending_tokens > 0 && pending_write_seq_len_ > 0) {
        auto &write_buf = is_k ? write_k_bufs_[layer_idx] : write_v_bufs_[layer_idx];
        size_t copy_bytes = pending_tokens * token_bytes;
        void *dst = static_cast<std::byte *>(buf->data()) + dst_offset;
        const void *src = write_buf->data();
        copy_region(dst, src, copy_bytes);
    }

    return buf->slice(0, 0, total_len);
}

tensor_t PagedKVCache::get_k_cache_slice(int layer_idx, int total_len) {
    validate_layer_idx(layer_idx);
    return gather_contiguous(layer_idx, total_len, true);
}

tensor_t PagedKVCache::get_v_cache_slice(int layer_idx, int total_len) {
    validate_layer_idx(layer_idx);
    return gather_contiguous(layer_idx, total_len, false);
}

tensor_t PagedKVCache::get_k_cache(int layer_idx) {
    validate_layer_idx(layer_idx);
    if (current_length_ == 0) return gather_k_bufs_[layer_idx];
    return gather_contiguous(layer_idx, current_length_, true);
}

tensor_t PagedKVCache::get_v_cache(int layer_idx) {
    validate_layer_idx(layer_idx);
    if (current_length_ == 0) return gather_v_bufs_[layer_idx];
    return gather_contiguous(layer_idx, current_length_, false);
}

// --- Scatter write buffer into blocks ---

void PagedKVCache::ensure_blocks_for_tokens(int total_tokens) {
    int bs = allocator_.block_size();
    int blocks_needed = (total_tokens + bs - 1) / bs;

    for (int layer = 0; layer < config_.num_layers; ++layer) {
        while (static_cast<int>(block_table_.k_blocks[layer].size()) < blocks_needed) {
            allocator_.extend_sequence(block_table_, layer, true);
        }
        while (static_cast<int>(block_table_.v_blocks[layer].size()) < blocks_needed) {
            allocator_.extend_sequence(block_table_, layer, false);
        }
    }
}

void PagedKVCache::scatter_to_blocks(int past_len, int new_tokens) {
    const int bs = allocator_.block_size();
    const size_t token_bytes = allocator_.pool().config().token_bytes();
    const auto &pool = allocator_.pool();

    for (int layer = 0; layer < config_.num_layers; ++layer) {
        auto *k_src = static_cast<const std::byte *>(write_k_bufs_[layer]->data());
        auto *v_src = static_cast<const std::byte *>(write_v_bufs_[layer]->data());

        for (int t = 0; t < new_tokens; ++t) {
            int global_pos = past_len + t;
            int block_idx = global_pos / bs;
            int offset_in_block = global_pos % bs;

            // K
            {
                void *dst = static_cast<std::byte *>(
                    pool.block_data(block_table_.k_blocks[layer][block_idx]))
                    + offset_in_block * token_bytes;
                copy_region(dst, k_src + t * token_bytes, token_bytes);
            }
            // V
            {
                void *dst = static_cast<std::byte *>(
                    pool.block_data(block_table_.v_blocks[layer][block_idx]))
                    + offset_in_block * token_bytes;
                copy_region(dst, v_src + t * token_bytes, token_bytes);
            }
        }
    }
}

void PagedKVCache::update_seq_len(int new_tokens) {
    if (new_tokens <= 0) return;

    int total_after = current_length_ + new_tokens;

    // Ensure we have enough blocks for the new total
    ensure_blocks_for_tokens(total_after);

    // Scatter write buffer data into blocks
    int past_len = (pending_write_past_len_ >= 0) ? pending_write_past_len_ : current_length_;
    scatter_to_blocks(past_len, new_tokens);

    // Update state
    current_length_ += new_tokens;
    block_table_.seq_len = current_length_;
    pending_write_past_len_ = -1;
    pending_write_seq_len_ = 0;
}

// --- Lifecycle ---

void PagedKVCache::reset() {
    allocator_.free_sequence(block_table_);
    current_length_ = 0;
    pending_write_past_len_ = -1;
    pending_write_seq_len_ = 0;

    // Re-allocate minimal blocks
    block_table_ = allocator_.allocate_sequence(256);
}

size_t PagedKVCache::memory_usage() const {
    // Count blocks actually allocated to this sequence
    size_t total_blocks = 0;
    for (const auto &layer_blocks : block_table_.k_blocks) {
        total_blocks += layer_blocks.size();
    }
    for (const auto &layer_blocks : block_table_.v_blocks) {
        total_blocks += layer_blocks.size();
    }
    return total_blocks * allocator_.pool().config().block_bytes();
}

float PagedKVCache::utilization() const {
    int capacity = allocated_capacity();
    if (capacity == 0) return 0.0f;
    return static_cast<float>(current_length_) / capacity;
}

void PagedKVCache::copy_region(void *dst, const void *src, size_t bytes) {
    if (config_.device_type == ZEDINFER_DEVICE_CPU) {
        std::memcpy(dst, src, bytes);
    } else {
        core::context().setDevice(config_.device_type, config_.device_id);
        core::context().runtime().api()->memcpy_sync(
            dst, src, bytes, ZEDINFER_MEMCPY_D2D);
    }
}

} // namespace zedinfer::kvcache
