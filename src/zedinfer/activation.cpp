#include "zedinfer/activation.hpp"
#include "backend/core/context/context.hpp"
#include "backend/tensor/tensor.hpp"
#include "utils/check.hpp"
#include "utils/types.hpp"

#include <cstddef>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace zedinfer {

// ============================================================================
// PositionIDsCache - Pre-computed position indices for attention
// ============================================================================

PositionIDsCache::PositionIDsCache(const ExecutorConfig &config)
    : config_(config) {
    initialize_cache();
}

void PositionIDsCache::initialize_cache() {
    // Create tensor [0, 1, 2, ..., max_seq_len-1]
    cache_ = Tensor::create(
        {static_cast<size_t>(config_.max_seq_len)},
        ZEDINFER_DTYPE_I64,
        config_.device_type,
        config_.device_id,
        false,
        nullptr);

    std::vector<int64_t> positions(config_.max_seq_len);
    for (size_t i = 0; i < config_.max_seq_len; ++i) {
        positions[i] = i;
    }

    cache_->load(positions.data());
}

tensor_t PositionIDsCache::get_slice(int start_pos, int seq_len) {
    if (start_pos < 0 || seq_len < 0) {
        throw std::invalid_argument("start_pos and seq_len must be non-negative");
    }

    if (start_pos + seq_len > (int)config_.max_seq_len) {
        throw std::out_of_range(
            "Position range [" + std::to_string(start_pos) + ", " + std::to_string(start_pos + seq_len) + ") exceeds cache size " + std::to_string(config_.max_seq_len));
    }

    // Zero-copy slice: returns view into cache
    return cache_->slice(0, start_pos, start_pos + seq_len);
}

// ============================================================================
// PrefillArena - Sequential allocator for prefill phase
// ============================================================================

PrefillArena::PrefillArena(const ExecutorConfig &config, size_t capacity)
    : ActivationAllocator(config),
      capacity_(capacity),
      current_offset_(0),
      peak_usage_(0) {
    // Allocate backing storage based on device type
    if (config_.device_type == ZEDINFER_DEVICE_CPU && core::context().runtime().deviceType() != ZEDINFER_DEVICE_CPU) {
        storage_ = core::context().runtime().allocateHostStorage(capacity);
    } else {
        core::context().setDevice(config_.device_type, config_.device_id);
        storage_ = core::context().runtime().allocateDeviceStorage(capacity);
    }
}

tensor_t PrefillArena::acquire(const std::vector<size_t> &shape) {
    size_t numel = std::accumulate(shape.begin(), shape.end(),
                                   static_cast<size_t>(1),
                                   std::multiplies<size_t>{});
    size_t nbytes = numel * utils::dsize(config_.data_type);

    // Check if allocation fits in remaining space
    if (current_offset_ + nbytes > capacity_) {
        throw std::runtime_error(
            "PrefillArena OOM: need " + std::to_string(nbytes) + " bytes, available " + std::to_string(capacity_ - current_offset_));
    }

    // Bump pointer allocation
    auto ptr = storage_->memory() + current_offset_;
    auto tensor = Tensor::create(
        shape,
        config_.data_type,
        config_.device_type,
        config_.device_id,
        true, // View into arena storage
        ptr);

    allocated_tensors_.push_back(tensor);
    current_offset_ += nbytes;
    peak_usage_ = std::max(peak_usage_, current_offset_);

    return tensor;
}

void PrefillArena::release(tensor_t) {
    ASSERT(false,
           "PrefillArena does not support individual release. "
           "Use reset() to reclaim all memory after prefill completes.");
}

void PrefillArena::reset() {
    current_offset_ = 0;
    allocated_tensors_.clear();
    // Note: peak_usage_ preserved for statistics
}

std::string PrefillArena::get_stats() const {
    std::ostringstream oss;
    const double mb_scale = 1024.0 * 1024.0;

    oss << "\n=== PrefillArena Statistics: ===\n";
    oss << "  Capacity:      " << capacity_ << " bytes ("
        << std::fixed << std::setprecision(2) << (capacity_ / mb_scale) << " MB)\n";
    oss << "  Current Usage: " << current_offset_ << " bytes ("
        << std::fixed << std::setprecision(2) << (current_offset_ / mb_scale) << " MB, "
        << std::fixed << std::setprecision(2) << (100.0 * current_offset_ / capacity_) << "%)\n";
    oss << "  Peak Usage:    " << peak_usage_ << " bytes ("
        << std::fixed << std::setprecision(2) << (peak_usage_ / mb_scale) << " MB, "
        << std::fixed << std::setprecision(2) << (100.0 * peak_usage_ / capacity_) << "%)\n";
    oss << "  Active Tensors: " << allocated_tensors_.size() << "\n";

    return oss.str();
}

// ============================================================================
// DecodePool - Reusable tensor pool for decode phase
// ============================================================================

DecodePool::DecodePool(const ExecutorConfig &config)
    : ActivationAllocator(config) {
    pool_.reserve(128); // Preallocate for common pool size
}

tensor_t DecodePool::acquire(const std::vector<size_t> &shape) {
    // Try to find available tensor with matching shape
    for (auto &slot : pool_) {
        if (!slot.in_use && slot.shape == shape) {
            slot.in_use = true;
            return slot.tensor;
        }
    }

    // No match found - allocate new tensor and add to pool
    auto tensor = Tensor::create(
        shape,
        config_.data_type,
        config_.device_type,
        config_.device_id,
        false,
        nullptr);

    pool_.push_back({tensor, shape, true});
    return tensor;
}

void DecodePool::release(tensor_t tensor) {
    for (auto &slot : pool_) {
        if (slot.tensor == tensor) {
            slot.in_use = false;
            return;
        }
    }
    ASSERT(false, "DecodePool::release: tensor not found in pool. "
                  "Only tensors acquired from this pool can be released.");
}

void DecodePool::reset() {
    for (auto &slot : pool_) {
        slot.in_use = false;
    }
}

size_t DecodePool::tensors_in_use() const {
    size_t count = 0;
    for (const auto &slot : pool_) {
        if (slot.in_use) {
            ++count;
        }
    }
    return count;
}

size_t DecodePool::total_memory() const {
    size_t total = 0;
    for (const auto &slot : pool_) {
        total += slot.tensor->numel() * utils::dsize(config_.data_type);
    }
    return total;
}

std::string DecodePool::get_stats() const {
    std::ostringstream oss;
    const double mb_scale = 1024.0 * 1024.0;

    oss << "\n=== DecodePool Statistics ===\n";
    oss << "  Total tensors: " << pool_.size() << std::endl;
    oss << "  In use: " << tensors_in_use() << std::endl;
    oss << "  Total memory: " << std::fixed << std::setprecision(2)
        << (total_memory() / mb_scale) << " MB" << std::endl;

    // Show shape distribution
    std::map<std::vector<size_t>, int> shape_dist;
    for (const auto &slot : pool_) {
        ++shape_dist[slot.shape];
    }

    if (!shape_dist.empty()) {
        oss << "  Shape distribution:" << std::endl;
        for (const auto &[shape, count] : shape_dist) {
            oss << "    [";
            for (size_t i = 0; i < shape.size(); ++i) {
                oss << shape[i];
                if (i + 1 < shape.size()) {
                    oss << ", ";
                }
            }
            oss << "]: " << count << " tensor(s)" << std::endl;
        }
    }

    return oss.str();
}

} // namespace zedinfer