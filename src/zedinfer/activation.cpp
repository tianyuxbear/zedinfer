#include "zedinfer/activation.hpp"
#include "backend/tensor/tensor.hpp"

#include <cstddef>
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

} // namespace zedinfer