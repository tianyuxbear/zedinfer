#pragma once

#include "backend/core/storage/storage.hpp" // IWYU pragma: keep
#include "backend/tensor/tensor.hpp"
#include "neollm.h"

#include <cstddef>

namespace neollm {

/**
 * Configuration for model execution environment.
 */
struct ExecutorConfig {
    NeollmDeviceType_t device_type;
    int device_id;
    NeollmDataType_t data_type; // Default dtype for activations
    size_t max_prefill_len;
    size_t max_seq_len;

    ExecutorConfig(NeollmDeviceType_t device_type = NEOLLM_DEVICE_CPU,
                   int device_id = 0,
                   NeollmDataType_t data_type = NEOLLM_DTYPE_F32,
                   size_t max_prefill_len = 128,
                   size_t max_seq_len = 16384)
        : device_type(device_type),
          device_id(device_id),
          data_type(data_type),
          max_prefill_len(max_prefill_len),
          max_seq_len(max_seq_len) {}
};

/**
 * Pre-allocated position IDs cache to avoid repeated allocation.
 */
class PositionIDsCache {
public:
    PositionIDsCache(const ExecutorConfig &config);

    /**
     * Get position_ids slice [start_pos, start_pos+1, ..., start_pos+seq_len-1].
     * @return Tensor of shape [seq_len]
     */
    tensor_t get_slice(int start_pos, int seq_len);

    int max_seq_len() const { return config_.max_seq_len; }

private:
    ExecutorConfig config_;
    tensor_t cache_; // Pre-allocated [0, 1, 2, ..., max_seq_len-1]

    void initialize_cache();
};

/**
 * Base class for activation tensor memory allocation strategies.
 */
class ActivationAllocator {
public:
    explicit ActivationAllocator(const ExecutorConfig &config)
        : config_(config) {}
    virtual ~ActivationAllocator() = default;

    // Core allocation interface
    virtual tensor_t acquire(const std::vector<size_t> &shape) = 0;
    virtual void release(tensor_t tensor) = 0;
    virtual void reset() = 0;

    // Statistics interface
    virtual size_t total_memory() const = 0;
    virtual std::string get_stats() const = 0;

protected:
    ExecutorConfig config_;
};

// ============================================================================
// Arena allocator for prefill phase - sequential allocation, bulk release
// ============================================================================
class PrefillArena : public ActivationAllocator {
public:
    explicit PrefillArena(const ExecutorConfig &config, size_t capacity);

    // Disable copy and move semantics
    PrefillArena(const PrefillArena &) = delete;
    PrefillArena &operator=(const PrefillArena &) = delete;
    PrefillArena(PrefillArena &&) = delete;
    PrefillArena &operator=(PrefillArena &&) = delete;

    // ActivationAllocator interface
    tensor_t acquire(const std::vector<size_t> &shape) override;
    void release(tensor_t tensor) override; // No-op for arena
    void reset() override;

    // Statistics interface
    size_t total_memory() const override { return storage_->size(); }
    size_t current_usage() const { return current_offset_; }
    size_t peak_usage() const { return peak_usage_; }
    std::string get_stats() const override;

private:
    core::storage_t storage_;
    size_t capacity_;
    size_t current_offset_;
    size_t peak_usage_;
    std::vector<tensor_t> allocated_tensors_; // Track tensors for cleanup
};

// ============================================================================
// Pool allocator for decode phase - reuse tensors across iterations
// ============================================================================
class DecodePool : public ActivationAllocator {
public:
    explicit DecodePool(const ExecutorConfig &config);

    // Disable copy and move semantics
    DecodePool(const DecodePool &) = delete;
    DecodePool &operator=(const DecodePool &) = delete;
    DecodePool(DecodePool &&) = delete;
    DecodePool &operator=(DecodePool &&) = delete;

    // ActivationAllocator interface
    tensor_t acquire(const std::vector<size_t> &shape) override;
    void release(tensor_t tensor) override; // Mark available for reuse
    void reset() override;

    // Statistics interface
    size_t total_tensors() const { return pool_.size(); }
    size_t tensors_in_use() const;
    size_t total_memory() const override;
    std::string get_stats() const override;

private:
    struct TensorSlot {
        tensor_t tensor;
        std::vector<size_t> shape;
        bool in_use;
    };

    std::vector<TensorSlot> pool_;
};

} // namespace neollm