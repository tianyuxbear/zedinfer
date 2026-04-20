#pragma once

#include "frontend/models/expert_weights.hpp"

#include <memory>

namespace zedinfer::model {

// Handle to an expert's FFN weights known to be on GPU (or on the target compute device).
// Same shape as ExpertFFN — the distinction is semantic: an ExpertGpuHandle is the result
// of ExpertPool::ensure_on_gpu and is guaranteed valid for compute at the point of return.
struct ExpertGpuHandle {
    // Quantized path
    tensor_t gate_packed = nullptr;
    tensor_t gate_scale = nullptr;
    tensor_t gate_g_idx = nullptr;

    tensor_t up_packed = nullptr;
    tensor_t up_scale = nullptr;
    tensor_t up_g_idx = nullptr;

    tensor_t down_packed = nullptr;
    tensor_t down_scale = nullptr;
    tensor_t down_g_idx = nullptr;

    // Dense path (used when the model is not quantized)
    tensor_t gate_weight = nullptr;
    tensor_t up_weight = nullptr;
    tensor_t down_weight = nullptr;

    bool is_quantized() const { return gate_packed != nullptr; }
};

enum class ExpertPoolStrategy {
    // Every expert permanently resident on GPU (current default, Phase 1 behavior).
    ALL_GPU,
    // Reserved for Phase 2 M2/M3: slot arena + CPU pinned storage + LRU.
    PINNED_LRU,
};

struct ExpertPoolConfig {
    ExpertPoolStrategy strategy = ExpertPoolStrategy::ALL_GPU;
    // Number of GPU slots when strategy == PINNED_LRU. -1 = auto-size from VRAM budget.
    // Ignored for ALL_GPU.
    int num_gpu_slots = -1;
};

// Manages GPU residency of expert weights for MoE models.
//
// M1 (current): ALL_GPU strategy only — every expert permanently resident; ensure_on_gpu
// is a trivial handle lookup. ExpertPool acts as a future-proof indirection layer.
//
// M2/M3 (planned): PINNED_LRU — a fixed-size arena of GPU slots is populated on demand
// from CPU pinned memory. ensure_on_gpu may trigger synchronous transfer (M2) or
// wait on an async prefetch event (M3).
class ExpertPool {
public:
    // Takes ownership of pre-loaded expert weights. For ALL_GPU strategy, tensors inside
    // `experts` are assumed to already live on the target compute device.
    ExpertPool(std::unique_ptr<ExpertWeights> experts, ExpertPoolConfig config);
    ~ExpertPool();

    ExpertPool(const ExpertPool&) = delete;
    ExpertPool& operator=(const ExpertPool&) = delete;

    // Return a GPU-ready handle for (layer, expert_id). M1 does no transfer. M2 will issue
    // a synchronous H2D if the expert isn't currently resident. M3 will wait on a
    // per-slot cudaEvent populated by the transfer stream.
    ExpertGpuHandle ensure_on_gpu(int layer, int expert_id);

    // Hint: start bringing (layer, expert_id) to GPU without blocking. No-op in M1.
    // M3 will trigger an async H2D on the runtime's transfer stream.
    void prefetch(int layer, int expert_id);

    // Diagnostics.
    int gpu_residents() const;
    size_t num_layers() const { return experts_->num_layers(); }
    size_t num_experts_per_layer() const { return experts_->num_experts_per_layer(); }
    ExpertPoolStrategy strategy() const { return config_.strategy; }

private:
    std::unique_ptr<ExpertWeights> experts_;
    ExpertPoolConfig config_;
};

} // namespace zedinfer::model
