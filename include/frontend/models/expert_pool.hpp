#pragma once

#include "frontend/models/expert_weights.hpp"
#include "zedinfer.h"

#include <cstdint>
#include <memory>
#include <vector>

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
    //
    // Returns by const reference: under ALL_GPU the handle is a member of a permanent
    // per-(layer, expert) cache; under PINNED_LRU it points into the assigned slot's
    // persistent handle field. Either way the reference is stable until the pool is
    // destroyed (ALL_GPU) or until the same caller's NEXT ensure_on_gpu call evicts
    // the slot in the same layer (PINNED_LRU). The compute path uses each returned
    // handle immediately and discards it — never holds across another ensure_on_gpu —
    // so the latter window is safe in practice.
    const ExpertGpuHandle& ensure_on_gpu(int layer, int expert_id);

    // Hint: start bringing (layer, expert_id) to GPU without blocking. No-op in M1.
    // M3 will trigger an async H2D on the runtime's transfer stream.
    void prefetch(int layer, int expert_id);

    // Metadata-only access to an expert's canonical tensor handles. Unlike ensure_on_gpu,
    // peek_expert never triggers a transfer or updates LRU — under PINNED_LRU it returns
    // the CPU-pinned copies, under ALL_GPU it returns the same GPU tensors that
    // ensure_on_gpu would. Use for init-time shape detection and parameter counting, where
    // we need tensor shapes/numel without wanting to pin a GPU slot.
    const ExpertFFN& peek_expert(int layer, int expert_id) const;

    // Cumulative ensure_on_gpu statistics. Useful for gauging hit rate after a run.
    // Under ALL_GPU, every call counts as a hit.
    struct Stats {
        std::uint64_t hits = 0;
        std::uint64_t misses = 0;
    };

    // Diagnostics.
    int gpu_residents() const;
    Stats stats() const { return {hits_, misses_}; }
    size_t num_layers() const { return experts_->num_layers(); }
    size_t num_experts_per_layer() const { return experts_->num_experts_per_layer(); }
    ExpertPoolStrategy strategy() const { return config_.strategy; }
    // Maximum number of experts callers can usefully prefetch before compute consumes
    // them. Under ALL_GPU returns 0 (no-op prefetch). Under PINNED_LRU returns the
    // per-layer slot count.
    int max_prefetch_depth() const {
        return config_.strategy == ExpertPoolStrategy::PINNED_LRU ? config_.num_gpu_slots : 0;
    }

    // Emit cumulative hit/miss stats to file + console. Idempotent — only the first
    // call on a given pool instance actually logs. Safe to call from a destructor or
    // an explicit end-of-session hook, or both.
    void log_stats() const;

private:
    // Per-slot state used under PINNED_LRU. Unused (empty) under ALL_GPU.
    // A Slot owns persistent GPU tensors sized to hold any single expert's FFN; over
    // time it is re-populated via cudaMemcpy from the CPU-pinned canonical copy.
    struct Slot {
        int expert_id = -1;                    // -1 when slot is empty
        std::uint64_t last_access = 0;
        ExpertGpuHandle gpu_handle;            // persistent GPU tensors; data is overwritten on miss
        zedinferEvent_t ready_event = nullptr; // M3: set by transfer stream after async H2D
        // M3.5: false after a prefetch populates the slot; flipped to true the first time
        // ensure_on_gpu hands this slot back to compute. LRU victim search skips slots
        // that are populated but not yet compute_touched (in-flight prefetches that compute
        // still expects to see). Prevents the sliding-window prefetcher from stomping on
        // its own unconsumed data.
        bool compute_touched = false;
    };

    // Internal helpers shared by ensure_on_gpu (miss path) and prefetch.
    // Both pick an LRU slot and issue an async H2D on the transfer stream; they
    // differ only in whether compute stream waits on the resulting ready_event.
    //
    // pick_lru_slot returns the LRU slot among those safe to evict: empty slots or
    // slots whose current expert has already been handed to compute. Pending-prefetch
    // slots (populated, not yet compute_touched) are skipped. Returns -1 when every
    // slot is a pending prefetch — caller decides whether to skip (prefetch) or fall
    // back to any-LRU (miss path, must make forward progress).
    int pick_lru_slot(std::size_t layer) const;
    // Issue the async H2D and record the slot's ready_event. Caller updates
    // residency/LRU. Does NOT emit stream_wait_event on compute.
    void start_async_transfer(std::size_t layer, std::size_t expert_id, std::size_t slot_idx);

    std::unique_ptr<ExpertWeights> experts_;
    ExpertPoolConfig config_;

    // ALL_GPU state. Pre-built ExpertGpuHandle per (layer, expert) so ensure_on_gpu
    // can return a stable const reference instead of constructing+copying a handle
    // (12 shared_ptr atomic ops) on every dispatch_expert_linear call. Empty under
    // PINNED_LRU (handles live in each slot).
    std::vector<std::vector<ExpertGpuHandle>> cached_handles_;

    // PINNED_LRU state. Left empty under ALL_GPU.
    std::vector<std::vector<Slot>> slots_;      // slots_[L] = num_gpu_slots slots for layer L
    std::vector<std::vector<int>> residency_;   // residency_[L][E] = slot idx in slots_[L], or -1
    std::vector<std::uint64_t> access_counter_; // monotonic per-layer counter for LRU

    // Cumulative ensure_on_gpu stats. Mutated on the compute path; single-threaded use.
    std::uint64_t hits_ = 0;
    std::uint64_t misses_ = 0;
    mutable bool stats_logged_ = false; // mutated by const log_stats()

    // Global "compute has reached here" barrier. Recorded on the compute stream
    // right before a miss evicts a slot; the transfer stream waits on it before
    // overwriting that slot. Ensures async H2D never clobbers in-flight GEMMs.
    // One event is enough because we only need "some point past all prior compute";
    // conservative (transfer waits for all compute, not just slot-relevant), but
    // correct and cheap. Overlap between prefetched transfers and compute still
    // happens because the transfer stream processes multiple H2Ds in order.
    zedinferEvent_t compute_barrier_ = nullptr;
};

} // namespace zedinfer::model
