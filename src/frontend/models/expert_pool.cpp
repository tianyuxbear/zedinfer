#include "frontend/models/expert_pool.hpp"

#include <plog/Log.h>
#include <stdexcept>

namespace zedinfer::model {

ExpertPool::ExpertPool(std::unique_ptr<ExpertWeights> experts, ExpertPoolConfig config)
    : experts_(std::move(experts)), config_(config) {
    if (!experts_) {
        throw std::runtime_error("ExpertPool: experts argument is null");
    }
    if (config_.strategy != ExpertPoolStrategy::ALL_GPU) {
        throw std::runtime_error("ExpertPool: only ALL_GPU strategy is implemented (M1). "
                                 "PINNED_LRU is planned for Phase 2 M2/M3.");
    }
    LOGI.printf("[ExpertPool] strategy=ALL_GPU, %zu layers × %zu experts, all resident on GPU",
                experts_->num_layers(), experts_->num_experts_per_layer());
}

ExpertPool::~ExpertPool() = default;

ExpertGpuHandle ExpertPool::ensure_on_gpu(int layer, int expert_id) {
    const auto& ffn = experts_->at(static_cast<size_t>(layer), static_cast<size_t>(expert_id));
    ExpertGpuHandle h;
    h.gate_packed = ffn.gate_packed;
    h.gate_scale = ffn.gate_scale;
    h.gate_g_idx = ffn.gate_g_idx;
    h.gate_weight = ffn.gate_weight;
    h.up_packed = ffn.up_packed;
    h.up_scale = ffn.up_scale;
    h.up_g_idx = ffn.up_g_idx;
    h.up_weight = ffn.up_weight;
    h.down_packed = ffn.down_packed;
    h.down_scale = ffn.down_scale;
    h.down_g_idx = ffn.down_g_idx;
    h.down_weight = ffn.down_weight;
    return h;
}

void ExpertPool::prefetch(int /*layer*/, int /*expert_id*/) {
    // M1: all experts permanently resident, nothing to do.
    // M3 will enqueue an async H2D on runtime.transfer_stream() and record a cudaEvent.
}

int ExpertPool::gpu_residents() const {
    // ALL_GPU: every (layer, expert_id) pair is resident.
    return static_cast<int>(experts_->num_layers() * experts_->num_experts_per_layer());
}

} // namespace zedinfer::model
