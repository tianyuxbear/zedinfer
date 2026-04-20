#include "frontend/models/expert_pool.hpp"

#include "backend/core/context/context.hpp"
#include "backend/device/runtime_api.hpp"
#include "backend/tensor/tensor.hpp"

#include <plog/Log.h>
#include <stdexcept>
#include <utility>

namespace zedinfer::model {

namespace {

// Allocate persistent GPU tensors sized to hold any single expert's FFN, based on a
// reference expert. Fields null in the template stay null in the slot (e.g. quantized
// models have no gate_weight; dense models have no gate_packed).
ExpertGpuHandle allocate_slot_tensors(const ExpertFFN& tmpl, zedinferDeviceType_t device_type, int device_id) {
    auto mk = [&](tensor_t ref) -> tensor_t {
        if (!ref) {
            return nullptr;
        }
        return Tensor::create(ref->shape(), ref->dtype(), device_type, device_id);
    };
    ExpertGpuHandle h;
    h.gate_packed = mk(tmpl.gate_packed);
    h.gate_scale = mk(tmpl.gate_scale);
    h.gate_g_idx = mk(tmpl.gate_g_idx);
    h.gate_weight = mk(tmpl.gate_weight);
    h.up_packed = mk(tmpl.up_packed);
    h.up_scale = mk(tmpl.up_scale);
    h.up_g_idx = mk(tmpl.up_g_idx);
    h.up_weight = mk(tmpl.up_weight);
    h.down_packed = mk(tmpl.down_packed);
    h.down_scale = mk(tmpl.down_scale);
    h.down_g_idx = mk(tmpl.down_g_idx);
    h.down_weight = mk(tmpl.down_weight);
    return h;
}

// D2H-copy every non-null tensor in an ExpertFFN from GPU into a freshly-allocated CPU
// pinned tensor (cudaMallocHost under the hood on NVIDIA). The original GPU tensor is
// released when its shared_ptr reference count drops to zero at reassignment, returning
// its storage to the pool. No-op for tensors already on CPU.
void migrate_expert_to_cpu(ExpertFFN& ffn, const ZedinferRuntimeAPI* api) {
    auto migrate = [&](tensor_t& t) {
        if (!t) {
            return;
        }
        if (t->deviceType() == ZEDINFER_DEVICE_CPU) {
            return;
        }
        auto cpu_copy = Tensor::create(t->shape(), t->dtype(), ZEDINFER_DEVICE_CPU, 0);
        api->memcpy_sync(cpu_copy->data(), t->data(), t->numel() * t->elementSize(), ZEDINFER_MEMCPY_D2H);
        t = std::move(cpu_copy);
    };
    migrate(ffn.gate_packed);
    migrate(ffn.gate_scale);
    migrate(ffn.gate_g_idx);
    migrate(ffn.gate_weight);
    migrate(ffn.up_packed);
    migrate(ffn.up_scale);
    migrate(ffn.up_g_idx);
    migrate(ffn.up_weight);
    migrate(ffn.down_packed);
    migrate(ffn.down_scale);
    migrate(ffn.down_g_idx);
    migrate(ffn.down_weight);
}

} // namespace

ExpertPool::ExpertPool(std::unique_ptr<ExpertWeights> experts, ExpertPoolConfig config)
    : experts_(std::move(experts)), config_(config) {
    if (!experts_) {
        throw std::runtime_error("ExpertPool: experts argument is null");
    }

    if (config_.strategy == ExpertPoolStrategy::ALL_GPU) {
        LOGI.printf("[ExpertPool] strategy=ALL_GPU, %zu layers × %zu experts, all resident on GPU",
                    experts_->num_layers(), experts_->num_experts_per_layer());
        return;
    }

    // PINNED_LRU: move all experts to CPU pinned memory, then allocate a per-layer GPU
    // slot arena. Slots will be populated on demand by ensure_on_gpu (Step 4).
    //
    // Peak VRAM during construction briefly holds the full expert set (as loaded by the
    // caller) plus the slot arena, because the pool does not immediately return freed
    // blocks to CUDA. This is fine for the INT4 Qwen3-30B-A3B validation on 24 GB; the
    // BF16 "doesn't-fit" scenario requires a streaming load path, which is out of scope
    // for M2.
    const size_t L = experts_->num_layers();
    const size_t E = experts_->num_experts_per_layer();

    if (config_.num_gpu_slots <= 0) {
        throw std::runtime_error("ExpertPool(PINNED_LRU): num_gpu_slots must be > 0");
    }
    if (static_cast<size_t>(config_.num_gpu_slots) > E) {
        throw std::runtime_error("ExpertPool(PINNED_LRU): num_gpu_slots exceeds num_experts_per_layer");
    }
    const int N = config_.num_gpu_slots;

    // Probe target GPU device from a non-null tensor of expert (0, 0).
    const ExpertFFN& probe = experts_->at(0, 0);
    tensor_t any = probe.gate_packed ? probe.gate_packed : probe.gate_weight;
    if (!any) {
        throw std::runtime_error("ExpertPool(PINNED_LRU): expert (0,0) has no gate tensor to probe device from");
    }
    const zedinferDeviceType_t gpu_type = any->deviceType();
    const int gpu_id = any->deviceId();
    if (gpu_type == ZEDINFER_DEVICE_CPU) {
        throw std::runtime_error("ExpertPool(PINNED_LRU): experts already on CPU; nothing to offload");
    }

    // Step A: D2H every expert into CPU pinned storage.
    auto* api = core::context().runtime().api();
    for (size_t l = 0; l < L; ++l) {
        for (size_t e = 0; e < E; ++e) {
            migrate_expert_to_cpu(experts_->at(l, e), api);
        }
    }
    LOGI.printf("[ExpertPool] D2H-migrated %zu experts to CPU pinned storage", L * E);

    // Step B: allocate per-layer GPU slot arena. Template shape taken from (now-CPU)
    // expert (0, 0) — all experts are homogeneous in shape for Qwen3 MoE.
    slots_.resize(L);
    residency_.assign(L, std::vector<int>(E, -1));
    access_counter_.assign(L, 0);
    const ExpertFFN& tmpl = experts_->at(0, 0);
    for (size_t l = 0; l < L; ++l) {
        slots_[l].resize(static_cast<size_t>(N));
        for (int s = 0; s < N; ++s) {
            slots_[l][s].gpu_handle = allocate_slot_tensors(tmpl, gpu_type, gpu_id);
        }
    }

    LOGI.printf("[ExpertPool] strategy=PINNED_LRU, %zu layers × %zu experts, %d GPU slots/layer", L, E, N);
}

ExpertPool::~ExpertPool() {
    // Fallback hook: emit stats here if no one called log_stats() explicitly. Today
    // the destructor does not run for pings due to a shared_ptr cycle in
    // InferenceEngine (see moe_session_handoff.md §3); callers should invoke
    // log_stats() explicitly. Kept here so the log still appears once that cycle
    // is fixed.
    log_stats();
}

void ExpertPool::log_stats() const {
    if (stats_logged_ || config_.strategy == ExpertPoolStrategy::ALL_GPU) {
        return;
    }
    stats_logged_ = true;
    const std::uint64_t total = hits_ + misses_;
    const double rate = (total > 0) ? (100.0 * static_cast<double>(hits_) / static_cast<double>(total)) : 0.0;
    LOGI.printf("[ExpertPool] stats: %llu hits, %llu misses (%.2f%% hit rate)",
                static_cast<unsigned long long>(hits_), static_cast<unsigned long long>(misses_), rate);
}

ExpertGpuHandle ExpertPool::ensure_on_gpu(int layer, int expert_id) {
    if (config_.strategy == ExpertPoolStrategy::ALL_GPU) {
        ++hits_;
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

    // PINNED_LRU: per-layer slot arena with LRU eviction + synchronous H2D on miss.
    const size_t L = static_cast<size_t>(layer);
    const size_t E = static_cast<size_t>(expert_id);
    auto& residents = residency_[L];
    auto& layer_slots = slots_[L];
    auto& counter = access_counter_[L];

    const int cached = residents[E];
    if (cached >= 0) {
        // Hit: refresh LRU timestamp and return the existing handle.
        ++hits_;
        ++counter;
        layer_slots[cached].last_access = counter;
        return layer_slots[cached].gpu_handle;
    }
    ++misses_;

    // Miss: pick the slot with the smallest last_access. Empty slots (last_access == 0)
    // are naturally preferred over populated ones.
    size_t victim = 0;
    for (size_t s = 1; s < layer_slots.size(); ++s) {
        if (layer_slots[s].last_access < layer_slots[victim].last_access) {
            victim = s;
        }
    }

    if (layer_slots[victim].expert_id >= 0) {
        // Evicting a populated slot: invalidate its residency entry before overwrite.
        residents[static_cast<size_t>(layer_slots[victim].expert_id)] = -1;
    }

    const ExpertFFN& src = experts_->at(L, E);
    auto& dst = layer_slots[victim].gpu_handle;
    auto* api = core::context().runtime().api();
    auto copy = [&](tensor_t s, tensor_t d) {
        if (!s || !d) {
            return;
        }
        api->memcpy_sync(d->data(), s->data(), s->numel() * s->elementSize(), ZEDINFER_MEMCPY_H2D);
    };
    copy(src.gate_packed, dst.gate_packed);
    copy(src.gate_scale, dst.gate_scale);
    copy(src.gate_g_idx, dst.gate_g_idx);
    copy(src.gate_weight, dst.gate_weight);
    copy(src.up_packed, dst.up_packed);
    copy(src.up_scale, dst.up_scale);
    copy(src.up_g_idx, dst.up_g_idx);
    copy(src.up_weight, dst.up_weight);
    copy(src.down_packed, dst.down_packed);
    copy(src.down_scale, dst.down_scale);
    copy(src.down_g_idx, dst.down_g_idx);
    copy(src.down_weight, dst.down_weight);

    layer_slots[victim].expert_id = expert_id;
    ++counter;
    layer_slots[victim].last_access = counter;
    residents[E] = static_cast<int>(victim);

    return dst;
}

void ExpertPool::prefetch(int /*layer*/, int /*expert_id*/) {
    // M1: all experts permanently resident, nothing to do.
    // M3 will enqueue an async H2D on runtime.transfer_stream() and record a cudaEvent.
}

const ExpertFFN& ExpertPool::peek_expert(int layer, int expert_id) const {
    return experts_->at(static_cast<size_t>(layer), static_cast<size_t>(expert_id));
}

int ExpertPool::gpu_residents() const {
    if (config_.strategy == ExpertPoolStrategy::ALL_GPU) {
        return static_cast<int>(experts_->num_layers() * experts_->num_experts_per_layer());
    }
    int count = 0;
    for (const auto& layer_slots : slots_) {
        for (const auto& slot : layer_slots) {
            if (slot.expert_id >= 0) {
                ++count;
            }
        }
    }
    return count;
}

} // namespace zedinfer::model
