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
        // Pre-build per-(layer, expert) ExpertGpuHandle from the ExpertFFN tensors so
        // ensure_on_gpu can hand back a const reference. Avoids constructing+copying a
        // handle (12 shared_ptr atomic ops) on every dispatch_expert_linear call.
        const size_t L = experts_->num_layers();
        const size_t E = experts_->num_experts_per_layer();
        cached_handles_.resize(L);
        for (size_t l = 0; l < L; ++l) {
            cached_handles_[l].resize(E);
            for (size_t e = 0; e < E; ++e) {
                const auto& ffn = experts_->at(l, e);
                auto& h = cached_handles_[l][e];
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
            }
        }
        LOGI.printf("[ExpertPool] strategy=ALL_GPU, %zu layers × %zu experts, all resident on GPU", L, E);
        return;
    }

    // PINNED_LRU: experts live in CPU pinned memory; a per-layer GPU slot arena is
    // populated on demand by ensure_on_gpu (or prefetch). Experts can arrive on GPU
    // (legacy path, INT4 small-model case) or on CPU pinned already (D.2 loader
    // routing, BF16 large-model case); the first branch below handles the former by
    // D2H-migrating in place.
    const size_t L = experts_->num_layers();
    const size_t E = experts_->num_experts_per_layer();

    if (config_.num_gpu_slots <= 0) {
        throw std::runtime_error("ExpertPool(PINNED_LRU): num_gpu_slots must be > 0");
    }
    if (static_cast<size_t>(config_.num_gpu_slots) > E) {
        throw std::runtime_error("ExpertPool(PINNED_LRU): num_gpu_slots exceeds num_experts_per_layer");
    }
    const int N = config_.num_gpu_slots;

    // Target GPU comes from the current runtime (where compute and slot arena live).
    auto& runtime = core::context().runtime();
    const zedinferDeviceType_t gpu_type = runtime.deviceType();
    const int gpu_id = runtime.deviceId();
    if (gpu_type == ZEDINFER_DEVICE_CPU) {
        throw std::runtime_error("ExpertPool(PINNED_LRU): runtime is CPU; PINNED_LRU requires a GPU runtime");
    }
    auto* api = runtime.api();

    // Probe where experts arrived from the loader. Two cases:
    //   (1) Experts loaded straight to CPU pinned (loader predicate, D.1/D.2 path) —
    //       nothing to migrate; slot arena gets allocated against the current runtime.
    //   (2) Experts loaded to GPU (legacy flow, small INT4 models) — D2H-migrate each
    //       one into CPU pinned storage and release the GPU originals to the pool.
    const ExpertFFN& probe = experts_->at(0, 0);
    tensor_t any = probe.gate_packed ? probe.gate_packed : probe.gate_weight;
    if (!any) {
        throw std::runtime_error("ExpertPool(PINNED_LRU): expert (0,0) has no gate tensor to probe device from");
    }
    const bool experts_on_gpu = (any->deviceType() != ZEDINFER_DEVICE_CPU);

    if (experts_on_gpu) {
        for (size_t l = 0; l < L; ++l) {
            for (size_t e = 0; e < E; ++e) { migrate_expert_to_cpu(experts_->at(l, e), api); }
        }
        LOGI.printf("[ExpertPool] D2H-migrated %zu experts to CPU pinned storage", L * E);
    } else {
        LOGI.printf("[ExpertPool] %zu experts already in CPU pinned memory (loader-routed); skipping D2H", L * E);
    }

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
            // Per-slot event used by M3 to sync async H2D with compute. Created here
            // so the cost is paid once at init, not per transfer. No-op on CPU backend.
            slots_[l][s].ready_event = api->create_event();
        }
    }
    // Single compute-stream barrier event; see field comment for rationale.
    compute_barrier_ = api->create_event();

    LOGI.printf("[ExpertPool] strategy=PINNED_LRU, %zu layers × %zu experts, %d GPU slots/layer", L, E, N);
}

ExpertPool::~ExpertPool() {
    // Emit stats first while the pool's fields are still valid (log_stats is const
    // and only reads counters; safe either way but avoids ambiguity).
    log_stats();

    // Release events created in the PINNED_LRU ctor path. No-op on ALL_GPU
    // (slots_ is empty, compute_barrier_ is null) and on CPU backend (all events nullptr).
    auto* api = core::context().runtime().api();
    for (auto& layer_slots : slots_) {
        for (auto& slot : layer_slots) {
            if (slot.ready_event) {
                api->destroy_event(slot.ready_event);
                slot.ready_event = nullptr;
            }
        }
    }
    if (compute_barrier_) {
        api->destroy_event(compute_barrier_);
        compute_barrier_ = nullptr;
    }
}

void ExpertPool::log_stats() const {
    if (stats_logged_ || config_.strategy == ExpertPoolStrategy::ALL_GPU) {
        return;
    }
    stats_logged_ = true;
    const std::uint64_t total = hits_ + misses_;
    const double rate = (total > 0) ? (100.0 * static_cast<double>(hits_) / static_cast<double>(total)) : 0.0;
    LOGI.printf("[ExpertPool] stats: %llu hits, %llu misses (%.2f%% hit rate)", static_cast<unsigned long long>(hits_),
                static_cast<unsigned long long>(misses_), rate);
}

const ExpertGpuHandle& ExpertPool::ensure_on_gpu(int layer, int expert_id) {
    if (config_.strategy == ExpertPoolStrategy::ALL_GPU) {
        ++hits_;
        return cached_handles_[static_cast<size_t>(layer)][static_cast<size_t>(expert_id)];
    }

    // PINNED_LRU: per-layer slot arena with LRU eviction + async H2D on miss.
    const size_t L = static_cast<size_t>(layer);
    const size_t E = static_cast<size_t>(expert_id);
    auto& residents = residency_[L];
    auto& layer_slots = slots_[L];
    auto& counter = access_counter_[L];
    auto* api = core::context().runtime().api();
    auto compute_stream = core::context().runtime().stream();

    const int cached = residents[E];
    if (cached >= 0) {
        // Hit: refresh LRU timestamp. The slot's ready_event was recorded by a prior
        // miss or prefetch. Waiting on it is a no-op if the H2D has already
        // completed, or blocks compute stream until it does.
        ++hits_;
        ++counter;
        layer_slots[cached].last_access = counter;
        layer_slots[cached].compute_touched = true;
        api->stream_wait_event(compute_stream, layer_slots[cached].ready_event);
        return layer_slots[cached].gpu_handle;
    }
    ++misses_;

    // Miss: issue async H2D then wait on compute stream before returning. When every
    // slot holds a pending prefetch (pathological caller), fall back to plain LRU so
    // the demand fetch can still make progress; the displaced prefetch's H2D is still
    // enqueued on transfer_stream and will complete, but the slot's ready_event gets
    // re-recorded for the new data, so correctness is preserved.
    int picked = pick_lru_slot(L);
    if (picked < 0) {
        picked = 0;
        for (std::size_t s = 1; s < layer_slots.size(); ++s) {
            if (layer_slots[s].last_access < layer_slots[picked].last_access) {
                picked = static_cast<int>(s);
            }
        }
    }
    const std::size_t victim = static_cast<std::size_t>(picked);
    start_async_transfer(L, E, victim);
    api->stream_wait_event(compute_stream, layer_slots[victim].ready_event);

    layer_slots[victim].expert_id = expert_id;
    ++counter;
    layer_slots[victim].last_access = counter;
    layer_slots[victim].compute_touched = true;
    residents[E] = static_cast<int>(victim);

    return layer_slots[victim].gpu_handle;
}

int ExpertPool::pick_lru_slot(std::size_t layer) const {
    const auto& layer_slots = slots_[layer];
    int victim = -1;
    for (std::size_t s = 0; s < layer_slots.size(); ++s) {
        const auto& slot = layer_slots[s];
        // Skip populated slots whose contents compute hasn't consumed yet — those
        // are pending prefetches; evicting them would force a redundant re-fetch.
        const bool evictable = (slot.expert_id < 0) || slot.compute_touched;
        if (!evictable) {
            continue;
        }
        if (victim < 0 || slot.last_access < layer_slots[static_cast<std::size_t>(victim)].last_access) {
            victim = static_cast<int>(s);
        }
    }
    return victim;
}

void ExpertPool::start_async_transfer(std::size_t layer, std::size_t expert_id, std::size_t slot_idx) {
    auto& layer_slots = slots_[layer];
    auto& residents = residency_[layer];
    auto& slot = layer_slots[slot_idx];

    if (slot.expert_id >= 0) {
        residents[static_cast<std::size_t>(slot.expert_id)] = -1;
    }
    // The slot is about to be overwritten with fresh data; reset the "consumed by
    // compute" flag so the prefetch window sees it as pending.
    slot.compute_touched = false;

    auto& runtime = core::context().runtime();
    auto* api = runtime.api();
    auto compute_stream = runtime.stream();
    auto transfer_stream = runtime.transfer_stream();

    // Barrier: transfer stream waits for compute to be done with the slot before
    // overwriting. Conservative (waits for ALL prior compute) but correct.
    api->record_event(compute_barrier_, compute_stream);
    api->stream_wait_event(transfer_stream, compute_barrier_);

    const ExpertFFN& src = experts_->at(layer, expert_id);
    auto& dst = slot.gpu_handle;
    auto copy = [&](tensor_t s, tensor_t d) {
        if (!s || !d) {
            return;
        }
        api->memcpy_async(d->data(), s->data(), s->numel() * s->elementSize(), ZEDINFER_MEMCPY_H2D, transfer_stream);
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

    api->record_event(slot.ready_event, transfer_stream);
}

void ExpertPool::prefetch(int layer, int expert_id) {
    if (config_.strategy != ExpertPoolStrategy::PINNED_LRU) {
        return;
    }
    const std::size_t L = static_cast<std::size_t>(layer);
    const std::size_t E = static_cast<std::size_t>(expert_id);
    auto& residents = residency_[L];
    auto& layer_slots = slots_[L];
    auto& counter = access_counter_[L];

    if (residents[E] >= 0) {
        // Already resident (or H2D in-flight from an earlier prefetch): bump LRU so
        // this slot isn't chosen as victim by a subsequent prefetch in the same layer.
        ++counter;
        layer_slots[residents[E]].last_access = counter;
        return;
    }

    // Not resident: reserve an evictable slot and start H2D on the transfer stream.
    // If the whole arena is already pending-prefetch (no compute_touched slot), this
    // prefetch is skipped — compute will demand-fetch the expert later, same as M2.
    const int picked = pick_lru_slot(L);
    if (picked < 0) {
        return;
    }
    const std::size_t victim = static_cast<std::size_t>(picked);
    start_async_transfer(L, E, victim);
    layer_slots[victim].expert_id = expert_id;
    ++counter;
    layer_slots[victim].last_access = counter;
    residents[E] = static_cast<int>(victim);
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
