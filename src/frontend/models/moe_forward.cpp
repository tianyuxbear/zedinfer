#include "frontend/models/moe_forward.hpp"
#include "backend/core/context/context.hpp"
#include "backend/ops/moe/topk_softmax.hpp"
#include "backend/ops/ops.hpp"
#include "frontend/models/decode_scratch.hpp"

#include <algorithm>
#include <functional>
#include <plog/Log.h>
#include <vector>

namespace zedinfer::model {

namespace {

// Tensor factory bound to exec_config device/dtype.
using MakeTensor = std::function<tensor_t(std::vector<size_t>)>;

// Persistent scratch for prefill indices/weights uploads. Grows monotonically; reused
// across moe_prefill calls and across MoE layers within a single inference.
//
// Why this exists: under ALL_GPU mode the BestFitMemoryPool is heavily populated by
// ~14 GB of expert weights, so per-layer Tensor::create / dealloc round-trips for these
// tiny indices/weights buffers hit the pool's slow path. PINNED_LRU was unaffected
// because its pool only holds ~150 MB of slot arena. Persistent buffers also let the
// upload run as memcpy_async on the compute stream, avoiding the per-layer global
// sync from cudaMemcpy.
//
// Lifetime caveat: thread_local persistence means these tensors hold shared_ptr to
// pool-backed storage until thread exit. If the engine (and its memory pool) is
// destroyed before the thread, these dangle. Acceptable for the current single-engine
// single-inference-thread serving model; revisit if multiple engines coexist.
struct PrefillIndexScratch {
    tensor_t indices_dev;  // device, I32, [capacity]
    tensor_t weights_dev;  // device, F32, [capacity]
    tensor_t indices_host; // pinned host (cudaMallocHost on NVIDIA), I32, [capacity]
    tensor_t weights_host; // pinned host, F32, [capacity]
    size_t capacity = 0;
    zedinferDeviceType_t device_type = ZEDINFER_DEVICE_CPU;
    int device_id = -1;
};

static thread_local PrefillIndexScratch s_prefill_scratch;

// Persistent host-side bucketing scratch for moe_prefill. Same motivation as
// PrefillIndexScratch: reuse capacity across calls so the per-layer bucket build
// doesn't allocate. Inner vectors are cleared (capacity preserved) each call.
struct PrefillBucketScratch {
    std::vector<std::vector<size_t>> expert_tokens;
    std::vector<std::vector<float>> expert_weights;
    std::vector<size_t> active_experts;
    std::vector<size_t> expert_offsets;
};

static thread_local PrefillBucketScratch s_prefill_buckets;

static void ensure_prefill_scratch(size_t needed, zedinferDeviceType_t device_type, int device_id) {
    if (s_prefill_scratch.capacity >= needed && s_prefill_scratch.device_type == device_type
        && s_prefill_scratch.device_id == device_id && s_prefill_scratch.indices_dev) {
        return;
    }
    // Grow with headroom so we don't reallocate every batch-size bump.
    const size_t new_cap = std::max(needed, s_prefill_scratch.capacity * 2);
    s_prefill_scratch.indices_dev = Tensor::create({new_cap}, ZEDINFER_DTYPE_I32, device_type, device_id);
    s_prefill_scratch.weights_dev = Tensor::create({new_cap}, ZEDINFER_DTYPE_F32, device_type, device_id);
    s_prefill_scratch.indices_host = Tensor::create({new_cap}, ZEDINFER_DTYPE_I32, ZEDINFER_DEVICE_CPU, 0);
    s_prefill_scratch.weights_host = Tensor::create({new_cap}, ZEDINFER_DTYPE_F32, ZEDINFER_DEVICE_CPU, 0);
    s_prefill_scratch.capacity = new_cap;
    s_prefill_scratch.device_type = device_type;
    s_prefill_scratch.device_id = device_id;
}

// Compute router logits on device, then bring them to host as F32 for top-k selection.
static ops::moe::TopKResult compute_router_topk(const ModelForwardConfig& model, tensor_t input, int layer_idx,
                                                tensor_t router_logits_buf, size_t top_k) {
    const size_t N = input->shape()[0];
    const size_t num_experts = model.num_experts;

    // Router weight is typically not quantized (small [num_experts, hidden_size] matrix).
    // Use the per-layer pointer cached at init time; falls back to dispatch_linear by
    // prefix when the layer's router happens to be quantized (entry is null).
    if (const auto& router_w = model.router_weights[static_cast<size_t>(layer_idx)]) {
        ops::linear(router_logits_buf, input, router_w, nullptr);
    } else {
        std::string gate_prefix = "layers." + std::to_string(layer_idx) + ".mlp.gate";
        model.dispatch_linear(router_logits_buf, input, gate_prefix, nullptr);
    }

    // D2H + dtype-to-F32 for CPU top-k (N * num_experts is small enough).
    //
    // PERF-TODO: this D2H copy forces a cudaDeviceSynchronize per MoE layer. For 48 layers
    // in decode this adds up. A GPU-side top-k kernel writing expert_ids/weights directly
    // to a pinned host buffer would eliminate the serialization point.
    tensor_t cpu = router_logits_buf;
    if (cpu->deviceType() != ZEDINFER_DEVICE_CPU) {
        cpu = cpu->to(ZEDINFER_DEVICE_CPU, 0);
    }
    if (cpu->dtype() != ZEDINFER_DTYPE_F32) {
        cpu = cpu->to(ZEDINFER_DTYPE_F32);
    }
    return ops::moe::topk_softmax(reinterpret_cast<const float*>(cpu->data()), N, num_experts, top_k,
                                  model.norm_topk_prob);
}

// Execute the N=1 decode path: loop over top-k experts, weighted accumulation into moe_output.
// Uses pre-allocated scratch buffers (zero Tensor::create per step).
static void moe_decode(const ModelForwardConfig& model, tensor_t moe_output, tensor_t input, int layer_idx,
                       const ops::moe::TopKResult& topk, DecodeScratch& scratch) {
    const size_t top_k = model.num_experts_per_tok;

    // M3 async prefetch: kick off H2D for every selected expert on the transfer stream
    // before the compute loop starts. Under ALL_GPU this is a no-op; under PINNED_LRU
    // the compute loop then mostly hits with already-populated slots, overlapping the
    // remaining H2Ds with previous experts' GEMMs. Order matches compute order so the
    // transfer stream's FIFO delivers e0 first, e1 second, etc.
    if (model.expert_pool) {
        for (size_t k = 0; k < top_k; ++k) { model.expert_pool->prefetch(layer_idx, topk.expert_ids[k]); }
    }

    for (size_t k = 0; k < top_k; ++k) {
        int expert_id = topk.expert_ids[k];
        float weight = topk.expert_weights[k];

        model.dispatch_expert_linear(scratch.expert_gate, input, layer_idx, expert_id, ExpertProj::Gate);
        model.dispatch_expert_linear(scratch.expert_up, input, layer_idx, expert_id, ExpertProj::Up);
        ops::swiglu(scratch.expert_act, scratch.expert_gate, scratch.expert_up);
        model.dispatch_expert_linear(scratch.expert_down, scratch.expert_act, layer_idx, expert_id, ExpertProj::Down);

        ops::add_scaled(moe_output, scratch.expert_down, weight);
    }
}

// Execute the N>1 prefill path: permute tokens by expert, run FFN per expert group, scatter back.
// Buffers (gathered, g_gate, g_up, g_act, g_down) are allocated ONCE at max group size = N
// and reused via slice views for each expert.
//
// Gather/scatter use single-launch CUDA kernels (`ops::gather_rows` / `ops::scatter_add_rows`)
// fed from a flat indices/weights upload built once at the start of prefill. Replaces the
// O(group_size) per-row memcpy and per-row add_scaled loops, cutting launch count from
// ~2 × group_size per expert to ~2 per expert.
static void moe_prefill(const ModelForwardConfig& model, tensor_t moe_output, tensor_t input, int layer_idx,
                        const ops::moe::TopKResult& topk, const MakeTensor& make) {
    const size_t N = input->shape()[0];
    const size_t hidden_size = model.config.hidden_size;
    const size_t num_experts = model.num_experts;
    const size_t top_k = model.num_experts_per_tok;
    const size_t moe_inter = model.moe_intermediate_size;

    // Bucket token indices by selected expert. Each expert receives at most N tokens
    // (since top-k picks are distinct per token), so max group size is N.
    // Reuse thread_local capacity across calls; inner clear() keeps allocated storage.
    auto& expert_tokens = s_prefill_buckets.expert_tokens;
    auto& expert_weights = s_prefill_buckets.expert_weights;
    expert_tokens.resize(num_experts);
    expert_weights.resize(num_experts);
    for (size_t e = 0; e < num_experts; ++e) {
        expert_tokens[e].clear();
        expert_weights[e].clear();
    }
    for (size_t n = 0; n < N; ++n) {
        for (size_t k = 0; k < top_k; ++k) {
            int eid = topk.expert_ids[n * top_k + k];
            expert_tokens[static_cast<size_t>(eid)].push_back(n);
            expert_weights[static_cast<size_t>(eid)].push_back(topk.expert_weights[n * top_k + k]);
        }
    }

    // Materialize the active-expert ordering used by the prefetcher and the compute loop
    // below, so indexing matches between the two.
    auto& active_experts = s_prefill_buckets.active_experts;
    active_experts.clear();
    for (size_t eid = 0; eid < num_experts; ++eid) {
        if (!expert_tokens[eid].empty()) {
            active_experts.push_back(eid);
        }
    }

    // Build flat indices/weights per active-expert order into the persistent pinned host
    // scratch. Sum of group sizes is exactly N × top_k by construction. Grow the device-
    // side scratch lazily if this prefill is bigger than any seen before on this thread.
    const size_t total = N * top_k;
    ensure_prefill_scratch(total, input->deviceType(), input->deviceId());

    auto* idx_host = reinterpret_cast<std::int32_t*>(s_prefill_scratch.indices_host->data());
    auto* w_host = reinterpret_cast<float*>(s_prefill_scratch.weights_host->data());

    auto& expert_offsets = s_prefill_buckets.expert_offsets;
    expert_offsets.clear();
    expert_offsets.push_back(0);
    size_t pos = 0;
    for (size_t idx = 0; idx < active_experts.size(); ++idx) {
        const size_t eid = active_experts[idx];
        const auto& tokens = expert_tokens[eid];
        const auto& wts = expert_weights[eid];
        for (size_t i = 0; i < tokens.size(); ++i) {
            idx_host[pos] = static_cast<std::int32_t>(tokens[i]);
            w_host[pos] = wts[i];
            ++pos;
        }
        expert_offsets.push_back(pos);
    }

    // Async H2D on the compute stream: keeps the upload ordered with the gather/scatter
    // kernels that consume it (FIFO on the same stream) without forcing a global sync.
    // Pinned host source guarantees DMA-overlap eligibility.
    auto& runtime = zedinfer::core::context().runtime();
    auto* api = runtime.api();
    auto compute_stream = runtime.stream();
    const auto kind_h2d = (input->deviceType() == ZEDINFER_DEVICE_CPU) ? ZEDINFER_MEMCPY_H2H : ZEDINFER_MEMCPY_H2D;
    api->memcpy_async(s_prefill_scratch.indices_dev->data(), s_prefill_scratch.indices_host->data(),
                      total * sizeof(std::int32_t), kind_h2d, compute_stream);
    api->memcpy_async(s_prefill_scratch.weights_dev->data(), s_prefill_scratch.weights_host->data(),
                      total * sizeof(float), kind_h2d, compute_stream);

    // Slice views over the prefix actually used this call (scratch capacity may be larger).
    auto indices_buf = s_prefill_scratch.indices_dev->slice(0, 0, total);
    auto weights_buf = s_prefill_scratch.weights_dev->slice(0, 0, total);

    // Pre-allocate buffers at max group size (N) and reuse across experts via slice views.
    auto gathered_full = make({N, hidden_size});
    auto g_gate_full = make({N, moe_inter});
    auto g_up_full = make({N, moe_inter});
    auto g_act_full = make({N, moe_inter});
    auto g_down_full = make({N, hidden_size});

    // M3.5 sliding-window prefetch. Keep `depth` transfers outstanding on the transfer
    // stream: first issue an initial burst of min(depth, active_experts.size()) calls,
    // then after each compute iteration issue one more so the window stays saturated.
    // Depth is capped at the per-layer slot count so prefetches don't evict each other
    // before compute consumes them. Under ALL_GPU depth==0 and all prefetch calls are
    // no-ops.
    const int depth = (model.expert_pool != nullptr) ? model.expert_pool->max_prefetch_depth() : 0;
    size_t prefetched_upto = 0;
    if (depth > 0) {
        const size_t burst = std::min(static_cast<size_t>(depth), active_experts.size());
        for (; prefetched_upto < burst; ++prefetched_upto) {
            model.expert_pool->prefetch(layer_idx, static_cast<int>(active_experts[prefetched_upto]));
        }
    }

    for (size_t idx = 0; idx < active_experts.size(); ++idx) {
        const size_t eid = active_experts[idx];
        const size_t group_size = expert_tokens[eid].size();
        const size_t off = expert_offsets[idx];

        auto idx_view = indices_buf->slice(0, off, off + group_size);
        auto w_view = weights_buf->slice(0, off, off + group_size);

        // Views for this expert's group size (contiguous: slicing dim 0 from 0).
        auto gathered = gathered_full->slice(0, 0, group_size);
        auto g_gate = g_gate_full->slice(0, 0, group_size);
        auto g_up = g_up_full->slice(0, 0, group_size);
        auto g_act = g_act_full->slice(0, 0, group_size);
        auto g_down = g_down_full->slice(0, 0, group_size);

        ops::gather_rows(gathered, input, idx_view);

        model.dispatch_expert_linear(g_gate, gathered, layer_idx, static_cast<int>(eid), ExpertProj::Gate);
        model.dispatch_expert_linear(g_up, gathered, layer_idx, static_cast<int>(eid), ExpertProj::Up);
        ops::swiglu(g_act, g_gate, g_up);
        model.dispatch_expert_linear(g_down, g_act, layer_idx, static_cast<int>(eid), ExpertProj::Down);

        ops::scatter_add_rows(moe_output, g_down, idx_view, w_view);

        // Slide the prefetch window forward: pull in the expert `depth` steps ahead.
        if (depth > 0 && prefetched_upto < active_experts.size()) {
            model.expert_pool->prefetch(layer_idx, static_cast<int>(active_experts[prefetched_upto]));
            ++prefetched_upto;
        }
    }
}

// Compute shared expert FFN (always active on all tokens) and add to moe_output.
// Writes final result (moe_output + shared) into `output`.
//
// Caller's invariant when has_shared_expert == false: `moe_output` and `output` alias
// the same buffer, so this function is a no-op in that case (no copy needed).
static void apply_shared_expert(const ModelForwardConfig& model, tensor_t output, tensor_t moe_output, tensor_t input,
                                int layer_idx, DecodeScratch* scratch, const MakeTensor& make) {
    if (!model.has_shared_expert) {
        return;
    }

    const size_t N = input->shape()[0];
    const size_t hidden_size = model.config.hidden_size;
    const size_t shared_inter = model.shared_expert_intermediate_size;
    const bool use_scratch = (scratch != nullptr && N == 1);

    auto sp = model.shared_expert_prefix(layer_idx);
    auto sh_gate = use_scratch ? scratch->shared_gate : make({N, shared_inter});
    auto sh_up = use_scratch ? scratch->shared_up : make({N, shared_inter});
    auto sh_act = use_scratch ? scratch->shared_act : make({N, shared_inter});
    auto sh_down = use_scratch ? scratch->shared_down : make({N, hidden_size});

    model.dispatch_linear(sh_gate, input, sp + "gate_proj", nullptr);
    model.dispatch_linear(sh_up, input, sp + "up_proj", nullptr);
    ops::swiglu(sh_act, sh_gate, sh_up);
    model.dispatch_linear(sh_down, sh_act, sp + "down_proj", nullptr);

    ops::add(output, moe_output, sh_down);
}

} // namespace

void moe_layer_forward(const ModelForwardConfig& model, tensor_t output, tensor_t input, int layer_idx,
                       const ExecutorConfig& exec_config, DecodeScratch* scratch) {
    const size_t N = input->shape()[0];
    const size_t hidden_size = model.config.hidden_size;
    const size_t num_experts = model.num_experts;
    const size_t top_k = model.num_experts_per_tok;
    const size_t moe_inter = model.moe_intermediate_size;

    MakeTensor make = [&](std::vector<size_t> shape) {
        return Tensor::create(shape, exec_config.data_type, exec_config.device_type, exec_config.device_id);
    };
    const bool use_scratch = (scratch != nullptr && N == 1);

    // Router logits buffer (scratch for decode, fresh alloc for prefill).
    auto router_logits = use_scratch ? scratch->router_logits : make({N, num_experts});
    auto topk = compute_router_topk(model, input, layer_idx, router_logits, top_k);

    // Accumulator for weighted sum of routed-expert outputs. When the layer has no
    // shared expert (e.g. Qwen3-30B-A3B), we accumulate directly into `output` —
    // apply_shared_expert becomes a no-op, saving a per-layer hidden_size D2D copy
    // + global sync. With shared expert, allocate a separate buffer so the shared
    // FFN result can be added without races.
    tensor_t moe_output;
    if (!model.has_shared_expert) {
        moe_output = output;
    } else if (use_scratch) {
        moe_output = scratch->moe_output;
    } else {
        moe_output = make({N, hidden_size});
    }
    ops::fill_zero(moe_output);

    if (use_scratch) {
        moe_decode(model, moe_output, input, layer_idx, topk, *scratch);
    } else {
        (void)moe_inter; // used inside moe_prefill
        moe_prefill(model, moe_output, input, layer_idx, topk, make);
    }

    apply_shared_expert(model, output, moe_output, input, layer_idx, scratch, make);
}

} // namespace zedinfer::model
