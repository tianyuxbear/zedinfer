#include "frontend/models/moe_forward.hpp"
#include "backend/core/context/context.hpp"
#include "backend/device/runtime_api.hpp"
#include "backend/ops/moe/topk_softmax.hpp"
#include "backend/ops/ops.hpp"
#include "backend/ops/shared_expert_gate/shared_expert_gate.hpp"
#include "frontend/models/decode_scratch.hpp"
#include "utils/types.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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

// Persistent N=2 scratch for spec-decode verify. Mirrors DecodeScratch's MoE
// fields but sized at [2, ...] instead of [1, ...]. Without this the N=2 path
// pays four [2, shared_inter] + one [2, num_experts] + one [2, H] BestFitPool
// round-trip per MoE layer, which under ALL_GPU (heavily populated with ~14 GB
// of expert weights) costs ~3.5 ms / layer — i.e. ~150 ms per forward step,
// dominating the 2-token verify timeline. Once allocated this buffer survives
// the lifetime of the inference thread; reallocation only happens if the model
// shape (hidden_size / shared_inter / num_experts) changes, which it doesn't.
struct MoeN2Scratch {
    tensor_t router_logits; // [2, num_experts]
    tensor_t moe_output;    // [2, H]
    tensor_t sh_gate;       // [2, shared_inter]
    tensor_t sh_up;         // [2, shared_inter]
    tensor_t sh_act;        // [2, shared_inter]
    tensor_t sh_down;       // [2, H]
    tensor_t sh_gate_logit; // [2, 1]

    size_t H = 0;
    size_t num_experts = 0;
    size_t shared_inter = 0;
    zedinferDeviceType_t device_type = ZEDINFER_DEVICE_CPU;
    int device_id = -1;
    zedinferDataType_t dtype = ZEDINFER_DTYPE_F32;
};
static thread_local MoeN2Scratch s_moe_n2;

static MoeN2Scratch& ensure_moe_n2(size_t H, size_t num_experts, size_t shared_inter, const ExecutorConfig& exec) {
    if (s_moe_n2.router_logits && s_moe_n2.H == H && s_moe_n2.num_experts == num_experts
        && s_moe_n2.shared_inter == shared_inter && s_moe_n2.device_type == exec.device_type
        && s_moe_n2.device_id == exec.device_id && s_moe_n2.dtype == exec.data_type) {
        return s_moe_n2;
    }
    auto mkf = [&](std::vector<size_t> shape) {
        return Tensor::create(shape, exec.data_type, exec.device_type, exec.device_id);
    };
    s_moe_n2.router_logits = mkf({2, num_experts});
    s_moe_n2.moe_output = mkf({2, H});
    if (shared_inter > 0) {
        s_moe_n2.sh_gate = mkf({2, shared_inter});
        s_moe_n2.sh_up = mkf({2, shared_inter});
        s_moe_n2.sh_act = mkf({2, shared_inter});
        s_moe_n2.sh_down = mkf({2, H});
        s_moe_n2.sh_gate_logit = mkf({2, 1});
    }
    s_moe_n2.H = H;
    s_moe_n2.num_experts = num_experts;
    s_moe_n2.shared_inter = shared_inter;
    s_moe_n2.device_type = exec.device_type;
    s_moe_n2.device_id = exec.device_id;
    s_moe_n2.dtype = exec.data_type;
    return s_moe_n2;
}

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

// Persistent host scratch for the router D2H + F32 conversion. Per call,
// router_logits_buf is small (num_experts × elem_size for N=1, larger for prefill).
// The previous implementation chained two Tensor::to() calls which each invoked
// cudaMallocHost (pinned host allocator, ~50–500 µs per call) and a synchronous
// memcpy_sync; for 40 MoE layers per decode token that pinned-allocator traffic
// alone burns several milliseconds per token. We pre-allocate a pinned D2H staging
// buffer and a plain f32 working buffer once per thread, sized to the largest N
// seen, and reuse them.
struct RouterHostScratch {
    std::vector<std::byte> raw; // pinned host bytes for the D2H copy (bf16/f16/f32 input)
    std::vector<float> as_f32;  // F32-converted view consumed by topk_softmax
    size_t raw_capacity = 0;    // in bytes
    size_t f32_capacity = 0;    // in floats
    void* pinned_ptr = nullptr;
};

static thread_local RouterHostScratch s_router_host;

struct RouterTopKScratch {
    tensor_t ids_dev;
    tensor_t weights_dev;
    tensor_t ids_host;
    tensor_t weights_host;
    size_t capacity = 0;
    zedinferDeviceType_t device_type = ZEDINFER_DEVICE_CPU;
    int device_id = 0;
};

static thread_local RouterTopKScratch s_router_topk;

static void ensure_router_topk_scratch(size_t total, zedinferDeviceType_t device_type, int device_id) {
    if (s_router_topk.ids_dev && s_router_topk.capacity >= total && s_router_topk.device_type == device_type
        && s_router_topk.device_id == device_id) {
        return;
    }
    const size_t new_cap = std::max(total, s_router_topk.capacity * 2);
    s_router_topk.ids_dev = Tensor::create({new_cap}, ZEDINFER_DTYPE_I32, device_type, device_id);
    s_router_topk.weights_dev = Tensor::create({new_cap}, ZEDINFER_DTYPE_F32, device_type, device_id);
    s_router_topk.ids_host = Tensor::create({new_cap}, ZEDINFER_DTYPE_I32, ZEDINFER_DEVICE_CPU, 0);
    s_router_topk.weights_host = Tensor::create({new_cap}, ZEDINFER_DTYPE_F32, ZEDINFER_DEVICE_CPU, 0);
    s_router_topk.capacity = new_cap;
    s_router_topk.device_type = device_type;
    s_router_topk.device_id = device_id;
}

#ifdef ENABLE_NVIDIA_API
static ops::moe::TopKResult router_topk_gpu_to_host(tensor_t router_logits_buf, size_t N, size_t top_k,
                                                    bool norm_topk_prob) {
    const size_t total = N * top_k;
    ensure_router_topk_scratch(total, router_logits_buf->deviceType(), router_logits_buf->deviceId());

    auto ids_dev = s_router_topk.ids_dev->slice(0, 0, total);
    auto weights_dev = s_router_topk.weights_dev->slice(0, 0, total);
    ops::moe::topk_softmax_gpu(ids_dev, weights_dev, router_logits_buf, top_k, norm_topk_prob);

    auto* api = core::context().runtime().api();
    api->memcpy_sync(s_router_topk.ids_host->data(), ids_dev->data(), total * sizeof(std::int32_t),
                     ZEDINFER_MEMCPY_D2H);
    api->memcpy_sync(s_router_topk.weights_host->data(), weights_dev->data(), total * sizeof(float),
                     ZEDINFER_MEMCPY_D2H);

    ops::moe::TopKResult result;
    result.expert_ids.resize(total);
    result.expert_weights.resize(total);
    std::memcpy(result.expert_ids.data(), s_router_topk.ids_host->data(), total * sizeof(std::int32_t));
    std::memcpy(result.expert_weights.data(), s_router_topk.weights_host->data(), total * sizeof(float));
    return result;
}
#endif

// Grow the pinned D2H staging buffer if needed. Uses cudaMallocHost via the runtime's
// host allocator so the buffer is DMA-pinned (matches the prior cudaMallocHost behavior
// of Tensor::to(CPU)). Freed on growth via cudaFreeHost.
static void ensure_router_pinned(size_t bytes) {
    if (s_router_host.pinned_ptr && s_router_host.raw_capacity >= bytes) {
        return;
    }
    auto* api = core::context().runtime().api();
    if (s_router_host.pinned_ptr) {
        api->free_host(s_router_host.pinned_ptr);
        s_router_host.pinned_ptr = nullptr;
    }
    const size_t new_cap = std::max(bytes, s_router_host.raw_capacity * 2);
    s_router_host.pinned_ptr = api->malloc_host(new_cap);
    s_router_host.raw_capacity = new_cap;
}

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

#ifdef ENABLE_NVIDIA_API
    if (router_logits_buf->deviceType() == ZEDINFER_DEVICE_NVIDIA && top_k <= 16) {
        return router_topk_gpu_to_host(router_logits_buf, N, top_k, model.norm_topk_prob);
    }
#endif

    const size_t total_elems = N * num_experts;
    const zedinferDataType_t src_dtype = router_logits_buf->dtype();
    const size_t src_bytes = total_elems * utils::dsize(src_dtype);
    auto* api = core::context().runtime().api();

    // D2H into thread_local pinned scratch. memcpy_sync still drains the compute stream
    // before initiating the copy, so this remains a per-layer serialization point — but
    // we no longer pay cudaMallocHost / cudaFreeHost on every call.
    if (router_logits_buf->deviceType() != ZEDINFER_DEVICE_CPU) {
        ensure_router_pinned(src_bytes);
        api->memcpy_sync(s_router_host.pinned_ptr, router_logits_buf->data(), src_bytes, ZEDINFER_MEMCPY_D2H);
    } else {
        // CPU runtime: data already host-side; skip the copy and dtype-convert in place if needed.
        ensure_router_pinned(src_bytes);
        std::memcpy(s_router_host.pinned_ptr, router_logits_buf->data(), src_bytes);
    }

    // Materialize F32 view consumed by topk_softmax. If router_logits are already F32,
    // reinterpret the pinned buffer; otherwise convert into the persistent f32 vector.
    const float* logits_f32 = nullptr;
    if (src_dtype == ZEDINFER_DTYPE_F32) {
        logits_f32 = reinterpret_cast<const float*>(s_router_host.pinned_ptr);
    } else {
        if (s_router_host.f32_capacity < total_elems) {
            s_router_host.as_f32.resize(total_elems);
            s_router_host.f32_capacity = total_elems;
        }
        if (src_dtype == ZEDINFER_DTYPE_BF16) {
            utils::bf16_to_fp32_batch(s_router_host.as_f32.data(),
                                      reinterpret_cast<const bf16_t*>(s_router_host.pinned_ptr), total_elems);
        } else if (src_dtype == ZEDINFER_DTYPE_F16) {
            utils::fp16_to_fp32_batch_f16c(s_router_host.as_f32.data(),
                                           reinterpret_cast<const fp16_t*>(s_router_host.pinned_ptr), total_elems);
        } else {
            throw std::runtime_error("compute_router_topk: unsupported router_logits dtype");
        }
        logits_f32 = s_router_host.as_f32.data();
    }
    return ops::moe::topk_softmax(logits_f32, N, num_experts, top_k, model.norm_topk_prob);
}

// Execute the small-N decode path (N ∈ {1, 2}): per-token loop over top-k experts,
// each expert pass is an M=1 GEMM into the [1, ...] scratch buffers. Zero
// Tensor::create per call (the small router/moe_output allocs are handled by
// moe_layer_forward, not here).
//
// N=2 is the spec-decode verify shape. Without this path, N=2 falls into
// moe_prefill which spins up gather/scatter kernels per active expert plus
// fresh [2, ...] allocs — those kernels are tuned for N≫top_k prefill batches
// and dominate when N is small (~20× slower per layer). Per-token decode keeps
// the launch count tractable: 2 × top_k × 3 expert GEMMs vs prefill's
// up-to-2×top_k expert GEMMs PLUS 2 gather + 2 scatter kernel launches per
// active expert PLUS per-call buffer creates.
static void moe_decode(const ModelForwardConfig& model, tensor_t moe_output, tensor_t input, int layer_idx,
                       const ops::moe::TopKResult& topk, DecodeScratch& scratch) {
    const size_t top_k = model.num_experts_per_tok;
    const size_t N = input->shape()[0];

    // M3 async prefetch: kick off H2D for every selected expert on the transfer stream
    // before the compute loop starts. Under ALL_GPU this is a no-op; under PINNED_LRU
    // the compute loop then mostly hits with already-populated slots, overlapping the
    // remaining H2Ds with previous experts' GEMMs. Order matches compute order so the
    // transfer stream's FIFO delivers e0 first, e1 second, etc. For N=2 prefetch across
    // both rows' selected experts.
    if (model.expert_pool) {
        for (size_t n = 0; n < N; ++n) {
            for (size_t k = 0; k < top_k; ++k) {
                model.expert_pool->prefetch(layer_idx, topk.expert_ids[n * top_k + k]);
            }
        }
    }

    for (size_t n = 0; n < N; ++n) {
        // Per-row views of [1, H] / [1, H]. For N=1 the view is the whole tensor
        // (no slice allocation); for N=2 slice produces a zero-copy view at row n.
        tensor_t input_n = (N == 1) ? input : input->slice(0, n, n + 1);
        tensor_t output_n = (N == 1) ? moe_output : moe_output->slice(0, n, n + 1);

        for (size_t k = 0; k < top_k; ++k) {
            int expert_id = topk.expert_ids[n * top_k + k];
            float weight = topk.expert_weights[n * top_k + k];

            model.dispatch_expert_linear(scratch.expert_gate, input_n, layer_idx, expert_id, ExpertProj::Gate);
            model.dispatch_expert_linear(scratch.expert_up, input_n, layer_idx, expert_id, ExpertProj::Up);
            ops::swiglu(scratch.expert_act, scratch.expert_gate, scratch.expert_up);
            model.dispatch_expert_linear(scratch.expert_down, scratch.expert_act, layer_idx, expert_id,
                                         ExpertProj::Down);

            ops::add_scaled(output_n, scratch.expert_down, weight);
        }
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
                                int layer_idx, DecodeScratch* scratch, const MakeTensor& make, const MoeN2Scratch* n2) {
    if (!model.has_shared_expert) {
        return;
    }

    const size_t N = input->shape()[0];
    const size_t hidden_size = model.config.hidden_size;
    const size_t shared_inter = model.shared_expert_intermediate_size;
    const bool use_n1_scratch = (scratch != nullptr && N == 1);

    auto sp = model.shared_expert_prefix(layer_idx);
    // For N=1: use the engine-wide DecodeScratch's [1, ...] buffers.
    // For N=2 (spec-decode verify): use the thread_local [2, ...] N2 cache —
    //   four BestFitPool fresh allocs per layer (one of which is at the FFN
    //   intermediate size) cost ~3.5 ms / layer under ALL_GPU.
    // Else (prefill N > 2): pay the per-call alloc as before.
    auto sh_gate = use_n1_scratch ? scratch->shared_gate : (n2 ? n2->sh_gate : make({N, shared_inter}));
    auto sh_up = use_n1_scratch ? scratch->shared_up : (n2 ? n2->sh_up : make({N, shared_inter}));
    auto sh_act = use_n1_scratch ? scratch->shared_act : (n2 ? n2->sh_act : make({N, shared_inter}));
    auto sh_down = use_n1_scratch ? scratch->shared_down : (n2 ? n2->sh_down : make({N, hidden_size}));

    model.dispatch_linear(sh_gate, input, sp + "gate_proj", nullptr);
    model.dispatch_linear(sh_up, input, sp + "up_proj", nullptr);
    ops::swiglu(sh_act, sh_gate, sh_up);
    model.dispatch_linear(sh_down, sh_act, sp + "down_proj", nullptr);

    // Qwen3.5: an additional per-token sigmoid gate scales the shared-expert
    // output before it joins the routed-expert sum. The gate weight is
    // layers.{L}.mlp.shared_expert_gate.weight of shape [1, hidden]; it is
    // absent in Qwen3-30B-A3B (and earlier), so we detect it by tensor lookup
    // and skip the gating when missing — matches the v0.2.0 behavior.
    //
    // HF: shared_expert_output = sigmoid(shared_expert_gate(hidden)) * shared_expert_output
    const std::string gate_w_name = "layers." + std::to_string(layer_idx) + ".mlp.shared_expert_gate.weight";
    if (model.weights.has_tensor(gate_w_name)) {
        auto gate_logits = (use_n1_scratch && scratch->shared_gate_logits) ? scratch->shared_gate_logits
                                                                           : (n2 ? n2->sh_gate_logit : make({N, 1}));
        ops::linear(gate_logits, input, model.weights.get_tensor(gate_w_name), nullptr);
        ops::shared_expert_gate(sh_down, gate_logits);
    }

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
    // Two gates:
    //   use_n1_scratch  — scratch.{router_logits,moe_output} are sized [1, ...];
    //                     true only when N=1 and we have a scratch context.
    //   use_decode_path — moe_decode is the per-token decode loop; valid for
    //                     N ∈ {1, 2}, gated on having a scratch (its [1, ...]
    //                     expert_{gate,up,act,down} buffers drive each token).
    //                     N=2 is the spec-decode verify shape; routing it
    //                     through moe_prefill (gather/scatter, [2, ...] fresh
    //                     allocs per layer) is what made spec mode 20× slower.
    static constexpr size_t kMaxSmallDecodeN = 2;
    const bool use_n1_scratch = (scratch != nullptr && N == 1);
    const bool use_decode_path = (scratch != nullptr && N <= kMaxSmallDecodeN);

    // Thread_local N=2 cache (router/moe_output/shared buffers), lazy-init.
    // Without this the N=2 spec-verify path pays ~6 fresh allocs per MoE layer,
    // each ~0.6 ms under ALL_GPU (BestFitPool is heavily populated by expert
    // weights), totaling ~140 ms per step and dominating the verify timeline.
    const MoeN2Scratch* n2 = nullptr;
    if (use_decode_path && N == 2) {
        n2 = &ensure_moe_n2(hidden_size, num_experts, model.shared_expert_intermediate_size, exec_config);
    }

    // Router logits buffer.
    //   N=1 + scratch  : scratch->router_logits     [1, num_experts]
    //   N=2 + scratch  : n2->router_logits          [2, num_experts]  (thread_local)
    //   N>2  prefill   : per-call make({N, num_experts})
    auto router_logits = use_n1_scratch ? scratch->router_logits : (n2 ? n2->router_logits : make({N, num_experts}));
    auto topk = compute_router_topk(model, input, layer_idx, router_logits, top_k);

    // Accumulator for weighted sum of routed-expert outputs. When the layer has no
    // shared expert (e.g. Qwen3-30B-A3B), we accumulate directly into `output` —
    // apply_shared_expert becomes a no-op, saving a per-layer hidden_size D2D copy
    // + global sync. With shared expert, allocate a separate buffer so the shared
    // FFN result can be added without races.
    tensor_t moe_output;
    if (!model.has_shared_expert) {
        moe_output = output;
    } else if (use_n1_scratch) {
        moe_output = scratch->moe_output;
    } else if (n2) {
        moe_output = n2->moe_output;
    } else {
        moe_output = make({N, hidden_size});
    }
    ops::fill_zero(moe_output);

    if (use_decode_path) {
        moe_decode(model, moe_output, input, layer_idx, topk, *scratch);
    } else {
        (void)moe_inter; // used inside moe_prefill
        moe_prefill(model, moe_output, input, layer_idx, topk, make);
    }

    apply_shared_expert(model, output, moe_output, input, layer_idx, scratch, make, n2);
}

} // namespace zedinfer::model
