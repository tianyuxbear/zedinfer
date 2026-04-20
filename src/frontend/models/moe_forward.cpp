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

// Compute router logits on device, then bring them to host as F32 for top-k selection.
static ops::moe::TopKResult compute_router_topk(const ModelForwardConfig& model, tensor_t input, int layer_idx,
                                                tensor_t router_logits_buf, size_t top_k) {
    const size_t N = input->shape()[0];
    const size_t num_experts = model.num_experts;

    // Router weight is typically not quantized (small [num_experts, hidden_size] matrix).
    std::string gate_name = model.router_weight_name(layer_idx);
    if (model.weights.has_tensor(gate_name)) {
        ops::linear(router_logits_buf, input, model.W(gate_name), nullptr);
    } else {
        // Fallback: quantized router path.
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
        for (size_t k = 0; k < top_k; ++k) {
            model.expert_pool->prefetch(layer_idx, topk.expert_ids[k]);
        }
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
// and reused via slice views for each expert. Phase 2's ExpertPool will manage these at a
// higher level.
static void moe_prefill(const ModelForwardConfig& model, tensor_t moe_output, tensor_t input, int layer_idx,
                        const ops::moe::TopKResult& topk, const MakeTensor& make) {
    const size_t N = input->shape()[0];
    const size_t hidden_size = model.config.hidden_size;
    const size_t num_experts = model.num_experts;
    const size_t top_k = model.num_experts_per_tok;
    const size_t moe_inter = model.moe_intermediate_size;

    // Bucket token indices by selected expert. Each expert receives at most N tokens
    // (since top-k picks are distinct per token), so max group size is N.
    std::vector<std::vector<size_t>> expert_tokens(num_experts);
    std::vector<std::vector<float>> expert_weights(num_experts);
    for (size_t n = 0; n < N; ++n) {
        for (size_t k = 0; k < top_k; ++k) {
            int eid = topk.expert_ids[n * top_k + k];
            expert_tokens[static_cast<size_t>(eid)].push_back(n);
            expert_weights[static_cast<size_t>(eid)].push_back(topk.expert_weights[n * top_k + k]);
        }
    }

    // Pre-allocate buffers at max group size (N) and reuse across experts via slice views.
    auto gathered_full = make({N, hidden_size});
    auto g_gate_full = make({N, moe_inter});
    auto g_up_full = make({N, moe_inter});
    auto g_act_full = make({N, moe_inter});
    auto g_down_full = make({N, hidden_size});

    auto* api = zedinfer::core::context().runtime().api();
    auto kind = (input->deviceType() == ZEDINFER_DEVICE_CPU) ? ZEDINFER_MEMCPY_H2H : ZEDINFER_MEMCPY_D2D;
    const size_t row_bytes = hidden_size * input->elementSize();

    // Materialize the active-expert ordering used by both the prefetcher and the
    // compute loop below, so indexing matches between the two.
    std::vector<size_t> active_experts;
    active_experts.reserve(num_experts);
    for (size_t eid = 0; eid < num_experts; ++eid) {
        if (!expert_tokens[eid].empty()) {
            active_experts.push_back(eid);
        }
    }

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
        const auto& tokens = expert_tokens[eid];
        size_t group_size = tokens.size();

        // Views for this expert's group size (contiguous: slicing dim 0 from 0).
        auto gathered = gathered_full->slice(0, 0, static_cast<int64_t>(group_size));
        auto g_gate = g_gate_full->slice(0, 0, static_cast<int64_t>(group_size));
        auto g_up = g_up_full->slice(0, 0, static_cast<int64_t>(group_size));
        auto g_act = g_act_full->slice(0, 0, static_cast<int64_t>(group_size));
        auto g_down = g_down_full->slice(0, 0, static_cast<int64_t>(group_size));

        // PERF-TODO: replace per-row cudaMemcpy with a single CUDA gather kernel. Current
        // implementation launches O(group_size) small copies per expert per layer; a gather
        // kernel would batch them into one launch and use coalesced loads.
        for (size_t i = 0; i < group_size; ++i) {
            void* dst = gathered->data() + static_cast<ptrdiff_t>(i * row_bytes);
            const void* src = input->data() + static_cast<ptrdiff_t>(tokens[i] * row_bytes);
            api->memcpy_sync(dst, src, row_bytes, kind);
        }

        model.dispatch_expert_linear(g_gate, gathered, layer_idx, static_cast<int>(eid), ExpertProj::Gate);
        model.dispatch_expert_linear(g_up, gathered, layer_idx, static_cast<int>(eid), ExpertProj::Up);
        ops::swiglu(g_act, g_gate, g_up);
        model.dispatch_expert_linear(g_down, g_act, layer_idx, static_cast<int>(eid), ExpertProj::Down);

        // Scatter weighted rows back.
        for (size_t i = 0; i < group_size; ++i) {
            auto out_row = moe_output->slice(0, static_cast<int64_t>(tokens[i]), static_cast<int64_t>(tokens[i]) + 1);
            auto down_row = g_down->slice(0, static_cast<int64_t>(i), static_cast<int64_t>(i) + 1);
            ops::add_scaled(out_row, down_row, expert_weights[eid][i]);
        }

        // Slide the prefetch window forward: pull in the expert `depth` steps ahead.
        if (depth > 0 && prefetched_upto < active_experts.size()) {
            model.expert_pool->prefetch(layer_idx, static_cast<int>(active_experts[prefetched_upto]));
            ++prefetched_upto;
        }
    }
}

// Compute shared expert FFN (always active on all tokens) and add to moe_output.
// Writes final result (moe_output + shared) into `output`.
// If no shared expert exists for this layer, copies moe_output to output.
static void apply_shared_expert(const ModelForwardConfig& model, tensor_t output, tensor_t moe_output, tensor_t input,
                                int layer_idx, DecodeScratch* scratch, const MakeTensor& make) {
    const size_t N = input->shape()[0];
    const size_t hidden_size = model.config.hidden_size;
    const size_t shared_inter = model.shared_expert_intermediate_size;
    const bool use_scratch = (scratch != nullptr && N == 1);

    if (!model.has_shared_expert) {
        auto* api = zedinfer::core::context().runtime().api();
        auto kind = (output->deviceType() == ZEDINFER_DEVICE_CPU) ? ZEDINFER_MEMCPY_H2H : ZEDINFER_MEMCPY_D2D;
        api->memcpy_sync(output->data(), moe_output->data(), output->numel() * output->elementSize(), kind);
        return;
    }

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

    // Accumulator for weighted sum of routed-expert outputs.
    auto moe_output = use_scratch ? scratch->moe_output : make({N, hidden_size});
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
