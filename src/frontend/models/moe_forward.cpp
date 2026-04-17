#include "frontend/models/moe_forward.hpp"
#include "backend/core/context/context.hpp"
#include "backend/ops/moe/topk_softmax.hpp"
#include "backend/ops/ops.hpp"
#include "frontend/models/decode_scratch.hpp"

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

    for (size_t eid = 0; eid < num_experts; ++eid) {
        const auto& tokens = expert_tokens[eid];
        if (tokens.empty()) {
            continue;
        }

        size_t group_size = tokens.size();

        // Views for this expert's group size (contiguous: slicing dim 0 from 0).
        auto gathered = gathered_full->slice(0, 0, static_cast<int64_t>(group_size));
        auto g_gate = g_gate_full->slice(0, 0, static_cast<int64_t>(group_size));
        auto g_up = g_up_full->slice(0, 0, static_cast<int64_t>(group_size));
        auto g_act = g_act_full->slice(0, 0, static_cast<int64_t>(group_size));
        auto g_down = g_down_full->slice(0, 0, static_cast<int64_t>(group_size));

        // Gather rows (TODO: replace with a single CUDA gather kernel for perf).
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
            auto out_row = moe_output->slice(0, static_cast<int64_t>(tokens[i]),
                                              static_cast<int64_t>(tokens[i]) + 1);
            auto down_row = g_down->slice(0, static_cast<int64_t>(i), static_cast<int64_t>(i) + 1);
            ops::add_scaled(out_row, down_row, expert_weights[eid][i]);
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

    auto sp = model.shared_expert_prefix(layer_idx);
    bool has_shared = model.weights.has_tensor(sp + "gate_proj.weight")
                   || model.has_quantized_linear(sp + "gate_proj");

    if (!has_shared) {
        auto* api = zedinfer::core::context().runtime().api();
        auto kind = (output->deviceType() == ZEDINFER_DEVICE_CPU) ? ZEDINFER_MEMCPY_H2H : ZEDINFER_MEMCPY_D2D;
        api->memcpy_sync(output->data(), moe_output->data(), output->numel() * output->elementSize(), kind);
        return;
    }

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
