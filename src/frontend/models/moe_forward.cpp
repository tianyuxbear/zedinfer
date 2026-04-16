#include "frontend/models/moe_forward.hpp"
#include "backend/core/context/context.hpp"
#include "backend/ops/moe/topk_softmax.hpp"
#include "backend/ops/ops.hpp"
#include "frontend/models/decode_scratch.hpp"

#include <plog/Log.h>
#include <vector>

namespace zedinfer::model {

// Helper: dispatch linear (quantized or dense) — same logic as transformer_forward.cpp
static void dispatch_linear(const ModelForwardConfig& model, tensor_t out, tensor_t in, const std::string& prefix,
                            tensor_t bias) {
    if (model.has_quantized_linear(prefix)) {
        auto quant = model.quant_linear(prefix);
        if (!bias) {
            bias = quant.bias;
        }
        ops::linear_quantized(out, in, quant.weight, bias, quant.scale, quant.g_idx, quant.num_bits, quant.group_size);
        return;
    }

    ops::linear(out, in, model.W(prefix + ".weight"), bias);
}

void moe_layer_forward(const ModelForwardConfig& model, tensor_t output, tensor_t input, int layer_idx,
                       const ExecutorConfig& exec_config, DecodeScratch* scratch) {
    const size_t N = input->shape()[0];
    const size_t hidden_size = model.config.hidden_size;
    const size_t num_experts = model.num_experts;
    const size_t top_k = model.num_experts_per_tok;
    const size_t moe_inter = model.moe_intermediate_size;
    const size_t shared_inter = model.shared_expert_intermediate_size;

    auto make = [&](std::vector<size_t> shape) {
        return Tensor::create(shape, exec_config.data_type, exec_config.device_type, exec_config.device_id);
    };

    const bool use_scratch = (scratch != nullptr && N == 1);

    // === Step 1: Router gating ===
    // Compute router logits: input @ gate_weight -> [N, num_experts]
    auto router_logits = use_scratch ? scratch->router_logits : make({N, num_experts});
    std::string gate_name = model.router_weight_name(layer_idx);

    // Router weight is typically not quantized (small [num_experts, hidden_size] matrix).
    if (model.weights.has_tensor(gate_name)) {
        ops::linear(router_logits, input, model.W(gate_name), nullptr);
    } else {
        // Fallback: quantized router path
        std::string gate_prefix = "layers." + std::to_string(layer_idx) + ".mlp.gate";
        dispatch_linear(model, router_logits, input, gate_prefix, nullptr);
    }

    // Copy router logits to CPU for top-k selection (only N * num_experts floats).
    tensor_t router_cpu = router_logits;
    if (router_logits->deviceType() != ZEDINFER_DEVICE_CPU) {
        router_cpu = router_logits->to(ZEDINFER_DEVICE_CPU, 0);
    }
    if (router_cpu->dtype() != ZEDINFER_DTYPE_F32) {
        router_cpu = router_cpu->to(ZEDINFER_DTYPE_F32);
    }

    auto topk = ops::moe::topk_softmax(reinterpret_cast<const float*>(router_cpu->data()), N, num_experts, top_k,
                                        model.norm_topk_prob);

    // === Step 2: Initialize MoE output accumulator ===
    auto moe_output = use_scratch ? scratch->moe_output : make({N, hidden_size});
    ops::fill_zero(moe_output);

    // === Step 3: Routed expert computation ===
    auto expert_gate = use_scratch ? scratch->expert_gate : make({N, moe_inter});
    auto expert_up = use_scratch ? scratch->expert_up : make({N, moe_inter});
    auto expert_act = use_scratch ? scratch->expert_act : make({N, moe_inter});
    auto expert_down = use_scratch ? scratch->expert_down : make({N, hidden_size});

    if (N == 1) {
        // Decode path: loop over top-k experts, no token permutation needed.
        for (size_t k = 0; k < top_k; ++k) {
            int expert_id = topk.expert_ids[k];
            float weight = topk.expert_weights[k];

            auto ep = model.expert_prefix(layer_idx, expert_id);

            dispatch_linear(model, expert_gate, input, ep + "gate_proj", nullptr);
            dispatch_linear(model, expert_up, input, ep + "up_proj", nullptr);
            ops::swiglu(expert_act, expert_gate, expert_up);
            dispatch_linear(model, expert_down, expert_act, ep + "down_proj", nullptr);

            // Weighted accumulation: moe_output += weight * expert_down
            ops::add_scaled(moe_output, expert_down, weight);
        }
    } else {
        // Prefill path: for each expert, gather tokens, compute FFN, scatter back.
        std::vector<std::vector<size_t>> expert_token_indices(num_experts);
        std::vector<std::vector<float>> expert_token_weights(num_experts);

        for (size_t n = 0; n < N; ++n) {
            for (size_t k = 0; k < top_k; ++k) {
                int expert_id = topk.expert_ids[n * top_k + k];
                float weight = topk.expert_weights[n * top_k + k];
                expert_token_indices[static_cast<size_t>(expert_id)].push_back(n);
                expert_token_weights[static_cast<size_t>(expert_id)].push_back(weight);
            }
        }

        for (size_t eid = 0; eid < num_experts; ++eid) {
            auto& token_indices = expert_token_indices[eid];
            if (token_indices.empty()) {
                continue;
            }

            size_t group_size = token_indices.size();
            auto ep = model.expert_prefix(layer_idx, static_cast<int>(eid));

            // Gather tokens for this expert (device-aware row copy).
            auto gathered = make({group_size, hidden_size});
            const size_t row_bytes = hidden_size * input->elementSize();
            auto kind = (input->deviceType() == ZEDINFER_DEVICE_CPU) ? ZEDINFER_MEMCPY_H2H : ZEDINFER_MEMCPY_D2D;
            auto* api = zedinfer::core::context().runtime().api();
            for (size_t i = 0; i < group_size; ++i) {
                void* dst = gathered->data() + static_cast<ptrdiff_t>(i * row_bytes);
                const void* src = input->data() + static_cast<ptrdiff_t>(token_indices[i] * row_bytes);
                api->memcpy_sync(dst, src, row_bytes, kind);
            }

            auto g_gate = make({group_size, moe_inter});
            auto g_up = make({group_size, moe_inter});
            auto g_act = make({group_size, moe_inter});
            auto g_down = make({group_size, hidden_size});

            dispatch_linear(model, g_gate, gathered, ep + "gate_proj", nullptr);
            dispatch_linear(model, g_up, gathered, ep + "up_proj", nullptr);
            ops::swiglu(g_act, g_gate, g_up);
            dispatch_linear(model, g_down, g_act, ep + "down_proj", nullptr);

            // Scatter weighted results back to moe_output.
            for (size_t i = 0; i < group_size; ++i) {
                float weight = expert_token_weights[eid][i];
                auto out_row = moe_output->slice(0, static_cast<int64_t>(token_indices[i]),
                                                 static_cast<int64_t>(token_indices[i]) + 1);
                auto down_row = g_down->slice(0, static_cast<int64_t>(i), static_cast<int64_t>(i) + 1);
                ops::add_scaled(out_row, down_row, weight);
            }
        }
    }

    // === Step 4: Shared expert (if present — always active on all tokens) ===
    auto sp = model.shared_expert_prefix(layer_idx);
    bool has_shared_expert = model.weights.has_tensor(sp + "gate_proj.weight")
                          || model.has_quantized_linear(sp + "gate_proj");

    if (has_shared_expert) {
        auto shared_gate = use_scratch ? scratch->shared_gate : make({N, shared_inter});
        auto shared_up = use_scratch ? scratch->shared_up : make({N, shared_inter});
        auto shared_act = use_scratch ? scratch->shared_act : make({N, shared_inter});
        auto shared_down = use_scratch ? scratch->shared_down : make({N, hidden_size});

        dispatch_linear(model, shared_gate, input, sp + "gate_proj", nullptr);
        dispatch_linear(model, shared_up, input, sp + "up_proj", nullptr);
        ops::swiglu(shared_act, shared_gate, shared_up);
        dispatch_linear(model, shared_down, shared_act, sp + "down_proj", nullptr);

        ops::add(output, moe_output, shared_down);
    } else {
        // No shared expert — copy routed output directly.
        auto* api = zedinfer::core::context().runtime().api();
        auto kind = (output->deviceType() == ZEDINFER_DEVICE_CPU) ? ZEDINFER_MEMCPY_H2H : ZEDINFER_MEMCPY_D2D;
        api->memcpy_sync(output->data(), moe_output->data(), output->numel() * output->elementSize(), kind);
    }
}

} // namespace zedinfer::model
