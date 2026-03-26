#pragma once

#include "backend/tensor/tensor.hpp"
#include "frontend/models/base.hpp"
#include "zedinfer/activation.hpp"

#include <memory>

namespace zedinfer::model {

/**
 * Pre-allocated scratch buffers for single-token decode (N=1).
 * Eliminates ~500 Tensor::create() / BestFitMemoryPool round-trips per decode step.
 * Allocated once at engine init (~400-500 KB), reused across all decode steps.
 */
struct DecodeScratch {
    // Inputs (overwritten each step via load)
    tensor_t ids;     // {1} I32
    tensor_t pos_ids; // {1} I64

    // Embedding / layer output (ping-pong pair)
    tensor_t hidden;     // {1, H}
    tensor_t hidden_out; // {1, H}

    // Per-layer reusable buffers
    tensor_t normed;      // {1, H}
    tensor_t q;           // {1, H}
    tensor_t k;           // {1, kv_dim}
    tensor_t v;           // {1, kv_dim}
    tensor_t q_normed;    // {nhead, head_dim}    (Qwen3 only, nullptr for Qwen2)
    tensor_t k_normed;    // {nkvhead, head_dim}  (Qwen3 only, nullptr for Qwen2)
    tensor_t q_rope;      // {1, nhead, head_dim}
    tensor_t k_rope;      // {1, nkvhead, head_dim}
    tensor_t attn_out;    // {1, nhead, head_dim}
    tensor_t o;           // {1, H}
    tensor_t h1;          // {1, H}
    tensor_t normed_post; // {1, H}
    tensor_t gate;        // {1, inter}
    tensor_t up;          // {1, inter}
    tensor_t act;         // {1, inter}
    tensor_t down;        // {1, H}

    // Output head
    tensor_t final_normed; // {1, H}
    tensor_t logits;       // {1, V}

    static std::unique_ptr<DecodeScratch> create(const ModelConfig& cfg, bool has_qk_norm,
                                                 const ExecutorConfig& exec_config);
};

} // namespace zedinfer::model
