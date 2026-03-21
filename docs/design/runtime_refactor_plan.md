# Runtime Refactor Plan

## Overview

This document details the transition from graph-based execution to direct model forward with pre-allocated scratch buffers, and the introduction of request/session/scheduler abstractions for multi-user serving.

## Phase 1: Direct Model Forward

### Current Path (to be bypassed)

```
engine.cpp:200  ->  executor_->forward(kvcache, input_ids, past_len)
executor.cpp:53 ->  walks graph nodes in topo order
executor.cpp:130->  execute_node() per node: gather inputs, alloc output, dispatch op
executor.cpp:213->  execute_op() switch on OpType
```

### New Path

```
engine.cpp      ->  model_->forward(input_ids, past_len, kvcache)
qwen2_forward.cpp-> layer-by-layer: norm -> attn -> residual -> norm -> mlp -> residual
                    ops::rms_norm(), ops::linear(), ops::rope(), ops::self_attention(), etc.
```

### Implementation: Qwen2Model::forward()

New file: `src/frontend/models/qwen2_forward.cpp`

```cpp
tensor_t Qwen2Model::forward(
    const std::vector<int> &input_ids,
    int past_len,
    kvcache::KVCache &kvcache,
    ScratchBuffers &scratch,
    const ForwardConfig &fwd_config) {

    const int seq_len = input_ids.size();
    const auto &cfg = config_;

    // 1. Prepare input_ids tensor
    auto ids_tensor = /* load input_ids into scratch or small temp */;

    // 2. Embedding
    auto hidden = scratch.hidden->view({seq_len, cfg.hidden_size});
    ops::embedding(hidden, ids_tensor, weights_.get_tensor("embed_tokens.weight"));

    // 3. Transformer layers
    for (int layer = 0; layer < cfg.num_hidden_layers; ++layer) {
        auto residual = hidden;  // alias, no copy

        // Input norm
        auto normed = scratch.norm_buf->view({seq_len, cfg.hidden_size});
        ops::rms_norm(normed, hidden, get_weight(layer, "input_layernorm.weight"), cfg.rms_norm_eps);

        // Q/K/V projections
        auto q = scratch.q_proj->view({seq_len, cfg.hidden_size});
        auto k = kvcache.get_k_write_slot(layer, past_len, seq_len);  // write directly to cache
        auto v = kvcache.get_v_write_slot(layer, past_len, seq_len);  // write directly to cache

        ops::linear(q, normed, get_weight(layer, "self_attn.q_proj.weight"),
                     get_weight(layer, "self_attn.q_proj.bias"));
        ops::linear(k, normed, get_weight(layer, "self_attn.k_proj.weight"),
                     get_weight(layer, "self_attn.k_proj.bias"));
        ops::linear(v, normed, get_weight(layer, "self_attn.v_proj.weight"),
                     get_weight(layer, "self_attn.v_proj.bias"));

        // RoPE
        auto q_rope = q->view({seq_len, nhead, head_dim});
        auto k_rope = k->view({seq_len, nkvhead, head_dim});
        ops::rope(q_rope, q_rope, position_ids, cfg.rope_theta);
        ops::rope(k_rope, k_rope, position_ids, cfg.rope_theta);

        // Attention
        auto k_full = kvcache.get_k_cache_slice(layer, past_len + seq_len);
        auto v_full = kvcache.get_v_cache_slice(layer, past_len + seq_len);
        auto attn_out = scratch.attn_out->view({seq_len, nhead, head_dim});
        ops::self_attention(attn_out, q_rope, k_full, v_full, scale);

        // O projection
        auto attn_flat = attn_out->view({seq_len, cfg.hidden_size});
        auto o_out = scratch.norm_buf->view({seq_len, cfg.hidden_size}); // reuse norm_buf
        ops::linear(o_out, attn_flat, get_weight(layer, "self_attn.o_proj.weight"), nullptr);

        // Residual 1
        ops::add(hidden, residual, o_out);

        // Post-attn norm
        residual = hidden;
        ops::rms_norm(normed, hidden, get_weight(layer, "post_attention_layernorm.weight"), cfg.rms_norm_eps);

        // MLP: gate + up -> swiglu -> down
        auto gate = scratch.gate->view({seq_len, cfg.intermediate_size});
        auto up = scratch.up->view({seq_len, cfg.intermediate_size});
        ops::linear(gate, normed, get_weight(layer, "mlp.gate_proj.weight"), nullptr);
        ops::linear(up, normed, get_weight(layer, "mlp.up_proj.weight"), nullptr);
        ops::swiglu(gate, gate, up);  // in-place on gate

        auto down = scratch.norm_buf->view({seq_len, cfg.hidden_size});
        ops::linear(down, gate, get_weight(layer, "mlp.down_proj.weight"), nullptr);

        // Residual 2
        ops::add(hidden, residual, down);
    }

    // 4. Final norm + LM head
    auto normed = scratch.norm_buf->view({seq_len, cfg.hidden_size});
    ops::rms_norm(normed, hidden, weights_.get_tensor("norm.weight"), cfg.rms_norm_eps);

    auto logits = scratch.logits->view({seq_len, cfg.vocab_size});
    ops::linear(logits, normed, weights_.get_tensor("lm_head.weight"), nullptr);

    kvcache.update_seq_len(seq_len);
    return logits;
}
```

### ScratchBuffers

```cpp
// include/zedinfer/scratch.hpp
struct ScratchBuffers {
    static ScratchBuffers allocate(
        const model::ModelConfig &config,
        int max_batch_tokens,
        zedinferDeviceType_t device,
        int device_id);

    // Each buffer sized for max_batch_tokens
    tensor_t hidden;
    tensor_t residual;  // may alias hidden via view if in-place is safe
    tensor_t norm_buf;
    tensor_t q_proj;
    tensor_t k_proj;    // only needed if KV cache doesn't provide write slots
    tensor_t v_proj;
    tensor_t attn_out;
    tensor_t gate;
    tensor_t up;
    tensor_t logits;
};
```

Scratch buffers are allocated once at engine initialization. `max_batch_tokens` is a configuration parameter (e.g., 2048). Every forward call views into these buffers at the needed size. Zero per-token allocation overhead.

### Buffer Reuse Analysis

Within a single layer, the following reuse is safe (no overlap in live ranges):
- `norm_buf` is consumed by q/k/v projections before being rewritten by o_proj
- `gate` is consumed by swiglu, then can be reused by down_proj input
- `q_proj` is consumed by attention before next layer's q_proj

## Phase 2: Request / Session Abstraction

### Request

```cpp
// include/zedinfer/request.hpp
struct InferenceRequest {
    uint64_t request_id;
    std::string session_id;           // links to session's KV cache
    std::vector<int> input_ids;       // tokenized input
    GenerationConfig config;

    // Scheduling state
    enum class Phase { PREFILL, DECODE, COMPLETE };
    Phase phase = Phase::PREFILL;
    int generated_count = 0;
    int last_token = -1;

    // Output collection
    std::vector<int> output_ids;
    std::function<void(const std::string&)> stream_callback;

    // Timing
    std::chrono::steady_clock::time_point arrival_time;
    GenerationStats stats;
};
```

### Session Registry

```cpp
// include/zedinfer/session_registry.hpp
class SessionRegistry {
public:
    InferenceSession* get_or_create(const std::string &session_id, const GenerationConfig &config);
    void remove(const std::string &session_id);
    void cleanup_expired(std::chrono::seconds max_idle);
private:
    std::unordered_map<std::string, std::unique_ptr<InferenceSession>> sessions_;
    std::mutex mutex_;
};
```

### Engine Becomes Stateless

Current `InferenceEngine` (`engine.hpp:25`):
- Remove `last_stats_` member
- `generate_tokens()` returns stats alongside output IDs
- `create_session()` delegates to `SessionRegistry`

```cpp
struct GenerationResult {
    std::vector<int> output_ids;
    GenerationStats stats;
};

GenerationResult generate_tokens(
    kvcache::KVCache &kvcache,
    const std::vector<int> &input_ids,
    const GenerationConfig &config);
```

## Phase 3: Scheduler Foundation

See `docs/continuous_batching_design.md` for full scheduler design.

The scheduler thread:
1. Receives `InferenceRequest` from HTTP handlers or CLI
2. Maintains a priority queue (FIFO by default)
3. Assembles a batch of decode requests + at most one prefill request
4. Calls `model->forward_batch(batch, kvcache_pool)` via the worker thread
5. Routes output logits to per-request samplers
6. Enqueues completed tokens to response streams

## Phase 4: Chat Template Extraction

Current hardcoded template in `session.cpp:38-65`:
```cpp
input += "<begin_of_sentence>";
input += "<User>" + user_input;
input += "<Assistant><think>\n";
```

Extract to model config:

```cpp
struct ChatTemplate {
    std::string bos;              // "<begin_of_sentence>"
    std::string user_prefix;      // "<User>"
    std::string user_suffix;      // ""
    std::string assistant_prefix; // "<Assistant><think>\n"
    std::string assistant_suffix; // ""
    bool add_bos_first_turn_only = true;
};
```

Load from `tokenizer_config.json` or a dedicated `chat_template.json` in the model directory.

## Files Affected Summary

| Phase | New Files | Modified Files | Removed Dependencies |
|-------|-----------|---------------|---------------------|
| 1 (Forward) | `src/frontend/models/qwen2_forward.cpp`, `qwen3_forward.cpp`, `include/zedinfer/scratch.hpp` | `include/frontend/models/{qwen2,qwen3}.hpp`, `include/zedinfer/engine.hpp`, `src/zedinfer/engine.cpp` | `GraphExecutor`, `GraphBuilder`, `ComputeGraph` bypassed |
| 2 (Request) | `include/zedinfer/request.hpp`, `include/zedinfer/session_registry.hpp` | `include/zedinfer/engine.hpp`, `src/zedinfer/engine.cpp` | `last_stats_` removed from engine |
| 3 (Scheduler) | `include/zedinfer/scheduler.hpp`, `src/zedinfer/scheduler.cpp` | `src/zedinfer/engine.cpp` (call scheduler instead of direct generate) | Direct generate loop replaced |
| 4 (Template) | `include/zedinfer/chat_template.hpp` | `src/zedinfer/session.cpp`, model loading | Hardcoded strings removed |
