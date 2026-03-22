# R1: Unify Model Forward - Detailed Refactoring Plan

**Goal:** Consolidate 4 near-identical forward files into a single shared transformer loop, with model-specific behavior expressed as configuration/hooks and execution mode (single/batch) expressed as a strategy object.

---

## 1. Problem Analysis

### 1.1 Current State: 4 Files, ~90% Identical

```
qwen2.cpp               → forward()         124 lines (layer loop)
qwen3.cpp               → forward()         128 lines (layer loop)
qwen2_forward_batch.cpp → forward_batch()    150 lines (layer loop)
qwen3_forward_batch.cpp → forward_batch()    155 lines (layer loop)
                                        Total: ~557 lines
```

### 1.2 Precise Diff: Qwen2 vs Qwen3

Only **2 differences** in the entire layer loop:

**Difference 1: QKV Bias**
```cpp
// Qwen2: has bias
ops::linear(q, normed, W("q_proj.weight"), W("q_proj.bias"));
ops::linear(k, normed, W("k_proj.weight"), W("k_proj.bias"));
ops::linear(v, normed, W("v_proj.weight"), W("v_proj.bias"));

// Qwen3: no bias
ops::linear(q, normed, W("q_proj.weight"), nullptr);
ops::linear(k, normed, W("k_proj.weight"), nullptr);
ops::linear(v, normed, W("v_proj.weight"), nullptr);
```

**Difference 2: Per-Head Q/K Norm (Qwen3 only)**
```cpp
// Qwen2: Q goes directly to RoPE
ops::rope(q_rope, q->view({sl, nhead, head_dim}), pos_ids, theta);

// Qwen3: Q/K get per-head RMSNorm before RoPE
auto q_normed = make({sl * nhead, head_dim});
ops::rms_norm(q_normed, q->view({sl * nhead, head_dim}), W("q_norm.weight"), eps);
auto k_normed = make({sl * nkvhead, head_dim});
ops::rms_norm(k_normed, k->view({sl * nkvhead, head_dim}), W("k_norm.weight"), eps);
ops::rope(q_rope, q_normed->view({sl, nhead, head_dim}), pos_ids, theta);
```

Everything else (embedding, RoPE, attention, O projection, residual, MLP, final norm, lm_head) is **identical**.

### 1.3 Precise Diff: forward() vs forward_batch()

**Input handling:**
```cpp
// forward(): single sequence
size_t sl = input_ids.size();
auto ids = make_typed({sl}, I32);
ids->load(input_ids.data());

// forward_batch(): concatenated batch
int total = batch.total_tokens();
auto ids = make_typed({total}, I32);
ids->load(batch.token_ids.data());
```

**KV write:**
```cpp
// forward(): via KVCache interface
auto v_slot = kvcache.get_v_cache_slice(L, past_len, sl);
ops::linear(v_slot->view({sl, kv_dim}), normed, W("v_proj.weight"), ...);

// forward_batch(): scatter to blocks per slot
ops::linear(v_all, normed, W("v_proj.weight"), ...);
for (slot : batch.slots) {
    scatter_kv_to_blocks(v_all.slice(slot), slot.request->block_table, ...);
}
```

**Attention:**
```cpp
// forward(): 3-way dispatch (paged decode / paged prefill / contiguous)
if (sl == 1 && kvcache.is_paged()) { paged_decode(...); }
else if (kvcache.is_paged()) { scatter + paged_prefill(...); }
else { self_attention(...); }

// forward_batch(): batched decode + per-request prefill
if (num_decode > 0) { paged_decode_batched(...); }
for (prefill_slot) { paged_prefill(...); }
```

**Post-forward:**
```cpp
// forward(): update KV cache length
kvcache.update_seq_len(sl);

// forward_batch(): nothing (scheduler updates block_table.seq_len)
```

**Per-token ops (norm, linear, rope, MLP) are byte-for-byte identical** — just `sl` vs `total` as the batch dimension.

---

## 2. Design

### 2.1 Core Idea

Split the forward into three orthogonal concerns:

```
Model Config     → what weights exist, Q/K bias, Q/K norm
                   (differs per model family)

Execution Mode   → how tokens are prepared, how KV is written, how attention runs
                   (differs per forward() vs forward_batch())

Transformer Loop → the shared layer computation
                   (identical across all 4 files)
```

### 2.2 ModelForwardConfig (model-specific, static)

```cpp
// include/frontend/models/forward_config.hpp

struct ModelForwardConfig {
    const model::ModelConfig &config;
    const model::ModelWeights &weights;

    // Model-specific features
    bool has_qkv_bias = false;       // Qwen2: true, Qwen3: false
    bool has_qk_norm = false;        // Qwen2: false, Qwen3: true

    // Weight name pattern (all current models use the same names)
    std::string weight_prefix(int layer) const {
        return "layers." + std::to_string(layer) + ".";
    }

    // Weight accessor
    tensor_t W(const std::string &name) const {
        return weights.get_tensor(name);
    }

    // QKV bias (nullptr if not has_qkv_bias)
    tensor_t q_bias(const std::string &prefix) const {
        return has_qkv_bias ? W(prefix + "self_attn.q_proj.bias") : nullptr;
    }
    tensor_t k_bias(const std::string &prefix) const {
        return has_qkv_bias ? W(prefix + "self_attn.k_proj.bias") : nullptr;
    }
    tensor_t v_bias(const std::string &prefix) const {
        return has_qkv_bias ? W(prefix + "self_attn.v_proj.bias") : nullptr;
    }
};
```

### 2.3 ForwardContext (execution-mode-specific, per-call)

```cpp
// include/frontend/models/forward_context.hpp

class ForwardContext {
public:
    virtual ~ForwardContext() = default;

    // Token count for this forward call
    virtual int num_tokens() const = 0;

    // Prepare input tensors (token IDs, position IDs)
    virtual void prepare_inputs(tensor_t &ids, tensor_t &pos_ids,
                                const ExecutorConfig &exec_config) = 0;

    // Write K/V for a layer after projection + RoPE
    virtual void write_kv(int layer, tensor_t k, tensor_t v) = 0;

    // Run attention for a layer, given Q
    virtual tensor_t attend(int layer, tensor_t q_rope, float scale,
                            const ExecutorConfig &exec_config,
                            int nhead, int nkvhead, int head_dim) = 0;

    // Post-forward finalization
    virtual void finalize(int num_tokens) = 0;
};
```

**Two implementations:**

```cpp
// Single-request context (wraps KVCache)
class SingleForwardContext : public ForwardContext {
    const std::vector<int> &input_ids_;
    int past_len_;
    kvcache::KVCache &kvcache_;
    // ...
    void write_kv(int layer, tensor_t k, tensor_t v) override {
        // Write to kvcache via get_k_cache_slice / direct-to-block
    }
    tensor_t attend(int layer, tensor_t q, ...) override {
        // 3-way dispatch: paged decode / paged prefill / contiguous
    }
    void finalize(int n) override { kvcache_.update_seq_len(n); }
};

// Batched context (wraps BatchContext + BlockPool)
class BatchedForwardContext : public ForwardContext {
    const BatchContext &batch_;
    kvcache::BlockAllocator &allocator_;
    // ...
    void write_kv(int layer, tensor_t k, tensor_t v) override {
        // Scatter per slot to blocks
    }
    tensor_t attend(int layer, tensor_t q, ...) override {
        // Batched decode + per-request prefill
    }
    void finalize(int n) override { /* noop, scheduler handles */ }
};
```

### 2.4 Unified Transformer Loop

```cpp
// src/frontend/models/transformer_forward.cpp

tensor_t transformer_forward(
    const ModelForwardConfig &model,
    ForwardContext &ctx,
    const ExecutorConfig &exec_config) {

    const auto &cfg = model.config;
    const int N = ctx.num_tokens();
    const size_t hidden = cfg.hidden_size;
    const size_t nhead = cfg.num_attention_heads;
    const size_t nkvhead = cfg.num_key_value_heads;
    const size_t head_dim = hidden / nhead;
    const size_t kv_dim = nkvhead * head_dim;
    const size_t inter = cfg.intermediate_size;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    auto make = [&](std::vector<size_t> shape) { ... };
    auto make_typed = [&](std::vector<size_t> shape, auto dtype) { ... };

    // Input
    tensor_t ids, pos_ids;
    ctx.prepare_inputs(ids, pos_ids, exec_config);

    // Embedding
    auto hidden_states = make({N, hidden});
    ops::embedding(hidden_states, ids, model.W("embed_tokens.weight"));

    // Layers
    for (size_t L = 0; L < cfg.num_hidden_layers; ++L) {
        auto p = model.weight_prefix(L);

        auto normed = make({N, hidden});
        ops::rms_norm(normed, hidden_states, model.W(p + "input_layernorm.weight"), cfg.rms_norm_eps);

        // Q/K/V projection
        auto q = make({N, hidden});
        ops::linear(q, normed, model.W(p + "self_attn.q_proj.weight"), model.q_bias(p));

        auto k = make({N, kv_dim});
        ops::linear(k, normed, model.W(p + "self_attn.k_proj.weight"), model.k_bias(p));

        auto v = make({N, kv_dim});
        ops::linear(v, normed, model.W(p + "self_attn.v_proj.weight"), model.v_bias(p));

        // Optional Q/K norm (Qwen3)
        if (model.has_qk_norm) {
            auto q_n = make({N * nhead, head_dim});
            ops::rms_norm(q_n, q->view({N * nhead, head_dim}),
                          model.W(p + "self_attn.q_norm.weight"), cfg.rms_norm_eps);
            auto k_n = make({N * nkvhead, head_dim});
            ops::rms_norm(k_n, k->view({N * nkvhead, head_dim}),
                          model.W(p + "self_attn.k_norm.weight"), cfg.rms_norm_eps);
            q = q_n->view({N, nhead, head_dim});  // use normed version
            k = k_n->view({N, nkvhead, head_dim});
        } else {
            q = q->view({N, nhead, head_dim});
            k = k->view({N, nkvhead, head_dim});
        }

        // RoPE
        auto q_rope = make({N, nhead, head_dim});
        ops::rope(q_rope, q, pos_ids, cfg.rope_theta);

        auto k_rope = make({N, nkvhead, head_dim});
        ops::rope(k_rope, k, pos_ids, cfg.rope_theta);

        // KV write (context-specific: KVCache or scatter-to-blocks)
        ctx.write_kv(L, k_rope, v);

        // Attention (context-specific: single/batched, paged/contiguous)
        auto attn = ctx.attend(L, q_rope, scale, exec_config, nhead, nkvhead, head_dim);

        // O projection + residual
        auto o = make({N, hidden});
        ops::linear(o, attn->view({N, hidden}), model.W(p + "self_attn.o_proj.weight"), nullptr);

        auto h1 = make({N, hidden});
        ops::add(h1, hidden_states, o);

        // MLP
        normed = make({N, hidden});
        ops::rms_norm(normed, h1, model.W(p + "post_attention_layernorm.weight"), cfg.rms_norm_eps);

        auto gate = make({N, inter});
        auto up = make({N, inter});
        ops::linear(gate, normed, model.W(p + "mlp.gate_proj.weight"), nullptr);
        ops::linear(up, normed, model.W(p + "mlp.up_proj.weight"), nullptr);

        auto act = make({N, inter});
        ops::swiglu(act, gate, up);

        auto down = make({N, hidden});
        ops::linear(down, act, model.W(p + "mlp.down_proj.weight"), nullptr);

        hidden_states = make({N, hidden});
        ops::add(hidden_states, h1, down);
    }

    // Output head
    auto normed = make({N, hidden});
    ops::rms_norm(normed, hidden_states, model.W("norm.weight"), cfg.rms_norm_eps);

    auto logits = make({N, cfg.vocab_size});
    ops::linear(logits, normed, model.W("lm_head.weight"), nullptr);

    ctx.finalize(N);
    return logits;
}
```

### 2.5 Model Class Becomes Thin

```cpp
// qwen2.cpp — after refactoring
tensor_t Qwen2Model::forward(
    const std::vector<int> &input_ids, int past_len,
    kvcache::KVCache &kvcache, const ExecutorConfig &exec_config) {

    ModelForwardConfig model_cfg{config_, *weights_, /*has_qkv_bias=*/true, /*has_qk_norm=*/false};
    SingleForwardContext ctx(input_ids, past_len, kvcache, exec_config);
    return transformer_forward(model_cfg, ctx, exec_config);
}

tensor_t Qwen2Model::forward_batch(
    const BatchContext &batch, kvcache::BlockAllocator &allocator,
    const ExecutorConfig &exec_config) {

    ModelForwardConfig model_cfg{config_, *weights_, true, false};
    BatchedForwardContext ctx(batch, allocator, exec_config);
    return transformer_forward(model_cfg, ctx, exec_config);
}

// qwen3.cpp — only differs in config flags
tensor_t Qwen3Model::forward(...) {
    ModelForwardConfig model_cfg{config_, *weights_, /*has_qkv_bias=*/false, /*has_qk_norm=*/true};
    SingleForwardContext ctx(input_ids, past_len, kvcache, exec_config);
    return transformer_forward(model_cfg, ctx, exec_config);
}
```

**Adding a new model (e.g., Llama):**
```cpp
tensor_t LlamaModel::forward(...) {
    ModelForwardConfig model_cfg{config_, *weights_, false, false}; // no bias, no Q/K norm
    SingleForwardContext ctx(...);
    return transformer_forward(model_cfg, ctx, exec_config);
}
```

One line of config. Zero new forward logic.

---

## 3. File Structure After Refactoring

### New Files

| File | Responsibility |
|------|---------------|
| `include/frontend/models/forward_config.hpp` | `ModelForwardConfig` struct |
| `include/frontend/models/forward_context.hpp` | `ForwardContext` base, `SingleForwardContext`, `BatchedForwardContext` |
| `src/frontend/models/forward_context.cpp` | Context implementations (KV write, attention dispatch) |
| `src/frontend/models/transformer_forward.cpp` | The single shared transformer loop |

### Modified Files

| File | Change |
|------|--------|
| `qwen2.cpp` | `forward()` → 5-line delegation to `transformer_forward` |
| `qwen3.cpp` | `forward()` → 5-line delegation |

### Deleted Files

| File | Reason |
|------|--------|
| `qwen2_forward_batch.cpp` | Merged into `transformer_forward` via `BatchedForwardContext` |
| `qwen3_forward_batch.cpp` | Same |

### Line Count

```
Before: 4 files, ~557 lines of layer loop code
After:  1 file (transformer_forward.cpp) ~100 lines
        + forward_context.cpp ~150 lines (write_kv, attend implementations)
        + forward_config.hpp ~40 lines
        = ~290 lines total, zero duplication
```

---

## 4. Implementation Tasks

### Task 1: Create ModelForwardConfig

- [ ] Define `ModelForwardConfig` in `include/frontend/models/forward_config.hpp`
- [ ] Weight accessor, bias helpers, QK norm flag
- [ ] Build to verify

### Task 2: Create ForwardContext hierarchy

- [ ] Define `ForwardContext` abstract base in `include/frontend/models/forward_context.hpp`
- [ ] Implement `SingleForwardContext` (wraps KVCache, handles paged/non-paged dispatch)
- [ ] Implement `BatchedForwardContext` (wraps BatchContext + BlockPool, handles scatter + batched attention)
- [ ] Move scatter_kv_to_blocks into `BatchedForwardContext::write_kv()`
- [ ] Build to verify

### Task 3: Extract transformer_forward loop

- [ ] Create `src/frontend/models/transformer_forward.cpp`
- [ ] Move the shared layer loop from `qwen2.cpp` into `transformer_forward()`
- [ ] Parameterize: `has_qkv_bias` → bias accessor, `has_qk_norm` → optional norm step
- [ ] Replace `sl`/`total` with `ctx.num_tokens()`
- [ ] Replace KV write with `ctx.write_kv()`
- [ ] Replace attention with `ctx.attend()`
- [ ] Build to verify

### Task 4: Convert Qwen2/Qwen3 to delegation

- [ ] Rewrite `Qwen2Model::forward()` → construct config + context, call `transformer_forward`
- [ ] Rewrite `Qwen2Model::forward_batch()` → same with `BatchedForwardContext`
- [ ] Rewrite `Qwen3Model::forward()` and `forward_batch()` → same with `has_qk_norm=true`
- [ ] Build to verify
- [ ] Run `ping` — verify bit-identical output
- [ ] Run `batch_bench` — verify batched output correct

### Task 5: Delete old files and cleanup

- [ ] Delete `qwen2_forward_batch.cpp`
- [ ] Delete `qwen3_forward_batch.cpp`
- [ ] Remove duplicate `scatter_kv_to_blocks` static functions
- [ ] Update xmake if needed (glob should auto-pick new files, auto-drop deleted files)
- [ ] Final build + test

---

## 5. Risks

| Risk | Mitigation |
|------|-----------|
| Virtual dispatch overhead in `ForwardContext` | Called once per layer per forward, not per token. ~28 calls vs millions of FLOPs. Negligible. |
| `has_qk_norm` branch in hot loop | Single branch per layer, predicted after first iteration. Zero measurable cost. |
| Tensor lifetime across context boundary | Contexts hold references, tensors are `shared_ptr`. No dangling. |
| Bit-identical output after refactoring | Run `ping` before/after. Logic is transplanted, not rewritten. |

---

## 6. Impact on Future Work

| Feature | Before (4 files) | After (unified) |
|---------|------------------|-----------------|
| Add Llama model | Write 4 forward files | Set 2 flags in `ModelForwardConfig` |
| Add Mistral model | Write 4 forward files | Set flags + optional sliding window hook |
| Integrate FlashAttention | Modify 4 attention dispatch blocks | Modify `ForwardContext::attend()` once |
| Add CUDA graph capture | 4 capture points | 1 capture point in `transformer_forward` |
| Add operator fusion | 4 identical MLP blocks to fuse | 1 MLP block in shared loop |
| KV cache INT8 | Modify 4 KV write paths | Modify `ForwardContext::write_kv()` once |
