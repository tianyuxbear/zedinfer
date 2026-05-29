# Per-Layer Weight Pre-Resolution (kill hot-path string lookups)

> Date: 2026-05-29
> Status: **Measured 2026-05-29 — NOT worth implementing now (deprioritized).** Kept as the design if a
> future profile shows the forward CPU path matters.
> Scope: `ModelForwardConfig`, `Model::forward_config()`, `transformer_forward.cpp`, the serving loop
> Risk: medium (touches the decode hot path; must be numerically validated against a real model)

---

## 0. Measurement verdict (why this is deprioritized)

Instrumented `serving_loop` step() to time `forward_config()` against the rest of the forward on a free B200:

| Model | `forward_config()` | rest of forward | fc share of step |
|-------|--------------------|-----------------|------------------|
| Qwen3-30B-A3B-GPTQ-Int4 (MoE) | ~8.1 µs/step | ~21–25 ms/step | **0.03–0.04%** |
| Qwen3-8B (dense) | ~0.03 µs/step | ~3.3 ms/step | **0.00%** |

So part §1.1 (the per-step `forward_config()` rebuild, including the 48-layer `router_weights` string loop)
is **0.03% of a decode step on MoE and ~0% on dense** — pure noise. Part §1.2 (the `W()` string lookups
inside `transformer_forward`) is bounded by the same per-lookup cost (~0.17 µs × a few hundred lookups ≈
tens of µs) against multi-millisecond steps, i.e. roughly ~1% at most on dense and far less on MoE.

Decode is dominated by the GEMMs — the MoE int4 path is launch-bound (GPU ~70% idle, ~990 tiny launches/token;
see roadmap Priority 3). Refactoring the forward CPU path for ≤1% while touching the core loop for every model
family is poor ROI. **The high-leverage work is the fused MoE GEMM (roadmap Priority 3) and FlashInfer planner
overhead (Priority 1), not this.** Revisit only if a profile ever shows the host-side forward path on the
critical path.

The rest of this document remains the implementation plan should that day come.

---

## 1. Problem (with code evidence)

### 1.1 `forward_config()` is rebuilt every decode step

`src/zedinfer/serving_loop.cpp:220` calls `engine_->model().forward_config()` **inside `step()`** —
i.e. a fresh `ModelForwardConfig` is constructed on **every forward pass**, which for decode is every
token. The hybrid path is the same (`serving_loop.cpp:178-179` builds `hybrid_forward_config*()` per step).

`Qwen3MoEModel::forward_config()` (`src/frontend/models/qwen3_moe.cpp:69-113`) does real per-call work:

- `detect_intermediate_size(...)` — string concat + `has_tensor` + `get_tensor` + `shape()` reads.
- `peek_expert(0,0)` metadata probe.
- **Rebuilds `router_weights` every call** (lines 106-110): a loop over *all* layers doing
  `router_weight_name(l)` (string concat) + `has_tensor` + `get_tensor` map lookups.

The in-code comment claims `router_weights` is a cache "so compute_router_topk can index instead of
doing a string-key map lookup every layer × decode step" — but because the whole `ModelForwardConfig`
is reconstructed every step, that cache is **rebuilt from scratch every step**. The intended saving is lost.

### 1.2 Dense forward still does string-keyed weight lookups per projection

`transformer_forward.cpp` resolves each weight via `model.W(prefix + ".weight")`
(`forward_config.hpp:80` → `ModelWeights::get_tensor`, `base.hpp:143`): a `std::string` concatenation
plus an `unordered_map<string,tensor_t>` lookup. For N=1 decode this runs ~7-9 times per layer per token
(~200+ string+hash ops/token on a 28-layer model). The GEMMs dominate wall-clock, but this is pure,
avoidable per-token overhead and it scales with layer count.

---

## 2. Goal

Resolve every weight tensor a layer needs **once at load time** into an indexed, pointer-based structure,
and stop reconstructing `ModelForwardConfig` per step. Decode-time weight access becomes `layer_weights_[L].q_proj`
(a pointer copy), with zero string work and zero map lookups in the hot path. Numerically identical output.

---

## 3. Proposed design

### 3.1 Cache `ModelForwardConfig` in the `Model`

`ModelForwardConfig` holds only model-stable state (`const ModelConfig&`, `const ModelWeights&`,
`ExpertPool*`, router pointers, quant params) — nothing per-request. So it can be built once and reused.

- Add a `mutable std::unique_ptr<ModelForwardConfig> cached_fwd_cfg_` (and hybrid equivalent) to each model.
- `forward_config()` builds it on first call and returns a `const ModelForwardConfig&` thereafter.
- Confirm no per-request fields leak into `HybridForwardConfig` before caching it (audit `ssm_pool`,
  any request-scoped members — those must stay as call-time arguments, not cached config).

### 3.2 Pre-resolved per-layer weight table

Add to `ModelForwardConfig` (populated once when the config is built):

```cpp
struct LayerWeights {
    tensor_t input_layernorm, post_attention_layernorm;
    tensor_t q_proj, k_proj, v_proj, o_proj;          // dense weights (null if quantized)
    QuantizedLinearRef q_proj_q, k_proj_q, ...;        // quantized refs (when packed)
    tensor_t q_bias, k_bias, v_bias;                   // null unless has_qkv_bias
    tensor_t q_norm, k_norm;                           // null unless has_qk_norm
    tensor_t gate_proj, up_proj, down_proj;            // dense MLP (null on MoE layers)
    // ... MoE router handled by existing router_weights[]
};
std::vector<LayerWeights> layer_weights;               // indexed by layer
tensor_t embed_tokens, final_norm, lm_head;            // top-level
```

`transformer_forward` then indexes `model.layer_weights[L]` instead of building prefixes. `dispatch_linear`
gains an overload taking a pre-resolved `(dense_weight | QuantizedLinearRef)` instead of a string prefix.

---

## 4. Migration steps (small, staged)

1. Cache `ModelForwardConfig`/`HybridForwardConfig` in the model; change `forward_config()` to return a
   const reference. Update call sites (`serving_loop.cpp:178-180,220`, `profiler.cpp:27,71`,
   `engine.cpp:348,436`). **Validate fingerprint** — this step alone removes the per-step rebuild.
2. Add `layer_weights` resolution to the config builder (populate from the existing string lookups, once).
3. Add `dispatch_linear` overloads that take resolved weights; switch `transformer_forward` to them.
4. Repeat (2)-(3) for `hybrid_transformer_forward` and `moe_forward`.

Each step is independently buildable and fingerprint-validatable.

---

## 5. Risks & validation

- **Numerical drift:** none expected (same tensors, same order). Validate by greedy A/B (see §6).
- **Lifetime:** cached config holds references to `ModelConfig`/`ModelWeights` owned by the model — safe as
  long as the config cache is a model member (same lifetime). Must not outlive the model.
- **Per-request state:** must stay out of the cached config (audit hybrid config).
- **Coverage:** the fingerprint harness covers one model/path at a time. Validate each family (Qwen2 dense,
  Qwen3 dense, Qwen3-MoE, Qwen3.5 hybrid, Qwen3.5-MoE) before declaring done.

---

## 6. Test plan

Deterministic greedy A/B fingerprint (already used to validate the refactor branch):

```
env CUDA_VISIBLE_DEVICES=<gpu> ZEDINFER_FORCE_ARGMAX=1 ZEDINFER_DUMP_TOKEN_IDS=1 \
  xmake run ping <model> --nvidia --gpu-memory-utilization 0.2 \
  --prompt "What is the capital of France? Answer in one sentence." --max-new-tokens 48 \
  2> tokens.txt
grep '\[zedinfer-tok\]' tokens.txt | md5sum   # must match the pre-refactor fingerprint
```

Baseline fingerprint (DeepSeek-R1-Distill-Qwen-1.5B, 48 tokens): `2b2c1e5121e3d94940cffafdfb9c7af1`.
Run per model family; the md5 must be unchanged before/after each migration step.
