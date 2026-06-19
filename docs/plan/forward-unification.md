# Unify `transformer_forward` and `hybrid_transformer_forward`

> Date: 2026-05-29
> Status: Draft (design only — not yet implemented)
> Scope: `src/frontend/models/transformer_forward.cpp`, `hybrid_transformer_forward.cpp`,
> `forward_config.hpp`, `moe_forward.cpp`
> Risk: high (the core forward loop for every model family; numerically sensitive; large surface)

---

## 1. Problem (with code evidence)

There are two parallel forward loops:

- `transformer_forward.cpp` (~155 lines) — dense / Qwen2 / Qwen3 / Qwen3-MoE.
- `hybrid_transformer_forward.cpp` (~715 lines) — Qwen3.5/3.6 hybrid (per-layer linear-attn vs full-attn).

They share the same skeleton — embed → for each layer { input_norm → attention → residual → post_norm →
MLP/MoE → residual } → final_norm → lm_head — and the same `use_scratch ? scratch->X : make(...)` buffer
pattern, and both call `moe_layer_forward` for MoE layers. But the skeleton is **duplicated**: the hybrid
file re-implements the residual structure, the scratch ping-pong, the MLP-vs-MoE dispatch, and the
final-norm/lm_head tail. The only real difference is the per-layer attention:

- dense: `ctx.write_kv` + `ctx.attend` (paged softmax attention), in `transformer_forward.cpp:104-113`.
- hybrid: per layer, `forward_linear_attn_layer` (GatedDeltaNet) **or** `forward_full_attn_layer`
  (`hybrid_transformer_forward.cpp:372-373`), selected by `ModelConfig::layer_types`.

Consequence: any change to the shared skeleton (a new norm, a residual tweak, a buffer-reuse optimization,
a bug fix) must be made twice and kept in sync, and they can silently drift.

---

## 2. Goal

One forward loop owns the shared skeleton; the per-layer attention is injected as a strategy. The dense path
and the hybrid path differ only in which attention step runs for a given layer. No numerical change.

---

## 3. Proposed design

Introduce a per-layer **attention step** abstraction the shared loop calls:

```cpp
// Runs the attention sub-block for layer L: consumes the normalized hidden state,
// returns the attention output (already o_proj'd) to be added to the residual.
// Implementations: PagedSoftmaxAttention (dense), GatedDeltaNetAttention + FullAttention (hybrid).
struct AttentionStep {
    virtual tensor_t run(size_t L, tensor_t normed_hidden, /* ctx, scratch, exec */ ...) = 0;
};
```

The shared loop (one function) does: embed (or external embeds) → for each layer { input_norm →
`attn_step.run(L, ...)` → residual → post_norm → `mlp_or_moe(L, ...)` → residual } → final_norm → lm_head.

Two concerns to resolve in the design:

1. **Performance.** CLAUDE.md forbids virtual dispatch on the hot path where a concrete type suffices. So the
   strategy must NOT be a per-token virtual call. Options: (a) template the loop on the attention policy
   (compile-time, zero overhead) and instantiate once per family; or (b) a per-*layer* (not per-token)
   function-pointer/tag chosen from `layer_types`, amortized across the whole token. Prefer (a) — the loop
   body is identical, only the attention functor type differs.

2. **Scratch shapes differ.** The hybrid path keeps extra thread-local scratch for the linear-attn/conv state
   (`LinearAttnDecodeScratch`, `FullAttnDecodeScratch`, `HybridN2Scratch` in `hybrid_transformer_forward.cpp`).
   The unified loop must let the attention policy own its own scratch while sharing the skeleton buffers
   (hidden/normed/o/h1/...). Keep `DecodeScratch` for the shared buffers; let each policy hold its own.

---

## 4. Migration steps (incremental, each fingerprint-validated)

1. Extract the dense attention sub-block (`write_kv` + `attend` + `o_proj`) into a `PagedSoftmaxAttention`
   functor; rewrite `transformer_forward` to call it. Validate dense fingerprint unchanged.
2. Extract the hybrid per-layer dispatch into `LinearAttn` / `FullAttn` functors; rewrite
   `hybrid_transformer_forward` to call them. Validate hybrid fingerprint unchanged.
3. Factor the now-identical skeletons (residual, scratch ping-pong, MLP/MoE dispatch, tail) into one
   templated `forward_impl<AttnPolicy>(...)`; have both entry points instantiate it. Validate both.

Stop after step 1 or 2 if the residual duplication is acceptable — even partial extraction reduces drift.

---

## 5. Risks & validation

- **Highest-risk refactor in the codebase** — it is the forward path for every model. Do it last, in small
  steps, each validated by the greedy fingerprint per family (Qwen2 dense, Qwen3 dense, Qwen3-MoE,
  Qwen3.5 hybrid, Qwen3.5-MoE, and a vision prompt for Qwen3.5-VL).
- **Do not regress decode latency.** Keep the policy compile-time (templated), not virtual. Re-run `bench`
  decode tok/s before/after; must be within noise.
- **MTP path** uses `MTPModule` separately but shares forward assumptions — re-run an MTP A/B
  (see `docs/plan/` MTP notes) after unification.

---

## 6. Recommendation

Land `docs/plan/per_layer_weight_resolution.md` and `docs/plan/refactor_sequence_block_table.md` first (smaller,
lower-risk, independently valuable). Tackle this unification only after those, and only with the full
per-family fingerprint harness in place.
