# Step 2.1: Marlin / Marlin-MoE integration

> Status: design doc. Implementation tracked separately under `Step 2.1.*`.
>
> Owner: feat/qwen3.5 branch.
>
> Predecessor: `docs/debug/stage_f_mtp_perf_pass.md` (perf breakdown showing
> MoE expert dispatch is the dominant remaining cost at N=2 spec verify).

---

## 1. Why Marlin

Stage F instrumentation breakdown for N=2 spec verify on Qwen3.5-35B-A3B-GPTQ-Int4:

| Layer kind        | N=1 baseline  | N=2 verify    | Δ per step |
| ----------------- | ------------- | ------------- | ---------- |
| linear-attn (30×) | 2.6 ms        | 3.3 ms        | +0.7 ms    |
| **full-attn (10×)** | 3.2 ms      | 14.0 ms       | +10.8 ms   |
| **MoE (40×)**     | 13.5 ms       | 24.5 ms       | +11.0 ms   |
| other             | ~2 ms         | ~2 ms         | 0          |
| **total**         | **~22 ms**    | **~41 ms**    | **+19 ms** |

`paged_attention_small_nq_kernel` (Step 2.0.5) attempted to close the
full-attn gap but ties with `paged_attention_prefill` — fa_attn at small n_q
is already launch-overhead-bound on H100/B200 and L2 hides the K/V reuse
savings the kernel was designed to exploit.

**MoE is the bigger remaining lever.** Per layer at N=2:

- 24 expert linear launches (top-8 × {gate, up, down}) at M=1 quantized GEMM.
- Per-token loop runs that twice for N=2 (48 launches/layer).
- 40 MoE layers × 48 launches = **~1920 small launches per spec verify step**.
- Each launch is launch-overhead-bound (5–10 µs cuBLAS plan lookup + kernel
  launch overhead), with actual compute being a tiny M=1 GPTQ-Int4 GEMM.

The fix shape is **fused gather + grouped quantized GEMM + scatter** — exactly
what Marlin-MoE provides for W4A16 (GPTQ-Int4) models.

Expected post-integration:

- 24 expert launches/layer → **1 fused launch/layer** (`marlin_gemm_moe`).
- MoE per layer: 0.61 ms → estimate **0.20–0.25 ms** (3× reduction).
- MoE total at N=2: 24.5 ms → estimate **~10–12 ms** (save ~12 ms).
- Spec mode main forward N=2: 41 ms → estimate **~28–30 ms**.
- Spec mode per token (50% accept rate): 30 + 5 / 1.5 ≈ **23 ms/token vs
  baseline 22 ms/token** — spec mode finally breaks even / wins by 1-2 ms.

The single-GEMM `gptq_marlin` kernel (separate from Marlin-MoE) also replaces
our 73 KB `linear_quantized_kernel.cuh`. That covers:

- Shared-expert FFN at every MoE layer (3 ops × 40 layers = 120 GEMMs/step).
- LM head (per step).
- Any other GPTQ-quantized dense layers (router weight is typically not
  quantized; depends on the GPTQ release).

---

## 2. SM compatibility (deployment constraint)

| GPU        | SM     | Status with Marlin |
| ---------- | ------ | ------------------ |
| RTX 3090   | SM86   | ✅ (covered by `mma.sync m16n8k16` baseline) |
| RTX 4090   | SM89   | ✅ |
| H100       | SM90   | ✅ |
| B200       | SM100  | ✅ (backward-compat; SM90+ specific paths in Marlin will be `__CUDA_ARCH__`-guarded and become no-ops on lower archs) |

Our `xmake/device/nvidia.lua` already sets
`add_cugencodes("sm_80", "sm_86", "sm_89", "sm_90", "sm_100")`, so Marlin
binaries get fat-binary'd for every target arch. Production loading on a
4090 dispatches to the `sm_89` SASS; the unused arch sections add a small
constant to library size but are not executed.

Marlin instructions reference (baseline, no SM90+ exclusive use):

- `mma.sync.aligned.m16n8k16.row.col.bf16.bf16.bf16.bf16` (SM80+)
- `ldmatrix.sync.aligned.x4.m8n8.shared.b16` (SM75+)
- `cp.async.ca.shared.global` (SM80+)

Marlin does **not** require `wgmma` or `cp.async.bulk` (SM90 exclusive). Newer
vllm versions may have added a `marlin_sm90_kernel.cu` that does — we'll
inspect at cherry-pick time and either include it (compiles only on sm_90+
under nvcc's auto-guard) or strip it if it breaks the multi-arch build.

---

## 3. What's in vllm/csrc/ that we want

Cherry-pick whole directories (no file-level selection):

```
vllm/csrc/quantization/gptq_marlin/      → third_party/marlin/gptq_marlin/
vllm/csrc/moe/marlin_moe/                → third_party/marlin/marlin_moe/
vllm/csrc/quantization/marlin_common/    → third_party/marlin/common/ (if exists)
```

Top-level entry points we'll call:

| Function | From | Purpose |
| -------- | ---- | ------- |
| `gptq_marlin_repack` | `gptq_marlin/gptq_marlin_repack.cu` | Offline: HF GPTQ weight layout → Marlin interleaved layout. Run at model load. |
| `gptq_marlin_gemm` | `gptq_marlin/gptq_marlin_kernels.cu` | Single W4A16 GEMM (BF16 act × INT4 weight). |
| `marlin_gemm_moe` | `marlin_moe/marlin_moe_ops.cu` | Fused gather + grouped W4A16 GEMM + scatter for top-K expert dispatch. |

Headers we'll include:

- `marlin_dtypes.cuh` (common dtype helpers)
- `gptq_marlin_dtypes.cuh` (GPTQ-specific traits)
- Various `_kernel.cuh` headers per the kernel families above

Files we'll exclude / not link:

- `vllm/csrc/quantization/awq/*` — AWQ, different format than GPTQ
- `vllm/csrc/quantization/fp8/*` — FP8, model is INT4
- `vllm/csrc/quantization/marlin_24/*` — 2:4 sparse, we don't have sparse weights
- Any `*.py` wrapper or PyTorch binding — we link the CUDA `.cu` files directly

---

## 4. Layout transition: HF GPTQ → Marlin layout

HF GPTQ INT4 weight layout per matrix:

```
qweight  [K/8, N]  int32 (8 INT4 values packed per int32 along K)
qzeros   [K/group_size, N/8]  int32 (8 INT4 zero-points per int32)
scales   [K/group_size, N]  bf16
g_idx    [K] int32   # group index per K position (column reorder)
```

Marlin internal layout (after `gptq_marlin_repack`):

```
B  (qweight repack)  int4 interleaved into mma fragments
B_scales  fp16/bf16 interleaved per warp tile
B_zp      (only if zero_points_per_tile mode)
```

The repack is a one-shot offline kernel — run during model load, store the
repacked tensor on GPU, discard the HF `qweight`/`qzeros`. Memory footprint
is identical (4 bits per weight).

For MoE: Marlin-MoE expects **all expert weights stacked** as
`[num_experts, K, N_packed]` (or similar), not per-expert separate tensors.
This means we need to either:

1. **At load time**: allocate one contiguous tensor `[num_experts, K, N]` and
   copy each HF expert weight into the corresponding slice, then run
   `gptq_marlin_repack` on the whole thing.
2. **Per-layer at first use**: lazy stack + repack on first MoE forward.

Option 1 is cleaner and avoids first-forward stalls. Memory delta is zero
(same total bytes, just one allocation instead of N).

`ExpertPool` (`include/frontend/models/expert_pool.hpp`) needs minor changes:

- Add a `marlin_repacked_weights` tensor per layer (3 of them: gate, up, down).
- Each tensor is `[num_experts, K, N_packed]` in Marlin layout.
- `ExpertGpuHandle` gets a new field for the Marlin handle (replaces or
  alongside the existing `gate_packed`/`gate_scale`/`gate_g_idx`).

---

## 5. dispatch_expert_linear refactor

Current path (`include/frontend/models/forward_config.hpp:130`):

```cpp
void dispatch_expert_linear(out, in, layer, expert_id, proj) {
    const ExpertGpuHandle& h = expert_pool->ensure_on_gpu(layer, expert_id);
    if (h.gate_packed) {
        ops::linear_quantized(out, in, *packed, ..., *scale, *g_idx, ...);
    } else if (h.gate_weight) {
        ops::linear(out, in, *weight, nullptr);
    }
}
```

Called per (layer, expert) per token, M=1 GEMM. **This is the hot path we
want to delete.** Step 2.1.3 removes `dispatch_expert_linear` entirely and
replaces the `forward_moe_mlp` MoE block with a single `marlin_gemm_moe`
call covering all top-K experts at once.

New `forward_moe_mlp` shape:

```cpp
// Inputs:  h_post       [N, hidden]  (per-token, N=1 or N=2)
//          expert_ids   [N, top_k]   (device int32, from GPU topk_softmax)
//          expert_weights [N, top_k] (device fp32)
//          marlin_gate_w [num_experts, hidden, moe_inter_packed] (Marlin layout)
//          marlin_up_w   [num_experts, hidden, moe_inter_packed]
//          marlin_down_w [num_experts, moe_inter, hidden_packed]
//          gate_scales / up_scales / down_scales (per-expert scales)
//
// Output:  moe_output [N, hidden]
//
// Implementation:
//   gate_out = marlin_gemm_moe(h_post, marlin_gate_w, expert_ids, expert_weights, scales)
//   up_out   = marlin_gemm_moe(h_post, marlin_up_w,   expert_ids, expert_weights, scales)
//   act      = swiglu(gate_out, up_out)
//   moe_output = marlin_gemm_moe(act, marlin_down_w, expert_ids, expert_weights, scales)
//                # marlin_gemm_moe internally does the weighted sum via
//                # expert_weights, so moe_output is already the topk-weighted sum.
```

Three `marlin_gemm_moe` calls per layer instead of 24 single-expert GEMMs.
Plus three `swiglu` calls (one per layer, dense bf16 op — keep hand-written).

`marlin_gemm_moe` signature (approximate, will refine when reading actual
code at integration):

```cpp
cudaError_t marlin_gemm_moe(
    const half*  A,            // [M, K] activation
    const int*   B_q,          // [num_experts, K/factor, N_packed]
    int*         sorted_ids,   // [M * top_k]  workspace
    const float* topk_weights, // [M, top_k]
    const int*   topk_ids,     // [M, top_k]
    const half*  B_scales,     // [num_experts, num_groups, N]
    int*         expert_offsets, // [num_experts + 1] prefix sum
    int*         workspace,
    half*        C,            // [M, N]  output (already topk-weighted sum)
    int M, int N, int K, int num_experts, int top_k,
    int group_size,
    cudaStream_t stream);
```

The `topk_ids` and `topk_weights` are the device-side outputs of our Step 2.0
`topk_softmax_gpu` kernel — which finally gets activated as part of this
work. The expert_offsets is a small CPU-side computation from topk_ids
(prefix sum), or a tiny GPU kernel.

---

## 6. MTP-specific note

The Qwen3.5-A3B-GPTQ-Int4 release explicitly excludes the MTP head from
quantization (HF GPTQ config's `dynamic` field has a `-:.*mtp.*` rule).

So MTP's `mtp_moe_one_token` continues to use `ops::linear` (BF16 cuBLASLt
M=1 GEMMs) — it's not in Marlin's scope. MTP forward time (~3 ms/call after
Stage F's MTPScratch fix) is no longer a bottleneck.

Future: if NextN > 1 makes MTP heavy, we can wire it up to FlashInfer's
`group_gemm` (BF16 dense grouped GEMM, already in `third_party/flashinfer/`).
Out of scope here.

---

## 7. Step-by-step plan

### Step 2.1.0: Vendor Marlin

- Cherry-pick:
  ```
  vllm/csrc/quantization/gptq_marlin/  → third_party/marlin/gptq_marlin/
  vllm/csrc/moe/marlin_moe/            → third_party/marlin/marlin_moe/
  ```
  Include any common headers Marlin's source files reference.
- Add to `xmake/device/nvidia.lua`:
  ```lua
  target("ops-nvidia")
      ...existing...
      add_files("../../third_party/marlin/**/*.cu")
      add_includedirs("../../third_party/marlin")
  ```
- xmake's existing `add_cugencodes("sm_80", "sm_86", "sm_89", "sm_90", "sm_100")`
  fat-binaries Marlin for every supported deploy GPU.
- Verify build clean on the dev machine (B200).
- Verify build clean on a 4090 / 3090 if available (or at least confirm
  `nvcc -arch=sm_89` compiles each `.cu` without SM90-exclusive errors).
- Commit checkpoint.

### Step 2.1.1: GPTQ repack at model load

- In `src/frontend/models/base.cpp` (or wherever weight tensors are
  finalized), detect GPTQ-Int4 weights and call `gptq_marlin_repack` once
  per matrix.
- For MoE experts: see Step 2.1.3 below — repack is part of expert_pool's
  reorganization.
- For dense GPTQ layers (LM head, shared expert, anything else):
  per-layer repack at load. Replace the stored `weight_packed` tensor with
  the repacked one. Keep the same key in `ModelWeights` so downstream
  lookups don't change.
- Confirm correctness: load model, do one prefill, compare a few logits
  against pre-Marlin output. Small bf16-rounding differences expected;
  bigger differences indicate repack bug.
- Commit checkpoint.

### Step 2.1.2: Single-GEMM replacement

- Modify `ops::linear_quantized` NVIDIA path to dispatch to
  `gptq_marlin_gemm` instead of the hand-written kernel.
- Delete `linear_quantized_kernel.cuh` (73 KB).
- Update `linear_nvidia.cu` — most of the file is the Marlin-equivalent
  hand-written kernel-selection logic; large parts can also be deleted.
- Bench: confirm baseline (no --mtp) decode tok/s improves (Marlin is faster
  than our hand-written quant kernel for M=1, especially at the LM head's
  M=1 N=vocab=248k K=hidden=2048 GEMM).
- Commit checkpoint.

### Step 2.1.3: MoE refactor

- `expert_pool.{hpp,cpp}` changes:
  - Per-layer stacked weight tensors: `gate_marlin_weight [num_experts, hidden, moe_inter_packed]`,
    `up_marlin_weight`, `down_marlin_weight`, plus corresponding `_scales` tensors.
  - At construction, for each layer iterate experts, allocate the stacked
    tensor, copy each HF expert weight into its slot, call
    `gptq_marlin_repack` once on the stacked tensor.
  - The per-expert `ExpertGpuHandle` becomes thin (just a layer pointer and
    expert id) since dispatch is now layer-wide.
- `forward_moe_mlp`:
  - Take `expert_ids[N, top_k]`, `expert_weights[N, top_k]` from
    `compute_router_topk` (now using the GPU `topk_softmax_gpu`).
  - Build `expert_offsets[num_experts + 1]` (small CPU loop + H2D, or tiny
    GPU kernel — depends on Marlin API requirement).
  - Call `marlin_gemm_moe` three times: gate, up, down. The middle swiglu
    stays as our existing GPU kernel.
- Delete `dispatch_expert_linear` and the per-token expert loop in
  `moe_decode` / `moe_decode_n2`. Both become single-kernel paths.
- `apply_shared_expert`: shared expert weight is also GPTQ-Int4. Use
  `gptq_marlin_gemm` (single-GEMM) here, NOT marlin_gemm_moe (only 1
  "expert"). Confirm M=2 verify path also works.
- Re-enable Step 2.0's `topk_softmax_gpu` — its output dtype/layout should
  match Marlin-MoE's expected `topk_ids` (int32) and `topk_weights` (fp32).
  Tie-breaking divergence with CPU is no longer an issue because MTP also
  uses GPU topk now (no two paths disagreeing).
- Commit checkpoint.

### Step 2.1.4: Validation + benchmark

- Run `tests/integration/test_qwen3_5_load.cpp` (or equivalent) to verify
  model load + first forward still produces sensible logits.
- Benchmark baseline + --mtp on Qwen3.5-35B-A3B-GPTQ-Int4:
  - Expected baseline: ~50 tok/s (up from 44, single GEMM improvements).
  - Expected --mtp: ~50 tok/s (matches baseline ± 2-3 tok/s — finally
    breaks even).
- Run on 4090 if available; confirm correctness + perf in fat-binary mode.
- Update `docs/debug/stage_f_mtp_perf_pass.md` perf section + add note.
- Commit checkpoint.

---

## 8. Risks and mitigations

| Risk | Likelihood | Mitigation |
| ---- | ---------- | ---------- |
| Marlin's expected weight layout requires repack details we miss | High | Validate end-to-end correctness on one prompt before moving to perf. Marlin has been deployed in vLLM for years — well-understood quirks are documented in its source comments. |
| `gptq_marlin_repack` doesn't match our HF GPTQ config (group_size, sym/asym) | Medium | Qwen GPTQ release uses standard config (group_size=128, sym=true, no act_order). Confirm by reading `quantize_config.json` in the model dir. |
| Marlin-MoE expects a specific `sorted_token_ids` and `expert_offsets` layout we have to compute | Medium | Read the Marlin-MoE source and reproduce the layout exactly. Probably a small custom kernel or a CPU loop. |
| Multi-arch build breaks on Marlin's SM90 specific paths | Low | xmake's existing fat-binary config handles this. If Marlin has `#if __CUDA_ARCH__ >= 900` guards, no-ops on lower archs. If it doesn't, we add ifdef guards in our copy. |
| Marlin kernel performance worse than hand-written at very small M (M=1) | Low | Marlin has explicit M=1 optimization. Reads inside vLLM's deploy show it beats their hand-written at all M. |
| Regression on B200 (Marlin's SM90 paths not tuned for SM100) | Low | Backward-compat at correctness. SM100-specific tuning is future work (not in this step). |

---

## 9. What's NOT in scope

- **CUDA Graph capture**: separate work item. Once Marlin reduces launch
  overhead per layer, CUDA Graph's benefit shrinks (was estimated 30% at
  current launch counts; will be smaller post-Marlin).
- **MTP MoE path**: stays on `ops::linear` (BF16 cuBLASLt). FlashInfer
  `group_gemm` could batch it, but MTP forward (~3 ms) isn't a hotspot.
- **Custom small-N attention**: `paged_attention_small_nq_kernel` (Step 2.0.5)
  is already in the tree. It's tied with `paged_attention_prefill` at small
  past_len; leaving it active for now.
- **AWQ / FP8 support**: Marlin only does W4A16 GPTQ. Different quant
  schemes need different kernels — not on the roadmap.
- **GPU topk_softmax tie-break alignment with CPU**: irrelevant post-2.1.3
  because main forward and MTP both use the GPU path. The CPU path stays
  as the CPU fallback.

---

## 10. Open questions

- Marlin-MoE's API has evolved across vLLM versions. We should pick a recent
  stable vLLM tag (e.g. v0.6.x) and lock to that. Are we OK with vendoring
  by tag and treating it as frozen (no upstream pull), or do we want git
  submodule for tracking?
- Does Marlin-MoE require the activation to be FP16, or does it support
  BF16 too? Qwen3.5 is BF16-native. If FP16 only, we add a per-layer
  `bf16 → fp16` upcast (cheap) and `fp16 → bf16` downcast at output.

The implementation step will resolve both during Step 2.1.0.

---

## 11. References

- Stage F perf: `docs/debug/stage_f_mtp_perf_pass.md`
- Marlin paper / blog: <https://blog.vllm.ai/2024/06/10/quantization.html>
- Marlin source (vLLM):
  - `vllm/csrc/quantization/gptq_marlin/`
  - `vllm/csrc/moe/marlin_moe/`
- Qwen3.5 GPTQ HF dynamic exclude rules (why MTP stays BF16):
  `Qwen3.5-35B-A3B-GPTQ-Int4/quantize_config.json`
