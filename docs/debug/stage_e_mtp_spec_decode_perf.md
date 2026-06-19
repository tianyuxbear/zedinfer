# Stage E: MTP speculative decoding — wall-clock perf findings

**Date:** 2026-05-27
**Branch:** `feat/qwen3.5`
**Model:** `Qwen3.5-35B-A3B-GPTQ-Int4`
**Hardware:** single NVIDIA H100 80GB

## TL;DR

Stage D.1 landed end-to-end speculative decoding via the MTP head. The token-
level accept rate looks fine (24%–57% depending on prompt), but the **wall-
clock decode throughput regresses ~10× under spec mode** (35 tok/s → 2.6–3.0
tok/s). Spec mode is therefore disabled by default — `ZEDINFER_MTP_SPEC=1` is
treated as a research toggle, not a production speedup.

This document records the measurements, the root-cause analysis behind the
regression, and the work that would actually be needed to make spec decode a
real wall-clock win on this stack.

## Measured numbers

Same prompt, `--max-new-tokens 80`, `ZEDINFER_FORCE_ARGMAX=1`, three trials
on three prompts. Decode-phase throughput only (prefill identical):

| Prompt                                      | Non-spec       | Spec (MTP)     | Slowdown |
| ------------------------------------------- | -------------- | -------------- | -------- |
| "Write a short story about a robot."        | 34.59 tok/s    | 2.85 tok/s     | 12.1×    |
| "Explain transformers in 2 sentences."      | 35.29 tok/s    | 3.01 tok/s     | 11.7×    |
| "What is the capital of France?"            | 35.28 tok/s    | 2.60 tok/s     | 13.6×    |

Per-step breakdown (instrumented with a temporary `ZEDINFER_MTP_PERF=1` probe
in `serving_loop.cpp` that wraps the MTP block):

```
SPEC mode (n_q=2 verify):
  main forward    : ~600 ms   ← !!
  MTP forward     : ~35 ms × (1 on reject, 2 on accept)
  total per step  : ~640 ms   → 1.5 tok/step → ~430 ms/token

NON-SPEC + MTP_DEBUG (n_q=1 main, MTP runs as observer):
  main forward    : ~30 ms
  MTP forward     : ~35 ms
  total per step  : ~65 ms    → 1.0 tok/step → ~65 ms/token

NON-SPEC pure (no MTP at all):
  main forward    : ~30 ms
  MTP forward     : 0 ms
  total per step  : ~30 ms    → 33 tok/s baseline
```

Two surprises jump out:

1. **The 2-token main forward is 20× slower than the 1-token main forward**
   (600 ms vs 30 ms). Adding one extra query token should add ≤10% of layer
   time, not 2000%.
2. **The MTP forward itself costs ~35 ms** — about as much as a whole main
   1-token decode. Even in pure-observer mode (`MTP_DEBUG`), MTP halves the
   throughput (33 → 15 tok/s ballpark).

Both bottlenecks need to be addressed before MTP can be a net win.

## Root cause #1 — MoE falls off the fast path at N>1

`src/frontend/models/moe_forward.cpp:392`:

```cpp
const bool use_scratch = (scratch != nullptr && N == 1);
```

This is the gate between `moe_decode` (the pre-allocated, N=1-optimized
scratch path) and `moe_prefill` (the general N>1 path that gathers tokens
by expert, runs grouped GEMMs, and scatters back). The intent is sound for
prefill — for N=512 the gather/group/scatter scheme is the right call. But
spec-decode verify hits `moe_layer_forward` with **N=2**, which is the
worst possible input to `moe_prefill`:

- 16 active experts in the worst case (2 tokens × top-8) vs 8 for N=1, so
  ~2× the per-expert kernel launches.
- Per-step allocations: `gathered_full`, `g_gate_full`, `g_up_full`,
  `g_act_full`, `g_down_full` are all `Tensor::create({N, ...})`, fresh per
  layer per step. `moe_decode` reuses `DecodeScratch`'s buffers.
- Two extra `memcpy_async` H2D per layer for indices/weights buffers.
- `gather_rows` + `scatter_add_rows` kernel launches per active expert.

At N=2 the gather/scatter overhead and the fresh allocations both dominate
the actual GEMM work. The compute time barely changes — it's the launch
overhead and per-step setup that explodes.

**Fix shape:** add a third path, "decode for small-N" (N ∈ {1, 2}). Keep
the scratch-style buffer reuse, just dispatch per-token through the
expert linear with M=1 GEMMs as `moe_decode` already does. Or pre-size
`DecodeScratch` for N ≤ MAX_SPEC_BATCH and use the existing fast loop.
The change is bounded to `moe_forward.cpp` plus growing `DecodeScratch`'s
buffer dimensions in `decode_scratch.cpp`.

## Root cause #2 — paged attention also drops to a slower kernel

`src/frontend/models/paged_forward_context.cpp:739` (the n_q>1 short-circuit
in `attend_decode_single`) bypasses the flashinfer cache entirely and goes
through `dispatch_paged_prefill` (the native paged-prefill kernel) for spec
verify. flashinfer prefill is already wired and tuned; the native path is
the unoptimized fallback.

This is smaller than the MoE issue per layer (attention is maybe 5% of layer
time at this model size), but it compounds.

Direct attempts to redirect spec verify through flashinfer are documented in
the Stage E.1 commit attempts on this branch (eventually reverted):

- **Single flashinfer prefill call with `seqlen_q=n_q`, `kv_batch_size=1`.**
  Math is correct on paper (queries align to last n_q kv positions, causal
  mask handles the rest), but in practice the kernel-internal reduction tree
  differs from the n_q=1 plan enough to flip argmax on borderline tokens. On
  the "story" prompt this dropped the empirical accept rate from 57% (D.1
  baseline with native paged_prefill) to 35% with flashinfer-as-spec.

- **N virtual decode slots sharing one block table** (`kv_batch_size=n_q`,
  each with `kv_len = past_len + k + 1`). Same outcome — the per-batch plan
  is not bit-identical to the single-slot plan that non-spec uses.

- **N sequential flashinfer calls, each `kv_batch_size=1`, with per-k scratch
  metadata.** Closer to non-spec's reduction tree, but still diverges. Output
  text degenerates within ~30 tokens on a 100-token generation.

The conclusion from those attempts: under bf16, matmul tiling for M=1 vs M=2
*also* differs (Q/K/V projections, MoE linears), so even if the attention
kernel itself were bit-identical, K/V cached at position `past_N` in spec
mode would not match K/V at the same position in non-spec mode. The drift
compounds layer by layer and step by step.

This is fundamental to **bf16 spec decode with batched verify**, not a bug
in our implementation. Production systems work around it with either:

1. **fp16 or fp32 for the verify path** (precise enough that batch-shape-
   dependent reduction differences vanish below the argmax threshold), or
2. **Custom kernels that pin tiling for both N=1 and N>1** so the reductions
   are bit-identical, or
3. **Accept the drift and use approximate verify** (accept whenever main's
   sampled token is plausible under the draft distribution, not bit-equal).

None of these are 1-PR fixes.

## Root cause #3 — MTP forward itself is slow

The MTP head is one transformer layer plus embedding/lm_head. In principle
that's ~1/30th of a main forward. We measured 35 ms — comparable to a full
30-layer main 1-token decode. Why:

`src/frontend/models/mtp_module.cpp`:

- **Two synchronous D2H memcpys per call** (`mtp_moe_one_token`): router
  logits (line 343) and shared-expert gate (line 393). Each drains the GPU
  stream. ~15-30 µs each, but the synchronization-induced pipeline stalls
  cost much more in steady-state.
- **CPU-resident top-k softmax** (line 350) — `ops::moe::topk_softmax` is a
  host-side function. The D2H sync is mandatory to feed it.
- **~50 small linear-launches per call**: 8 routed experts × 4 ops + shared
  expert × 4 + Q/K/V/O projections + 2 norms. Each launch is ~5-10 µs of
  cuBLAS overhead. For M=1 GEMMs the overhead dominates the actual compute.
- **Per-call fresh allocations** for q_raw, gate, k_raw, v, q_normed,
  k_normed, gate_buf, up_buf, act_buf, down_buf, sh_*, etc. Same kind of
  allocation pressure as the MoE prefill path.

**Fix shape:** mirror what `moe_decode` + `DecodeScratch` did for the main
forward — pre-allocate buffers in a `MTPDecodeScratch`, route top-k onto the
GPU (or batch the D2H into a single async transfer), fuse small linears
where possible. The MTP module is ~400 lines so this is a contained refactor.

## What spec decode would need to be a real win

Even after fixing #1 (MoE small-N), #2 (paged attention), and #3 (MTP
overhead), the break-even math is brutal. Let's say after fixes:

- Main 2-token forward: 35 ms (only +5 ms over 1-token, since the extra
  query is a single MoE pass plus one extra attention row).
- MTP forward: 10 ms (after scratch reuse + GPU top-k).
- Per spec step: 35 + 1.5×10 = 50 ms, emitting 1.5 tokens at 50% accept
  → **33 ms/token**.

Versus non-spec 30 ms/token, that's still a wash. You need the accept rate
above ~60% OR the MTP forward below ~5 ms before MTP-style speculation pays
for itself on this model+hardware combo. Multi-token MTP heads (predict t+2,
t+3) push the breakeven the other way, but Qwen3.5's released MTP only
predicts t+2.

For a real wall-clock speedup, the more promising direction is **CUDA Graph
capture for the main decode path** (eliminates per-layer launch overhead on
the dominant path) before reinvesting in MTP. The graph win is independent
of spec decode and benefits all decoding.

## What's checked in

- Stage D.1 (`fcb204f`) — the end-to-end spec decode path, env-gated on
  `ZEDINFER_MTP_SPEC=1`.
- This document.

Spec decode is **disabled by default** (no env var) and the documentation
already calls it a research toggle. No code changes from Stage E land on
the branch — every attempted optimization was reverted because it either
regressed accept rate (E.1 flashinfer-per-k) or kept the same regression
(E.1 batched-virtual-slots).

## References

- Stage D.1 commit: `fcb204f` "Stage D.1 — end-to-end speculative decoding via MTP"
- Earlier perf write-up (kernel-precision angle only): `docs/debug/qwen3_5_sampler_and_thinking_drift.md`
- MoE forward: `src/frontend/models/moe_forward.cpp`
- MTP forward: `src/frontend/models/mtp_module.cpp`
- Paged attention dispatch: `src/frontend/models/paged_forward_context.cpp`
