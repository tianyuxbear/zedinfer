# Stage F: MTP performance pass — N=2 scratch caches + MTPScratch

**Date:** 2026-05-27
**Branch:** `feat/qwen3.5`
**Model:** `Qwen3.5-35B-A3B-GPTQ-Int4`
**Hardware:** single NVIDIA H100 80GB

## TL;DR

Stage E (commit `e841908`) shipped end-to-end MTP speculative decoding but
recorded a ~12× wall-clock regression — main 2-token verify forward took
~600 ms, MTP forward itself took ~35 ms, total 430 ms/token vs baseline 30
ms/token. Stage F is a focused performance pass that closes most of that
gap without touching any attention or expert kernels.

End state: **MTP mode is now ~9.75× faster than the Stage E broken state**,
but **still ~20% slower than no-MTP baseline**. The remaining gap is
fundamental to small-N kernel dispatch (paged_prefill at n_q=2,
cuBLAS plan selection at M=2 for Q/K/V/O projections) and would need
kernel-level work to close (CUDA Graph capture, custom small-batch
attention, batched expert GEMM).

Spec decode remains **off by default**, opt-in via `--mtp` on the CLI
(or `ZEDINFER_MTP_SPEC=1` env var). Recommended for research / when the
trade-offs are understood; not yet a production speedup.

## Measurements

Three prompts, `--max-new-tokens 80`, `ZEDINFER_FORCE_ARGMAX=1`, warmup +
3 runs each, averaged. Decode-phase throughput only (prefill unchanged):

| Mode                          | tok/s (avg)  | vs Stage E baseline |
| ----------------------------- | ------------ | ------------------- |
| Baseline (no MTP)             | 44.8 tok/s   | reference           |
| Stage E spec mode (broken)    | 3.7 tok/s    | 0.083× (regression) |
| Stage F spec mode             | 35.8 tok/s   | 0.80× (close)       |

Per-step breakdown (the `ZEDINFER_STEP_PERF=1` and `ZEDINFER_LAYER_PERF=1`
toggles used during investigation are now gone; numbers captured during
development):

```
Baseline (n_q=1):
  main forward    : ~22 ms   (la_attn 2.6 + fa_attn 3.2 + moe 13.5 + other)
  per step total  : ~22 ms

Stage F spec (n_q=2):
  main forward    : ~41 ms   (la_attn 3.3 + fa_attn 14 + moe 24.5 + other)
  MTP forward     : ~3 ms × {1 reject, 2 accept}
  per step total  : ~46 ms   → 1.5 tokens (50% accept) → ~30 ms/token
```

Wall-clock end-to-end:

| Mode                | Total wall (80 tokens) | tok/s |
| ------------------- | ---------------------- | ----- |
| Baseline            | 1.90 s                 | 42.1  |
| Stage F MTP         | 2.71 s                 | 29.5  |

## What landed (Stage F)

### F.1 MoE small-N fast path

`src/frontend/models/moe_forward.cpp`

- Extended `moe_decode` (formerly N=1 only) to handle N ∈ {1, 2} via a
  per-token loop. The `[1, ...]` scratch buffers (`expert_gate/up/act/down`)
  are reused per token; each token contributes top-k expert M=1 GEMMs.
- `moe_layer_forward` gate changed from `use_scratch = (... && N == 1)` to
  `use_decode_path = (... && N <= 2)` and `use_n1_scratch = (... && N == 1)`.
  N=2 fresh-allocates the [2, num_experts] router and [2, H] moe_output
  (two small allocs per layer, totally affordable) but stays on the fast
  per-token expert path instead of falling into `moe_prefill`'s
  gather/group/scatter pipeline.
- Added `MoeN2Scratch` (thread_local) covering router_logits, moe_output,
  and the shared-expert buffers (sh_gate/up/act/down + sh_gate_logit) at
  [2, ...] dim. This was the single largest win: under ALL_GPU the shared
  expert's 4 fresh [2, shared_inter] / [2, H] allocs per layer cost ~3.5
  ms — ~140 ms / step across 40 MoE layers. Threading these through
  thread_local persistent buffers dropped per-layer MoE time from ~6.7 ms
  to ~0.6 ms.

### F.2 Hybrid forward N=2 scratch

`src/frontend/models/hybrid_transformer_forward.cpp`

- Extended `s_fa_scratch` and `s_la_scratch` buffer dims from `[1, ...]`
  to `[kMaxSmallDecodeBatch=2, ...]`. Use sites now call a new
  `scratch_view(buf, N)` helper that returns `slice(0, 0, N)` (zero-copy
  contiguous view of the first N rows) — works for both N=1 decode and
  N=2 verify.
- The `use_fa_scratch` / `use_la_scratch` gates flipped from `(N == 1)`
  to `(N <= 2)`. N=2 verify now reuses the same 7-12 per-layer scratch
  tensors instead of doing 7-12 fresh BestFitPool round-trips per layer.
- Added `HybridN2Scratch` (thread_local) covering the layer-wide ids,
  pos_ids, hidden(+out), normed, h1, normed_post, final_normed, logits,
  and hidden_snap (MTP residual snapshot) at [2, ...] dim. The outer
  loop's `use_n2_scratch` flag mirrors the use_scratch=(N==1) pattern.
  The hidden ping-pong compares `tensor->data()` (pointer equality on
  the backing storage) instead of `shared_ptr` identity, since slice
  views are different shared_ptrs that share storage.

### F.3 MTPScratch

`src/frontend/models/mtp_module.cpp`

- Added `MTPScratch` (thread_local) covering all ~28 fresh allocs that
  MTPModule::forward + mtp_attention_decode + mtp_moe_one_token used to
  do per call. First call lazy-allocates from `main_cfg_`; subsequent
  calls just reuse.
- Replaced the synchronous D2H + host-sigmoid on the [1, 1]
  `sh_gate_logit` with the existing `ops::shared_expert_gate(sh_down,
  gate_logits)` GPU op — one less stream drain per MTP call.
- Replaced four `memcpy_sync` (2× D2D in concat, 2× D2D in K/V cache
  write) with `memcpy_async` on the compute stream. Same FIFO ordering;
  no per-call serialization point.
- Replaced the per-call `std::vector<uint16_t/float>` heap allocs for
  router top-k with thread_local persistent buffers.

Net effect on MTP forward time: ~35 ms → ~3 ms (~12× faster).

### F.4 CLI flag

- `--mtp` on `serve`, `bench`, `chat`, `ping` (default off).
- `SchedulerConfig::mtp_enabled` plumbs through to `ServingLoop`.
- Existing `ZEDINFER_MTP_SPEC` and `ZEDINFER_MTP_DEBUG` env vars still
  work as research toggles.

## Where the remaining 20% gap lives

After all of F.1–F.3, the 2-token main forward is `~41 ms` and the
1-token main forward is `~22 ms`. Per-layer breakdown at N=2 (across
40 layers):

| Layer kind         | N=2 total | N=1 total | ratio | per-layer extra |
| ------------------ | --------- | --------- | ----- | --------------- |
| linear-attn (30×)  | 3.3 ms    | 2.6 ms    | 1.27× | +0.02 ms        |
| full-attn   (10×)  | 14.0 ms   | 3.2 ms    | 4.4×  | +1.08 ms        |
| MoE         (40×)  | 24.5 ms   | 13.5 ms   | 1.8×  | +0.27 ms        |

The full-attn N=2 cost is the bulk of what remains. Root cause: the
native paged_prefill kernel is grid-tuned for n_q≫1 (prefill) and
launch-bound at n_q=2. The flashinfer prefill kernel was also tried
(commit on this branch then reverted) but is *slower* than the native
prefill at n_q=2 on H100 — its plan also targets large batches.

For spec mode to actually beat the baseline 22 ms/token, main 2-token
forward would need to drop to ≤ ~30 ms (so 30/1.5 ≈ 20 ms/token at 50%
accept), and that requires either:

1. **CUDA Graph capture for decode** — amortizes launch overhead across
   all kernels in the forward, agnostic to n_q. This is the cleanest
   path and benefits both N=1 and N=2 paths.
2. **Custom small-batch paged attention kernel** — tuned grid for
   n_q ∈ {1, 2, 4} verify shapes.
3. **Batched expert GEMM** — collapse the per-token, per-expert M=1
   GEMM loop into a single grouped/strided GEMM call. ~24 expert linear
   launches per MoE layer → ~3.

None of these is a one-PR fix. They are the natural follow-up work and
documented in `docs/roadmap.md` as the next decode-perf priorities.

## References

- Initial spec implementation: `docs/debug/qwen3_5_mtp_spec_decode.md`
- Stage E perf write-up: `docs/debug/stage_e_mtp_spec_decode_perf.md`
- Stage F commits: this PR/branch.
- Files touched: `moe_forward.cpp`, `mtp_module.cpp`,
  `hybrid_transformer_forward.cpp`, `serving_loop.cpp` (CLI plumbing),
  `examples/{serve,bench,chat,ping}.cpp` (CLI flag), `scheduler.hpp`
  (config field), `serving_loop.hpp` (cached flag).
