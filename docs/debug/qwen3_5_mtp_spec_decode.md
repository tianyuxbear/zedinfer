# Qwen3.5 MTP speculative decoding — adaptation log and findings

**Branch:** `feat/qwen3.5`
**Date range:** stages A → E, completed 2026-05-27
**Model:** `Qwen3.5-35B-A3B-GPTQ-Int4`
**Hardware:** single NVIDIA H100 80GB

This document is the end-to-end writeup of adapting Qwen3.5's Multi-Token
Prediction (MTP) head to zedinfer, from the weight-loader plumbing through
the full speculative-decoding integration and the eventual wall-clock
investigation. It captures the design decisions, the surprises (some that
worked, several that did not), and the perf findings that ultimately drove
the decision to ship MTP as an env-gated research toggle rather than a
default-on speedup.

For the perf-only summary, see [`stage_e_mtp_spec_decode_perf.md`](stage_e_mtp_spec_decode_perf.md).

---

## 1. Background

Qwen3.5 releases ship an extra "MTP" decoder layer alongside the main
transformer. At inference, the main model produces hidden state `h_t` for
position `t`; the MTP head consumes `h_t` plus token `t+1` and predicts
`t+2`. Used naively this just helps the model train; used at serve time
it enables one-extra-token speculative decoding (Medusa-style with `K=1`
draft).

Goal: integrate MTP so the main 1-token decode forward becomes a 2-token
verify forward whose second logit row is "free" — the draft was already
proposed by MTP at the previous step. On accept we emit 2 tokens per step
instead of 1; on reject we emit 1 and discard the draft K/V.

The expected payoff under perfect kernel parity: ~1.5× throughput at 50%
accept rate. The realized payoff: see Stage E.

---

## 2. Staged plan (executed)

| Stage | Commit | Subject |
| ----- | ------ | ------- |
| A     | `8f7f7d8` | MTPModule weight loader + verifier (forward still stubbed) |
| B.1   | `5fd067b` | MTPModule::forward end-to-end (no past KV yet) |
| C.1   | `f73d333` | MTP K/V cache + proper attention (decode-only) |
| C.2   | `875f3a0` | MTP prompt prefill, accept rate 26% → 50% |
| D.0   | `575cb88` | Per-Request MTP K/V state (multi-request safe) |
| D.1   | `fcb204f` | End-to-end speculative decoding via MTP |
| E     | `e841908` | Findings doc — spec decode is currently a perf regression |

Each stage was a small, reviewable step with its own test plan and revert
boundary. The staging stayed conservative on purpose: every previous-stage
commit was independently functional (e.g. C.2 could run as a non-spec
observer printing draft-vs-actual on every token), so a regression in a
later stage never required reverting everything.

---

## 3. Stage A — weight loader + verifier

**Files added:** `include/frontend/models/mtp_module.hpp`,
`src/frontend/models/mtp_module.cpp` (skeleton).

**What it does:** parses MTP weights out of the safetensors release into a
new `MTPModule` instance owned by `Qwen3_5MoeModel`. Implements
`mtp_module()->ready()` for the engine to gate on, plus a stub
`MTPModule::forward(...)` that throws.

**Two surprises during loader work:**

1. **Two release variants of the expert weights.** The original Qwen3.5
   release ships per-expert tensors (`mtp.experts.{i}.gate_proj.weight`,
   etc., 256 of each). The Qwen3.6 35B-A3B preview release fuses them into
   a single 3-D tensor `mtp.experts.gate_proj.weight` of shape
   `[256, hidden, moe_inter]`. `wire_mtp_experts()` now detects which
   layout is present and either slices the fused tensor or wires up
   per-expert pointers. INT4 GPTQ weights only ship per-expert.

2. **`Qwen3_5MoeRMSNorm` uses `output * (1 + weight)`**, not the standard
   `output * weight`. Missed this in stage A and got non-zero norm outputs
   on stage B even with everything else right. Fixed by threading
   `add_one_to_weight=true` through to `ops::rms_norm`.

---

## 4. Stage B — `MTPModule::forward` end-to-end (no past K/V)

**Goal:** make MTP forward produce *some* output for a single
`(hidden, token)` pair, ignoring the K/V cache for now. The next-token
prediction will be junk on its own, but the kernel plumbing is exercised.

**Layout of `MTPModule::forward`:**

```
embed(token)                  // [1, hidden]
pre_fc_norm_embedding(embed)
pre_fc_norm_hidden(hidden)
concat([norm_h, norm_e]) -> [1, 2*hidden]
fc(concat) -> [1, hidden]     // projection back to hidden dim
attention(...)                 // STAGE B: writes K/V but reads none
moe(...)                       // shared + routed experts
final_norm()
lm_head() -> logits
```

The MoE branch reuses `compute_router_topk` and the per-expert `ops::linear`
pattern from `moe_forward.cpp`, but inlined into `mtp_moe_one_token` so it
operates on a single row (N=1) without going through `moe_layer_forward`'s
dispatch.

**One stumble:** `Qwen3_5MoEConfig` does NOT carry `norm_topk_prob` —
that field lives on `HybridForwardConfig`. Hardcoded `true` in
`mtp_moe_one_token` since that's the Qwen3.5/3.6 release setting.

---

## 5. Stage C.1 — MTP K/V cache + proper attention

The MTP head has its own attention layer that needs its own K/V cache —
the main model's K/V cache stores main-layer projections, which are
dimensionally different from MTP's.

**Cache layout chosen:** `[max_kv_len, num_kv_heads * head_dim]` bf16
contiguous, accessed via a **single-page paged-attention trick**:
`block_size = max_kv_len`, `page_table = [0]`. This lets the existing
`ops::attention` paged-decode kernel handle the read with zero new CUDA
code — the cache is treated as one big "page" of length max_kv_len.

**Why this over a flat dense cache:** the attention kernel already handles
the paged layout; introducing a parallel non-paged dispatch path would
have been more code than just configuring the paged path with a
one-element page table.

**The attention call inside MTP** (`mtp_attention_decode`) lives at line
221 of `mtp_module.cpp`:

```cpp
params.seq_len  = past + 1;
params.seqlen_q = 1;
ops::attention(params);   // dispatches to native paged_decode kernel
```

Plus the per-call ceremony of MRoPE on Q/K (using `mrope_3d` with all three
T/H/W positions equal — text-only path), `rms_norm` with the (1+w) Qwen3.5
weight form, and gate * attn output.

After C.1 the MTP forward produces sensible per-step predictions but only
in isolation (no prompt-prefill cache built yet).

---

## 6. Stage C.2 — MTP prompt prefill

After C.1, the very first MTP call on a fresh request had no past K/V, so
its first ~5 predictions were noise until the cache warmed up. C.2 added
`MTPModule::prefill`, called once after the main prefill:

- Iterate over prompt positions 0..P-1.
- For each, run `MTPModule::forward(hidden_main_row_i, next_token_i)`.
- This pre-populates MTP's K/V at positions 0..P-1.

**Measured impact:** offline accept rate (MTP-debug observer mode comparing
`mtp_top1` of step N to `main_argmax` of step N+1) went from **26% to 50%**
on the "2+2" prompt. Stage C.2 commit message captures this.

The 50% number became the reference point that Stage D was meant to hit
in production spec mode. It did not — see Stage D.1 and E.

---

## 7. Stage D.0 — per-Request K/V state

Up to C.2, MTP's K/V buffers were module-globals: one allocation reused
across requests. Stage D.0 moved them onto `InferenceRequest`:

```cpp
struct InferenceRequest {
    // ...
    tensor_t mtp_k_cache;             // [max_kv_len, Hkv * Dh] bf16
    tensor_t mtp_v_cache;
    tensor_t mtp_page_table_dev;      // [1] int32 = {0}
    int      mtp_past_seq_len = 0;
    int      mtp_pending_draft = -1;
    int      mtp_accept_count = 0;
    int      mtp_reject_count = 0;
    int      mtp_last_n_committed = 0;
};
```

`mtp_reset_request_state()` is called from `serving_loop.cpp` when a fresh
request first hits the MTP path (right at prefill). Allocation is amortized
over the request's lifetime; only `mtp_past_seq_len` resets between turns.

No throughput change from this stage — purely a correctness move so two
concurrent requests don't trample each other's MTP K/V.

---

## 8. Stage D.1 — end-to-end speculative decoding

This is where MTP graduates from "observer" to "active drafter."

**Wiring** (`serving_loop.cpp` + `scheduler.cpp` + `batch_context.cpp`):

1. After each main forward, MTP runs and writes its top-1 prediction to
   `req.mtp_pending_draft`.
2. At the *next* scheduler step, `batch_context.cpp` detects
   `req.mtp_pending_draft >= 0` and pushes 2 tokens into the batch
   (`last_token` at position `seq_len`, `draft` at `seq_len+1`),
   `num_tokens=2`.
3. The main forward runs as 2-token verify. The new piece in
   `paged_forward_context.cpp` is `attend_decode_single` — when `n_q > 1`
   it falls out of the flashinfer decode dispatch and into the native
   `paged_prefill` kernel (this matters for Stage E perf).
4. `process_results` reads the two logit rows:
   - If `argmax(logits[0]) == draft`: **accept** — commit
     `[draft, sample(logits[1])]`, seq_len += 2, `mtp_last_n_committed = 2`.
   - Else: **reject** — commit `[argmax(logits[0])]`, seq_len += 1,
     `mtp_last_n_committed = 1`. The dirty K/V at position `seq_len+1`
     will be overwritten by the next step's input.
5. Serving loop advances MTP's K/V by `mtp_last_n_committed` positions
   using the corresponding rows of `hidden_main` and the emitted tokens.
6. Final logit's argmax becomes the next step's `mtp_pending_draft`.

**Smoke test (D.1 commit):**

```
spec verify steps: 21    accepts: 5    rejects: 16
accept rate:        5/21  = 23.8%
tokens-per-step:    26/21 = 1.24x
output:  "The sum of 2* plus 2, is, 4.\n\n**2 + **2** = **4**<|im_end|>"
```

Two things were noted in the commit message and triaged into Stage E:

- **Accept rate dropped 50% → 24%** between C.2's offline measurement and
  D.1's production. Hypothesized: kernel-precision drift between
  `paged_prefill` (used by 2-token verify) and `paged_decode` (used by
  non-spec) — bf16 reduction order differs.
- **Output diverges from non-spec** ("The sum of 2* plus 2..." vs the
  cleaner "The sum of 2 and 2 is **4**.").

Both became Stage E's input.

---

## 9. Stage E — wall-clock investigation

Stage E was meant to push accept rate from 24% back toward 50% and confirm
the wall-clock speedup. Neither outcome materialized.

### 9.1 Three attempts at fixing kernel-precision drift

The thinking was: if `n_q=2` verify and `n_q=1` non-spec use different
attention kernels, their bf16 reduction trees differ, and argmax flips on
borderline tokens. Three implementations were tried and all reverted:

**Attempt 1 — single flashinfer prefill call with `seqlen_q=n_q`,
`kv_batch_size=1`.** Math is correct on paper (queries align to the last
`n_q` kv positions, causal mask handles the rest), but on the
"transformer" prompt:

- Offline (MTP-debug): 47.6% accept rate, output matches non-spec.
- D.1 baseline (native paged_prefill, no flashinfer): 24% accept rate, output diverges.
- Attempt 1 (single flashinfer call): 23.8% accept rate, output also diverges.

Switching the kernel did *not* recover the offline accept rate. The first 5
emitted tokens matched non-spec, then diverged at step 6 just like the D.1
baseline. The kernel itself wasn't the bottleneck.

**Attempt 2 — n_q virtual decode slots sharing one block table.** Treat
the `n_q` new query positions as `n_q` separate single-query decode slots,
all pointing to the same block table but with `kv_len = past_len + k + 1`
varying per virtual slot. flashinfer handles each as an independent query
with `kv_batch_size = n_q`.

This worked correctness-wise (52% accept rate on "transformer", matching
offline) but the per-batch plan layout differed from non-spec's
`kv_batch_size=1` plan, so query 0's reduction still diverged from
non-spec. Output: "The **Transformer** is a revolutionary, architecture,
become, the, foundation, of, modern, large, language, models" — semantic
but obviously degraded.

**Attempt 3 — N sequential flashinfer calls, each `kv_batch_size=1`.** Most
conservative approach: per-query flashinfer call with its own scratch
metadata, identical plan shape to non-spec's call. Output: identical
divergence pattern as Attempt 2. The kernel call shape matched non-spec
for query 0, but the **K/V values at position `past_N` themselves differ**
between modes — because Q/K/V projections are bf16 GEMMs whose tiling
depends on batch shape (M=1 vs M=2), and the resulting K[past_N] is not
bit-identical between modes. The error compounds through layers.

**Conclusion of attempt 1–3:** the kernel-precision drift is not in the
attention dispatch — it's in *every* per-layer linear that processes the
extra query token. Under bf16 with batch-shape-dependent tiling, n_q=2
verify cannot be made bit-identical to n_q=1 non-spec without either:

1. Forcing matmul tiling to be invariant under M=1/M=2 (custom kernel work).
2. Running verify in fp16/fp32 (precision higher than the gap).
3. Approximating the verify — accept if the draft is "plausible" under the
   main distribution, not bit-equal to argmax.

None are a one-PR fix. Reverted Attempts 1–3 and moved to wall-clock.

### 9.2 Wall-clock measurement

Three prompts, `--max-new-tokens 80`, `ZEDINFER_FORCE_ARGMAX=1`:

| Prompt                                      | Non-spec    | Spec        | Slowdown |
| ------------------------------------------- | ----------- | ----------- | -------- |
| "Write a short story about a robot."        | 34.59 tok/s | 2.85 tok/s  | 12.1×    |
| "Explain transformers in 2 sentences."      | 35.29 tok/s | 3.01 tok/s  | 11.7×    |
| "What is the capital of France?"            | 35.28 tok/s | 2.60 tok/s  | 13.6×    |

Spec mode is ~12× **slower**, not faster. Per-step instrumentation
(temporary `ZEDINFER_MTP_PERF=1` flag wrapping the MTP block) explained it:

```
SPEC mode (n_q=2 verify):
  main forward    : ~600 ms     ← !!
  MTP forward     : ~35 ms × (1 on reject, 2 on accept)
  per step total  : ~640 ms  → ~430 ms/token at 50% accept

NON-SPEC + MTP_DEBUG:
  main forward    : ~30 ms
  MTP forward     : ~35 ms
  per step total  : ~65 ms

NON-SPEC pure:
  main forward    : ~30 ms
  per step total  : ~30 ms
```

The 2-token main forward is **20× slower** than the 1-token main forward —
not the expected 1.1×. Root-causing that is where the actual finding lives.

### 9.3 Root cause: MoE falls off the fast path at N>1

`src/frontend/models/moe_forward.cpp:392`:

```cpp
const bool use_scratch = (scratch != nullptr && N == 1);
```

This is the gate between `moe_decode` (N=1, pre-allocated scratch, M=1
expert linears via `dispatch_expert_linear`) and `moe_prefill` (N>1,
gather-by-expert + grouped GEMMs + scatter-back). The latter is tuned for
N=512 prompt processing, not N=2 verify:

- 16 active experts in the worst case (2 tokens × top-8) vs 8 for N=1.
- Per-call fresh `Tensor::create` for `gathered_full`, `g_gate_full`,
  `g_up_full`, `g_act_full`, `g_down_full` — `moe_decode` reuses
  `DecodeScratch`'s buffers.
- Two extra async H2D memcpys per layer.
- `gather_rows` + `scatter_add_rows` kernel launches per active expert.

At N=2 the gather/scatter overhead dominates the actual GEMM work. The
GEMM time barely changes — launch and setup explode.

**Fix shape** (deferred): add a third path "decode for small-N" (N ∈
{1, 2}) that keeps scratch-style buffer reuse and dispatches per-token
through `dispatch_expert_linear` with M=1 GEMMs, the way `moe_decode`
already does. Pre-size `DecodeScratch`'s buffers for N ≤ MAX_SPEC_BATCH.

### 9.4 Secondary cost: MTP forward itself is ~35 ms

For one transformer layer worth of work, 35 ms is way too high. Looking at
`mtp_module.cpp`:

- **Two synchronous D2H memcpys per call** in `mtp_moe_one_token` —
  router logits (line 343) and shared-expert gate (line 393). Each drains
  the stream; the pipeline stalls cost more than the transfers themselves.
- **CPU-resident top-k softmax** (line 350). Forces the D2H sync.
- **~50 small linear launches per call**: top-8 experts × 4 ops + shared
  expert × 4 + Q/K/V/O projections + 2 norms. Each launch is ~5–10 µs of
  cuBLAS overhead, which dominates the actual compute at M=1.
- **Per-call fresh allocations** for every intermediate buffer.

**Fix shape** (deferred): mirror what `moe_decode` + `DecodeScratch` did
for the main forward — pre-allocate a `MTPDecodeScratch`, move top-k to
GPU, fuse small linears.

### 9.5 Break-even analysis

Even after fixing both bottlenecks:

- Main 2-token forward: ~35 ms (5 ms over baseline, just the extra row
  through the MoE pass and one extra attention row).
- MTP forward: ~10 ms (after scratch reuse + GPU top-k).
- Per spec step: 35 + 1.5×10 = 50 ms, emitting 1.5 tokens at 50% accept
  → **~33 ms/token**.

vs non-spec pure: **~30 ms/token**.

Still a wash. To net-win on this model/hardware, MTP would need either
accept rate >60% (unlikely under bf16) or sub-5 ms forward time. The more
promising direction is **CUDA Graph capture** for the main decode path —
eliminates per-layer launch overhead universally, independent of spec
decode. That's where decode perf work should go next.

---

## 10. What shipped vs what was reverted

**Shipped (in `feat/qwen3.5` branch):**

- Stages A → D.1 (commits `8f7f7d8` → `fcb204f`): full MTP weight load,
  forward, K/V cache, prefill, speculative integration. Env-gated on
  `ZEDINFER_MTP_SPEC=1` (active spec) or `ZEDINFER_MTP_DEBUG=1`
  (observer-only).
- Stage E findings doc (`e841908`): this writeup and the perf summary.

**Reverted (not on the branch):**

- All three Stage E.1 attempts at attention kernel changes.
- A temporary perf-instrumentation probe in `serving_loop.cpp` (was used
  to produce the §9.2 numbers).
- A naive "treat n_q>1 as flashinfer prefill" change in
  `build_flashinfer_decode_cache` that broke output text quality.

**Default behavior:** unchanged. Without env vars, the engine runs
non-spec exactly as before Stage A.

---

## 11. Lessons applicable to future spec-decode work

1. **Measure wall-clock before celebrating accept rate.** Stage D.1's
   24% accept rate looked like a regression from 50%; the actual problem
   was that *every* spec step was 20× slower than a non-spec step, which
   no accept rate could compensate for. The token-level metric was a
   distraction from the kernel-level reality.

2. **Bf16 + batched verify ≠ bf16 + unbatched non-spec.** Even with
   identical attention kernels, the per-layer Q/K/V/O linears have
   batch-shape-dependent tiling. K/V values diverge at every position the
   verify writes. This is a property of the kernel selection, not a bug.

3. **The MoE fast path is structured around N=1.** Anything else falls
   into a path tuned for the opposite extreme (N=512 prompt processing).
   Any future small-batch optimization (spec decode, multi-request
   continuous batching where slot count goes 1 → 2 → 4 → ...) will hit
   this same cliff and need the same fix.

4. **Stage gating saved a lot of pain.** Because C.2 was a usable
   observer (`MTP_DEBUG=1` printing draft vs actual), Stage E could
   compare offline vs production accept rates on the *same* code path.
   That comparison is what pinpointed kernel precision as a candidate
   root cause — a hypothesis that turned out wrong, but cheap to test.

---

## 12. References

- Commits: `8f7f7d8`, `5fd067b`, `f73d333`, `875f3a0`, `575cb88`,
  `fcb204f`, `e841908` (in order).
- Perf summary: [`stage_e_mtp_spec_decode_perf.md`](stage_e_mtp_spec_decode_perf.md).
- Adjacent debug write-up (sampler/thinking drift, not spec-decode but
  related to Qwen3.5 GPTQ-Int4 numerical quirks):
  [`qwen3_5_sampler_and_thinking_drift.md`](qwen3_5_sampler_and_thinking_drift.md).
- Source files touched: `mtp_module.{hpp,cpp}`, `qwen3_5_moe.cpp`,
  `serving_loop.cpp`, `scheduler.cpp`, `batch_context.cpp`,
  `paged_forward_context.cpp`, `request.hpp`, `hybrid_transformer_forward.cpp`,
  `chat_template.cpp`.
