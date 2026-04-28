# Phase 3+ Directions

> Forward-looking menu of work that goes beyond Phase 2 (heterogeneous MoE +
> two rounds of polish). Nothing here is committed scope — items are sized,
> ranked, and tagged with thesis-fit so future sessions can pick by deadline
> and time budget.
>
> Status when this was written: Phase 2 is done. M1 / M2 / M3 + BF16 D-series
> + auto-N + gather/scatter kernels + four CPU-side perf cleanups all landed.
> The MoE path's last non-trivial CPU-side cost is the router logits D2H sync
> (see §A.1 below).

## A. High ROI, narrow scope

### A.1 GPU-side top-k kernel for router — **~½ day**

**Problem.** `compute_router_topk` runs the router gate on GPU but pulls logits
back to host (D2H), softmaxes + top-k on CPU, then ships expert IDs/weights to
the per-step bucketing. Each MoE layer's `cudaMemcpy(D2H)` forces a global
device sync. Over 48 layers in decode this serializes the pipeline.

**Fix.** Write a CUDA kernel that fuses softmax + top-k + (optional) renorm
and writes `expert_ids[N×top_k]` (int32) + `expert_weights[N×top_k]` (f32)
directly into a pinned host buffer via `cudaMemcpyAsync`. Compute path stays
on the compute stream; CPU bucketing reads the pinned buffer on demand.

**Files.** New `include/backend/ops/moe/topk_softmax.hpp` GPU entry +
`src/backend/ops/moe/nvidia/topk_softmax.cu`; modify `compute_router_topk`
in `src/frontend/models/moe_forward.cpp` to dispatch to GPU when
`input->deviceType() == ZEDINFER_DEVICE_NVIDIA`.

**Expected.** Decode `+几%` (every layer saves one sync). Closes out the
MoE path's CPU-side perf list.

**Thesis fit.** Same vein as the polish round 2 commits. Strengthens the
"heterogeneous MoE on consumer GPU" perf story.

### A.2 Speculative decoding — **~3-5 days**

**Problem.** Decode is launch-bound and memory-bandwidth-bound on consumer
GPUs. Even with ExpertPool sliding-window prefetch, decode tok/s caps where
single-token forward latency caps.

**Fix.** Standard draft+verify spec decoding:
- Load a small draft model (e.g. Qwen3-1.5B or self-distillation head)
- Draft generates N candidate tokens autoregressively
- Main model does ONE batched forward pass on all N candidates
- Accept the longest matching prefix; rebase on first divergence

Variants worth comparing in the paper:
- **Vanilla** (chain draft): simplest, baseline
- **EAGLE-style** (tree draft): higher acceptance rate
- **Self-spec** (last-layer-only draft head): no separate model, less RAM

**Files.** Bigger refactor:
- New draft model loader path (or share `Model::parse`)
- Verify-only forward path in `transformer_forward` that takes N candidate
  tokens at once (existing prefill path with paged attention may already
  handle this — check)
- Acceptance logic + KV cache rollback in `Scheduler` / `ServingLoop`

**Expected.** Decode 1.5-3× depending on prompt structure and draft quality.

**Thesis fit.** Strong. Stands as its own chapter, perf gain is dramatic and
visualizable, fits the "consumer GPU inference" framing. Independent of MoE
work — works for both dense (Qwen3) and MoE (Qwen3-MoE) targets.

## B. Thesis-title coverage

### B.1 Multi-request / multi-user scheduling — **~3-5 days**

**Problem.** The thesis title mentions "multi-user". `ServingLoop` already
runs continuous batching, but the actual benchmarks and validations are
single-stream. We've never stress-tested concurrent requests with the
ExpertPool path.

**Fix.** Audit and harden the multi-request flow:
- HTTP server already accepts concurrent submissions via cpp-httplib thread
  pool → Scheduler. Verify request isolation under MoE.
- Test fairness: long prompt + short prompt simultaneously, ensure neither
  starves.
- KV cache eviction policy under memory pressure — check that prefix cache
  + block pool LRU works reliably.
- Add a multi-request benchmark target (e.g. `batch_bench` extended to
  spawn N concurrent client streams).

**Risk.** Some Phase 2 paths (PINNED_LRU's slot arena) were designed for
single-stream. Concurrent ensure_on_gpu calls might race on slot eviction
or LRU counters. Needs explicit thread-safety review.

**Thesis fit.** Closes the "multi-user" gap in the title literally. Without
it, the thesis is technically about "single-user heterogeneous MoE on
consumer GPU".

### B.2 Fiddler-style CPU expert execution — **~5 days, low ROI**

> Already discussed and skipped twice (handoff §5 item 3). Listed here for
> completeness because it's the only thing that makes "异构推理" mean
> "compute heterogeneity" and not just "memory heterogeneity".

**Problem.** Current PINNED_LRU still does compute on GPU; cold experts get
shipped over PCIe. For very-cold or thrashing-prone experts, computing on
CPU directly (using the already-pinned host weights) skips PCIe entirely.

**Why deferred.**
- Need a CPU INT4 GEMM (oneDNN doesn't ship one; would need to write or
  vendor llama.cpp's q4_K kernels)
- Scheduler decision (when to pick CPU vs GPU per expert) needs latency
  modeling
- ExpertPool's hit rate at N=32 is already 99.88% on the validation
  workload, so cold-expert traffic is ~0.12% — absolute speedup is small

**When to revisit.** If the thesis framing requires "compute heterogeneity"
literally, or if a target model has much lower hit rate (extreme
fan-out / non-stationary routing).

## C. Model coverage (paper experiment table)

Each item ~0.5-1 day if the architecture is close to existing supported
families.

### C.1 Llama-family (3.x / 3.1 / 3.2)
Dense, very similar to Qwen3 dense. Mostly weight-name mapping in
`load_weights` + `Model::parse` branch. Validates the dense path beyond Qwen.

### C.2 Mixtral 8x7B / 8x22B
MoE with 8 experts top-2. Same shape family as Qwen3-MoE; mainly different
config field names + tokenizer. Adds an "MoE we support" data point.

### C.3 DeepSeek V2 / V3
MoE with **multi-head latent attention (MLA)** — different attention
shape (low-rank K/V), would need attention path changes. Higher cost but
high paper value (these are state-of-the-art MoE models).

### C.4 GLM4 / Yi / Phi
Various dense families. Each is a small adapter PR.

## D. Engineering cleanup (post-defense PR-able)

### D.1 Real Jinja2 chat template via vendored minja
Already in handoff §5 item 5. Replaces hardcoded `default_qwen_chatml` /
`default_deepseek_r1`. Modern Qwen3's `tokenizer_config.json` uses Jinja
features (namespace, slicing, filters) for tool calling and thinking-mode
that we currently skip.

### D.2 HuggingFace token-level alignment harness
Already in handoff §5 item 4. Python script:
1. Load HF model with `transformers.AutoModelForCausalLM`, greedy generate
   N tokens for a fixed prompt
2. Run zedinfer on the same prompt with argmax sampler
3. Compare token IDs; report first-divergence position and per-position
   logit max-abs-diff

Adds a "we match HF on first 50 tokens" line to the thesis correctness
section. Closes the "is the output really right" question evaluators
always ask.

### D.3 Address `compute_router_topk` PERF-TODO via §A.1
Same item as A.1, listed here only because it's flagged as PERF-TODO in
`moe_forward.cpp:79`.

## E. Recommended ordering by deadline pressure

If thesis defense is **soon (< 2 weeks)**:
1. **A.1** GPU top-k kernel (~½ day) — closes MoE perf story
2. **D.2** HF token alignment (~1 day) — answers correctness question
3. **C.1 + C.2** add Llama + Mixtral (~1-2 days) — extends paper experiment
   table breadth
4. Defer A.2 / B.1 to "future work" section

If thesis defense is **medium (2-6 weeks)**:
1. A.1 + D.2 + C.1 + C.2 (above, ~4 days)
2. **A.2** Speculative decoding (~3-5 days) — its own paper chapter

If thesis defense is **further out**:
1. All of the above
2. **B.1** Multi-request hardening (~3-5 days) — closes thesis-title gap
3. **C.3** DeepSeek V2/V3 with MLA (~2-3 days) — strongest model coverage

Post-defense PR-able anytime: **D.1 (Jinja2)**, **B.2 (Fiddler)**.
