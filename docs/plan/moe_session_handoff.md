# MoE Session Handoff (Phase 2 in progress)

> Carries state between sessions on `feat/hetero-moe-support`. Update at the end of
> every session that touches the MoE path.

## 1. Current state at a glance

- ✅ **Phase 1**: Qwen3-30B-A3B-GPTQ-Int4 runs end-to-end on NVIDIA GPU, all experts
  GPU-resident. Validated ("Who are you?" → coherent "Qwen..." output).
- ✅ **Cleanup rounds**: code organization, ops layout, dispatch unification,
  `ExpertWeights` indexed storage, per-layer MoE decision, HF-matching topk_softmax,
  `moe_forward.cpp` split into `compute_router_topk` / `moe_decode` / `moe_prefill` /
  `apply_shared_expert`.
- ✅ **Phase 2 runtime infrastructure**: transfer stream in `Runtime`,
  `register_pinned` / `unregister_pinned` API on CPU and NVIDIA backends
  (`nvidia_runtime_api.cu:113`), pre-allocated prefill buffers.
- ✅ **M1 — ExpertPool MVP (ALL_GPU)** — commit `36e73cc`.
- ✅ **M2 — CPU pinned storage + synchronous H2D + `ZEDINFER_MOE_GPU_SLOTS`**.
- ✅ **M3 — Async prefetch on transfer stream + sliding-window**. Async H2D on
  `Runtime::transfer_stream()` guarded by per-slot `cudaEvent_t`; `moe_decode` and
  `moe_prefill` issue prefetch batches before the compute loop (prefill uses a
  sliding window sized to `max_prefetch_depth()` to avoid self-eviction). At N=32
  on Qwen3-30B-A3B-GPTQ-Int4: prefill latency 1.36× baseline, decode ~1× baseline.
  See `heterogeneous_moe.md §6 M3` for the numbers.
- ⏳ **M4 (optional) — Predictive prefetch** — not needed for current workloads;
  revisit only if a model with extreme fan-out shows under-utilized compute.

Phase 2 core is functionally done. Remaining open items are out-of-scope-for-M3
polish: streaming expert load (BF16 60GB support), auto N sizing, CPU-side Fiddler
execution of cold experts.

Key code entry points:
- `include/frontend/models/expert_pool.hpp` + `.cpp` — ExpertPool, slot arena, LRU,
  async H2D, sliding-window support via `compute_touched` flag + `pick_lru_slot`
  returning `-1` when no evictable slot.
- `include/backend/device/runtime_api.hpp` — event API (`create_event` /
  `destroy_event` / `record_event` / `stream_wait_event`).
- `include/frontend/models/forward_config.hpp:52` — `ExpertPool* expert_pool`.
- `forward_config.hpp:118-153` — `dispatch_expert_linear` calls `ensure_on_gpu`.
- `src/frontend/models/moe_forward.cpp` — `moe_decode` batch prefetch;
  `moe_prefill` active-experts ordering + sliding window.
- `src/frontend/models/qwen3_moe.cpp` — `choose_pool_config()` parses
  `ZEDINFER_MOE_GPU_SLOTS`; ctor constructs pool.

## 2. Validation strategy

Two models:

1. `Qwen/Qwen3-30B-A3B-GPTQ-Int4` (~15 GB, downloaded)
   - Bit-exact comparison vs. ALL_GPU for correctness.
   - Force offloading via `ZEDINFER_MOE_GPU_SLOTS=N` for `N < num_experts`.
   - Already validated through M2 at `N=32`.

2. `Qwen/Qwen3-30B-A3B` BF16 (~60 GB, user may not have downloaded)
   - Real "doesn't fit on 24 GB GPU" scenario — requires a streaming load path
     (experts read from safetensors mmap directly into CPU pinned, not via GPU).
   - Milestone validation only; ground truth via HuggingFace `transformers`.
   - Also the first test of the dense (non-quantized) `gate_weight` / `up_weight` /
     `down_weight` path through `allocate_slot_tensors` and `dispatch_expert_linear`.

## 3. Pre-existing caveats (still relevant)

1. **Qwen3-30B-A3B `head_dim = 128`** even though `hidden_size / num_heads = 64`.
   The forward loop reads `head_dim` from config.

2. **Qwen3-30B-A3B has no shared expert** despite HF naming. `apply_shared_expert`
   respects `has_shared_expert` (`src/frontend/models/moe_forward.cpp:151`).

3. **GPTQ format quirks**: `qweight` transposed in loader (`[K/8, N]` → `[N, K/8]`);
   zero-point convention adjusted in `process_gptq_group`.

4. **`moe_prefill` per-row memcpy gather**: known perf TODO
   (`moe_forward.cpp:118-125`). A CUDA gather kernel would batch launches. Not on
   critical path for M3.

5. **Router logits D2H sync**: per-MoE-layer `cudaDeviceSynchronize` forced by
   `compute_router_topk`'s copy to CPU. GPU top-k kernel would eliminate it.
   Separate optimization.

6. **M2 peak VRAM**: PINNED_LRU ctor assumes experts arrive on GPU then D2H-migrates.
   Peak briefly holds full expert set on GPU. Enough for INT4 30B on 24 GB; BF16 60B
   requires the streaming-load path (see §5 item 1).

7. *(Resolved)* Shared_ptr cycle in `InferenceEngine` — `ServingLoop` and `Profiler`
   now hold a non-owning `InferenceEngine*`, so the engine destructor runs cleanly
   and the ExpertPool stats log lands in `logs/ping.log` from `~ExpertPool`.

## 4. Build and run

```bash
# Configure + build (clean build after struct layout changes)
xmake f -m release --nv-gpu=y --onednn=y
xmake build

# ALL_GPU (unchanged behavior)
xmake run ping ~/data/models/Qwen3-30B-A3B-GPTQ-Int4 --nvidia

# PINNED_LRU: force offloading
ZEDINFER_MOE_GPU_SLOTS=32 xmake run ping ~/data/models/Qwen3-30B-A3B-GPTQ-Int4 --nvidia

# Edge of per-layer slots: top_k=8 is the minimum before correctness-preserving thrashing
ZEDINFER_MOE_GPU_SLOTS=8 xmake run ping ~/data/models/Qwen3-30B-A3B-GPTQ-Int4 --nvidia

# Iteration shortcut
ZEDINFER_DISABLE_WARMUP=1 xmake run ping ~/data/models/Qwen3-30B-A3B-GPTQ-Int4 --nvidia
```

## 5. Suggested next work (Phase 2 polish, any order)

Phase 2 M1–M3 are landed and validated on Qwen3-30B-A3B-GPTQ-Int4. These items are
worth doing but not blocking the thesis's core claim:

1. **Streaming expert load (BF16 60GB support).** Current PINNED_LRU ctor D2H-migrates
   experts from GPU to CPU pinned, which assumes the loader first fit everything on
   GPU. Needed for models that genuinely don't fit (BF16 30B on 24GB, or 70B+ MoE).
   Touches `src/frontend/loader/safetensor.cpp` and `src/frontend/models/base.cpp` to
   let expert tensors be read straight to CPU pinned while non-experts still go to
   GPU. Also unlocks dense-path validation through `allocate_slot_tensors` and
   `dispatch_expert_linear` (GPTQ path is well-tested; dense path is unexercised).

2. **Auto N sizing.** When `ZEDINFER_MOE_GPU_SLOTS` is unset, pick N from
   `get_memory_info()` − KV budget − non-expert weights. Removes the "guess a number"
   experience that currently blocks first-run users. Implementation note: do it after
   the block_pool init, so we know KV actual footprint.

3. **Fix the prefill per-row gather** (`moe_forward.cpp:140-147`). PERF-TODO predates
   M3 but surfaces now as the dominant remaining prefill cost — M3 sliding window
   exposed it. A single CUDA gather kernel batching the per-row copies is the obvious
   fix.

4. **Fiddler-style CPU execution** of cold experts. `docs/plan/heterogeneous_moe.md §9`
   kept this out of Phase 2 scope; revisit only if a model/config appears where
   transfer still dominates after M3.

## 6. Pre-existing issues worth a future pass

1. **Non-quantized MoE path unvalidated.** The dense (`gate_weight`/`up_weight`/
   `down_weight`) path in `allocate_slot_tensors` and `dispatch_expert_linear` compiled
   and runs through the motions but has never seen a BF16 MoE model. Gated on item 1
   above.

2. **`xmake run` drops env vars.** Observed during bench (see commit messages around
   the `CUDA_VISIBLE_DEVICES` investigation). Workaround: invoke binaries directly
   with `LD_LIBRARY_PATH=$PWD/python/zedinfer:$LD_LIBRARY_PATH`. No plan to change
   xmake's behavior.

## 7. Development rules

- Small, reversible steps. `xmake build && ping` per change, compare text output.
- Preserve the per-layer slot scope — it's the right fit for sequential-layer decode.
- `log_stats()` fires from the destructor now (shared_ptr cycle fixed). Writes to
  `logs/ping.log`. Useful for sanity-checking hit rates after a test.
- At session end, **update this file**.
