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
- ✅ **M2 — CPU pinned storage + synchronous H2D + `ZEDINFER_MOE_GPU_SLOTS`** — this
  session. Qwen3-30B-A3B-GPTQ-Int4 runs with `N=32` slots/layer: text output matches
  ALL_GPU baseline; decode is synchronous-H2D slow (expected). Hit/miss stats logged
  in destructor.
- 🚧 **M3 — Async prefetch on transfer stream** — not started. Next milestone.
- ⏳ **M4 (optional) — Predictive prefetch** — gated on M3 profiling.

Key code entry points:
- `include/frontend/models/expert_pool.hpp` + `.cpp` — `ExpertPool`, slot arena, LRU.
  Per-layer `slots_` / `residency_` / `access_counter_`. `hits_` / `misses_` cumulative
  stats in destructor.
- `include/frontend/models/forward_config.hpp:52` — `ExpertPool* expert_pool`.
- `forward_config.hpp:118-153` — `dispatch_expert_linear` calls `ensure_on_gpu`.
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
   requires the streaming-load path (see §2).

7. **Shared_ptr cycle in InferenceEngine** (pre-existing, surfaced by M2 diagnostics):
   `InferenceEngine` uses `enable_shared_from_this`; its owned `unique_ptr<ServingLoop>`
   and `unique_ptr<Profiler>` each hold a `shared_ptr<InferenceEngine>` (from
   `shared_from_this()`). When the outer shared_ptr in `main()` releases, refcount
   drops from 2 → 1 and the engine is never destroyed — Model and ExpertPool never
   hit their destructors. Workaround in place: `Model::log_runtime_stats()` virtual
   + explicit call from `examples/ping.cpp` before `return 0`. Proper fix (out of
   M2 scope): change ServingLoop/Profiler to hold a raw pointer or reference to
   InferenceEngine instead of shared_ptr.

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

## 5. Suggested first step for the next session (M3)

Read in order:
1. This file.
2. `docs/plan/heterogeneous_moe.md` §6 M3 and §7 (env vars).
3. `include/frontend/models/expert_pool.hpp` / `.cpp` — today's PINNED_LRU
   implementation. M3 modifies `ensure_on_gpu` miss path and `prefetch`.
4. `include/backend/core/runtime/runtime.hpp` — `transfer_stream()` accessor.
5. `src/frontend/models/moe_forward.cpp` — call site where M3 will insert
   `prefetch(layer, expert_id)` after router emits top-k.

M3 strategy:
- Add `cudaEvent_t ready_event` to each `Slot` (or backend-agnostic `zedinferEvent_t`
  if we want a CPU backend story).
- In miss path: enqueue `cudaMemcpyAsync` on `runtime.transfer_stream()`; record event.
  Mark slot as "populating" so the compute path knows to wait.
- Compute stream: `cudaStreamWaitEvent` on slot event before calling the expert GEMM.
- `prefetch(layer, expert_id)` from `moe_decode` right after router: for each selected
  expert not resident, kick off async H2D.
- Benchmark decode latency vs. M1 ALL_GPU baseline.

Development rules:
- Small, reversible steps. `xmake build && ping` per change, compare text output.
- Preserve the per-layer slot scope — it's the right fit for sequential-layer decode.
- Hit-rate log in destructor is a cheap sanity check; use it to verify prefetch hits.
- At session end, **update this file**.
