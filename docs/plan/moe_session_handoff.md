# MoE Session Handoff (Phase 2 in progress)

> Carries state between sessions on `feat/hetero-moe-support`. Update at the end of
> every session that touches the MoE path.

## 1. Current state at a glance

- ✅ **Phase 1**: Qwen3-30B-A3B-GPTQ-Int4 runs end-to-end on a single NVIDIA GPU,
  all experts GPU-resident. Validated ("Who are you?" → coherent "Qwen..." output).
- ✅ **Cleanup rounds**: code organization, ops layout, dispatch unification,
  `ExpertWeights` indexed storage, per-layer MoE decision, HF-matching topk_softmax,
  `moe_forward.cpp` split into `compute_router_topk` / `moe_decode` / `moe_prefill` /
  `apply_shared_expert`.
- ✅ **Phase 2 runtime infrastructure**: transfer stream in `Runtime`,
  `register_pinned` / `unregister_pinned` API on both CPU and NVIDIA backends
  (`nvidia_runtime_api.cu:113`), pre-allocated prefill buffers in `moe_prefill`.
- ✅ **M1 — ExpertPool MVP (ALL_GPU)**: landed at commit `36e73cc`. Output verified.
  Key code:
  - `include/frontend/models/expert_pool.hpp` — `ExpertPool`, `ExpertGpuHandle`,
    `ExpertPoolStrategy { ALL_GPU, PINNED_LRU }`, `ExpertPoolConfig`.
  - `src/frontend/models/expert_pool.cpp` — `ALL_GPU` only; `PINNED_LRU` branch
    throws.
  - `include/frontend/models/forward_config.hpp:52` — `ExpertPool* expert_pool`.
  - `forward_config.hpp:118-153` — `dispatch_expert_linear` goes through pool.
  - `src/frontend/models/qwen3_moe.cpp:138-144` — `Qwen3MoEModel` constructs
    `ExpertPool` with default `ALL_GPU` config.
- 🚧 **M2 — CPU pinned storage + synchronous H2D**: not started. Next step.
- ⏳ **M3 — Async prefetch on transfer stream**: pending M2.

## 2. Validation strategy

Two models during Phase 2:

1. `Qwen/Qwen3-30B-A3B-GPTQ-Int4` (~15 GB, already downloaded)
   - Fast iteration. Bit-exact comparison vs. Phase 1 for correctness.
   - Force offloading via `ZEDINFER_MOE_GPU_SLOTS=N` for `N < num_experts` (once M2
     parses the env var).

2. `Qwen/Qwen3-30B-A3B` BF16 (~60 GB, user may not have downloaded)
   - Real "doesn't fit on 24 GB GPU" scenario. Milestone validation only.
   - Ground truth: HuggingFace `transformers` on the same model.
   - Exercises the dense (non-quantized) expert path — currently untested.

## 3. Pre-existing caveats (still relevant)

1. **Qwen3-30B-A3B `head_dim = 128`** even though `hidden_size / num_heads = 64`.
   The forward loop reads `head_dim` from config.

2. **Qwen3-30B-A3B has no shared expert** despite HF naming. `apply_shared_expert`
   respects `has_shared_expert` (`src/frontend/models/moe_forward.cpp:151`).

3. **GPTQ format quirks**: `qweight` transposed in loader (`[K/8, N]` → `[N, K/8]`);
   zero-point convention adjusted in `process_gptq_group`.

4. **`moe_prefill` per-row memcpy gather**: known perf TODO
   (`moe_forward.cpp:118-125`). A CUDA gather kernel would batch launches. Not on
   critical path for M2.

5. **Router logits D2H sync**: per-MoE-layer `cudaDeviceSynchronize` forced by
   `compute_router_topk`'s copy to CPU. GPU top-k kernel would eliminate it.
   Separate optimization.

6. **Design-doc inconsistency on slot scope**: `docs/plan/heterogeneous_moe.md`
   section 7 says `ZEDINFER_MOE_GPU_SLOTS` is "N per layer"; section 5.2's
   `residency_` map key `layer * num_experts + expert_id` suggests a global arena.
   Per-layer is the right fit for sequential-layer decode; M2 will implement
   per-layer slots and update the design doc to match.

## 4. Build and run

```bash
# Configure + build (do a clean build when struct layouts change)
xmake f -m release --nv-gpu=y --onednn=y
xmake build

# Current (M1, ALL_GPU)
xmake run ping ~/data/models/Qwen3-30B-A3B-GPTQ-Int4 --nvidia

# Once M2 is implemented: force N slots per layer
ZEDINFER_MOE_GPU_SLOTS=32 xmake run ping ~/data/models/Qwen3-30B-A3B-GPTQ-Int4 --nvidia

# Iteration shortcut
ZEDINFER_DISABLE_WARMUP=1 xmake run ping ~/data/models/Qwen3-30B-A3B-GPTQ-Int4 --nvidia
```

## 5. Suggested first step for the next session

Start M2 (CPU pinned storage + synchronous on-demand H2D).

Read in order:
1. This file.
2. `docs/plan/heterogeneous_moe.md` sections 5–6 (ExpertPool design, M2 tasks).
3. `include/frontend/models/expert_pool.hpp` + `.cpp` — the ALL_GPU-only skeleton
   that M2 extends.
4. `src/frontend/models/qwen3_moe.cpp:110-144` — `extract_expert_weights` and the
   pool construction site.
5. `src/backend/device/nvidia/nvidia_runtime_api.cu:113-125` —
   `registerPinned`/`unregisterPinned` implementation.

M2 design choices (see Section 6 of the design doc for rationale):
- **Per-layer slot arena.** Each layer owns a vector of `N` slots (`N` from env
  var, default = num_experts = ALL_GPU degenerate). Decode access is strictly
  sequential by layer, so per-layer arenas avoid cross-layer eviction churn.
- **Expert weight storage on CPU pinned memory.** Simplest path: extract experts
  after load as today (they're on GPU), D2H copy to pinned host buffers, free the
  GPU originals. Costs a transient VRAM spike during load but avoids a loader
  rewrite. Revisit if the transient pressure is prohibitive on small GPUs.
- **`ensure_on_gpu` contract unchanged.** Returns `ExpertGpuHandle` pointing into
  the assigned slot. M2 blocks on `cudaMemcpy` inside the call when the expert
  isn't resident; M3 will swap that for a `cudaStreamWaitEvent` on the compute
  stream.
- **LRU per layer.** Single `uint64_t access_counter_` per layer, slots record
  `last_access`. Eviction picks the smallest `last_access`.

Development rules:
- Small, reversible steps. After each step: `xmake build && xmake run ping …`.
- Clean build (`rm -rf build .xmake`) when changing struct layouts across TUs.
- Keep `dispatch_expert_linear` unchanged — the pool absorbs all residency logic.
- At the end of each session, **update this file** so the next session lands
  running.
