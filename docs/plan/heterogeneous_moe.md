# Heterogeneous MoE Design

> Status: Phase 1 (GPU-only MoE) complete. Phase 2 (CPU↔GPU expert offloading) is the
> subject of this document.

## 1. Goal

Run large MoE models (Qwen3-30B-A3B class and beyond) on consumer GPUs (≤24 GB VRAM) by
storing cold experts in CPU pinned memory and transferring them to GPU on demand. The
thesis differentiator is combining INT4 quantization with per-expert CPU/GPU placement on
a **single consumer machine** — not datacenter multi-GPU.

Non-goal for this phase: Fiddler-style CPU execution of experts (keep weights on CPU, send
activation there, compute on CPU). Deferred — slot-based offloading is enough to demonstrate
the core idea.

## 2. Phase 1 — what's already in place

The current code runs Qwen3-30B-A3B-GPTQ-Int4 end-to-end with **all** experts resident on
GPU. The key abstractions that Phase 2 will build on:

| Component | Location | Role |
|-----------|----------|------|
| `ExpertWeights` | `include/frontend/models/expert_weights.hpp` | `(layer, expert_id) → ExpertFFN` indexed storage |
| `ExpertFFN` | same | `tensor_t` handles for gate / up / down (quantized or dense) |
| `ModelForwardConfig::dispatch_expert_linear(...)` | `forward_config.hpp` | Calls `ops::linear_quantized` on the ExpertFFN's tensors |
| `Runtime::transfer_stream()` | `backend/core/runtime/runtime.hpp` | Second CUDA stream dedicated to H2D / D2H |
| `register_pinned` / `unregister_pinned` | `runtime_api.hpp` | `cudaHostRegister` wrappers for pinned host memory |
| Pre-allocated prefill buffers | `moe_forward.cpp::moe_prefill` | Reused across experts; not per-call allocations |

**The Phase 2 hook**: `dispatch_expert_linear` is the single entry point for expert weight
access in the forward path. Phase 2 replaces the direct `ExpertFFN` lookup with a call
through `ExpertPool`:

```cpp
// Phase 1 (current):
const auto& ffn = experts->at(layer, expert_id);
ops::linear_quantized(out, in, ffn.gate_packed, ...);

// Phase 2 (planned):
auto gpu_handle = expert_pool.ensure_on_gpu(layer, expert_id);
ops::linear_quantized(out, in, gpu_handle.gate_packed, ...);
```

## 3. Model profile: Qwen3-30B-A3B-GPTQ-Int4

Grounded in the actual config, not the original design-doc estimates:

| Field | Value |
|-------|-------|
| `num_hidden_layers` | 48 |
| `hidden_size` | 2048 |
| `num_attention_heads` | 32 |
| `num_key_value_heads` | 4 |
| `head_dim` | 128 |
| `num_experts` | 128 |
| `num_experts_per_tok` | 8 |
| `moe_intermediate_size` | 768 |
| `shared_expert_intermediate_size` | — (not used for this model) |
| Quantization | GPTQ INT4, group_size=128, sym=true |

Per-expert weight size (INT4, group_size=128, per expert, 3 projections):

```
gate_proj: [2048 → 768] qweight [256, 768] int32 = 768 KB
           scales [16, 768] fp16 = 24 KB
           (qzeros, g_idx: ignored / small)
up_proj:   same = 768 KB + 24 KB
down_proj: [768 → 2048] qweight [96, 2048] int32 = 768 KB
           scales [6, 2048] fp16 = 24 KB

Total per expert: ~2.4 MB  (INT4 packed)
```

Full expert weights: 128 experts × 48 layers × 2.4 MB ≈ **14.7 GB**
Non-expert (attention, norms, router, embeddings): ≈ **1.5 GB**

So on 24 GB VRAM everything fits. Heterogeneous mode is only needed when:
- GPU has less VRAM (16 GB, 12 GB, 8 GB)
- Model is larger (70B+ MoE)
- We want to reserve more VRAM for KV cache (long contexts)

## 4. Expert transfer cost

```
PCIe 4.0 x16: ~25 GB/s theoretical, ~20 GB/s practical (pinned memory)
                                    ~6 GB/s pageable

Per-expert transfer (INT4, ~2.4 MB):
  Pinned:   ~0.12 ms
  Pageable: ~0.40 ms

Per decode step (8 experts × 48 layers = 384 expert uses):
  If ALL cold: 384 × 0.12 ms = 46 ms transfer (with pinned)
  If 50% cold: 192 × 0.12 ms = 23 ms transfer
  If prefetched and overlapped with compute: potentially 0 ms overhead
```

Single-expert GEMM compute (INT4, K=2048, N=768, M=1) on a modern GPU is ~0.05-0.2 ms.
Transfer dominates unless overlapped — **prefetching is essential**.

## 5. ExpertPool design

### 5.1 Architecture

```
┌───────────────────────────────────────────────────────────────┐
│  ExpertPool                                                   │
├───────────────────────────────────────────────────────────────┤
│                                                               │
│   GPU slot arena  [slot 0] [slot 1] ... [slot K-1]            │
│        ↑              ↑                                        │
│        │              │                                        │
│        └── ensure_on_gpu(L, E)  ──── LRU metadata              │
│                                  (hits, last_access)           │
│                                                                │
│   CPU pinned storage                                           │
│        ExpertFFN ffn[L][E]  (canonical copy, always valid)    │
│                                                               │
│   Transfer management                                          │
│        cudaMemcpyAsync on runtime.transfer_stream()            │
│        cudaEvent_t per slot (compute waits on this)            │
│                                                               │
└───────────────────────────────────────────────────────────────┘
```

**Invariants**:
- Every expert has a valid CPU-pinned copy. GPU copies are cached.
- Each GPU slot holds at most one `(layer, expert_id)` at a time.
- When compute reads a GPU slot, it waits on that slot's event (set by the transfer stream).

### 5.2 Data structures (as implemented through M2)

Actual location: `include/frontend/models/expert_pool.hpp`. Slot scope is **per-layer**
(not the global arena the earlier draft showed). MoE decode visits layers strictly in
order, so a slot reused across layers would thrash; per-layer arenas avoid it.

```cpp
// include/frontend/models/expert_pool.hpp

struct ExpertGpuHandle {
    // Handles into a GPU slot. Valid for the window between ensure_on_gpu returning
    // and the next ensure_on_gpu call that evicts the same slot (same layer, different expert).
    tensor_t gate_packed, gate_scale, gate_g_idx;
    tensor_t up_packed,   up_scale,   up_g_idx;
    tensor_t down_packed, down_scale, down_g_idx;
    tensor_t gate_weight, up_weight, down_weight;  // dense path (unused for GPTQ)
};

enum class ExpertPoolStrategy { ALL_GPU, PINNED_LRU };

struct ExpertPoolConfig {
    ExpertPoolStrategy strategy = ExpertPoolStrategy::ALL_GPU;
    int num_gpu_slots = -1;   // per layer; ignored for ALL_GPU
};

class ExpertPool {
public:
    // ALL_GPU: experts arrive on GPU, pool is a thin handle layer.
    // PINNED_LRU: experts arrive on GPU, ctor D2H-migrates to CPU pinned + allocates
    // a per-layer GPU slot arena.
    ExpertPool(std::unique_ptr<ExpertWeights> experts, ExpertPoolConfig config);
    ~ExpertPool();

    // Return a GPU-ready handle. M2: hit = O(1); miss = LRU eviction + synchronous
    // cudaMemcpy H2D. M3 will swap the sync copy for a transfer-stream async copy
    // plus cudaStreamWaitEvent on the compute stream.
    ExpertGpuHandle ensure_on_gpu(int layer, int expert_id);

    // Prefetch hint. M1/M2: no-op. M3: async H2D on runtime.transfer_stream().
    void prefetch(int layer, int expert_id);

    // Metadata-only tensor access (shape/numel) that never triggers a transfer.
    const ExpertFFN& peek_expert(int layer, int expert_id) const;

    struct Stats { std::uint64_t hits = 0; std::uint64_t misses = 0; };
    Stats stats() const;
    int gpu_residents() const;
    ExpertPoolStrategy strategy() const;

private:
    struct Slot {
        int expert_id = -1;           // -1 when slot is empty
        std::uint64_t last_access = 0;
        ExpertGpuHandle gpu_handle;   // persistent GPU tensors, data overwritten on miss
    };

    std::unique_ptr<ExpertWeights> experts_;
    ExpertPoolConfig config_;

    // PINNED_LRU state; empty under ALL_GPU.
    std::vector<std::vector<Slot>> slots_;        // slots_[L] = num_gpu_slots slots for layer L
    std::vector<std::vector<int>>  residency_;    // residency_[L][E] = slot idx, or -1
    std::vector<std::uint64_t>     access_counter_; // one LRU counter per layer

    std::uint64_t hits_ = 0;
    std::uint64_t misses_ = 0;
};
```

The slot index is an opaque per-layer ordinal. Residency is a dense `residency_[L][E]`
vector (O(1) lookup) rather than a hashmap — `num_experts_per_layer` is small and
dense indexing is cheaper. Eviction is a linear scan over `slots_[L]` (N slots, small).

### 5.3 Slot size calculation

A GPU slot must hold the largest expert's tensors. For homogeneous models (our case),
size is fixed per model:

```
slot_bytes = sizeof(gate_qweight) + sizeof(gate_scales)
           + sizeof(up_qweight)   + sizeof(up_scales)
           + sizeof(down_qweight) + sizeof(down_scales)
```

For Qwen3-30B-A3B-GPTQ-Int4: slot_bytes ≈ 2.4 MB. Arena of K=64 slots ≈ 150 MB.

### 5.4 Integration with `ModelForwardConfig`

`ModelForwardConfig` currently holds `const ExpertWeights* experts`. Replace with
`ExpertPool*`:

```cpp
struct ModelForwardConfig {
    // ...
    ExpertPool* expert_pool = nullptr;

    void dispatch_expert_linear(tensor_t out, tensor_t in, int layer, int expert_id,
                                 ExpertProj proj) const {
        if (expert_pool) {
            auto handle = expert_pool->ensure_on_gpu(layer, expert_id);
            // ... use handle.gate_packed etc.
            return;
        }
        // Fallback removed (or kept for tests)
    }
};
```

Ownership: `Qwen3MoEModel` constructs an `ExpertPool` from its `ExpertWeights` at init time
instead of exposing the raw `ExpertWeights`.

## 6. Phase 2 milestones

### M1 — ExpertPool MVP (functional, no offloading yet) — ✅ DONE (commit 36e73cc)

`ExpertPool` introduced with ALL_GPU strategy. `Qwen3MoEModel` owns the pool;
`ModelForwardConfig::expert_pool` replaces the raw `experts` field; every
`dispatch_expert_linear` call goes through `ensure_on_gpu`. Output bit-exact vs. Phase 1.

### M2 — CPU pinned storage + on-demand synchronous transfer — ✅ DONE

- Constructor branches on `PINNED_LRU`: D2H-migrates every expert into CPU pinned tensors
  (`cudaMallocHost` via `Tensor::create(..., ZEDINFER_DEVICE_CPU, 0)`), then allocates a
  per-layer GPU slot arena shaped from the template expert (0, 0).
- Residency is a dense `residency_[L][E]` int vector (O(1) hit); eviction is a linear
  `argmin(last_access)` over `slots_[L]` (N small). Empty slots (last_access=0) are
  naturally preferred.
- Miss path: `cudaMemcpy` H2D synchronous for every non-null tensor in the expert's FFN.
  Pool-returned storage of the evicted expert is reused by the memory pool on next alloc.
- `peek_expert(layer, expert_id)` added for init-time shape/numel queries so
  `forward_config()` and `calculate_num_parameters()` don't thrash the arena.
- `ZEDINFER_MOE_GPU_SLOTS=N` env var: unset or `N >= num_experts` → ALL_GPU;
  `1 <= N < num_experts` → PINNED_LRU(N). Validated end-to-end on
  `Qwen3-30B-A3B-GPTQ-Int4` with `N=32`: output text matches ALL_GPU, decode visibly
  slower (expected).

Known limitations carried into M3:
- **Peak VRAM during construction**: loader still puts experts on GPU first, then ctor
  D2H-migrates. Works for INT4 30B on 24GB. BF16 60GB needs a streaming load path (out
  of scope for M2).
- **Synchronous compute-stream stall** on every miss (`cudaMemcpy` is fully synchronous).
  M3's purpose.

### M3 — Async prefetch (compute / transfer overlap) — ✅ DONE

- Runtime API extended with `zedinferEvent_t` + `create_event` / `destroy_event` /
  `record_event` / `stream_wait_event` (`backend/device/runtime_api.hpp`; NVIDIA impl
  uses `cudaEventCreateWithFlags(DisableTiming)` + `cudaStreamWaitEvent`).
- Each `Slot` owns a `ready_event`; the pool owns a single `compute_barrier_` event
  used to prevent the transfer stream from overwriting a slot whose in-flight compute
  hasn't passed.
- `ensure_on_gpu` miss path moves from `cudaMemcpy` (synchronous) to `cudaMemcpyAsync`
  on `Runtime::transfer_stream()`, records `slot.ready_event`, then has the compute
  stream `cudaStreamWaitEvent` on it before returning. Hit path also waits — so
  in-flight prefetches block compute only as long as the transfer genuinely needs.
- `prefetch(layer, expert_id)` became the real async version: claim an LRU slot,
  kick H2D on the transfer stream, do not wait on compute.
- `moe_decode` and `moe_prefill` call `prefetch` for all selected experts before the
  compute loop starts. `moe_prefill` uses a sliding window of `max_prefetch_depth()`
  (= per-layer slot count) outstanding transfers — initial burst plus one additional
  prefetch per finished compute iteration — so prefetches never evict each other
  before compute consumes them.
- To support the sliding window, `Slot` gained a `compute_touched` flag;
  `pick_lru_slot` skips populated-but-unconsumed slots when choosing a victim.

**Measurements — Qwen3-30B-A3B-GPTQ-Int4, single-card NVIDIA RTX A6000 (48GB),
`-p 128 -d 128 -r 3`, `--gpu-memory-utilization 0.5`.**

| Config | Prefill (tok/s) | Prefill ratio | Decode (tok/s) | Decode ratio |
|---|---|---|---|---|
| ALL_GPU | 122.04 | 1.00× | 26.25 | 1.00× |
| PINNED_LRU N=32 (M3) | 87.75 | **0.72×** (1.39× slowdown) | 31.10 | **1.19×** (0.84× latency) |

Success criterion from the plan ("decode ≤ 1.5× baseline, prefill similar order") is
met on both dimensions.

The **decode-faster-than-ALL_GPU** result is consistent, not noise (the same pattern
held on earlier noisier shared-GPU runs too). Two plausible causes:

1. **L2 / TLB locality.** ALL_GPU keeps 14.7 GB of expert weights scattered across
   VRAM; GEMMs for the 8 routed experts per step read from 14.7 GB of address space.
   PINNED_LRU keeps only ~3.7 GB of slot arena on GPU (32 × 48 × ~2.4 MB), with
   slot tensors allocated in one batched call at ctor time — denser, warmer caches.

2. **GPU clock-state warmup ordering.** The PINNED_LRU run always follows the ALL_GPU
   run in the same process chain, and the GPU is already at high clock state by then.
   Re-running in swapped order would pin down how much of the 18% is ordering vs.
   locality. Pending.

Either way, M3's qualitative claim is established: async prefetch + sliding window
makes decode effectively free of PCIe tax, and cuts the prefill regression from the
2× that M2 showed to 1.39×.

Remaining prefill gap is dominated by the per-row `memcpy_sync` gather inside
`moe_prefill` (existing PERF-TODO, independent of expert offloading).

### M4 (optional, stretch) — Predictive prefetch / speculative prefetch

Start prefetching layer L+1's likely experts during layer L's MLP compute. Use last token's
routing as a prior. Only worth doing if M3 profiling shows compute is under-utilized during
expert FFN.

## 7. Config exposure

Current surface — one env var controls everything:

```
ZEDINFER_MOE_GPU_SLOTS = N   N per layer.
                             N unset OR N >= num_experts → ALL_GPU, experts GPU-resident.
                             N < num_experts             → PINNED_LRU with N slots/layer.
                                                           Experts are loaded straight to
                                                           CPU pinned memory via the loader
                                                           predicate (see §10); the GPU
                                                           slot arena is populated on demand.
```

The same knob couples the loader's routing predicate and the pool strategy, so users
don't need to set them independently. Auto-sizing N based on VRAM budget is listed as
future polish in `moe_session_handoff.md §5`.

## 8. Risks and mitigations

| Risk | Mitigation |
|------|-----------|
| PCIe bandwidth saturation when many cold experts are hit together | Start with fewer tokens per batch during prefill; rely on GPU-resident hot experts |
| Prefetch accuracy low → transfers wasted | Start with non-speculative prefetch (M3): only prefetch experts actually selected by router |
| Pinned memory pressure limits other host work | Make pinning optional (fallback to pageable); keep total pinned to expert weights only |
| CPU→GPU copy serializing on transfer stream → bottleneck | Use cudaMemcpyAsync with pinned memory (only path that actually runs in parallel) |
| LRU thrashing when routing is uniform | Measure expert access distribution; if too uniform, fall back to ALL_GPU for that model |
| Prefill path has many unique experts per step | M2 was 2× slowdown; M3 sliding-window prefetch closed the gap to 1.36× at N=32. Further improvements blocked on the per-row gather PERF-TODO in `moe_prefill`. |

## 9. Out of scope for Phase 2

- **Fiddler-style CPU execution** of cold experts (compute on CPU instead of transfer).
  Revisit after M3 if transfer dominates latency even with overlap.
- **Disk-backed storage** for extreme memory pressure.
- **Learned prefetch predictor** (ProMoE-style MLP). Worth exploring in a follow-up once
  non-speculative prefetch (M3) establishes a baseline.
- **Scheduler integration** (expert-aware batching). Only useful if we serve multiple
  requests concurrently — single-user decode doesn't need it.

## 10. Large-model support — BF16 30B on 24GB VRAM — ✅ DONE

M1-M3 establish PINNED_LRU as a working path. But M2's construction still assumes the
whole expert set arrived on GPU first (loader default), with a D2H migration step
happening inside the pool's ctor. For models that genuinely don't fit — BF16
Qwen3-30B-A3B is ~60GB on a 24GB card — that initial GPU load OOMs before D2H can run.

### 10.1 Loader routing predicate

`Model::load_weights` gained an optional `std::function<bool(const std::string&)>
to_cpu_pinned` parameter (default always-false → pre-change behavior). When the
predicate matches a tensor name, the loader allocates its destination via
`Tensor::create(shape, dtype, ZEDINFER_DEVICE_CPU, 0)` instead of `target_device`.
On the NVIDIA runtime, CPU tensors go through `cudaMallocHost` and are thus pinned
— eligible for direct `cudaMemcpyAsync` onto the transfer stream with full PCIe
bandwidth.

`Model::parse` builds the predicate from `ZEDINFER_MOE_GPU_SLOTS`: when
`N < num_experts` on a `qwen3_moe` model, every tensor whose mapped name contains
`.mlp.experts.` routes to CPU pinned. Non-expert tensors (attention, router, embeds,
norms, etc.) still go to the target GPU. Loader log:

```
[Model] ZEDINFER_MOE_GPU_SLOTS=16 < num_experts=128: routing MoE experts to CPU pinned memory
[Loader] Routed 55296 tensors to CPU pinned memory via predicate
```

### 10.2 ExpertPool auto-detection

`ExpertPool::ExpertPool` probes `experts_->at(0, 0)`'s device to pick its
construction path:

- `experts_on_gpu = false` → experts arrived already pinned on host. Skip the D2H
  migration loop entirely; allocate the GPU slot arena against the current runtime.
  Construction time collapses from ~25 s to <1 s.
- `experts_on_gpu = true` → legacy flow for INT4 small models where everything fits
  on GPU. D2H-migrate each expert in place, same as M2.

Both branches converge to the same post-condition: `experts_` holds CPU-pinned
tensors; per-layer slot arena is ready on GPU.

### 10.3 Validation

Machine: NVIDIA RTX A6000 (48 GB physical). Weight + KV cache budget limited to
**24 GB** via `--gpu-memory-utilization 0.5` — the engine computes
`allowed_bytes = total × utilization`, and `init_block_pool` sizes the KV cache as
`allowed - used_after_weights_loaded`. This closely matches a 4090-24GB scenario
for the pieces that the utilization knob controls (weights + KV cache). What it
does *not* reproduce: forward-pass activation headroom — on A6000 activations draw
from the remaining 24 GB outside the budget, whereas on a real 24 GB card they
have to fit within the same 24 GB. For ping-sized workloads (12-token prompt)
activations are under a few hundred MB so the distinction is cosmetic here; for
long prompts or wide batches on a true 24 GB card, budget the KV cache a little
tighter to leave headroom.

Command:
```
ZEDINFER_MOE_GPU_SLOTS=16 \
./build/.../ping ~/data/models/Qwen3-30B-A3B --nvidia --gpu-memory-utilization 0.5
```

Observations from the log:
- `Device memory: free=37836 MB, total=48541 MB` — after model load, ~10.7 GB GPU
  used (3 GB non-experts + 7.2 GB slot arena of 16 slots × 48 layers × 9.4 MB).
- `Creating KV page pool: 434097 pages × 16 tokens, 13565 MB` — leftover 13.3 GB
  went to KV cache (matches 24 GB allowed − 10.7 GB used).
- Coherent output on "Who are you?".

Runtime throughput (12-token ping prompt, 153 generated tokens):

| | Prefill tok/s | Decode tok/s |
|---|---|---|
| INT4 30B, N=32 (D.3) | 17.1 | 21.5 |
| BF16 30B, N=16 (D.5) | 5.6 | 5.3 |

The ~3-4× slowdown vs INT4 matches the 4× increase in per-expert bytes (9.4 MB
BF16 vs 2.4 MB INT4) — PCIe transfer dominates for BF16 decode. Functional result
is the main deliverable here; perf work would go after the prefill per-row gather
PERF-TODO and any further M3-family prefetch heuristics.

**What this proves**: a 60 GB MoE model runs on a single 24 GB GPU without
Python-side orchestration, with first-class async prefetch + sliding-window
overlap, using one env var as the user-visible knob. First exercise of the dense
(non-quantized) expert path through `ExpertPool` and `dispatch_expert_linear` —
passed without code changes.

## 11. File plan

| File | Status | Purpose |
|------|--------|---------|
| `include/backend/moe/expert_pool.hpp` | new (M1) | `ExpertPool` class |
| `src/backend/moe/expert_pool.cpp` | new (M1) | Pool implementation, LRU |
| `include/frontend/models/expert_weights.hpp` | exists | Phase 1 data structure; still used internally by pool |
| `include/frontend/models/forward_config.hpp` | modify (M1) | Replace `experts` field with `expert_pool` |
| `include/frontend/models/qwen3_moe.hpp` | modify (M1) | Owns `ExpertPool` instead of `ExpertWeights` |
| `src/frontend/models/moe_forward.cpp` | modify (M3) | Call `pool.prefetch(...)` after routing |
| `xmake/backend.lua` | modify (M1) | Add `moe` target or fold into existing |
