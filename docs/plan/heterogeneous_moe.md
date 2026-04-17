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

### 5.2 Data structures

```cpp
// include/backend/moe/expert_pool.hpp

struct ExpertGpuHandle {
    // Pointers into a GPU slot. Valid only after ensure_on_gpu / prefetch completes.
    tensor_t gate_packed, gate_scale, gate_g_idx;
    tensor_t up_packed,   up_scale,   up_g_idx;
    tensor_t down_packed, down_scale, down_g_idx;
    tensor_t gate_weight, up_weight, down_weight;  // dense path (unused for GPTQ models)
};

struct ExpertPoolConfig {
    int num_gpu_slots = 0;     // 0 = auto (fill available VRAM after non-expert weights + KV cache)
    enum Strategy { ALL_GPU, LRU, PINNED_LRU };
    Strategy strategy = Strategy::LRU;
};

class ExpertPool {
public:
    // Takes ownership of the ExpertWeights (CPU-resident, from Qwen3MoEModel).
    // Pins CPU memory regions and allocates GPU slot arena.
    ExpertPool(std::unique_ptr<ExpertWeights> cpu_experts, ExpertPoolConfig cfg,
               zedinferDeviceType_t gpu_device, int gpu_device_id);

    ~ExpertPool();  // unpins CPU memory, frees GPU slots

    // Bring expert (layer, expert_id) to GPU (blocking on transfer if necessary).
    // Returns handles into the assigned GPU slot.
    ExpertGpuHandle ensure_on_gpu(int layer, int expert_id);

    // Start an H2D transfer for this expert on the transfer stream without blocking
    // the compute stream. Safe to call for experts already resident (no-op + marks LRU).
    void prefetch(int layer, int expert_id);

    // Stats for diagnostics / scheduler integration.
    int gpu_residents() const;
    int transfers_this_step() const;
    void begin_step();  // resets per-step counters

private:
    struct Slot {
        int layer = -1;
        int expert_id = -1;
        bool valid = false;
        uint64_t last_access = 0;
        // GPU storage (pre-allocated at fixed size per slot).
        ExpertGpuHandle gpu;
        // Event marking when transfer into this slot completed.
        cudaEvent_t ready_event = nullptr;
    };

    std::unique_ptr<ExpertWeights> cpu_experts_;    // pinned host copies
    std::vector<Slot> gpu_slots_;
    // Map from (layer, expert) to slot index. -1 = not resident.
    std::unordered_map<uint64_t, int> residency_;   // key = layer * num_experts + expert_id
    uint64_t access_counter_ = 0;
    ExpertPoolConfig cfg_;
};
```

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

### M1 — ExpertPool MVP (functional, no offloading yet)

Goal: introduce the abstraction without changing behavior. Every expert has a permanently
assigned GPU slot. `ensure_on_gpu` is a simple map lookup. No CPU copy, no transfers.

Tasks:
- Add `ExpertPool` class (skeleton: all experts have GPU slots)
- `Qwen3MoEModel` owns `ExpertPool` instead of `ExpertWeights`
- `ModelForwardConfig::experts` replaced by `expert_pool`
- `dispatch_expert_linear` goes through the pool
- Verify perplexity / decode output unchanged against Phase 1 baseline

Success criteria: same output as Phase 1, same perf (±5%). No VRAM reduction yet.

### M2 — CPU pinned storage + on-demand synchronous transfer

Goal: reduce VRAM by moving some experts to CPU pinned memory. Transfers are synchronous
(compute stream blocks on cudaMemcpy). Not fast, but proves the data path.

Tasks:
- Load all expert weights into CPU pinned memory (call `register_pinned` after mmap,
  or allocate+copy into `cudaMallocHost`'d region)
- GPU arena with `num_gpu_slots` < total experts
- LRU eviction when arena is full
- `ensure_on_gpu`: synchronous `cudaMemcpy` from CPU slot to GPU slot
- Add env var / config knob: `ZEDINFER_MOE_GPU_SLOTS=N`

Success criteria: runs Qwen3-30B-A3B with only (say) 32 GPU slots per layer
× 48 layers = 1536 slots total (vs 128 × 48 = 6144 permanently resident). Output
matches Phase 1. Decode latency degrades — that's expected, M3 fixes it.

### M3 — Async prefetch (compute / transfer overlap)

Goal: eliminate the transfer stall by prefetching cold experts on the transfer stream
while the current layer's attention runs.

Tasks:
- Per-slot `cudaEvent_t`; transfer stream records event when H2D completes
- `ensure_on_gpu` returns a handle immediately; compute stream does `cudaStreamWaitEvent`
  before using the slot
- Prefetch strategy (start simple):
  1. At end of layer L's routing, for each selected expert not yet on GPU, issue
     `prefetch(layer=L, expert_id)` on transfer stream
  2. Experts selected but already resident: just update LRU timestamp, no transfer
- Benchmark: compare M2 vs M3 decode latency

Success criteria: decode latency within 1.5× of Phase 1 (all-GPU) for 32-slot config.
Prefill latency similar order.

### M4 (optional, stretch) — Predictive prefetch / speculative prefetch

Start prefetching layer L+1's likely experts during layer L's MLP compute. Use last token's
routing as a prior. Only worth doing if M3 profiling shows compute is under-utilized during
expert FFN.

## 7. Config exposure

User-visible knobs (env vars initially, CLI flags later):

```
ZEDINFER_MOE_GPU_SLOTS          N per layer, or -1 for all (=ALL_GPU)
ZEDINFER_MOE_STRATEGY           "all_gpu" | "lru" | "pinned_lru" (default)
ZEDINFER_MOE_PREFETCH           0 = synchronous transfer only
                                1 = prefetch on current-layer routing
```

Default behavior: if total expert VRAM fits after non-expert allocation, use ALL_GPU.
Otherwise, use PINNED_LRU with slots sized to fit remaining VRAM.

## 8. Risks and mitigations

| Risk | Mitigation |
|------|-----------|
| PCIe bandwidth saturation when many cold experts are hit together | Start with fewer tokens per batch during prefill; rely on GPU-resident hot experts |
| Prefetch accuracy low → transfers wasted | Start with non-speculative prefetch (M3): only prefetch experts actually selected by router |
| Pinned memory pressure limits other host work | Make pinning optional (fallback to pageable); keep total pinned to expert weights only |
| CPU→GPU copy serializing on transfer stream → bottleneck | Use cudaMemcpyAsync with pinned memory (only path that actually runs in parallel) |
| LRU thrashing when routing is uniform | Measure expert access distribution; if too uniform, fall back to ALL_GPU for that model |
| Prefill path has many unique experts per step | M2/M3 will be slow for prefill; consider a separate "prefill all-on-GPU" mode |

## 9. Out of scope for Phase 2

- **Fiddler-style CPU execution** of cold experts (compute on CPU instead of transfer).
  Revisit after M3 if transfer dominates latency even with overlap.
- **Disk-backed storage** for extreme memory pressure.
- **Learned prefetch predictor** (ProMoE-style MLP). Worth exploring in a follow-up once
  non-speculative prefetch (M3) establishes a baseline.
- **Scheduler integration** (expert-aware batching). Only useful if we serve multiple
  requests concurrently — single-user decode doesn't need it.

## 10. File plan

| File | Status | Purpose |
|------|--------|---------|
| `include/backend/moe/expert_pool.hpp` | new (M1) | `ExpertPool` class |
| `src/backend/moe/expert_pool.cpp` | new (M1) | Pool implementation, LRU |
| `include/frontend/models/expert_weights.hpp` | exists | Phase 1 data structure; still used internally by pool |
| `include/frontend/models/forward_config.hpp` | modify (M1) | Replace `experts` field with `expert_pool` |
| `include/frontend/models/qwen3_moe.hpp` | modify (M1) | Owns `ExpertPool` instead of `ExpertWeights` |
| `src/frontend/models/moe_forward.cpp` | modify (M3) | Call `pool.prefetch(...)` after routing |
| `xmake/backend.lua` | modify (M1) | Add `moe` target or fold into existing |
