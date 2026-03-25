# ZedInfer Roadmap

> Implementation status and future plans. Reference for continuing development.

---

## Completed Features

### Core Infrastructure
| Feature | PR/Commit | Key Files |
|---------|-----------|-----------|
| Direct model forward (no graph) | PR-2 | `transformer_forward.cpp` |
| cuBLAS for GPU linear | PR-3a | `linear_cublas.cu` |
| oneDNN for CPU linear | PR-3b | `linear_cpu.cpp` |
| Stateless engine + request types | PR-4 | `engine.hpp`, `request.hpp` |
| Configurable chat template | PR-5 | `chat_template.hpp/.cpp` |
| Scheduler (single-request) | PR-6 | `scheduler.hpp/.cpp` |
| Paged KV cache + block pool | PR-7 | `block_pool.hpp/.cpp` |
| Paged attention kernels | PR-8 | `paged_attention_nvidia.cu` |
| Continuous batching | PR-9 | `scheduler.cpp`, `batch_context.cpp` |
| HTTP API + Web UI + SSE | PR-10 | `http_server.cpp`, `web/index.html` |

### Refactoring (R1-R6)
| Refactor | Description |
|----------|-------------|
| R1: Unified forward | 4 forward files -> 1 `transformer_forward.cpp` |
| R2: Unified KV management | Session + batch mode -> single block table path |
| R3: Clean KVCache interface | Removed KVCache base class, DynamicKVCache standalone |
| R4: Unified attention dispatch | 4 attention functions -> 1 `ops::attention(AttentionParams)` |
| R5: Split engine | Engine -> Engine + ServingLoop + Profiler |
| R6: Dead code removal | Graph execution code deleted |

### Recent Optimizations
| Optimization | Impact | Key Change |
|-------------|--------|------------|
| ArgmaxSampler pinned buffer | Eliminated 570us/call cudaMallocHost | Pre-allocated host buffer |
| Paged decode kernel smem optimization | ~16% decode improvement | Precomputed physical addresses in shared memory |
| Prefix caching | Skip redundant prefill for shared prefixes | Block ref counting + chain hash |
| DecodeScratch pre-allocation | Eliminates ~500 Tensor::create/step | Fixed-address decode buffers |
| BlockPool O(1) counters | Fast admission control | Incremental free/evictable counters |

### Code Health
| Cleanup | Description |
|---------|-------------|
| Removed cuDNN integration | Only used in warmup, not serving |
| Removed ContiguousForwardContext | Warmup now uses paged path |
| Removed DynamicKVCache + KVCache base | No longer needed |
| ForwardContext de-virtualization | PagedForwardContext used directly |
| Request ownership simplification | `borrow_block_table()` / `own_block_table()` API |
| Scheduler decomposition | Extracted `allocate_blocks_for_request()` |
| Engine slimming | Delegation methods -> `serving_loop()` / `profiler()` accessors |
| attend() decomposition | Split into decode_single/decode_batched/prefill |
| ensure_blocks extraction | Shared by scheduler and profiler |
| CPU memcpy kind fix | Accept all memcpy kinds on CPU |

---

## Upcoming Work

### Priority 1: FlashInfer Integration

**Goal:** Replace custom paged attention kernels with FlashInfer's optimized kernels. Expected 3-5x decode speedup, 2-4x prefill speedup.

**Prerequisite:** K/V block unification (currently separate `k_blocks`/`v_blocks` per layer; FlashInfer needs shared page indices).

**Approach:** AOT (Ahead-Of-Time) kernel generation — run Python codegen offline, compile generated .cu files as regular CUDA code. No Python at runtime.

**Plan:** `docs/plan/flashinfer_integration.md`

**Key changes:**
1. `SequenceBlockTable`: k_blocks + v_blocks -> unified pages
2. `AttentionParams`: k_block_table + v_block_table -> unified page_table
3. `BlockPool`: single pool -> separate K/V pools with shared page IDs
4. All consumers: scheduler, session, paged_forward_context, prefix_cache, batch_context

### Priority 2: CUDA Graph (Phase 2)

**Goal:** Capture decode forward pass as CUDA graph, replay with single launch. Saves ~2ms/step kernel launch overhead.

**Status:** Phase 1 (DecodeScratch) complete. Phase 2 (capture/replay) not started.

**Consideration:** Not compatible with MoE/heterogeneous inference. Keep implementation simple, don't over-invest.

**Plan:** `docs/plan/cuda_graph.md`

### Priority 3: INT8 Quantization

**Goal:** 2x model memory reduction, enabling larger batch sizes or bigger models on same hardware.

**Approach:**
- Per-channel symmetric quantization for linear weights
- `QuantizationConfig` in model config
- `QuantizedLinearWeight` storage (int8 weight + fp16 scale)
- Dispatch: `ops::linear_quantized()` for CPU (oneDNN INT8) and GPU (cuBLAS INT8)

**Plan:** `docs/plan/quantization.md`

### Priority 4: INT4 Quantization (GPTQ/AWQ)

**Goal:** 4x model memory reduction. Run 30B+ models on 24GB GPUs.

**Depends on:** INT8 infrastructure (same quantized weight path, extended to 4-bit).

### Priority 5: Heterogeneous CPU/GPU Inference

**Goal:** Mixed device execution — part of model on GPU, part on CPU with pinned memory transfers.

**Key design:**
- Per-layer device placement
- Pinned host memory for CPU-resident weights
- Async prefetch with compute overlap

**Plan:** `docs/plan/heterogeneous_moe.md`

### Priority 6: MoE Expert Offloading

**Goal:** Run large MoE models (Qwen-30B-A3B) on 24GB GPUs by dynamically loading experts.

**Depends on:** Quantization (INT4 to fit), heterogeneous inference (CPU/GPU transfer), paged KV cache (memory efficiency).

---

## Known Technical Debt

| Issue | Severity | Blocked by |
|-------|----------|-----------|
| K/V block separation | High | Blocks FlashInfer integration |
| Paged prefill kernel naive (no IO tiling) | High | Needs FlashInfer |
| Paged decode kernel 5x slower than contiguous | High | Needs FlashInfer |
| No operator correctness tests | Medium | Tested via Python bindings externally |
| No scheduler/serving integration tests | Medium | — |
| build_decode_cache per-context rebuild | Low | FlashInfer will replace |

---

## Performance Reference

**Machine:** B200 180GB VRAM
**Model:** DeepSeek-R1-Distill-Qwen-1.5B (BF16)

| Config | Metric | Value |
|--------|--------|-------|
| Single decode (paged, p=128 d=128) | Decode throughput | ~129 tok/s |
| Batch=4 | Total throughput | ~672 tok/s |
| Batch=32 | Total throughput | ~1,820 tok/s |
| Batch=128 | Total throughput | ~2,170 tok/s |

**Bottleneck:** Paged attention decode kernel (64% of GPU time, avg 129us/call). FlashInfer integration is the primary optimization target.
