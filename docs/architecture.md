# ZedInfer Architecture

> Single source of truth for the current system design. Updated as the codebase evolves.

---

## Overview

ZedInfer is a C++17 LLM inference engine supporting CPU (x86 AVX-512) and NVIDIA GPU (CUDA) backends. No Python in the serving path.

**Supported models:** Qwen2 (DeepSeek-R1-Distill-Qwen-1.5B), Qwen3 (DeepSeek-R1-0528-Qwen3-8B)
**Data types:** BF16, FP16, FP32
**Interfaces:** CLI (bench/chat/ping), HTTP API with Web UI

---

## System Topology

```
HTTP Clients / CLI
       |
  HttpServer (cpp-httplib thread pool)
       |  submit_async()
       v
  ServingLoop (engine thread)
    |-- Scheduler (batch assembly, admission, prefix cache)
    |-- transformer_forward() (shared forward loop)
    |-- PagedForwardContext (KV scatter, paged attention dispatch)
       |
  Operators (ops::*)
    linear    : cuBLAS/cuBLASLt (GPU), oneDNN (CPU)
    attention : custom paged kernels (decode/prefill/batched)
    rms_norm, rope, add, swiglu, embedding, argmax
       |
  PagedKVCache
    BlockPool (static VRAM allocation)
    BlockAllocator (per-request block management)
    PrefixCache (cross-request block sharing)
       |
  Runtime: BestFitMemoryPool, Device API, Tensor
```

---

## Key Components

### InferenceEngine (`include/zedinfer/engine.hpp`)
Resource container and factory. Owns model, tokenizer, sampler, block pool, prefix cache, decode scratch. Exposes `serving_loop()` and `profiler()` for callers.

### ServingLoop (`include/zedinfer/serving_loop.hpp`)
Drives inference: `submit_async()` for HTTP, `generate()` for CLI sessions. Contains the Scheduler and runs `schedule() -> step() -> process_results()` loop.

### Scheduler (`include/zedinfer/scheduler.hpp`)
- Decode-first scheduling with chunked prefill
- Block-based admission control via `BlockAllocator::available_blocks()`
- Prefix cache integration: matches prompt prefix on admission, skips redundant prefill
- `allocate_blocks_for_request()` handles prefix match + allocation + extension

### PagedForwardContext (`include/frontend/models/paged_forward_context.hpp`)
Concrete execution context for `transformer_forward()`. No virtual dispatch (ForwardContext base class removed). Handles:
- KV scatter to blocks (`write_kv`)
- Paged attention dispatch: `attend_decode_single()`, `attend_decode_batched()`, `attend_prefill()`
- GPU block table caching across layers

### transformer_forward (`src/frontend/models/transformer_forward.cpp`)
Single shared forward loop parameterized by `ModelForwardConfig` (bias, Q/K norm flags). Takes `PagedForwardContext&` directly. When `DecodeScratch*` is provided and N=1, uses pre-allocated buffers (zero Tensor::create per decode step).

### DecodeScratch (`include/frontend/models/decode_scratch.hpp`)
Pre-allocated GPU buffers for single-token decode (~400KB). Eliminates ~500 `Tensor::create`/`BestFitMemoryPool` round-trips per decode step.

### BlockPool (`include/backend/kvcache/block_pool.hpp`)
Fixed-size KV cache block pool. Features:
- O(1) stats via `free_count_` / `evictable_count_` counters
- Reference counting: `share()` / `release()` for prefix caching
- LRU eviction for memory pressure
- Content hash + immutable flags for cached prefix blocks

### PrefixCache (`include/backend/kvcache/prefix_cache.hpp`)
Chain-hashed prefix matching. When multiple requests share the same token prefix, KV blocks are shared (ref_count > 1) instead of recomputed.

### InferenceRequest (`include/zedinfer/request.hpp`)
Unified ownership model: `borrow_block_table()` (session mode) or `own_block_table()` (batch mode). Single `block_table()` accessor, no dual-path branching.

### Profiler (`include/zedinfer/profiler.hpp`)
Warmup and benchmarking using the paged path (same kernels as actual serving). Uses DecodeScratch for decode measurement.

---

## Operator Backends

| Operator | CPU | GPU |
|----------|-----|-----|
| linear | oneDNN GEMM (BF16/FP32 native) | cuBLAS/cuBLASLt (auto-tuned, fused bias) |
| attention (decode) | Paged GQA, OMP parallel | Custom paged kernel, online softmax, smem address precompute |
| attention (prefill) | Paged GQA, causal mask | Custom paged kernel (naive, no IO-aware tiling) |
| attention (batched decode) | Loop over single decode | Grid=(num_reqs, nhead), per-request block table |
| rms_norm | AVX-512 vectorized | Vectorized 128-bit packed, warp+block reduction |
| rope | CPU scalar + OMP | Per-token CUDA kernel |
| add, swiglu | AVX-512 vectorized | Vectorized 128-bit packed |
| embedding | OMP memcpy | Vectorized lookup |
| argmax | OMP reduction | Block-scope reduction, pre-allocated pinned host buffer |

---

## Build System

- **Tool:** XMake (`xmake.lua`)
- **C++ Standard:** C++17
- **GPU:** Optional (`--nv-gpu=y`), links cuBLAS/cuBLASLt
- **CPU GEMM:** Optional oneDNN (`--onednn=y`)
- **Targets:** Static libraries (zedinfer, frontend, backend, ops, etc.) + example binaries (bench, chat, ping, serve, batch_bench)

---

## Test Suite

9 test binaries, ~130 test cases:
- `test-memorypool`, `test-storage` — core infrastructure
- `test-tensor` — tensor ops (shape, slice, permute, device transfer)
- `test-tokenizer`, `test-loader` — frontend I/O (env var `ZEDINFER_TEST_MODEL_PATH`)
- `test-blockpool`, `test-prefixcache` — KV cache + prefix caching
- `test-sampler` — argmax + general sampler
- `test-chattemplate` — chat template formatting
