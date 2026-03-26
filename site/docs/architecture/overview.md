# Architecture Overview

ZedInfer is a C++17 LLM inference engine. All serving-path code is native C++ -- no Python runtime dependency.

---

## System Topology

```
HTTP Clients / CLI
       |
  HttpServer (thread pool)
       |
  ServingLoop (engine thread)
    |-- Scheduler (batch assembly, admission, prefix cache)
    |-- Forward Loop (shared transformer forward)
    |-- Paged Attention (KV scatter + attention dispatch)
       |
  Operators
    linear    : cuBLAS/cuBLASLt (GPU), oneDNN (CPU)
    attention : custom paged kernels (decode/prefill/batched)
    rms_norm, rope, add, swiglu, embedding, argmax
       |
  Paged KV Cache
    BlockPool (static VRAM allocation)
    BlockAllocator (per-request block management)
    PrefixCache (cross-request block sharing)
       |
  Runtime: Memory Pool, Device API, Tensor
```

---

## Request Lifecycle

1. **Submit** -- HTTP request arrives, tokenized into input IDs, wrapped as an `InferenceRequest`
2. **Schedule** -- Scheduler checks block availability, performs prefix matching, admits request into the active batch
3. **Prefill** -- All input tokens processed in one forward pass (or chunked across multiple passes for long prompts), KV cache populated
4. **Decode** -- Tokens generated one at a time. Each step: forward pass with single new token, sample next token, append to KV cache
5. **Complete** -- Stop token or max length reached. Response returned (or streamed token-by-token via SSE)

---

## Key Components

### Inference Engine

Resource container and factory. Owns model weights, tokenizer, sampler, block pool, prefix cache, and pre-allocated decode buffers. Provides `serving_loop()` and `profiler()` accessors to callers.

### Serving Loop

Drives the inference pipeline on a dedicated thread. Accepts requests via `submit_async()` (HTTP) or `generate()` (CLI), then continuously runs the core loop:

```
schedule() → step() → process_results()
```

Each iteration assembles a batch from the scheduler, executes one forward pass, and dispatches results (completed tokens or finish signals) back to waiting callers.

### Scheduler

Implements decode-first continuous batching:

- **Decode priority** -- Active decode requests are always included first, ensuring low per-token latency
- **Chunked prefill** -- Long prompts are split into chunks and interleaved with decode steps, preventing head-of-line blocking
- **Block-based admission** -- New requests are admitted only if the block allocator has enough free blocks for the estimated KV cache requirement
- **Prefix cache integration** -- On admission, the scheduler checks if any prefix of the new prompt's token IDs matches cached blocks. Matched blocks are shared (ref count incremented), and only the unmatched suffix is scheduled for prefill

### Forward Loop

A single parameterized function handles all supported model architectures. Instead of per-model forward implementations, the loop is configured via `ModelForwardConfig` flags:

- Bias presence in attention/MLP projections
- Q/K normalization
- Tied embeddings

For single-token decode, pre-allocated `DecodeScratch` buffers provide fixed-address GPU memory, eliminating ~500 tensor allocation round-trips per step.

### Paged KV Cache

The KV cache is managed as a pool of fixed-size blocks (similar to virtual memory pages):

- **BlockPool** -- Statically allocated at engine startup. Provides O(1) allocation, deallocation, and statistics via incremental counters. Supports reference counting for block sharing and LRU eviction for memory reclamation.
- **BlockAllocator** -- Per-request block management. Extends block tables as sequences grow, one block at a time.
- **PrefixCache** -- Chain-hashed content addressing. Each block is identified by a hash of its token content plus the hash of the preceding block. When a new request's prefix matches existing blocks, they are shared instead of recomputed.

### Paged Attention

Three dispatch paths, selected automatically based on batch composition:

- **Decode single** -- Single active request, single new token. Optimized kernel with shared memory address precomputation.
- **Decode batched** -- Multiple active requests, each generating one token. Grid dimensions = (num_requests, num_heads).
- **Prefill** -- One or more requests with multiple input tokens. Causal masking applied.

All paths operate on non-contiguous KV blocks via block tables (indirection arrays mapping logical positions to physical block IDs).

---

## Operator Backends

| Operator | CPU | GPU |
|----------|-----|-----|
| linear | oneDNN GEMM (BF16/FP32, runtime ISA dispatch) | cuBLAS/cuBLASLt (auto-tuned, fused bias) |
| attention (decode) | Paged GQA, OpenMP parallel | Custom paged kernel, online softmax |
| attention (prefill) | Paged GQA, causal mask | Custom paged kernel |
| attention (batched) | Loop over single decode | Grid=(num_reqs, nhead) |
| rms_norm | AVX vectorized | Warp + block reduction |
| rope | Scalar + OpenMP | Per-token CUDA kernel |
| add, swiglu | AVX vectorized | Vectorized CUDA |
| embedding | OpenMP memcpy | Vectorized lookup |
| argmax | OpenMP reduction | Block-scope reduction, pinned host buffer |

---

## Design Principles

- **No graph execution** -- Direct forward calls, no intermediate representation, no compilation step
- **No virtual dispatch on hot paths** -- Concrete types used directly (e.g., `PagedForwardContext` instead of abstract `ForwardContext`)
- **No Python in serving** -- Python is only used for offline tools (operator testing, codegen). The entire serving path is native C++.
- **Minimize per-step allocation** -- Pre-allocated buffers (`DecodeScratch`, pinned argmax buffer) eliminate memory allocation from the decode loop
- **Unified code paths** -- One forward function for all models, one block table path for sessions and batch mode, one attention dispatch interface
