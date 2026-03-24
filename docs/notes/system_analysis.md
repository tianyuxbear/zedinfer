# ZedInfer System Analysis: Single-GPU Multi-User Serving

> Stage checkpoint after PR-9 (Continuous Batching).
> Focus: single-machine, single-GPU, multi-user inference.
> Scope: excludes quantization (INT8/INT4) and heterogeneous CPU/GPU inference.

---

## 1. Current Architecture Overview

```
HTTP Clients (future PR-10)
        |
        v
  ┌─────────────────────────────────────────────────┐
  │  InferenceEngine                                 │
  │  ┌───────────┐  ┌──────────┐  ┌──────────────┐  │
  │  │ Scheduler  │  │  Model   │  │  BlockPool   │  │
  │  │            │  │ (Qwen2/3)│  │  (VRAM pool) │  │
  │  │ schedule() │  │          │  │              │  │
  │  │ process_   │  │ forward()│  │ BlockAlloc   │  │
  │  │ results()  │  │ forward_ │  │              │  │
  │  │            │  │ batch()  │  │ Per-request   │  │
  │  │ waiting_q  │  │          │  │ block tables │  │
  │  │ active_reqs│  │          │  │              │  │
  │  └───────────┘  └──────────┘  └──────────────┘  │
  │                                                   │
  │  ┌───────────┐  ┌──────────┐  ┌──────────────┐  │
  │  │ Tokenizer │  │ Sampler  │  │ ChatTemplate │  │
  │  │ (HF BPE)  │  │ (Argmax) │  │ (Configurable│  │
  │  └───────────┘  └──────────┘  └──────────────┘  │
  └─────────────────────────────────────────────────┘
                        |
                        v
  ┌─────────────────────────────────────────────────┐
  │  Backend                                         │
  │  ┌──────────────────────────────────────────┐   │
  │  │ Operators (ops::*)                        │   │
  │  │  linear    : cuBLAS (GPU) / oneDNN (CPU) │   │
  │  │  attention : paged decode/prefill (GPU)  │   │
  │  │             cuDNN FlashAttn prefill (GPU) │   │
  │  │  rms_norm  : custom CUDA / OMP CPU       │   │
  │  │  rope      : custom CUDA / CPU           │   │
  │  │  embedding, add, swiglu, argmax          │   │
  │  └──────────────────────────────────────────┘   │
  │  ┌──────────────────────────────────────────┐   │
  │  │ KV Cache                                  │   │
  │  │  PagedKVCache : block pool + block table │   │
  │  │  DynamicKVCache : contiguous (fallback)  │   │
  │  └──────────────────────────────────────────┘   │
  │  ┌──────────────────────────────────────────┐   │
  │  │ Runtime: Memory Pool, Device API, Tensor │   │
  │  └──────────────────────────────────────────┘   │
  └─────────────────────────────────────────────────┘
```

---

## 2. Applied Optimizations

### 2.1 Operator Level

| Operator | GPU Optimization | CPU Optimization | Status |
|----------|-----------------|------------------|--------|
| **linear** (GEMM/GEMV) | cuBLAS/cuBLASLt with auto-tuning, fused bias epilogue | oneDNN GEMM (BF16/FP32 native) | Done (PR-3a/3b) |
| **attention (decode)** | Custom paged attention kernel, online softmax, tiled KV, block-table direct read, zero-copy KV write | OMP parallel per-head | Done (PR-8) |
| **attention (prefill)** | Custom paged prefill kernel (block-table direct read). cuDNN FlashAttention exists as fallback for non-paged path only. | OMP naive GQA | Done (PR-8) |
| **rms_norm** | Vectorized 128-bit packed, warp+block reduction | AVX512/AVX2 vectorized | Already optimized |
| **rope** | Per-token CUDA kernel, on-the-fly sincos | CPU scalar | Already adequate (memory-bound) |
| **add, swiglu, embedding** | Vectorized 128-bit packed CUDA | AVX vectorized | Already bandwidth-limited |

### 2.2 Memory Management

| Feature | Description | Status |
|---------|-------------|--------|
| **Paged KV Cache** | Fixed-size block pool, per-request block table | Done (PR-7) |
| **Static VRAM Budget** | `total * gpu_memory_utilization - used` at init, one-time allocation | Done (PR-7) |
| **Zero-copy Decode** | `Tensor::create(is_mmap=true, block_ptr)` — model writes K/V directly into block | Done (PR-8) |
| **Memory Pool** | BestFit allocator for intermediate tensors, 64B alignment | Already existed |
| **No per-forward allocation** | Direct model forward with reusable tensor shapes | Done (PR-2) |

### 2.3 Scheduling & Batching

| Feature | Description | Status |
|---------|-------------|--------|
| **Continuous Batching** | Requests enter/leave batch independently at every iteration | Done (PR-9) |
| **Decode-First Scheduling** | Active decode requests always scheduled before new prefills | Done (PR-9) |
| **Chunked Prefill** | Long prompts split across iterations, bounded by `max_prefill_tokens` | Done (PR-9) |
| **Block-based Admission** | New requests admitted only if enough free blocks available | Done (PR-9) |
| **Batched Decode Attention** | Grid=(num_reqs, nhead), all decode requests in one kernel | Done (PR-9) |
| **Concat Tokens (No Padding)** | Per-token ops process `[total_tokens, dim]` — no wasted compute | Done (PR-9) |

### 2.4 Model Execution

| Feature | Description | Status |
|---------|-------------|--------|
| **Direct Forward** | `model.forward()` / `forward_batch()` — no graph execution overhead | Done (PR-2) |
| **Stateless Engine** | No mutable state on engine — safe for multi-session | Done (PR-4) |
| **Configurable Chat Template** | Per-model-family template, loaded from config | Done (PR-5) |

---

## 3. Performance Characteristics (Measured)

Model: DeepSeek-R1-Distill-Qwen-1.5B (BF16), GPU: NVIDIA B200

| Metric | Value |
|--------|-------|
| Single-request prefill throughput | ~1,125 tok/s (128 tokens) |
| Single-request decode throughput | ~306 tok/s |
| Batch=4 total throughput | ~672 tok/s (2.2x) |
| Batch=32 total throughput | ~1,820 tok/s (5.9x) |
| Batch=128 total throughput | ~2,170 tok/s (7.1x) |

Scaling trend: sub-linear, converging around batch=64-128. Expected for a small model on a large GPU — compute saturates quickly.

---

## 4. Strengths

1. **End-to-end paged pipeline.** From block pool allocation to paged attention kernels, the entire KV cache path avoids contiguous allocation and gather/copy overhead during decode.

2. **Zero-copy decode path.** Model writes K/V directly into block memory via `Tensor::create(is_mmap)`. Paged attention reads from the same memory. No scatter, no gather, no copy.

3. **Clean separation of concerns.** Scheduler owns request lifecycle and block allocation. Model owns compute. Pool owns memory. No circular dependencies.

4. **Backward compatibility.** `run_one()` still works for CLI tools (bench/chat/ping). `DynamicKVCache` fallback available. Paged and non-paged paths coexist.

5. **Library-backed compute.** cuBLAS for GEMM (auto-tuned for target GPU), oneDNN for CPU linear, cuDNN FlashAttention for non-paged prefill. Not reinventing the wheel for the compute-intensive ops.

---

## 5. Weaknesses & Missing Optimizations

### 5.1 Attention Kernel Quality

| Gap | Impact | Difficulty |
|-----|--------|-----------|
| **No FlashAttention for paged prefill** | Current paged prefill kernel is naive (no IO-aware tiling, no Q-Q reuse). For long prompts (2K+), it's 2-4x slower than FlashAttention-2. | High — requires flash-attn C++ integration with `block_table` + `cu_seqlens` support |
| **Prefill requests not merged** | Each prefill request launches a separate kernel. 10 concurrent prefills = 10 launches per layer. | Medium — need varlen API (`cu_seqlens`) |
| **Decode kernel V aggregation** | Block size 256 threads, head_dim=128 → `dv_group=2`. Each V element is touched by 2 threads with strided access. Sub-optimal for large batch. | Low — tune block size or restructure V loop |

### 5.2 Memory Efficiency

| Gap | Impact | Difficulty |
|-----|--------|-----------|
| **Per-layer block table upload** | In `forward_batch()`, block tables are uploaded to GPU as tensors every layer. Redundant — block tables don't change between layers. | Low — upload once, reuse across layers |
| **No KV cache quantization** | KV stored in BF16/FP16. INT8 KV cache would halve memory, allowing 2x more concurrent requests. | Medium — need quantized block pool + dequant in attention kernel |
| **No prefix caching** | Identical prompt prefixes across requests are prefilled independently. Could share KV blocks for common prefixes (system prompt). | Medium-High — needs block reference counting + copy-on-write |

### 5.3 Scheduling

| Gap | Impact | Difficulty |
|-----|--------|-----------|
| **No preemption** | If blocks run out, new requests queue indefinitely. Should preempt low-priority requests (swap KV to CPU, re-prefill later). | Medium — block swap to CPU pinned memory |
| **No priority scheduling** | FIFO only. No support for latency-sensitive vs throughput-oriented requests. | Low — add priority field and priority queue |
| **No speculative decoding** | Each decode step generates 1 token. Speculative decoding (draft model + verify) could generate 3-5 tokens per step. | High — needs draft model integration |
| **No dynamic batching adjustment** | Batch composition is recomputed each iteration, but no adaptive `max_batch_tokens` based on GPU utilization. | Low-Medium |

### 5.4 Compute

| Gap | Impact | Difficulty |
|-----|--------|-----------|
| **No operator fusion** | `rms_norm + linear`, `linear + bias + residual_add` run as separate kernels. Each has kernel launch overhead + extra memory roundtrip. | Medium — cuDNN graph API or custom fused kernels |
| **No CUDA graph capture** | The forward pass launches many small kernels. CUDA graph could batch them into a single launch for decode (fixed shape). | Medium — decode path has fixed shapes, suitable for graph capture |
| **Scatter kernel is per-token memcpy** | KV scatter in `forward_batch()` copies one token at a time with `memcpy_sync`. A custom scatter kernel could do all tokens in one launch. | Low — write a simple scatter CUDA kernel |

### 5.5 System

| Gap | Impact | Difficulty |
|-----|--------|-----------|
| **No HTTP API** | Can't serve real users yet. Only CLI and programmatic interface. | Medium (PR-10) |
| **No streaming via SSE** | `stream_callback` exists but no HTTP SSE transport. | Low (part of PR-10) |
| **Single-threaded engine loop** | `run_loop()` is single-threaded. Submit from HTTP threads, but engine processes sequentially. Fine for single GPU, but limits CPU overhead hiding. | Low — acceptable for single GPU |

---

## 6. Optimization Priority Ranking (Single-GPU Multi-User)

| Priority | Optimization | Expected Impact | Effort |
|----------|-------------|----------------|--------|
| **P0** | HTTP API (PR-10) | Enables real multi-user serving | Medium |
| **P1** | FlashAttention C++ API for paged prefill | 2-4x prefill speedup for long prompts | High |
| **P1** | CUDA graph capture for decode | 20-30% decode latency reduction (kernel launch overhead) | Medium |
| **P2** | Block table upload optimization | ~5% per-iteration overhead reduction | Low |
| **P2** | KV cache INT8 quantization | 2x more concurrent requests | Medium |
| **P2** | Custom KV scatter kernel | Minor — scatter is already small vs compute | Low |
| **P3** | Prefix caching | Significant for shared system prompts | Medium-High |
| **P3** | Operator fusion (norm+linear, etc.) | 10-15% decode latency improvement | Medium |
| **P3** | Preemption with CPU swap | Prevents queue starvation under memory pressure | Medium |
| **P4** | Speculative decoding | 2-3x decode throughput per request | High |
| **P4** | Priority scheduling | Better SLO management | Low |

---

## 7. Comparison with Production Systems

| Feature | ZedInfer (current) | vLLM | TensorRT-LLM |
|---------|-------------------|------|---------------|
| Paged KV cache | Yes | Yes | Yes |
| Continuous batching | Yes | Yes | Yes |
| Paged attention kernel | Custom (adequate) | Optimized (v1/v2) | Flash-based |
| FlashAttention prefill | No (custom naive paged kernel; cuDNN as non-paged fallback only) | flash-attn (paged + varlen) | Fused kernel |
| CUDA graphs | No | Yes | Yes |
| Operator fusion | No | Limited | Extensive |
| Prefix caching | No | Yes | Yes |
| Speculative decoding | No | Yes | Yes |
| INT8/INT4 KV cache | No | Yes | Yes |
| HTTP API | No | Yes (OpenAI-compat) | Triton backend |
| Multi-GPU tensor parallel | No | Yes | Yes |
| Preemption/swap | No | Yes | N/A |
| Language | C++ | Python + C++ kernels | C++ |
| Python-free serving | Yes | No (Python runtime) | Yes |

**ZedInfer's differentiator:** Pure C++ serving path with no Python runtime dependency. This matters for embedded/edge deployment and latency-sensitive scenarios where Python GIL and startup overhead are unacceptable.

---

## 8. Recommended Next Steps

### Phase 1: Enable Serving (prerequisite for everything else)

| # | Item | Why | Effort |
|---|------|-----|--------|
| 1 | **PR-10: HTTP API** (OpenAI-compatible) | Without this, multi-user serving is impossible. All subsequent optimizations are meaningless without a way to accept real requests. | Medium |

### Phase 2: Decode Latency (directly felt by every user, every token)

| # | Item | Why | Effort |
|---|------|-----|--------|
| 2 | **CUDA Graph capture for decode** | Decode is the hot loop (runs thousands of times per request). Current launch overhead: ~10us × ~50 kernels × per step. CUDA graph eliminates this entirely. Expected: 20-30% decode latency reduction with minimal code change. | Medium |
| 3 | **Operator fusion (norm+linear, linear+bias+residual)** | Reduces kernel launch count and memory roundtrips. Complements CUDA graph — fewer kernels to capture, less memory traffic. Expected: 10-15% additional decode improvement. | Medium |

### Phase 3: Prefill Throughput (felt during prompt processing)

| # | Item | Why | Effort |
|---|------|-----|--------|
| 4 | **FlashAttention C++ API for paged prefill** | Current paged prefill kernel is naive (no IO-aware tiling). For 2K+ prompts, FlashAttention-2 is 2-4x faster. Also enables `cu_seqlens` varlen API to batch multiple prefill requests in one kernel launch. | High |
| 5 | **Prefix caching** | For multi-user serving, many requests share the same system prompt (e.g., "You are a helpful assistant..."). Without prefix caching, each request re-prefills the common prefix. With it, KV blocks for the prefix are computed once and shared via reference counting. Reduces redundant prefill by 50-90% in typical chat serving. | Medium-High |

### Phase 4: Capacity & Model Size (run larger models, serve more users)

| # | Item | Why | Effort |
|---|------|-----|--------|
| 6 | **INT8/INT4 weight quantization** (PR-11/12) | 2-4x model memory reduction. Enables 8B models on 24GB GPUs. More VRAM left for KV cache = more concurrent users. | Medium |
| 7 | **KV cache INT8 quantization** | Halves KV cache memory. At batch=128, KV cache dominates VRAM. INT8 KV doubles the number of concurrent requests at the same memory budget. | Medium |

### Phase 5: Resilience & Advanced Scheduling

| # | Item | Why | Effort |
|---|------|-----|--------|
| 8 | **Preemption with CPU swap** | Under memory pressure, swap cold request KV to CPU pinned memory instead of dropping. Resume without re-prefill. Prevents queue starvation. | Medium |
| 9 | **Priority scheduling** | Different SLO tiers: latency-sensitive (interactive chat) vs throughput-oriented (batch processing). FIFO is insufficient for production. | Low |

### Phase 6: Throughput Multipliers (advanced)

| # | Item | Why | Effort |
|---|------|-----|--------|
| 10 | **Speculative decoding** | Draft model proposes 3-5 tokens per step, main model verifies in one forward pass. 2-3x effective decode throughput per request. Requires a small draft model + verification logic. | High |
| 11 | **Multi-GPU tensor parallelism** | Run models too large for one GPU (30B+). Split weight matrices across GPUs with NCCL all-reduce. | Very High |

### Dependency Graph

```
Phase 1:  HTTP API
              |
Phase 2:  CUDA Graph ──> Operator Fusion
              |
Phase 3:  FlashAttention Paged ──> Prefix Caching
              |
Phase 4:  INT8/INT4 Weights ──> KV Cache INT8
              |
Phase 5:  Preemption ──> Priority Scheduling
              |
Phase 6:  Speculative Decoding ──> Multi-GPU TP
```

Phases 2-4 can proceed in parallel. Phase 5 depends on HTTP API being in production. Phase 6 is independent but high-effort.

---

## 9. Architecture Principles Validated

Through PR-1 to PR-9, the following design decisions have proven correct:

1. **Paged KV before continuous batching** — the block pool/table abstraction was ready when batching needed it. No throwaway intermediate code.

2. **Direct model forward (no graph)** — `forward()` and `forward_batch()` are straightforward C++ methods. Easy to debug, profile, and extend. No graph abstraction overhead.

3. **Scheduler as the control plane** — the scheduler owns request lifecycle, block allocation, and batch assembly. The model is a pure compute function that takes a `BatchContext` and returns logits. Clean separation.

4. **Static VRAM budget** — querying free memory after model loading and pre-allocating the entire block pool eliminates runtime allocation failures. The scheduler uses `available_blocks()` for admission control — deterministic and predictable.

5. **`run_one()` backward compatibility** — CLI tools (bench/chat/ping) continue to work unchanged through all refactoring. Development velocity was never blocked by serving-mode changes.
