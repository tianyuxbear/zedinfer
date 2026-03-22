# Post-Cleanup Status Report

> Final review after R1-R6 refactoring + optimization passes.

---

## 1. Architecture Overview

```
InferenceEngine (resource owner + factory)
  ├── Model (config + weights + forward_config())
  ├── Tokenizer (HF BPE)
  ├── Sampler (Argmax)
  ├── BlockPool (static VRAM allocation)
  ├── BlockAllocator (per-request block management)
  ├── ServingLoop (schedule + step + run_loop/run_serving)
  │     └── Scheduler (batch assembly, block-based admission, process_results)
  └── Profiler (warmup/profile with DynamicKVCache)

transformer_forward() — single shared forward loop
  ├── ModelForwardConfig (has_qkv_bias, has_qk_norm)
  ├── PagedForwardContext (serving: paged blocks, scatter, paged attention)
  └── ContiguousForwardContext (warmup/profile: DynamicKVCache, self_attention)

ops::attention(AttentionParams) — unified dispatch
  ├── Contiguous (warmup/profile)
  ├── Paged decode single
  ├── Paged decode batched
  └── Paged prefill
```

---

## 2. Applied Optimizations

| Technology | Description | Status |
|-----------|-------------|--------|
| cuBLAS/cuBLASLt for linear | Auto-tuned GEMM/GEMV, fused bias epilogue | Done |
| oneDNN for CPU linear | BF16/FP32 native GEMM | Done |
| Paged KV cache | Static block pool, per-request block table | Done |
| Zero-copy decode | Tensor::create(is_mmap) writes K/V directly to block memory | Done |
| Paged attention decode | Custom CUDA kernel, block-table-indexed K/V, online softmax | Done |
| Paged attention prefill | Custom CUDA kernel, block-table-indexed, causal mask | Done |
| Batched paged decode | Grid=(num_reqs, nhead), all decode requests in one kernel | Done |
| Continuous batching | Decode-first scheduling, concat tokens (no padding) | Done |
| Chunked prefill | Long prompts split across iterations, bounded by max_prefill_tokens | Done |
| Block table GPU caching | Per-layer GPU tensors built once, reused across all transformer layers | Done |
| Async KV scatter | GPU memcpy_async for block scatter, eliminates CPU blocking | Done |
| Unified execution path | Session and batch mode both use schedule()+step(), no dual path | Done |
| Block-based admission control | can_admit() checks available blocks, accounts for existing session blocks | Done |
| Dynamic block extension | Blocks extended on-demand during decode and multi-turn prefill | Done |
| ServingLoop idle-wait | condition_variable for HTTP serving mode (run_serving/stop) | Done |
| Session borrowed block table | block_table_ref avoids copy, process_results updates in-place | Done |
| Explicit prefill chunk_start | ScheduledBatch stores chunk offset, no implicit coupling with prefill_progress | Done |

---

## 3. Code Health

| Metric | Status |
|--------|--------|
| Forward loop duplications | 0 (single transformer_forward.cpp) |
| KV management paths | 1 (block table everywhere, borrowed or owned) |
| Attention dispatch functions | 1 (ops::attention with AttentionParams) |
| Execution paths | 1 (schedule+step for all generation) |
| Dead code | 0 (graph execution, run_one, qwen3 #if 0, unused members — all removed) |
| Compiler warnings | 0 |
| Model virtual methods | 5 (config, weights, model_type, num_params, forward_config) |
| Engine lines | ~140 (resource owner only) |

---

## 4. Remaining Accepted Debt

| Item | Reason to Keep | Impact |
|------|---------------|--------|
| DynamicKVCache + ContiguousForwardContext | Warmup/profile path; cuDNN warmup may affect GPU runtime state (unresolved perf mystery) | No serving impact |
| cuDNN FlashAttention code | Optional compile flag, only triggered during warmup prefill | Can be removed once paged warmup is implemented |
| CPU paged attention kernel | Written but not wired into model forward for CPU | Ready when CPU serving is needed |

---

## 5. Evolution Roadmap

### Phase 1: Enable Serving

| # | Item | Readiness |
|---|------|-----------|
| 1 | **HTTP API (PR-10)** | ServingLoop has run_serving/stop/submit_async. Session management ready. |

### Phase 2: Decode Latency

| # | Item | Readiness |
|---|------|-----------|
| 2 | **CUDA Graph capture** | Block table GPU cache eliminates dynamic allocation in decode loop. Fixed kernel sequence. |
| 3 | **Operator fusion** | Single forward loop — one place to add fused kernels. |

### Phase 3: Prefill Throughput

| # | Item | Readiness |
|---|------|-----------|
| 4 | **flash-attn C++ API (varlen paged)** | AttentionParams can add cu_seqlens field. Unified dispatch adds one branch. |
| 5 | **Prefix caching** | Unified block table path. Need: BlockPool ref counting + hash table. |

### Phase 4: Capacity

| # | Item | Readiness |
|---|------|-----------|
| 6 | **INT8/INT4 weight quantization** | ModelForwardConfig can add quant flags. linear dispatch adds quantized path. |
| 7 | **KV cache INT8** | BlockPool can store INT8 blocks. Attention kernels need dequant. |

### Phase 5: Resilience

| # | Item | Readiness |
|---|------|-----------|
| 8 | **Preemption (CPU swap)** | Block granularity ready. Need: pinned memory pool + swap logic. |
| 9 | **Priority scheduling** | Scheduler can add priority field to InferenceRequest. |

### Phase 6: Advanced

| # | Item | Readiness |
|---|------|-----------|
| 10 | **Speculative decoding** | AttentionParams can add verification mode. |
| 11 | **Multi-GPU tensor parallelism** | Weight sharding + NCCL all-reduce. |

---

## 6. Adding a New Model

After all refactoring, adding a new model family (e.g., Llama):

```cpp
// include/frontend/models/llama.hpp
struct LlamaConfig : public ModelConfig { ... };

class LlamaModel : public Model {
    ModelForwardConfig forward_config() const override {
        return {config_, *weights_, /*has_qkv_bias=*/false, /*has_qk_norm=*/false};
    }
    // ... config(), weights(), model_type(), num_parameters()
};
```

Zero forward logic needed. The shared `transformer_forward()` handles everything parameterized by `ModelForwardConfig`.

---

## 7. Benchmark Reference (B200, DeepSeek-R1-Distill-Qwen-1.5B, BF16)

| Config | Throughput |
|--------|-----------|
| Single request, decode | ~306 tok/s |
| Batch=4 | ~672 tok/s (2.2x) |
| Batch=32 | ~1,820 tok/s (5.9x) |
| Batch=128 | ~2,170 tok/s (7.1x) |
