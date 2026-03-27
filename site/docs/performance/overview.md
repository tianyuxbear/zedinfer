# Performance Overview

Benchmark results on **NVIDIA RTX 4090** (24GB VRAM), BF16 precision.

---

## DeepSeek-R1-Distill-Qwen-1.5B

### Prefill (tok/s)

| Prompt Length | ZedInfer | llama.cpp | vLLM | SGLang |
|--------------|----------|-----------|------|--------|
| 128 | 4,661 | 15,076 | 13,332 | 5,967 |
| 256 | 5,329 | 21,949 | 21,961 | 12,178 |
| 512 | 5,566 | 24,647 | 29,476 | 23,853 |
| 1024 | 5,612 | 24,259 | 36,611 | 34,162 |

### Decode (tok/s)

| Prompt Length | ZedInfer | ZedInfer (batch=4) | llama.cpp | vLLM | SGLang |
|--------------|----------|--------------------|-----------|------|--------|
| 128 | 189.3 | 513.7 | 245.4 | 209.8 | 247.4 |
| 256 | 170.2 | 453.8 | 245.6 | 209.8 | 246.4 |
| 512 | 141.2 | 370.2 | 241.5 | 209.2 | 245.8 |
| 1024 | 105.3 | 268.6 | 241.0 | 208.3 | 244.4 |

---

## DeepSeek-R1-0528-Qwen3-8B

### Prefill (tok/s)

| Prompt Length | ZedInfer | llama.cpp | vLLM | SGLang |
|--------------|----------|-----------|------|--------|
| 128 | 2,980 | 4,442 | 4,666 | 3,001 |
| 256 | 3,374 | 6,601 | 7,467 | 5,246 |
| 512 | 2,914 | 7,430 | 8,563 | 6,860 |
| 1024 | 2,261 | 7,040 | 9,284 | 8,213 |

### Decode (tok/s)

| Prompt Length | ZedInfer | ZedInfer (batch=4) | llama.cpp | vLLM | SGLang |
|--------------|----------|--------------------|-----------|------|--------|
| 128 | 54.0 | 183.3 | 58.3 | 57.0 | 59.5 |
| 256 | 50.4 | 171.5 | 58.3 | 56.7 | 59.2 |
| 512 | 43.3 | 149.5 | 57.7 | 56.5 | 58.9 |
| 1024 | 35.3 | 125.2 | 57.8 | 56.1 | 58.3 |

---

## Key Observations

**Prefill gap**: ZedInfer's prefill throughput is 2-6x behind vLLM/SGLang, and the gap widens with sequence length. Root cause: the current paged prefill kernel has no IO-aware tiling -- K/V data is loaded from global memory independently per query position with no shared memory reuse.

**Decode degradation**: Other engines maintain near-constant decode throughput across all sequence lengths. ZedInfer's decode throughput drops as sequence length grows (189 → 105 tok/s for 1.5B at 128 → 1024). Root cause: the decode kernel uses Grid=(nhead,) -- only 12 blocks for 1.5B, utilizing 9% of RTX 4090's 128 SMs. Longer sequences mean more serial work per block.

**Batched decode**: The batch=4 numbers show that batching effectively multiplies throughput (2.5-2.7x vs single), confirming the continuous batching scheduler works correctly. However, per-request throughput still degrades with sequence length.

!!! note "FlashInfer integration is the primary optimization target"
    Replacing the custom paged attention kernels with FlashInfer is expected to deliver **3-5x decode speedup** and **2-4x prefill speedup** through IO-aware tiling (prefill) and split-K flash-decoding (decode).

---

## Optimizations Applied (v0.1.0)

| Optimization | Impact |
|-------------|--------|
| DecodeScratch pre-allocation | Eliminates ~500 Tensor allocations per decode step |
| Paged decode shared memory precompute | ~16% decode improvement (breaks dependent-load chain) |
| ArgmaxSampler pinned buffer | Eliminated 570 us/call cudaMallocHost |
| Prefix caching | Skips redundant prefill for shared token prefixes |
| BlockPool O(1) counters | Fast admission control for continuous batching |

---

## Reproducing Benchmarks

```bash
# Single-request latency
docker run --gpus all -v /path/to/models:/models \
    --entrypoint /app/bench \
    tianyuxbear/zedinfer:latest /models/DeepSeek-R1-Distill-Qwen-1.5B --nvidia -p 128 -d 128 -r 3

# Batched throughput (4 concurrent requests)
docker run --gpus all -v /path/to/models:/models \
    --entrypoint /app/batch_bench \
    tianyuxbear/zedinfer:latest /models/DeepSeek-R1-Distill-Qwen-1.5B --nvidia -b 4 -p 128 -d 128
```

### Comparison engine versions

| Engine | Version |
|--------|---------|
| ZedInfer | v0.1.0 |
| llama.cpp | e4832e3 |
| vLLM | 0.13.0 |
| SGLang | 0.5.7 |
