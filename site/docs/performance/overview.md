# Performance Overview

Benchmark results on NVIDIA B200 (180GB VRAM).

---

## End-to-End Throughput

**Model:** DeepSeek-R1-Distill-Qwen-1.5B (BF16)
**Config:** prefill=128, decode=128

| Batch Size | Decode Throughput (tok/s) | Notes |
|-----------|--------------------------|-------|
| 1 | ~129 | Single request |
| 4 | ~672 | |
| 32 | ~1,820 | |
| 128 | ~2,170 | Near memory-bandwidth bound |

!!! note "Performance is actively improving"
    The current paged attention kernels are the primary bottleneck (64% of GPU time). FlashInfer integration is expected to deliver 3-5x decode speedup.

---

## Bottleneck Analysis

| Component | % GPU Time | Avg Latency | Status |
|-----------|-----------|-------------|--------|
| Paged attention (decode) | 64% | 129 us/call | Bottleneck -- FlashInfer will replace |
| cuBLAS linear | ~30% | Varies | Optimized (auto-tuned) |
| Other kernels | ~6% | < 10 us | Not bottleneck |

---

## Optimizations Applied

| Optimization | Impact |
|-------------|--------|
| ArgmaxSampler pinned buffer | Eliminated 570 us/call cudaMallocHost |
| Paged decode shared memory precompute | ~16% decode improvement |
| Prefix caching | Skips redundant prefill for shared prefixes |
| DecodeScratch pre-allocation | Eliminates ~500 Tensor allocations per decode step |
| BlockPool O(1) counters | Fast admission control |

---

## Reproducing Benchmarks

```bash
# Single-request latency
docker run --gpus all -v /path/to/models:/models \
    --entrypoint /app/bench \
    zedinfer:latest /models/DeepSeek-R1-Distill-Qwen-1.5B --nvidia -p 128 -d 128 -r 3

# Batched throughput
docker run --gpus all -v /path/to/models:/models \
    --entrypoint /app/batch_bench \
    zedinfer:latest /models/DeepSeek-R1-Distill-Qwen-1.5B --nvidia -b 32 -p 128 -d 128
```
