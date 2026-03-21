# Benchmark Plan

## Hardware Targets

| Name | GPU | CPU | RAM | PCIe |
|------|-----|-----|-----|------|
| Primary (server) | NVIDIA RTX 4090 (24GB) | Intel Xeon Silver 4310 | 128GB | PCIe 4.0 x16 |
| Secondary (future) | NVIDIA RTX 3090 (24GB) | Same or equivalent | 128GB | PCIe 4.0 x16 |
| CPU-only | - | Intel Xeon Silver 4310 (AVX-512) | 128GB | N/A |

## Models Under Test

| Model | Size | Layers | Hidden | KV Heads | Head Dim |
|-------|------|--------|--------|----------|----------|
| DeepSeek-R1-Distill-Qwen-1.5B | 1.5B | 28 | 1536 | 2 | 128 |
| DeepSeek-R1-0528-Qwen3-8B | 8B | 36 | 4096 | 8 | 128 |

## Benchmark Configurations

### Standard Configurations

| Config | Prefill Len | Decode Len | Rounds | Purpose |
|--------|-------------|-----------|--------|---------|
| A: Decode-heavy | 32 | 256 | 5 | Chatbot response latency |
| B: Balanced | 128 | 128 | 5 | General benchmark |
| C: Prefill-heavy | 512 | 64 | 5 | Document processing |
| D: Long context | 2048 | 128 | 3 | Long context performance |
| E: Short decode | 16 | 32 | 10 | Minimum latency scenario |

### Per-Operator Benchmarks

| Operator | Shapes (M, N, K) | Purpose |
|----------|------------------|---------|
| linear (decode) | (1, 1536, 1536), (1, 4096, 4096), (1, 11008, 4096) | Decode-phase GEMV |
| linear (prefill) | (128, 1536, 1536), (128, 4096, 4096), (512, 4096, 4096) | Prefill-phase GEMM |
| self_attention (decode) | seq=1, total_len={128,512,2048,8192} | Decode attention |
| self_attention (prefill) | seq={128,512,2048}, total_len=seq | Prefill attention |
| rms_norm | (128, 1536), (128, 4096), (1, 4096) | Normalization |
| rope | (128, 32, 128), (1, 32, 128) | Position encoding |
| swiglu | (128, 11008), (1, 11008) | Activation |

## Metrics

### Latency Metrics

| Metric | Unit | How Measured |
|--------|------|-------------|
| Prefill latency | ms | Time for first forward pass (all prompt tokens) |
| Decode latency (per token) | ms | Average time per decode step |
| Time to First Token (TTFT) | ms | Prefill latency + first decode step |
| Total generation time | ms | End-to-end from prompt to last token |

### Throughput Metrics

| Metric | Unit | How Measured |
|--------|------|-------------|
| Prefill throughput | tokens/s | prompt_tokens / prefill_latency |
| Decode throughput | tokens/s | generated_tokens / decode_latency |
| Total throughput | tokens/s | total_tokens / total_time |
| Batch throughput (multi-user) | requests/s | completed_requests / wall_time |

### Resource Metrics

| Metric | Unit | How Measured |
|--------|------|-------------|
| Peak GPU memory | MB | nvidia-smi or CUDA API |
| Peak CPU memory | MB | /proc/self/status VmPeak |
| KV cache memory | MB | kvcache.memory_usage() |
| GPU utilization | % | nvidia-smi (sampled) |

## Before/After Methodology

### Per-Stage Protocol

For each implementation stage:

1. **Before**: Run all standard configs (A-E) on both models, record results
2. **Implement**: Make the changes for this stage
3. **After**: Run identical configs, record results
4. **Compare**: Generate comparison table

### Comparison Format

```
================ Benchmark Comparison: Stage N ================
Model: DeepSeek-R1-0528-Qwen3-8B
Device: NVIDIA RTX 4090
Config: B (prefill=128, decode=128)

                    Before      After       Delta    Status
Prefill (ms):       45.23       42.10       -6.9%    PASS
Decode (ms):        312.50      295.80      -5.3%    PASS
Prefill (tok/s):    2830        3040        +7.4%    PASS
Decode (tok/s):     409.6       432.7       +5.6%    PASS
Peak GPU (MB):      8192        7850        -4.2%    PASS
============================================================

PASS threshold: no regression >2% on latency metrics
WARN threshold: regression 2-5%
FAIL threshold: regression >5%
```

### Statistical Rigor

- Run each config `rounds` times (minimum 3, default 5)
- Report mean and standard deviation
- Discard first run as warmup (already handled by engine warmup)
- Use paired comparison (same hardware, same model state)
- Report p-value for before/after difference (paired t-test, if rounds >= 5)

## Per-Stage Benchmark Focus

| Stage | Primary Metric | Configs | Expected Outcome |
|-------|---------------|---------|-----------------|
| 0: Test infra | Baseline capture | All | Record baselines |
| 1: Direct forward | Decode latency | A, B | 5-15% decode improvement (fewer allocations) |
| 2a: cuBLAS linear | Linear operator time | Operator bench | 2-5x speedup for GEMM |
| 2b: oneDNN linear | Linear operator time | Operator bench (CPU) | 2-5x speedup for CPU GEMM |
| 3: Multi-user | (no single-user change) | B | No regression |
| 4: Scheduler | TTFT, batch throughput | Multi-user bench | Higher throughput |
| 5: Continuous batching | Batch throughput | Multi-user bench | 2-4x throughput |
| 6: Paged KV cache | Memory usage | D (long context) | Reduced peak memory |
| 7: Paged attention | Decode latency + memory | A, D | No latency regression, better memory |
| 8: HTTP API | Request throughput | HTTP load test | Baseline for serving |
| 9: INT8 | All metrics | All | ~50% memory reduction, ~1.5x throughput |
| 10: INT4 | All metrics | All | ~75% memory reduction |

## Multi-User Benchmarks (After Stage 5+)

| Config | Concurrent Requests | Prompt Len | Max Tokens | Metric |
|--------|-------------------|-----------|-----------|--------|
| MU-A | 4 | 128 | 128 | Batch throughput, avg TTFT |
| MU-B | 16 | 64 | 64 | Batch throughput, avg TTFT |
| MU-C | 32 | 32 | 32 | Max throughput, P99 TTFT |
| MU-D | 4 | 2048 | 128 | Long-context multi-user |

### HTTP Load Test (After Stage 8)

Use `wrk` or `hey` to generate load:
```bash
hey -n 100 -c 10 -m POST \
    -H "Content-Type: application/json" \
    -d '{"model":"qwen3-8b","messages":[{"role":"user","content":"Hi"}],"max_tokens":64}' \
    http://localhost:8080/v1/chat/completions
```

Metrics: requests/sec, P50/P90/P99 latency, error rate.

## Benchmark Automation

### Script: `scripts/benchmark.sh`

```bash
#!/bin/bash
# Run standard benchmarks and save results
MODEL_PATH=$1
DEVICE=${2:-nvidia}
OUTPUT_DIR=${3:-benchmarks/$(date +%Y%m%d_%H%M%S)}

mkdir -p $OUTPUT_DIR

for config in A B C D E; do
    case $config in
        A) P=32;  D=256; R=5 ;;
        B) P=128; D=128; R=5 ;;
        C) P=512; D=64;  R=5 ;;
        D) P=2048; D=128; R=3 ;;
        E) P=16;  D=32;  R=10 ;;
    esac

    FLAGS=""
    if [ "$DEVICE" = "nvidia" ]; then FLAGS="--nvidia"; fi

    xmake run bench $MODEL_PATH -p $P -d $D -r $R $FLAGS \
        | tee $OUTPUT_DIR/config_${config}.txt
done
```

### Results Storage

Benchmark results saved as JSON in `benchmarks/`:
```json
{
    "timestamp": "2026-03-21T10:00:00Z",
    "stage": "stage_1_direct_forward",
    "hardware": {"gpu": "RTX 4090", "cpu": "Xeon Silver 4310"},
    "model": "DeepSeek-R1-0528-Qwen3-8B",
    "configs": {
        "B": {
            "prefill_len": 128,
            "decode_len": 128,
            "rounds": 5,
            "prefill_ms": {"mean": 42.1, "stddev": 1.2},
            "decode_ms": {"mean": 295.8, "stddev": 3.5},
            "prefill_tps": {"mean": 3040, "stddev": 85},
            "decode_tps": {"mean": 432.7, "stddev": 5.1}
        }
    }
}
```
