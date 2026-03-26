# Operator Benchmarks

Per-operator performance data. Updated as optimizations land.

---

!!! info "Coming soon"
    Detailed per-operator benchmarks (latency, throughput, dtype comparison) for both CPU and GPU backends will be published here.

## Planned Content

- **Linear**: cuBLAS vs oneDNN, BF16/FP16/FP32, various matrix sizes
- **Attention**: Decode single vs batched, prefill scaling with sequence length
- **RMSNorm**: CPU (AVX2/AVX-512) vs GPU, varying hidden dimensions
- **RoPE**: CPU vs GPU, throughput vs sequence length
- **SwiGLU / Add**: Elementwise operation throughput
- **Embedding / Argmax**: Lookup and reduction performance

## Test Framework

Operator correctness is verified via Python bindings against PyTorch reference implementations. Each operator is tested across:

- Multiple shapes (small to large)
- Multiple data types (FP32, FP16, BF16)
- Both CPU and GPU backends
