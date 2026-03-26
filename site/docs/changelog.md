# Changelog

---

## v0.1.0 (2026-03-26)

First release.

### Features

- Direct model forward -- no graph execution overhead, single shared forward loop
- Paged KV cache -- block pool with reference counting, LRU eviction, prefix caching
- Continuous batching -- decode-first scheduling with chunked prefill
- OpenAI-compatible HTTP API with SSE streaming and Web UI
- Docker deployment -- multi-stage build for closed-source distribution

### Supported Models

- Qwen2: DeepSeek-R1-Distill-Qwen-1.5B, Qwen2.5-Math-1.5B-Instruct
- Qwen3: DeepSeek-R1-0528-Qwen3-8B, Qwen3-8B

### Operator Backends

| | CPU | GPU |
|---|---|---|
| Linear | oneDNN (runtime ISA dispatch) | cuBLAS / cuBLASLt |
| Attention | Paged GQA (decode/prefill/batched) | Custom paged kernels |
| RMSNorm, RoPE, SwiGLU, Add | AVX vectorized | CUDA vectorized |

### Performance (B200, DeepSeek-R1-Distill-Qwen-1.5B BF16)

| Batch Size | Throughput |
|-----------|-----------|
| 1 | ~129 tok/s |
| 4 | ~672 tok/s |
| 32 | ~1,820 tok/s |
| 128 | ~2,170 tok/s |
