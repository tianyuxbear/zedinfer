# ZedInfer

A C++17 LLM inference engine with CPU (x86) and NVIDIA GPU (CUDA) backends. No Python in the serving path.

---

## Core Features

### Direct Model Forward
No graph compilation, no IR lowering. A single shared forward loop executes transformer layers directly, parameterized by model config flags. This eliminates graph-level overhead and keeps the execution path transparent and debuggable.

### Paged KV Cache
GPU memory is pre-allocated as a fixed-size block pool. Each request's KV cache is backed by non-contiguous blocks, enabling efficient memory utilization across varying sequence lengths. The block pool supports reference counting for sharing, LRU eviction under memory pressure, and O(1) statistics for fast admission decisions.

### Prefix Caching
When multiple requests share the same token prefix (e.g., system prompt or shared conversation history), their KV cache blocks are shared via content hashing rather than recomputed. This significantly reduces prefill latency for multi-turn conversations.

### Continuous Batching
Decode-first scheduling with chunked prefill. New requests are admitted while existing requests continue generating, maximizing GPU utilization. The scheduler performs block-based admission control to prevent out-of-memory conditions.

### Optimized Operator Backends
- **GPU**: cuBLAS/cuBLASLt for linear layers (auto-tuned), custom paged attention kernels with online softmax and shared memory optimization
- **CPU**: oneDNN for linear (runtime ISA dispatch -- automatically selects AVX2/AVX-512/AMX), AVX-vectorized elementwise ops with OpenMP parallelism

### OpenAI-Compatible HTTP API
Drop-in replacement for OpenAI's `/v1/chat/completions` endpoint with SSE streaming support. Includes a built-in Web UI for interactive chat. Stateful sessions enable multi-turn conversations with KV cache persistence across requests.

### DecodeScratch Pre-allocation
Pre-allocated fixed-address GPU buffers for single-token decode eliminate ~500 memory allocation round-trips per decode step, reducing per-token overhead to near zero.

---

## Supported Models

| Model | Architecture | Parameters |
|-------|-------------|-----------|
| DeepSeek-R1-Distill-Qwen-1.5B | Qwen2 | 1.5B |
| Qwen2.5-Math-1.5B-Instruct | Qwen2 | 1.5B |
| DeepSeek-R1-0528-Qwen3-8B | Qwen3 | 8B |
| Qwen3-8B | Qwen3 | 8B |

Adding a new Qwen-family model requires only defining a `ModelForwardConfig` -- zero forward logic changes needed.

## Data Types

BF16, FP16, FP32

## Quick Start

```bash
docker run --gpus all -p 8080:8080 \
    -v /path/to/models:/models \
    tianyuxbear/zedinfer:latest /models/DeepSeek-R1-Distill-Qwen-1.5B --nvidia
```

Then open [http://localhost:8080](http://localhost:8080) for the Web UI, or call the API:

```bash
curl http://localhost:8080/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"messages":[{"role":"user","content":"Hello"}]}'
```

See the [Quick Start Guide](guide/quickstart.md) for details.
