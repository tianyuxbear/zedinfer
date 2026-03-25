<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="web/images/logo-dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="web/images/logo-light.svg">
    <img alt="ZedInfer Logo" src="web/images/logo-dark.svg" width="400">
  </picture>
</p>

<p align="center">
  <strong>High-performance LLM inference engine built from scratch in C++17</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/language-C++17-blue.svg" alt="Language"/>
  <img src="https://img.shields.io/badge/build-XMake-green.svg" alt="Build"/>
  <img src="https://img.shields.io/badge/platform-Linux-lightgrey.svg" alt="Platform"/>
  <img src="https://img.shields.io/badge/license-MIT-orange.svg" alt="License"/>
</p>

<p align="center">
  <a href="README_ZH.md">中文文档</a>
</p>

---

## ✨ Features

### Implemented

- **Multi-user serving** — Continuous batching scheduler with decode-first policy and chunked prefill
- **Paged KV cache** — Fixed-size block pool with static VRAM budget, O(1) block allocation
- **Paged attention** — Custom CUDA kernels for decode (single/batched) and prefill with block-table indexing
- **Prefix caching** — Cross-request KV block sharing via chain-hashed content matching and reference counting
- **HTTP API** — OpenAI-compatible `/v1/chat/completions` with SSE streaming and embedded web chat UI
- **Stateful sessions** — Server-side KV cache reuse across multi-turn conversations
- **Optimized operators** — cuBLAS/cuBLASLt for GPU linear, oneDNN for CPU linear, pre-allocated decode scratch buffers
- **Direct model forward** — No graph execution overhead; single shared `transformer_forward()` loop
- **Zero Python runtime** — Pure C++ serving path, no Python dependency at inference time

### Planned (see `docs/plan/`)

- 🔜 **FlashInfer integration** — Optimized paged attention kernels (3-5x decode speedup expected)
- 🔜 **CUDA Graph** — Capture/replay decode forward pass (Phase 1 DecodeScratch done)
- 📋 **INT8/INT4 quantization** — Weight-only quantization for 2-4x memory reduction
- 📋 **Heterogeneous inference** — CPU/GPU mixed execution with expert offloading for MoE models

---

## 🏗️ Supported Models

| Model | Architecture | Parameters | Tested |
|-------|-------------|-----------|--------|
| DeepSeek-R1-Distill-Qwen-1.5B | Qwen2 | 1.5B | ✅ |
| DeepSeek-R1-0528-Qwen3-8B | Qwen3 | 8B | ✅ |
| Qwen2.5-Math-1.5B-Instruct | Qwen2 | 1.5B | ✅ |
| Qwen3-8B | Qwen3 | 8B | ✅ |

Adding a new Qwen-family model requires only defining a `ModelForwardConfig` (bias/Q-K norm flags) — zero forward logic needed.

---

## 📦 Dependencies

### Third-party Headers (vendored in `third_party/include/`)

| Library | Purpose | License |
|---------|---------|---------|
| [nlohmann/json](https://github.com/nlohmann/json) | JSON parsing (model config, tokenizer, HTTP API) | MIT |
| [plog](https://github.com/SergiusTheBest/plog) | Lightweight logging framework | MIT |
| [argparse](https://github.com/p-ranav/argparse) | CLI argument parsing | MIT |
| [cpp-httplib](https://github.com/yhirose/cpp-httplib) | HTTP server (OpenAI-compatible API) | MIT |
| [dbg-macro](https://github.com/sharkdp/dbg-macro) | Debug printing utility | MIT |

### System / Package Manager Dependencies

| Library | Purpose | License | Required |
|---------|---------|---------|----------|
| [ICU4C](https://icu.unicode.org/) | Unicode regex for BPE tokenizer | Unicode License | Yes |
| [linenoise](https://github.com/antirez/linenoise) | Interactive CLI line editing (chat) | BSD-2-Clause | Vendored |
| [Google Test](https://github.com/google/googletest) | Unit testing framework | BSD-3-Clause | Test only |
| [pybind11](https://github.com/pybind/pybind11) | Python operator test bindings | BSD-3-Clause | Optional |

### GPU / Compute Libraries

| Library | Purpose | License | Required |
|---------|---------|---------|----------|
| [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) | GPU runtime, kernel compilation | NVIDIA EULA | GPU only |
| [cuBLAS / cuBLASLt](https://developer.nvidia.com/cublas) | Optimized GPU GEMM (part of CUDA Toolkit) | NVIDIA EULA | GPU only |
| [oneDNN](https://github.com/oneapi-src/oneDNN) | Optimized CPU GEMM (BF16/FP32 native) | Apache-2.0 | Optional |

---

## 🔧 Build

### Prerequisites

- Linux (tested on Ubuntu 22.04+)
- GCC 11+ with C++17 support
- [XMake](https://xmake.io/) build system
- CUDA Toolkit 12.0+ (for GPU support)

### Build Commands

```bash
# CPU only
xmake f -m release
xmake build

# CPU + NVIDIA GPU
xmake f -m release --nv-gpu=y
xmake build

# CPU + GPU + oneDNN
xmake f -m release --nv-gpu=y --onednn=y
xmake build
```

---

## 🚀 Run

### Interactive Chat

```bash
xmake run chat /path/to/model --nvidia
```

### HTTP Server

```bash
xmake run serve /path/to/model --nvidia --port 8080
# Open http://localhost:8080 for the web chat UI
```

### Quick Test

```bash
xmake run ping /path/to/model --nvidia
```

### Benchmark

```bash
# Single-request benchmark
xmake run bench /path/to/model --nvidia -p 128 -d 128 -r 3

# Multi-request batch benchmark
xmake run batch_bench /path/to/model --nvidia -p 128 -d 128 --batch 4
```

---

## 🧪 Test

### C++ Unit Tests

```bash
# Build all tests
xmake build -g test

# Run individual test suites
xmake run test-blockpool       # KV cache block pool + ref counting
xmake run test-prefixcache     # Prefix caching hash match
xmake run test-sampler         # Argmax + general sampler
xmake run test-chattemplate    # Chat template formatting
xmake run test-tensor          # Tensor ops (shape, slice, permute, device transfer)
xmake run test-memorypool      # Memory pool allocation
xmake run test-storage         # Storage layer

# Tests requiring model files (set env var)
ZEDINFER_TEST_MODEL_PATH=/path/to/model xmake run test-tokenizer
ZEDINFER_TEST_MODEL_PATH=/path/to/model xmake run test-loader
```

### Operator Correctness Tests (Python)

Operators are tested via Python bindings against PyTorch reference implementations:

```bash
# Build with Python bindings
xmake f -m release --nv-gpu=y --onednn=y --pytest=y
xmake build zedinfer_ops

# Install Python package
uv pip install -e python/

# Run operator tests (compare against PyTorch)
uv run tests/python/test_add.py
uv run tests/python/test_linear.py
uv run tests/python/test_self_attention.py
uv run tests/python/test_rms_norm.py
uv run tests/python/test_rope.py
uv run tests/python/test_swiglu.py
uv run tests/python/test_embedding.py
uv run tests/python/test_argmax.py
```

---

## 📊 Performance

> WIP — Comprehensive benchmarks coming soon.

---

## 📁 Project Structure

```
zedinfer/
├── include/
│   ├── zedinfer/                    # Engine, scheduler, session, request
│   │   ├── engine.hpp
│   │   ├── scheduler.hpp
│   │   ├── serving_loop.hpp
│   │   ├── session.hpp
│   │   ├── request.hpp
│   │   ├── http_server.hpp
│   │   ├── chat_template.hpp
│   │   └── ...
│   ├── backend/
│   │   ├── core/                    # Runtime, memory pool, storage, context
│   │   ├── device/                  # Device abstraction (CPU / NVIDIA)
│   │   ├── tensor/                  # Tensor (shared_ptr, view, slice, permute)
│   │   ├── kvcache/                 # Block pool, prefix cache
│   │   │   ├── block_pool.hpp
│   │   │   └── prefix_cache.hpp
│   │   └── ops/                     # Operator dispatch + kernel headers
│   │       ├── ops.hpp
│   │       ├── attention_params.hpp
│   │       └── {add,argmax,embedding,linear,rms_norm,rope,self_attention,swiglu}/
│   └── frontend/
│       ├── models/                  # Model config, forward loop, decode scratch
│       │   ├── paged_forward_context.hpp
│       │   └── decode_scratch.hpp
│       ├── tokenizer/               # HuggingFace BPE tokenizer
│       ├── sampler/                 # Argmax + general (temp/top-k/top-p)
│       └── loader/                  # SafeTensors mmap loader
├── src/                             # Implementation (.cpp / .cu)
│   ├── zedinfer/                    # Engine, scheduler, serving loop, HTTP server
│   ├── backend/                     # Operators, KV cache, runtime
│   └── frontend/                    # Models, tokenizer, sampler, loader
├── examples/
│   ├── bench.cpp                    # Single-request benchmark
│   ├── batch_bench.cpp              # Multi-request batch benchmark
│   ├── chat.cpp                     # Interactive multi-turn chat (linenoise)
│   ├── ping.cpp                     # Quick single-turn test
│   └── serve.cpp                    # HTTP server entry point
├── tests/
│   ├── core/                        # Memory pool, storage tests
│   ├── tensor/                      # Tensor operation tests
│   ├── kvcache/                     # Block pool, prefix cache tests
│   ├── sampler/                     # Sampler tests
│   ├── zedinfer/                    # Chat template tests
│   ├── tokenizer/                   # Tokenizer encode/decode tests
│   ├── loader/                      # SafeTensors loader tests
│   └── python/                      # Operator correctness tests (PyTorch ref)
├── web/
│   ├── index.html                   # Single-page chat UI
│   └── images/                      # Icons, logo
├── third_party/include/             # Vendored header-only libraries
├── docs/
│   ├── architecture.md              # Current system design
│   ├── roadmap.md                   # Status + future plans
│   └── plan/                        # Design docs for upcoming features
├── xmake.lua                        # Build configuration
└── CLAUDE.md                        # AI assistant working rules
```

---

## 📄 License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).

Copyright (c) 2025-2026 ZedInfer Contributors. See individual dependency licenses in the [Dependencies](#-dependencies) section.
