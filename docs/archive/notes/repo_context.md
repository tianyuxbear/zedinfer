# Repository Context

## Overview

**ZedInfer** is a lightweight C++17 LLM inference engine built from scratch, targeting edge/AIPC scenarios. It supports CPU (x86_64 AVX512/AVX2) and NVIDIA GPU (CUDA) backends with FP32/FP16/BF16 precision. Currently adapted for Qwen2 and Qwen3 model families (DeepSeek-R1 distillations).

## Build System

- **Tool**: [XMake](https://xmake.io/) (`xmake.lua`)
- **Standard**: C++17
- **Dependencies**:
  - `icu4c` - Unicode processing (tokenizer regex)
  - `readline` - Interactive CLI (chat example)
  - `gtest` - Unit testing
  - `pybind11` - Optional Python operator test bindings
  - `nlohmann/json` - JSON parsing (model config, tokenizer)
- **Third-party headers** (`third_party/include/`):
  - `plog` - Logging framework
  - `sharkdp/dbg.h` - Debug printing
  - `argparse` - Command-line argument parsing
- **Build options**:
  - `--nv-gpu=y` - Enable NVIDIA GPU support (defines `ENABLE_NVIDIA_API`)
  - `--pytest` - Build Python operator test bindings
- **Targets**: `zedinfer` (static lib), `frontend`, `backend`, `utils`, plus examples and tests

## Directory Layout

```
zedinfer/
  include/
    zedinfer.h                      # C API: device types, data types, memcpy kinds
    zedinfer/
      engine.hpp                    # InferenceEngine (stateless, shared across sessions)
      session.hpp                   # InferenceSession (per-session state)
      executor.hpp                  # GraphExecutor (graph walker)
      activation.hpp                # ExecutorConfig, PositionIDsCache
      generation_types.hpp          # GenerationConfig, GenerationStats, GenerationMode
    backend/
      core/
        core.hpp                    # Forward declarations (Storage, Runtime, Context)
        context/context.hpp         # Thread-local Context (device/runtime management)
        runtime/runtime.hpp         # Runtime (allocation, streams, device binding)
        storage/storage.hpp         # Storage (memory block with device affinity)
        memory/
          allocator.hpp             # MemoryAllocator (abstract)
          memory_pool.hpp           # BestFitMemoryPool (size-class based)
          naive_allocator.hpp       # NaiveAllocator (malloc/free passthrough)
          pooled_allocator.hpp      # PooledAllocator (pool-backed)
      device/
        device.hpp                  # Device class (type + id)
        runtime_api.hpp             # ZedinferRuntimeAPI (function pointer table)
      tensor/tensor.hpp             # Tensor (shared_ptr, view/slice/permute/to)
      kvcache/
        base.hpp                    # KVCache base class
        dynamic.hpp                 # DynamicKVCache (auto-growth, contiguous)
      ops/
        ops.hpp                     # Operator dispatch interface (add, linear, etc.)
        {add,argmax,embedding,linear,rearrange,rms_norm,rope,self_attention,swiglu}/
          cpu/*.hpp                 # CPU kernel headers
          nvidia/*.cuh              # NVIDIA kernel headers
    frontend/
      graph/
        graph.hpp                   # ComputeGraph, GraphNode, OpType enum
        builder.hpp                 # GraphBuilder (Qwen2/Qwen3 builders)
        shape.hpp                   # ShapeTemplate, ShapeDim, ExecutionContext
      models/
        base.hpp                    # Model, ModelConfig, ModelWeights
        qwen2.hpp                   # Qwen2Model, Qwen2Config
        qwen3.hpp                   # Qwen3Model, Qwen3Config
      loader/
        interface.hpp               # IModelLoader, IModelFile, TensorInfo
        safetensors.hpp             # SafeTensorsLoader (mmap-based)
      tokenizer/
        base.hpp                    # Tokenizer, Config, SpecialTokens
        hf_tokenizer.hpp            # HFTokenizer (BPE with ICU regex)
        byte_level.hpp              # ByteLevelDecoder
      sampler/sampler.hpp           # Sampler (Argmax, General with temp/top-k/top-p)
    utils/
      types.hpp                     # Type utilities (bf16_t, fp16_t, cast, dsize)
      check.hpp                     # Assertion/check macros
      logging.hpp                   # Logging utilities
      ops.hpp                       # Operator utility types
      system_info.hpp               # System information queries
      nvidia/                       # CUDA utility headers (common, math, memory, types)
  src/                              # Implementations matching include/ structure
  examples/
    bench.cpp                       # Benchmarking (prefill/decode latency measurement)
    chat.cpp                        # Interactive multi-turn chat with readline
    ping.cpp                        # Quick single-turn test
  tests/
    core/test_memory_pool.cpp       # Memory pool tests
    core/test_storage.cpp           # Storage tests
    loader/test_safetensors.cpp     # SafeTensors loader tests
    tensor/test_tensor.cpp          # Tensor tests
    tokenizer/test_hf.cpp           # Tokenizer tests
    python/bindings/zedinfer_ops.cpp  # Python operator bindings
  third_party/include/              # Header-only dependencies
  xmake.lua                        # Build configuration
  xmake/                            # Modular build scripts (device, tests, examples, etc.)
```

## Entry Points

| Binary | Source | Purpose |
|--------|--------|---------|
| `bench` | `examples/bench.cpp` | Performance benchmarking (configurable prefill/decode lengths, rounds, device) |
| `chat` | `examples/chat.cpp` | Interactive multi-turn chat with readline (hardcoded NVIDIA device) |
| `ping` | `examples/ping.cpp` | Quick single-turn test (hardcoded model path and NVIDIA device) |

## Supported Models

| Model | Config Class | Builder | Weight Names |
|-------|-------------|---------|--------------|
| Qwen2 (DeepSeek-R1-Distill-Qwen-1.5B) | `Qwen2Config` | `Qwen2GraphBuilder` | q/k/v with bias, no per-head norm |
| Qwen3 (DeepSeek-R1-0528-Qwen3-8B) | `Qwen3Config` | `Qwen3GraphBuilder` | q/k/v without bias, per-head q_norm/k_norm |

## Operator Inventory

| Operator | CPU | NVIDIA | Notes |
|----------|-----|--------|-------|
| `add` | Yes (+ AVX optimized variant) | Yes | Element-wise addition |
| `argmax` | Yes | Yes | Used by ArgmaxSampler |
| `embedding` | Yes | Yes | Token embedding lookup |
| `linear` | Yes (handwritten matmul/vecmul) | Yes (custom PTX kernels) | BF16/FP16 convert to FP32 on CPU |
| `rearrange` | Yes | - | Used for contiguous() copy |
| `rms_norm` | Yes (with sdot helper) | Yes | RMS normalization |
| `rope` | Yes | Yes | Rotary position encoding |
| `self_attention` | Yes (naive GQA + causal mask) | Yes (custom CUDA kernels) | FP32 accumulation |
| `swiglu` | Yes | Yes | SiLU(gate) * up |

## Test Coverage

- `test_memory_pool` - BestFitMemoryPool allocation/deallocation/fragmentation
- `test_storage` - Storage creation and device affinity
- `test_safetensors` - SafeTensors file loading and tensor extraction
- `test_tensor` - Tensor creation, view, slice, permute, contiguous
- `test_hf` - HFTokenizer encode/decode

No integration tests. No operator correctness tests. No end-to-end generation tests.
