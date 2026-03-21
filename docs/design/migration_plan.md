# Migration Plan

## Guiding Principles

1. Each stage is independently testable and benchmarkable
2. No stage breaks existing functionality until the replacement is validated
3. Performance is measured before and after each stage
4. Each stage has a rollback approach
5. Stages are ordered to maximize incremental value

## Stage 0: Test Infrastructure

**Goal**: Establish correctness and performance baselines before any refactoring.

**Scope**:
- Operator correctness tests against PyTorch reference
- End-to-end generation snapshot tests (greedy, fixed prompts)
- Performance baselines for all standard benchmark configs
- Extend Python bindings for operator testing

**Files affected**:
- New: `tests/ops/`, `tests/e2e/`, `tests/perf/`, `scripts/benchmark.sh`
- Modified: `tests/python/bindings/zedinfer_ops.cpp` (expose more ops)
- Modified: `xmake.lua` or `xmake/tests.lua` (new test targets)

**Public interfaces changed**: None

**Test plan**: The tests ARE the deliverable. Validate they pass on current code.

**Benchmark plan**: Capture baseline for all configs (A-E) on both models.

**Risks**: Model paths may differ across servers. Use environment variable.

**Rollback**: N/A (additive only)

---

## Stage 1: Direct Model Forward

**Goal**: Replace graph-based execution with direct `forward()` for Qwen2/Qwen3. Eliminate per-forward allocation overhead.

**Scope**:
- Add `forward()` method to `Qwen2Model` and `Qwen3Model`
- Implement `ScratchBuffers` allocation
- Modify `InferenceEngine` to call `model->forward()` instead of `executor->forward()`
- Keep graph code intact but unused for these models

**Files affected**:
- New: `src/frontend/models/qwen2_forward.cpp`, `qwen3_forward.cpp`
- New: `include/zedinfer/scratch.hpp`, `src/zedinfer/scratch.cpp`
- Modified: `include/frontend/models/base.hpp` (add `forward()` virtual)
- Modified: `include/frontend/models/qwen2.hpp`, `qwen3.hpp` (add `forward()`)
- Modified: `include/zedinfer/engine.hpp`, `src/zedinfer/engine.cpp` (call model->forward)

**Public interfaces changed**:
- `Model` gains `forward()` virtual method
- `InferenceEngine` constructor changes (no longer needs GraphExecutor)

**Test plan**:
- E2E generation test: output must be bit-identical to graph-based path
- Operator output: same (forward calls same ops::* functions)

**Benchmark plan**: Config A-E before/after. Expect 5-15% decode latency improvement.

**Risks**: Off-by-one in tensor view dimensions. Mitigated by bit-identical E2E test.

**Rollback**: Revert engine to call executor->forward(). Graph code still intact.

---

## Stage 2: Operator Performance

**Goal**: Replace handwritten kernels with optimized libraries.

### Stage 2a: cuBLAS for NVIDIA Linear

**Scope**:
- Integrate cuBLAS for `ops::nvidia::linear()`
- cuBLAS handle created per Runtime, reused across calls
- Keep custom kernels as fallback (#ifdef)

**Files affected**:
- Modified: `src/backend/ops/linear/nvidia/linear_nvidia.cu`
- Modified: `src/backend/core/runtime/runtime.cpp` (cuBLAS handle lifecycle)
- Modified: `include/backend/core/runtime/runtime.hpp` (cuBLAS handle member)
- Modified: `xmake.lua` (link cuBLAS)

**Test plan**: Operator correctness: linear output within tolerance of current output.

**Benchmark plan**: Operator-level linear benchmark (decode M=1, prefill M=128/512). Config B E2E.

**Rollback**: `#ifdef USE_CUBLAS` / `#else` current kernels.

### Stage 2b: oneDNN for CPU Linear

**Scope**:
- Integrate oneDNN for `ops::cpu::linear()`
- Handle initialization, memory descriptors
- BF16/FP16 support via oneDNN's built-in conversion

**Files affected**:
- Modified: `src/backend/ops/linear/cpu/linear_cpu.cpp`
- Modified: `xmake.lua` (add oneDNN dependency, optional)

**Test plan**: Same as 2a for CPU.

**Benchmark plan**: CPU operator benchmarks. CPU E2E.

**Rollback**: `#ifdef USE_ONEDNN` / `#else` current handwritten kernels.

### Stage 2c: CPU Attention Improvement

**Scope**: Tiled/blocked attention with better cache locality.

**Files affected**: `src/backend/ops/self_attention/cpu/self_attention_cpu.cpp`

---

## Stage 3: Multi-User Foundation

**Goal**: Make engine safe for concurrent session access.

**Scope**:
- Remove `last_stats_` from `InferenceEngine` (return stats per-call)
- Add `GenerationResult` return type
- Add `InferenceRequest` struct
- Add `SessionRegistry` for session management
- Extract chat template to configuration

**Files affected**:
- New: `include/zedinfer/request.hpp`, `include/zedinfer/session_registry.hpp`
- New: `include/zedinfer/chat_template.hpp`
- Modified: `include/zedinfer/engine.hpp` (remove last_stats_, new return type)
- Modified: `src/zedinfer/engine.cpp` (return GenerationResult)
- Modified: `src/zedinfer/session.cpp` (use chat template config)
- Modified: `examples/chat.cpp`, `examples/ping.cpp` (adapt to new API)

**Public interfaces changed**: `generate_tokens()` returns `GenerationResult` instead of `vector<int>`.

**Test plan**: E2E tests pass. Multi-session test (2 sessions, verify independent output).

**Benchmark plan**: Config B. No regression expected.

**Rollback**: Revert engine API to old signature.

---

## Stage 4: Scheduler Foundation

**Goal**: Request queue and simple single-request scheduler.

**Scope**:
- Implement `Scheduler` class (single-request mode)
- Submit/schedule/process_results lifecycle
- Wire scheduler into engine

**Files affected**:
- New: `include/zedinfer/scheduler.hpp`, `src/zedinfer/scheduler.cpp`
- Modified: `src/zedinfer/engine.cpp` (use scheduler for generate)

**Test plan**: Same E2E results as without scheduler (single-request mode).

**Benchmark plan**: Config B. No regression expected (scheduler overhead < 0.1ms).

---

## Stage 5: Continuous Batching

**Goal**: Multiple sequences in a single forward pass.

**Scope**:
- `BatchContext` assembly in scheduler
- `Model::forward_batch()` implementation
- Batch-aware operator kernels (primarily attention)
- Decode-first scheduling with chunked prefill

**Files affected**:
- New: `include/zedinfer/batch_context.hpp`
- Modified: `include/frontend/models/base.hpp` (add forward_batch)
- Modified: `src/frontend/models/qwen2_forward.cpp`, `qwen3_forward.cpp` (batch path)
- Modified: `src/backend/ops/self_attention/` (batched attention)
- Modified: `src/zedinfer/scheduler.cpp` (batch assembly)

**Public interfaces changed**: `Model::forward_batch()`, new `BatchContext` type.

**Test plan**: Multi-session correctness (4 concurrent, verify each). Batch throughput test.

**Benchmark plan**: MU-A, MU-B configs. Expect 2-4x throughput improvement.

**Risks**: Batch attention correctness with variable sequence lengths. Mitigated by per-sequence verification.

---

## Stage 6: Paged KV Cache

**Goal**: Block-based KV cache management.

**Scope**:
- `BlockPool`, `BlockAllocator`, `PagedKVCache`
- `SequenceKVMeta` block table per request
- Scheduler integration (block allocation on admit, free on complete)
- Contiguous fallback for non-paged attention (initially)

**Files affected**:
- New: `include/backend/kvcache/paged.hpp`, `src/backend/kvcache/paged.cpp`
- New: `include/backend/kvcache/block_pool.hpp`, `src/backend/kvcache/block_pool.cpp`
- Modified: `src/zedinfer/scheduler.cpp` (block allocation)
- Modified: `src/zedinfer/engine.cpp` (create block allocator)

**Test plan**: Block pool unit tests. KV cache append/read correctness. E2E with paged cache.

**Benchmark plan**: Config D (long context). Measure memory usage improvement.

**Rollback**: `DynamicKVCache` remains usable.

---

## Stage 7: Paged Attention Kernels

**Goal**: Attention kernels that read K/V from non-contiguous blocks.

**Scope**:
- `paged_attention_decode()` for GPU (adapt vLLM kernels)
- `paged_attention_prefill()` for GPU
- CPU paged attention (block-iterating variant)
- Remove contiguous fallback from PagedKVCache

**Files affected**:
- New: `src/backend/ops/self_attention/nvidia/paged_attention_nvidia.cu`
- New: `src/backend/ops/self_attention/cpu/paged_attention_cpu.cpp`
- Modified: `include/backend/ops/ops.hpp` (new function signatures)
- Modified: `src/frontend/models/qwen2_forward.cpp`, `qwen3_forward.cpp` (use paged attention)

**Test plan**: Paged attention output matches non-paged reference for same inputs.

**Benchmark plan**: Config A, D. Compare decode latency against non-paged.

---

## Stage 8: HTTP / OpenAPI Server

**Goal**: Serve inference requests via HTTP.

**Scope**:
- cpp-httplib integration
- `/v1/chat/completions` (streaming and non-streaming)
- `/v1/models`, `/health`
- `examples/serve.cpp` entry point

**Files affected**:
- New: `include/zedinfer/http_server.hpp`, `src/zedinfer/http_server.cpp`
- New: `include/zedinfer/api_types.hpp`
- New: `examples/serve.cpp`
- New: `third_party/include/httplib.h` (or xmake package)

**Test plan**: HTTP endpoint tests with curl. Streaming correctness.

**Benchmark plan**: HTTP load test (wrk/hey). Config MU-B via HTTP.

---

## Stage 9: INT8 Quantization

**Goal**: INT8 weight-only quantization support.

**Scope**:
- `QuantizationConfig` in model config
- `QuantizedLinearWeight` storage
- INT8 weight loading from SafeTensors
- `ops::linear_quantized()` dispatch (CPU + GPU)
- GPU: cuBLAS INT8 GEMM
- CPU: oneDNN INT8 or dequant-multiply

**Files affected**:
- New: `include/backend/tensor/quantized_tensor.hpp`
- New: `src/backend/ops/linear/cpu/linear_int8.cpp`
- New: `src/backend/ops/linear/nvidia/linear_int8.cu`
- Modified: `include/frontend/models/base.hpp` (QuantizationConfig)
- Modified: `src/frontend/models/base.cpp` (quantized loading)
- Modified: `include/backend/ops/ops.hpp` (linear_quantized)

**Test plan**: INT8 output quality tests (KL divergence < 0.01). E2E with INT8 model.

**Benchmark plan**: All configs. Expect ~50% memory reduction, ~1.5x throughput.

---

## Stage 10: INT4 Quantization

**Goal**: INT4 GPTQ/AWQ quantization support.

**Scope**:
- Add `ZEDINFER_DTYPE_I4`
- GPTQ weight unpacking
- INT4 dequant kernels (CPU + GPU)
- AWQ format support

**Files affected**:
- Modified: `include/zedinfer.h` (new dtype)
- New: `src/backend/ops/linear/cpu/linear_int4.cpp`
- New: `src/backend/ops/linear/nvidia/linear_int4.cu`
- New: `src/frontend/loader/gptq_loader.cpp`

**Test plan**: INT4 quality tests (KL divergence < 0.05). E2E with GPTQ model.

**Benchmark plan**: All configs. Expect ~75% memory reduction.

---

## Stage 11: Heterogeneous CPU/GPU Inference

**Goal**: Mixed device execution with pinned memory transfers.

**Scope**:
- `malloc_pinned` / `free_pinned` in RuntimeAPI
- Per-layer device placement
- Async transfer with compute overlap

**Files affected**:
- Modified: `include/backend/device/runtime_api.hpp` (pinned memory)
- Modified: `src/backend/device/nvidia/nvidia_runtime_api.cu`
- Modified: `src/frontend/models/base.cpp` (per-layer placement)

---

## Stage 12: MoE Expert Offloading

**Goal**: Run MoE models with GPU + CPU expert weight management.

**Scope**:
- MoE model class
- Expert pool (GPU + CPU)
- Expert routing (top-K selection)
- Gather/scatter by expert
- Predictive prefetching

**Files affected**: New MoE module under `include/backend/moe/` and `src/backend/moe/`.

---

## Implementation Order Summary

```
Stage 0: Test Infrastructure                    [No code changes, additive]
    |
Stage 1: Direct Model Forward                  [Core refactor]
    |
Stage 2a: cuBLAS for GPU Linear               [Operator improvement]
Stage 2b: oneDNN for CPU Linear               [Parallel with 2a]
    |
Stage 3: Multi-User Foundation                 [API changes]
    |
Stage 4: Scheduler Foundation                  [New component]
    |
Stage 5: Continuous Batching                   [Major feature]
    |
Stage 6: Paged KV Cache                       [Memory management]
    |
Stage 7: Paged Attention Kernels              [Kernel work]
    |
Stage 8: HTTP / OpenAPI Server                 [Integration]
    |
Stage 9: INT8 Quantization                    [Extension]
    |
Stage 10: INT4 Quantization                   [Extension]
    |
Stage 11: Heterogeneous CPU/GPU               [Advanced]
    |
Stage 12: MoE Expert Offloading               [Advanced]
```

Each stage can be merged independently. Stages 2a/2b can proceed in parallel. The critical path is: 0 -> 1 -> 3 -> 4 -> 5 -> 6 -> 7.
