# Risk Register

## R1: Attention Redesign Scope

**Category**: Technical
**Likelihood**: High
**Impact**: High

**Description**: The current `self_attention` operator (`src/backend/ops/self_attention/`) is a naive implementation with tight assumptions: contiguous K/V buffers, single-sequence, non-batched. Paged attention, continuous batching, and FlashAttention integration each independently require substantial changes to this operator. Combined, they may require a full rewrite of the attention path.

**Affected files**:
- `src/backend/ops/self_attention/cpu/self_attention_cpu.cpp` - Full rewrite needed
- `src/backend/ops/self_attention/nvidia/self_attention_nvidia.cu` - Full rewrite needed
- `src/zedinfer/executor.cpp:150-178` - KV cache injection logic
- `include/backend/kvcache/` - New paged cache interface

**Mitigation**:
- Design the new attention interface once, covering all three features (paged, batched, efficient)
- Stage implementation: first non-paged batched attention, then paged attention on top
- Use well-tested reference implementations (FlashAttention-2/3 for GPU, xformers-style for CPU)
- Maintain the current naive implementation as a correctness reference during the transition

---

## R2: Thread Safety of Engine/Executor

**Category**: Technical
**Likelihood**: High
**Impact**: Medium

**Description**: `InferenceEngine::last_stats_` (`engine.hpp:92`) is mutable and written during `generate_tokens()`. If multiple sessions call `generate()` concurrently on the same engine, data races occur. Additionally, `BestFitMemoryPool` (`memory_pool.hpp:62`) is documented as single-threaded. The `core::Context` is thread-local, so different threads get different runtimes, but within a thread, concurrent coroutine-style execution would conflict.

**Affected files**:
- `include/zedinfer/engine.hpp:92` - `last_stats_` member
- `include/backend/core/memory/memory_pool.hpp:62` - Single-threaded pool
- `src/zedinfer/executor.cpp:65` - Local activation map (safe per call, but pool is shared)

**Mitigation**:
- Move `last_stats_` to per-session or per-request scope
- Either make memory pool thread-safe (mutex) or use per-thread pools
- Define a clear threading model: one GPU stream per request, or serialized through a scheduler
- Document thread safety contracts for each class

---

## R3: Memory Allocation Overhead in Hot Path

**Category**: Performance
**Likelihood**: High
**Impact**: Medium

**Description**: Every `GraphExecutor::forward()` call allocates fresh tensors for each compute node's output (`executor.cpp:183-189`). For decode phase (single token, 60+ nodes per layer, 28+ layers), this means 500+ allocations per token. Even with the memory pool, the allocation/deallocation overhead is non-trivial.

**Mitigation**:
- Replace graph execution with direct model forward using pre-allocated scratch buffers
- Or implement graph-level memory planning (`optimize_memory()`) to reuse buffers
- Pre-allocate a fixed set of activation buffers sized for max sequence length
- The direct forward approach is simpler and eliminates this entirely

---

## R4: KV Cache Growth Stalls

**Category**: Performance
**Likelihood**: Medium
**Impact**: High

**Description**: `DynamicKVCache::grow_cache()` (`dynamic.cpp:61-97`) allocates new tensors and copies all existing data. For a 8B model with BF16, growing from 256 to 512 tokens copies `28 layers * 2 (K+V) * 512 * 8 * 128 * 2 bytes = ~28 MB`. Larger sequences mean larger copies. This causes latency spikes during generation.

**Mitigation**:
- Paged KV cache eliminates growth entirely (allocate new blocks, no copy)
- Short-term: Pre-allocate generous initial capacity to reduce growth events
- Consider the "aggressive" growth strategy to minimize growth count

---

## R5: cuBLAS / oneDNN Integration Complexity

**Category**: Technical
**Likelihood**: Medium
**Impact**: Medium

**Description**: Replacing handwritten linear kernels with cuBLAS/oneDNN introduces external library dependencies. cuBLAS requires careful handle management and workspace allocation. oneDNN requires memory descriptor setup. Both require dtype-specific dispatch logic different from the current simple function-pointer approach.

**Affected files**:
- `src/backend/ops/linear/cpu/linear_cpu.cpp` - Replace matmul/vecmul calls
- `src/backend/ops/linear/nvidia/linear_nvidia.cu` - Replace custom kernels
- `xmake.lua` - Add library dependencies

**Mitigation**:
- Wrap cuBLAS/oneDNN behind a thin abstraction layer matching the current `ops::linear` signature
- Initialize handles at engine creation time, not per-call
- Keep current kernels as fallback during transition
- Benchmark before and after to verify improvement

---

## R6: Chat Template Fragility

**Category**: Technical
**Likelihood**: Medium
**Impact**: Low

**Description**: `InferenceSession::chat()` (`session.cpp:38-65`) hardcodes the DeepSeek-R1 chat template with specific Unicode tokens (`<begin_of_sentence>`, `<User>`, `<Assistant><think>`). This breaks for any model that uses a different chat format.

**Mitigation**:
- Extract chat template to a configuration (part of model config or tokenizer config)
- Support Jinja2-style templates or a simple template DSL
- This is low priority compared to other changes but should be addressed before adding new model families

---

## R7: Regression Risk During Refactoring

**Category**: Process
**Likelihood**: High
**Impact**: High

**Description**: Current test coverage is limited to unit tests for memory pool, storage, tensor, tokenizer, and SafeTensors loader. There are no end-to-end generation correctness tests, no operator correctness tests against reference implementations, and no performance regression tests.

**Mitigation**:
- Before starting any major refactoring, establish:
  1. Operator correctness tests (compare output against PyTorch reference for each op)
  2. End-to-end generation test (compare generated text for fixed prompt + seed)
  3. Performance baseline (prefill/decode latency for standard configurations)
- Run these tests after every significant change
- The existing `bench` example can serve as the performance baseline tool
- Python operator test bindings (`tests/python/bindings/zedinfer_ops.cpp`) suggest this was planned

---

## R8: Paged Attention Kernel Complexity

**Category**: Technical
**Likelihood**: Medium
**Impact**: High

**Description**: Paged attention kernels are significantly more complex than standard attention. They require indirect memory access through a block table, which complicates GPU kernel optimization (memory coalescing, occupancy). Custom CUDA paged attention kernels are error-prone and hard to optimize.

**Mitigation**:
- Consider using vLLM's paged attention kernels (Apache 2.0 licensed) as a starting point
- For CPU: implement paged attention as a modification of the existing naive kernel first
- Maintain non-paged attention as a correctness reference
- Extensive testing with varying sequence lengths and block sizes

---

## R9: Quantization Accuracy Degradation

**Category**: Technical
**Likelihood**: Medium
**Impact**: Medium

**Description**: INT8 and especially INT4 quantization can degrade model output quality. The degree of degradation varies by model, quantization method (per-tensor vs per-channel vs per-group), and calibration data. Without careful validation, quantized models may produce noticeably worse results.

**Mitigation**:
- Start with well-established quantization formats (GPTQ, AWQ) that have known quality characteristics
- Implement per-group quantization for INT4 (better quality than per-channel)
- Build a quality evaluation pipeline (perplexity on standard benchmarks)
- Support multiple quantization configs to let users trade quality vs speed

---

## R10: Heterogeneous Inference Transfer Overhead

**Category**: Performance
**Likelihood**: Medium
**Impact**: High

**Description**: CPU-GPU data transfers over PCIe are slow (12-16 GB/s bidirectional for PCIe 4.0 x16). If expert weights are moved between CPU and GPU during inference, the transfer time may dominate over the compute time saved, especially for small experts.

**Mitigation**:
- Use pinned (page-locked) host memory for all CPU-resident weights
- Implement async prefetch: start loading next expert while computing current one
- Track expert access patterns and keep hot experts on GPU
- Only offload when VRAM is genuinely insufficient
- Benchmark transfer overhead vs compute gain for target model/hardware configurations

---

## R11: Build System Complexity with New Dependencies

**Category**: Process
**Likelihood**: Low
**Impact**: Medium

**Description**: Adding cuBLAS, oneDNN, FlashAttention, and an HTTP library significantly increases build complexity. Cross-platform support (Linux/Windows) may be harder to maintain. Some dependencies may conflict or require specific compiler versions.

**Mitigation**:
- Make all new dependencies optional (similar to `--nv-gpu` flag)
- Use xmake's package management for external libraries
- Pin versions of critical dependencies
- Test both CPU-only and GPU builds in CI

---

## Risk Summary

| Risk | Likelihood | Impact | Priority | Mitigation Status |
|------|-----------|--------|----------|-------------------|
| R1: Attention redesign scope | High | High | Critical | Design-first approach |
| R2: Thread safety | High | Medium | High | Architecture decision needed |
| R3: Allocation overhead | High | Medium | High | Direct forward approach |
| R4: KV cache growth stalls | Medium | High | High | Paged cache resolves |
| R5: Library integration | Medium | Medium | Medium | Incremental migration |
| R6: Chat template | Medium | Low | Low | Configuration-based templates |
| R7: Regression risk | High | High | Critical | Test infrastructure first |
| R8: Paged attention complexity | Medium | High | High | Reference implementations |
| R9: Quantization accuracy | Medium | Medium | Medium | Established formats |
| R10: Transfer overhead | Medium | High | Medium | Pinned memory + prefetch |
| R11: Build complexity | Low | Medium | Low | Optional dependencies |
