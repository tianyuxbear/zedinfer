# Implementation Plan

> Detailed, PR-sized stages. Each stage is independently mergeable, testable, and benchmarkable.
> No stage breaks the existing `bench`, `chat`, or `ping` binaries until its replacement is validated.
>
> Reference designs live under `docs/design/`. Repository analysis under `docs/notes/`.

---

## Ordering Rationale

The user-suggested order is:

1. Simplify model execution path
2. Request / session abstraction
3. Scheduler
4. Continuous batching
5. Paged KV cache
6. Paged attention operator path
7. HTTP API
8. INT8 quantization
9. INT4 quantization
10. Heterogeneous MoE support
11. Backend / kernel optimization

I propose moving **kernel optimization (cuBLAS / oneDNN)** earlier, between steps 1 and 2,
for two reasons specific to this codebase:

- The current handwritten `linear` on GPU (`src/backend/ops/linear/nvidia/linear_nvidia.cu`)
  and CPU (`src/backend/ops/linear/cpu/linear_cpu.cpp`) are significant bottlenecks. Replacing
  them has **zero coupling** to any other stage and produces an immediate, measurable win.
- Every later stage (batching, paged attention, quantization) benefits from a fast linear baseline;
  without it, benchmarks conflate linear slowness with new-feature overhead.

The resulting order is:

```
PR-1   Test infrastructure & baselines
PR-2   Direct model forward (bypass graph)
PR-3a  cuBLAS for NVIDIA linear           \
PR-3b  oneDNN for CPU linear               > parallelizable
PR-4   Stateless engine & request types
PR-5   Chat template extraction
PR-6   Scheduler (single-request)
PR-7   Continuous batching
PR-8   Paged KV cache
PR-9   Paged attention kernels
PR-10  HTTP / OpenAPI server
PR-11  INT8 quantization
PR-12  INT4 quantization
PR-13  Heterogeneous inference & pinned memory
PR-14  MoE model support & expert offloading
```

---

## PR-1: Test Infrastructure and Performance Baselines

### Objective

Establish operator correctness tests, an end-to-end generation snapshot test, and
record latency baselines, all before any refactoring begins.

### Affected Files

| Action | File |
|--------|------|
| New | `tests/ops/test_linear_correctness.cpp` |
| New | `tests/ops/test_attention_correctness.cpp` |
| New | `tests/ops/test_elementwise_correctness.cpp` |
| New | `tests/e2e/test_generation_snapshot.cpp` |
| New | `tests/e2e/reference/qwen2_1.5b_greedy.json` |
| New | `tests/e2e/reference/qwen3_8b_greedy.json` |
| New | `scripts/benchmark.sh` |
| New | `scripts/generate_reference.py` (one-time, offline) |
| Modify | `tests/python/bindings/zedinfer_ops.cpp` — expose `ops::linear`, `ops::self_attention`, `ops::rms_norm`, `ops::rope`, `ops::swiglu` |
| Modify | `xmake/tests.lua` — new test targets `test-ops`, `test-e2e` |

### Interfaces Added / Changed / Removed

None. This PR is purely additive.

### Dependency

None (first PR).

### Correctness Tests

- Per-operator: generate random inputs on CPU, run through `ops::*`, compare against
  PyTorch reference (via Python bindings or saved expected outputs).
  Tolerances: FP32 1e-5, BF16/FP16 1e-2.
- Shapes: seq_len {1, 128, 512}, hidden {1536, 4096}, covering both Qwen2-1.5B and Qwen3-8B dimensions.

### Regression Tests

- E2E snapshot: load model, run greedy generation on `"What is 2+2?"` for 32 tokens,
  compare output token IDs against `tests/e2e/reference/*.json`.

### Benchmarks

Capture baselines with `scripts/benchmark.sh` for configs A-E
(see `docs/design/benchmark_plan.md`).
Save results under `benchmarks/baseline/`.

### Risks

- Model files may not be present in CI. Use `ZEDINFER_TEST_MODEL_PATH` env var;
  skip E2E if unset.
- Python bindings build may fail if pybind11 config is stale.

### Rollback

Additive only — delete new files.

---

## PR-2: Direct Model Forward (Bypass Graph)

### Objective

Add `Model::forward()` virtual method and concrete implementations in `Qwen2Model` /
`Qwen3Model`. Pre-allocate `ScratchBuffers` at engine init. Wire `InferenceEngine` to
call `model_->forward()` instead of `executor_->forward()`.

The graph code (`frontend/graph/`, `zedinfer/executor.hpp`) stays in the tree but is
no longer on the inference hot path.

### Affected Files

| Action | File |
|--------|------|
| New | `include/zedinfer/scratch.hpp` — `ScratchBuffers` struct + `ScratchBuffers::allocate()` |
| New | `src/zedinfer/scratch.cpp` — allocation implementation |
| New | `src/frontend/models/qwen2_forward.cpp` — `Qwen2Model::forward()` |
| New | `src/frontend/models/qwen3_forward.cpp` — `Qwen3Model::forward()` |
| Modify | `include/frontend/models/base.hpp:68` — add `virtual tensor_t forward(...)` |
| Modify | `include/frontend/models/qwen2.hpp` — declare `forward()` override |
| Modify | `include/frontend/models/qwen3.hpp` — declare `forward()` override |
| Modify | `include/zedinfer/engine.hpp` — add `ScratchBuffers scratch_` member; keep `executor_` for warmup/profile but stop using it for generation |
| Modify | `src/zedinfer/engine.cpp:183-269` — `generate_tokens()` calls `model_->forward()` |
| Modify | `xmake/frontend.lua` or `xmake.lua` — add new .cpp files to `models` target |

### Interfaces Added / Changed / Removed

**Added:**

```cpp
// include/frontend/models/base.hpp
class Model {
    // ... existing ...
    virtual tensor_t forward(
        const std::vector<int> &input_ids,
        int past_len,
        kvcache::KVCache &kvcache,
        ScratchBuffers &scratch,
        const ExecutorConfig &config) = 0;
};
```

```cpp
// include/zedinfer/scratch.hpp
struct ScratchBuffers {
    tensor_t hidden;     // [max_tokens, hidden_size]
    tensor_t norm_buf;   // [max_tokens, hidden_size]
    tensor_t q_proj;     // [max_tokens, hidden_size]
    tensor_t attn_out;   // [max_tokens, hidden_size]
    tensor_t gate;       // [max_tokens, intermediate_size]
    tensor_t up;         // [max_tokens, intermediate_size]
    tensor_t logits;     // [max_tokens, vocab_size]

    static ScratchBuffers allocate(
        const model::ModelConfig &config,
        int max_tokens,
        zedinferDeviceType_t device,
        int device_id,
        zedinferDataType_t dtype);
};
```

**Changed:**

- `InferenceEngine` constructor: `graph_` and `executor_` become optional
  (kept for `warmup`/`profile`, set to nullptr for new forward path).
- `generate_tokens()` no longer calls `executor_->forward()`.

**Removed from hot path** (not deleted):

- `GraphExecutor::forward()` — still exists, just not called during generation.

### Dependency

PR-1 (need snapshot tests to verify bit-identical output).

### Correctness Tests

- Run E2E snapshot test (`tests/e2e/test_generation_snapshot.cpp`) — output token IDs
  must be **bit-identical** to PR-1 reference. This is the single most important gate.
- Run operator correctness tests (unchanged, sanity check).

### Regression Tests

- All existing GTest targets must still pass (`test-tokenizer`, `test-loader`,
  `test-memorypool`, `test-storage`, `test-tensor`).
- `bench`, `chat`, `ping` must still compile and run.

### Benchmarks

Before/after on configs A-E for both models.
Expected: 5-15 % decode latency improvement from eliminating per-node allocation.

### Risks

- Tensor view dimensions may differ from graph executor's implicit reshaping
  (e.g., `executor.cpp:161-170` reshapes q_norm/k_norm outputs). Mitigated by
  bit-identical E2E test.
- Scratch buffer size may exceed GPU memory for very large `max_tokens`.
  Default `max_tokens = 2048`; configurable.

### Rollback

Set a boolean flag `use_direct_forward_` in engine. When false, fall back to
`executor_->forward()`. Both paths co-exist.

---

## PR-3a: cuBLAS for NVIDIA Linear

### Objective

Replace the custom PTX-based CUDA linear kernels with cuBLAS GEMM calls.
Keep custom kernels behind a compile flag as fallback.

### Affected Files

| Action | File |
|--------|------|
| Modify | `include/backend/core/runtime/runtime.hpp` — add `cublasHandle_t cublas_handle_` member |
| Modify | `src/backend/core/runtime/runtime.cpp` — create / destroy cuBLAS handle alongside stream |
| Modify | `src/backend/ops/linear/nvidia/linear_nvidia.cu` — call `cublasGemmEx` / `cublasLtMatmul` instead of custom kernels |
| Modify | `xmake/device/nvidia.lua` — link `-lcublas -lcublasLt` |

### Interfaces Added / Changed / Removed

**Added:**

- `Runtime::cublas_handle()` accessor (NVIDIA runtime only, guarded by `ENABLE_NVIDIA_API`).

**Changed:**

- Internal implementation of `ops::nvidia::linear()` — callers see no change.

**Removed:**

- Nothing. Custom kernels remain under `#ifndef USE_CUBLAS`.

### Dependency

PR-1 (operator correctness test for linear).

### Correctness Tests

- `tests/ops/test_linear_correctness.cpp` on NVIDIA: output within tolerance of CPU reference.
- E2E snapshot on GPU must still match.

### Regression Tests

Existing tests unaffected (cuBLAS is strictly internal).

### Benchmarks

Operator-level: `linear` for M={1, 128, 512}, N={1536, 4096, 11008}, K={1536, 4096}.
Expected: 2-5x speedup, especially for prefill (large M).

### Risks

- cuBLAS workspace memory. Use `cublasSetWorkspace()` with a pre-allocated buffer.
- Column-major vs row-major confusion. Current tensors are row-major; cuBLAS expects
  column-major. Handle via transposition flags in `cublasGemmEx`.

### Rollback

`#ifndef USE_CUBLAS` restores custom kernels. Build flag in `xmake.lua`.

---

## PR-3b: oneDNN for CPU Linear

### Objective

Replace handwritten `matmul()` / `vecmul()` with oneDNN GEMM primitives for
FP32, and potentially BF16 native GEMM (eliminating the current convert-compute-convert path
in `linear_cpu.cpp:30-78`).

### Affected Files

| Action | File |
|--------|------|
| Modify | `src/backend/ops/linear/cpu/linear_cpu.cpp` — call oneDNN instead of `matmul`/`vecmul` |
| Modify | `xmake.lua` — add optional `onednn` dependency |
| Keep | `src/backend/ops/linear/cpu/matmul.cpp`, `vecmul.cpp` — retained as fallback |

### Interfaces Added / Changed / Removed

None externally. Only internal dispatch changes.

### Dependency

PR-1 (operator correctness test for CPU linear).

### Correctness Tests

Same as PR-3a but on CPU device.

### Benchmarks

CPU-only configs A-E. Operator-level for same shapes as PR-3a.
Expected: 2-5x for FP32; larger gains for BF16 (native BF16 GEMM vs convert path).

### Risks

- oneDNN adds a non-trivial dependency (~50 MB). Make it optional
  (`#ifdef USE_ONEDNN`). Document the zero-dependency fallback.
- Thread contention: oneDNN uses OpenMP internally; may conflict with
  existing `#pragma omp` in other operators. Set `OMP_NUM_THREADS` consistently.

### Rollback

`#ifndef USE_ONEDNN` restores handwritten kernels.

---

## PR-4: Stateless Engine and Request Types

### Objective

Make `InferenceEngine` truly stateless by removing `last_stats_` and returning
results per-call. Introduce `InferenceRequest` and `GenerationResult` types
that will later plug into the scheduler.

### Affected Files

| Action | File |
|--------|------|
| New | `include/zedinfer/request.hpp` — `InferenceRequest`, `GenerationResult` |
| Modify | `include/zedinfer/engine.hpp:71,91-92` — remove `last_stats_`, change `generate_tokens` return type |
| Modify | `src/zedinfer/engine.cpp:183-269` — return `GenerationResult` |
| Modify | `include/zedinfer/session.hpp` — `chat()` reads stats from `GenerationResult` |
| Modify | `src/zedinfer/session.cpp:38-65` — adapt to new engine API |
| Modify | `examples/bench.cpp` — adapt to new stats access |
| Modify | `examples/ping.cpp` — adapt to new stats access |

### Interfaces Added / Changed / Removed

**Added:**

```cpp
// include/zedinfer/request.hpp
struct GenerationResult {
    std::vector<int> output_ids;
    GenerationStats stats;
};

enum class RequestPhase { QUEUED, PREFILL, DECODE, COMPLETE };

struct InferenceRequest {
    uint64_t request_id;
    std::string session_id;
    std::vector<int> input_ids;
    GenerationConfig config;
    RequestPhase phase = RequestPhase::QUEUED;
    int generated_count = 0;
    int last_token = -1;
    std::vector<int> output_ids;
};
```

**Changed:**

```cpp
// engine.hpp — BEFORE:
std::vector<int> generate_tokens(kvcache::KVCache &, const std::vector<int> &, const GenerationConfig &);
const GenerationStats &last_stats() const;

// engine.hpp — AFTER:
GenerationResult generate_tokens(kvcache::KVCache &, const std::vector<int> &, const GenerationConfig &);
// last_stats() removed
```

**Removed:**

- `InferenceEngine::last_stats_` member.
- `InferenceEngine::last_stats()` accessor.
- `InferenceEngine::update_stats_prefill()`, `update_stats_decode()` private helpers
  (stats are now local to `generate_tokens`).

### Dependency

PR-2 (direct forward must be working).

### Correctness Tests

- E2E snapshot: output tokens unchanged.
- `bench` still prints correct metrics.

### Regression Tests

All existing tests, plus `bench`, `chat`, `ping` compile and run.

### Benchmarks

Config B before/after. Expected: no measurable change.

### Risks

Low. Pure API reshaping.

### Rollback

Revert `engine.hpp` signature; re-add `last_stats_`.

---

## PR-5: Chat Template Extraction

### Objective

Replace the hardcoded DeepSeek-R1 chat template in `session.cpp:42-49` with
a configurable `ChatTemplate` loaded from the model directory or a default.

### Affected Files

| Action | File |
|--------|------|
| New | `include/zedinfer/chat_template.hpp` — `ChatTemplate` struct |
| Modify | `include/zedinfer/session.hpp` — add `ChatTemplate template_` member |
| Modify | `src/zedinfer/session.cpp:38-49` — use `template_` instead of literals |
| Modify | `src/zedinfer/engine.cpp` — load template in `create()`, pass to session |
| Modify | `include/frontend/models/base.hpp` — add optional `chat_template` field to `ModelConfig` |

### Interfaces Added / Changed / Removed

**Added:**

```cpp
struct ChatTemplate {
    std::string bos_token;
    std::string user_prefix;
    std::string user_suffix;
    std::string assistant_prefix;
    std::string assistant_suffix;
    bool add_bos_first_turn_only = true;

    static ChatTemplate load(const std::string &model_path);
    static ChatTemplate default_deepseek_r1();
};
```

### Dependency

PR-4.

### Correctness Tests

E2E snapshot still passes (default template matches current hardcoded one).

### Regression Tests

`chat` example still produces coherent output.

### Benchmarks

None (no performance impact).

### Risks

Minimal. Template mismatch could produce garbled output; mitigated by
`default_deepseek_r1()` fallback.

### Rollback

Revert to hardcoded strings.

---

## PR-6: Scheduler (Single-Request Mode)

### Objective

Introduce a `Scheduler` class that wraps the generate loop. Initially
processes one request at a time (no batching). This lays the groundwork
for continuous batching without changing any operator kernels.

### Affected Files

| Action | File |
|--------|------|
| New | `include/zedinfer/scheduler.hpp` — `Scheduler`, `SchedulerConfig` |
| New | `src/zedinfer/scheduler.cpp` — implementation |
| Modify | `include/zedinfer/engine.hpp` — add `scheduler_` member, new `submit()` method |
| Modify | `src/zedinfer/engine.cpp` — wire scheduler into `generate` / `generate_tokens` |

### Interfaces Added / Changed / Removed

**Added:**

```cpp
struct SchedulerConfig {
    int max_batch_tokens = 2048;
    int max_batch_requests = 64;
    int max_prefill_tokens = 512;
    int max_queue_size = 256;
};

class Scheduler {
public:
    Scheduler(SchedulerConfig config);
    void submit(std::unique_ptr<InferenceRequest> request);
    bool has_work() const;
    int pending_count() const;
    int active_count() const;

    // In single-request mode, runs the full generate loop for one request.
    // In batched mode (PR-7), assembles and returns a ScheduledBatch.
    GenerationResult run_one(
        model::Model &model,
        kvcache::KVCache &kvcache,
        ScratchBuffers &scratch,
        const ExecutorConfig &exec_config,
        sampler::Sampler &sampler,
        tokenizer::Tokenizer &tokenizer);
};
```

**Changed:**

- `InferenceEngine::generate_tokens()` delegates to `scheduler_->run_one()`.

### Dependency

PR-4 (request types).

### Correctness Tests

- E2E snapshot: identical output.
- Submit 3 sequential requests, verify independent results.

### Regression Tests

`bench`, `chat`, `ping` all still work.

### Benchmarks

Config B. Expected: < 0.5 ms overhead from scheduler indirection.

### Risks

Over-abstraction for single-request mode. Keep `run_one()` thin.

### Rollback

Remove scheduler, revert engine to direct generate loop.

---

## PR-7: Continuous Batching

### Objective

Extend the scheduler to assemble multi-sequence batches. Add `Model::forward_batch()`
and batch-aware attention. Decode-first scheduling with chunked prefill.

This is the largest single PR. It may be split further during implementation.

### Affected Files

| Action | File |
|--------|------|
| New | `include/zedinfer/batch_context.hpp` — `BatchContext`, `BatchSlot` |
| New | `src/zedinfer/batch_context.cpp` — assembly logic |
| Modify | `include/frontend/models/base.hpp` — add `virtual forward_batch()` |
| New | `src/frontend/models/qwen2_forward_batch.cpp` |
| New | `src/frontend/models/qwen3_forward_batch.cpp` |
| Modify | `include/zedinfer/scheduler.hpp` — `schedule()` returns `ScheduledBatch` |
| Modify | `src/zedinfer/scheduler.cpp` — batch assembly, decode-first, chunked prefill |
| Modify | `src/backend/ops/self_attention/cpu/self_attention_cpu.cpp` — batched attention (multi-sequence with offsets) |
| Modify | `src/backend/ops/self_attention/nvidia/self_attention_nvidia.cu` — batched attention |
| Modify | `include/backend/ops/ops.hpp` — add `self_attention_batched()` |

### Interfaces Added / Changed / Removed

**Added:**

```cpp
struct BatchSlot {
    InferenceRequest *request;
    int seq_offset;
    int seq_len;
};

struct ScheduledBatch {
    std::vector<BatchSlot> prefill_slots;
    std::vector<BatchSlot> decode_slots;
    int total_tokens() const;
};

struct BatchContext {
    std::vector<int> token_ids;          // flattened
    struct SlotInfo { int request_idx; int start_pos; int seq_len; int past_len; bool is_prefill; };
    std::vector<SlotInfo> slots;
    int total_tokens() const;
};

// ops.hpp
void self_attention_batched(
    tensor_t out, tensor_t q,
    const std::vector<tensor_t> &k_caches,
    const std::vector<tensor_t> &v_caches,
    const std::vector<int> &seq_lens,
    float scale);
```

```cpp
// Model
virtual tensor_t forward_batch(const BatchContext &batch, ...) = 0;
```

### Dependency

PR-6 (scheduler foundation), PR-2 (direct forward), PR-3a/3b (fast linear for meaningful batch benchmarks).

### Correctness Tests

- Run 4 requests concurrently with different prompts. Verify each output matches
  the single-request reference for that prompt.
- Compare batched attention output against sequential per-request attention.

### Regression Tests

Single-request path must still produce identical output (batch size 1 = no regression).

### Benchmarks

Multi-user configs MU-A (4 requests), MU-B (16 requests).
Expected: 2-4x throughput improvement vs sequential.

### Risks

- Batched attention correctness with variable sequence lengths.
- Prefill/decode mixing in a single forward pass.
- Memory for scratch buffers scales with `max_batch_tokens`.

### Rollback

Scheduler falls back to `run_one()` (single-request mode).

---

## PR-8: Paged KV Cache

### Objective

Implement block-based KV cache with fixed-size blocks, a block pool, and a block
allocator. The scheduler allocates/frees blocks per request.
Attention kernels in this PR still use a **contiguous gather** fallback
(copy blocks into a temp contiguous buffer before attention). True paged
attention kernels come in PR-9.

### Affected Files

| Action | File |
|--------|------|
| New | `include/backend/kvcache/block_pool.hpp` — `BlockConfig`, `BlockPool` |
| New | `src/backend/kvcache/block_pool.cpp` |
| New | `include/backend/kvcache/paged.hpp` — `PagedKVCache`, `SequenceKVMeta`, `BlockAllocator` |
| New | `src/backend/kvcache/paged.cpp` |
| Modify | `src/zedinfer/scheduler.cpp` — allocate blocks on admit, free on complete |
| Modify | `src/zedinfer/engine.cpp` — create `BlockAllocator` at init |

### Interfaces Added / Changed / Removed

**Added:**

```cpp
struct BlockConfig { int block_size; int num_kv_heads; int head_dim; ... };

class BlockPool {
    int allocate();
    void free(int block_id);
    std::byte* block_ptr(int block_id) const;
    int free_blocks() const;
};

class BlockAllocator {
    SequenceKVMeta allocate_initial(int num_layers, int estimated_tokens);
    int allocate_block();
    void free_sequence(SequenceKVMeta &meta);
    int available_blocks() const;
};

struct SequenceKVMeta {
    std::vector<std::vector<int>> k_block_table;  // [layer][block_idx]
    std::vector<std::vector<int>> v_block_table;
    int seq_len;
};

class PagedKVCache {
    void append(int layer, tensor_t k_new, tensor_t v_new, SequenceKVMeta &meta);
    tensor_t gather_contiguous_k(int layer, const SequenceKVMeta &meta);
    tensor_t gather_contiguous_v(int layer, const SequenceKVMeta &meta);
};
```

**Existing `DynamicKVCache` is NOT removed.** Sessions can use either.

### Dependency

PR-7 (scheduler manages per-request KV state).

### Correctness Tests

- Block pool: allocate all, free all, re-allocate. Verify no leaks.
- PagedKVCache: append tokens, gather contiguous, compare against `DynamicKVCache` slice.
- E2E with paged cache (via contiguous gather): output identical to non-paged.

### Regression Tests

Existing `DynamicKVCache` path still works.

### Benchmarks

Config D (long context). Measure peak memory vs `DynamicKVCache`.
Expected: ~30 % less peak memory (no over-allocation / growth copies).

### Risks

- Contiguous gather fallback negates most of the latency benefit. Acceptable because
  PR-9 adds true paged attention.
- Block fragmentation with many short sequences.

### Rollback

Engine config flag `use_paged_kvcache = false` reverts to `DynamicKVCache`.

---

## PR-9: Paged Attention Kernels

### Objective

Implement attention kernels that read K/V directly from block-table-indexed
memory, eliminating the contiguous gather in PR-8. Separate decode and
prefill attention paths.

### Affected Files

| Action | File |
|--------|------|
| New | `include/backend/ops/self_attention/paged.hpp` — paged attention declarations |
| New | `src/backend/ops/self_attention/cpu/paged_attention_cpu.cpp` |
| New | `src/backend/ops/self_attention/nvidia/paged_attention_nvidia.cu` |
| Modify | `include/backend/ops/ops.hpp` — add `paged_attention_decode`, `paged_attention_prefill` |
| Modify | `src/frontend/models/qwen2_forward_batch.cpp` — use paged attention |
| Modify | `src/frontend/models/qwen3_forward_batch.cpp` — same |
| Modify | `src/backend/kvcache/paged.cpp` — remove contiguous gather methods (or keep as debug fallback) |

### Interfaces Added / Changed / Removed

**Added:**

```cpp
void paged_attention_decode(
    tensor_t out,            // [num_seqs, nhead, head_dim]
    tensor_t query,          // [num_seqs, nhead, head_dim]
    tensor_t k_cache_pool,   // [total_blocks * block_size, nkvhead, head_dim]
    tensor_t v_cache_pool,
    tensor_t block_tables,   // [num_seqs, max_blocks_per_seq] int32
    tensor_t seq_lens,       // [num_seqs] int32
    float scale,
    int block_size);

void paged_attention_prefill(
    tensor_t out,
    tensor_t query,
    tensor_t k_cache_pool,
    tensor_t v_cache_pool,
    tensor_t block_table,    // [max_blocks] int32
    int seq_len,
    int past_len,
    float scale,
    int block_size);
```

### Dependency

PR-8 (paged KV cache, block tables).

### Correctness Tests

- Paged decode: compare output against non-paged `self_attention` with same Q/K/V data.
- Paged prefill: same comparison.
- E2E with fully paged pipeline: output identical to non-paged reference.

### Regression Tests

Non-paged attention path still works for single-request mode.

### Benchmarks

Config A (decode-heavy), Config D (long context).
Expected: no latency regression on decode; significant memory improvement.

### Risks

- GPU paged attention kernel is complex. Consider adapting vLLM paged attention
  kernels (Apache-2.0) for initial implementation.
- CPU paged attention may be slower than contiguous due to indirect access.
  Mitigated by block-iterating loop that processes one block at a time for cache locality.

### Rollback

Engine flag reverts to non-paged attention + contiguous gather.

---

## PR-10: HTTP / OpenAPI Server

### Objective

Add an HTTP server with OpenAI-compatible endpoints. Wire to scheduler for
request submission. Support SSE streaming.

### Affected Files

| Action | File |
|--------|------|
| New | `include/zedinfer/http_server.hpp` |
| New | `src/zedinfer/http_server.cpp` |
| New | `include/zedinfer/api_types.hpp` — OpenAI-compatible JSON types |
| New | `examples/serve.cpp` — HTTP serving entry point |
| Add dep | `third_party/include/httplib.h` (cpp-httplib, header-only, MIT) or xmake package |
| Modify | `xmake/examples.lua` — new `serve` target |

### Interfaces Added / Changed / Removed

**Added:**

```cpp
struct ServerConfig { std::string host; int port; int max_connections; int request_timeout_ms; };

class HttpServer {
    HttpServer(ServerConfig, std::shared_ptr<Scheduler>, std::shared_ptr<InferenceEngine>);
    void start();
    void stop();
};
```

Endpoints: `POST /v1/chat/completions`, `GET /v1/models`, `GET /health`.

### Dependency

PR-7 (scheduler with batching).

### Correctness Tests

- Unit test: parse request JSON -> `InferenceRequest` -> format response JSON.
- Integration: `curl` against running server, verify response structure.
- Streaming: verify SSE chunks, `[DONE]` terminator.

### Regression Tests

`bench`, `chat`, `ping` unaffected.

### Benchmarks

HTTP load test with `wrk` or `hey`: 10 concurrent connections, 64-token responses.

### Risks

- cpp-httplib's thread-per-connection model may bottleneck at high concurrency.
  Acceptable for initial deployment; can switch to async library later.

### Rollback

Delete `serve` target. No other code depends on it.

---

## PR-11: INT8 Quantization

### Objective

Support INT8 per-channel symmetric weight-only quantization. Load INT8 weights
from SafeTensors. Implement quantized linear dispatch on CPU (oneDNN INT8 or
dequant-multiply) and GPU (cuBLAS INT8 GEMM).

### Affected Files

| Action | File |
|--------|------|
| New | `include/backend/tensor/quantized_weight.hpp` — `QuantizedLinearWeight`, `QuantizationConfig` |
| New | `src/backend/ops/linear/cpu/linear_int8.cpp` |
| New | `src/backend/ops/linear/nvidia/linear_int8.cu` |
| Modify | `include/frontend/models/base.hpp:17-40` — add `QuantizationConfig` to `ModelConfig` |
| Modify | `include/frontend/models/base.hpp:42-65` — `ModelWeights` gains quantized weight storage |
| Modify | `src/frontend/models/base.cpp:106-160` — detect INT8 weights during loading |
| Modify | `include/backend/ops/ops.hpp` — add `linear_quantized()` |
| Modify | `src/frontend/models/qwen2_forward.cpp` — dispatch to `linear_quantized` when weights are INT8 |
| Modify | `src/frontend/models/qwen3_forward.cpp` — same |

### Interfaces Added / Changed / Removed

**Added:**

```cpp
struct QuantizationConfig {
    enum class Method { NONE, INT8_PER_CHANNEL, INT4_GPTQ, INT4_AWQ };
    Method method = Method::NONE;
    int group_size = 128;
    bool is_quantized() const;
};

struct QuantizedLinearWeight {
    tensor_t weight;    // int8[N, K]
    tensor_t scales;    // float16[N]
    tensor_t zeros;     // nullptr for symmetric
    QuantizationConfig config;
};

void linear_quantized(tensor_t out, tensor_t in, const QuantizedLinearWeight &qw);
```

### Dependency

PR-3a/3b (cuBLAS/oneDNN foundation), PR-2 (direct forward).

### Correctness Tests

- Load INT8-quantized model, compare logits against FP16 reference.
  KL divergence per token < 0.01.
- Operator-level: `linear_quantized` vs `linear` with dequantized weights.

### Regression Tests

FP16/BF16/FP32 models must still load and run identically.

### Benchmarks

All configs. Expected: ~50 % memory reduction, 1.2-1.8x throughput improvement
(less memory bandwidth for weights).

### Risks

- INT8 cuBLAS GEMM may require specific tensor layouts (column-major, specific alignment).
- Quality degradation on some models. Mitigate with per-model validation.

### Rollback

`QuantizationConfig::Method::NONE` is default; INT8 path only activates when
INT8 weights are detected.

---

## PR-12: INT4 Quantization (GPTQ / AWQ)

### Objective

Support INT4 per-group quantization. Load GPTQ and AWQ formatted weights.
Implement INT4 dequant-on-the-fly linear kernels.

### Affected Files

| Action | File |
|--------|------|
| Modify | `include/zedinfer.h` — add `ZEDINFER_DTYPE_I4 = 14` |
| New | `src/frontend/loader/gptq_loader.cpp` — GPTQ weight unpacking |
| New | `src/backend/ops/linear/cpu/linear_int4.cpp` |
| New | `src/backend/ops/linear/nvidia/linear_int4.cu` |
| Modify | `src/frontend/models/base.cpp` — GPTQ/AWQ weight detection and loading |
| Modify | `include/backend/tensor/quantized_weight.hpp` — INT4 fields |

### Interfaces Added / Changed / Removed

**Added:**

- `ZEDINFER_DTYPE_I4` enum value.
- GPTQ weight loading utilities.

**Changed:**

- `QuantizedLinearWeight` gains `zeros` tensor for asymmetric INT4.
- `linear_quantized()` dispatches to INT4 kernels when `config.method == INT4_GPTQ`.

### Dependency

PR-11 (INT8 infrastructure; reuses `QuantizedLinearWeight`, `linear_quantized` dispatch).

### Correctness Tests

- Load GPTQ-quantized model, compare logits. KL divergence < 0.05.
- Operator-level: INT4 dequant matches manual dequantization.

### Benchmarks

All configs. Expected: ~75 % memory reduction vs FP16.
Target: Qwen3-8B in ~4 GB, leaving ample room for KV cache on 24 GB GPU.

### Risks

- GPTQ packing format varies across quantization libraries. Pin to `auto_gptq` format.
- INT4 GPU performance depends heavily on kernel quality. Start with dequant + cuBLAS FP16;
  optimize later with CUTLASS or Marlin kernels.

### Rollback

INT4 path behind `QuantizationConfig::Method` check. No impact on existing dtypes.

---

## PR-13: Heterogeneous Inference and Pinned Memory

### Objective

Add pinned host memory support to the runtime API. Enable per-layer device
placement so some weights can live on CPU while others live on GPU.
Add async H2D transfer with compute overlap.

### Affected Files

| Action | File |
|--------|------|
| Modify | `include/backend/device/runtime_api.hpp` — add `malloc_pinned`, `free_pinned` |
| Modify | `src/backend/device/nvidia/nvidia_runtime_api.cu` — implement pinned memory via `cudaMallocHost` |
| Modify | `include/backend/core/runtime/runtime.hpp` — add `allocatePinnedStorage()` |
| Modify | `src/backend/core/runtime/runtime.cpp` — implement |
| Modify | `src/frontend/models/base.cpp` — per-layer device placement during weight loading |
| New | `include/zedinfer/device_placement.hpp` — placement policy configuration |

### Interfaces Added / Changed / Removed

**Added:**

```cpp
// runtime_api.hpp
typedef void *(*malloc_pinned_api)(size_t);
typedef void (*free_pinned_api)(void *);

// In ZedinferRuntimeAPI struct:
malloc_pinned_api malloc_pinned;
free_pinned_api free_pinned;
```

```cpp
// runtime.hpp
storage_t allocatePinnedStorage(size_t size);
```

```cpp
struct DevicePlacement {
    struct LayerPlacement { zedinferDeviceType_t device; int device_id; };
    LayerPlacement attention;   // where attention weights live
    LayerPlacement mlp;         // where MLP weights live
    static DevicePlacement all_gpu(int device_id);
    static DevicePlacement split(int gpu_layers, int total_layers);
};
```

### Dependency

PR-2 (direct forward, which controls per-layer execution).

### Correctness Tests

- Load model with split placement, run E2E. Output must match all-GPU reference.
- Pinned memory allocation / deallocation stress test.

### Benchmarks

Measure transfer latency. With async overlap, expect < 5 % latency increase
for models that mostly fit on GPU.

### Risks

- Cross-device data movement in the forward loop adds complexity.
- Pinned memory can reduce available system memory.

### Rollback

`DevicePlacement::all_gpu()` is default. No cross-device code runs unless
explicitly configured.

---

## PR-14: MoE Model Support and Expert Offloading

### Objective

Add MoE model class, expert routing, expert weight pool (GPU + CPU), and
dynamic expert offloading with async prefetch.

### Affected Files

| Action | File |
|--------|------|
| New | `include/frontend/models/moe_base.hpp` — MoE model config |
| New | `include/backend/moe/expert_pool.hpp` |
| New | `include/backend/moe/router.hpp` — top-K expert selection |
| New | `src/backend/moe/expert_pool.cpp` |
| New | `src/backend/moe/router.cpp` |
| New | `src/backend/ops/moe/gather_scatter.cpp` — token gather/scatter by expert |
| New | `src/frontend/models/qwen3_moe.hpp`, `qwen3_moe_forward.cpp` |
| Modify | `src/frontend/models/base.cpp` — detect MoE config, instantiate MoE model |
| Modify | `src/zedinfer/scheduler.cpp` — expert-aware scheduling hints |

### Interfaces Added / Changed / Removed

**Added (key types):**

```cpp
class ExpertPool {
    ExpertWeightPtrs ensure_on_gpu(int layer, int expert_id);
    void prefetch_to_gpu(int layer, int expert_id, zedinferStream_t stream);
    void evict_lru();
};

struct MoEConfig {
    int num_experts;
    int top_k;
    int expert_intermediate_size;
};
```

### Dependency

PR-13 (pinned memory), PR-12 (INT4 to fit model), PR-7 (continuous batching for multi-user).

### Correctness Tests

- Router: top-K selection matches PyTorch `torch.topk`.
- Expert forward: compare per-expert FFN output against dense FFN with same weights.
- E2E with MoE model: compare against HuggingFace Transformers output.

### Benchmarks

Qwen-30B-A3B (INT4) on 24 GB GPU. Measure:
- All experts on GPU: baseline decode latency
- 80 % on GPU, 20 % on CPU: measure overhead
- Report prefetch hit rate, avg transfer time

### Risks

- Expert routing correctness is critical; wrong expert selection produces garbage.
- Transfer scheduling interacts with continuous batching in complex ways.
- Qwen-30B-A3B model may not be available for testing initially.

### Rollback

MoE model class is a separate `Model` subclass. Dense models are unaffected.

---

## Critical Path

```
PR-1 ──> PR-2 ──> PR-4 ──> PR-6 ──> PR-7 ──> PR-8 ──> PR-9 ──> PR-10
              \                                                      \
              PR-3a ─────────────────────────────> PR-11 ──> PR-12 ──> PR-13 ──> PR-14
              PR-3b ──────────────────────────────/
              \
              PR-5 (can merge anytime after PR-4)
```

PRs 3a and 3b can proceed in parallel with each other and with PR-4/5.
PR-5 can be merged any time after PR-4.
PR-10 (HTTP) can merge any time after PR-7.
PR-11/12 (quantization) can proceed independently once PR-3a/3b and PR-2 are in.
PR-13/14 (heterogeneous/MoE) depend on quantization and batching.

## Summary Table

| PR | Title | Est. Complexity | Key Dependency |
|----|-------|----------------|----------------|
| 1 | Test infrastructure & baselines | Low | None |
| 2 | Direct model forward | Medium | PR-1 |
| 3a | cuBLAS for NVIDIA linear | Low-Medium | PR-1 |
| 3b | oneDNN for CPU linear | Low-Medium | PR-1 |
| 4 | Stateless engine & request types | Low | PR-2 |
| 5 | Chat template extraction | Low | PR-4 |
| 6 | Scheduler (single-request) | Medium | PR-4 |
| 7 | Continuous batching | High | PR-6 |
| 8 | Paged KV cache | Medium-High | PR-7 |
| 9 | Paged attention kernels | High | PR-8 |
| 10 | HTTP / OpenAPI server | Medium | PR-7 |
| 11 | INT8 quantization | Medium | PR-3a, PR-3b |
| 12 | INT4 quantization | Medium | PR-11 |
| 13 | Heterogeneous inference | Medium | PR-2 |
| 14 | MoE expert offloading | High | PR-13, PR-12, PR-7 |
