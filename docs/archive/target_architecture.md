# Target Architecture

## System Topology

```
                        HTTP / OpenAPI
                             |
                     +-------v--------+
                     |  HttpServer     |  /v1/chat/completions, /v1/models
                     |  (SSE stream)   |
                     +-------+--------+
                             |
                     +-------v--------+
                     |  Scheduler      |  Continuous batching, admission, fairness
                     |  RequestQueue   |  Token-level scheduling
                     +-------+--------+
                             |
          +------------------v-------------------+
          |         BatchedExecutor               |
          |  Assembles batch -> Model::forward()  |
          +------------------+-------------------+
                             |
          +------------------v-------------------+
          |         Model::forward()              |
          |  Qwen2Model / Qwen3Model / MoEModel  |
          |  Direct layer-by-layer execution      |
          +------------------+-------------------+
                             |
     +---+---+---+---+------v------+---+---+---+
     |ops:: |ops:: |ops:: |ops::   |ops:: |ops::|
     |linear|attn  |norm  |embed   |rope  |swiglu|
     +--+---+--+---+--+---+--+----+--+---+--+--+
        |      |      |      |       |      |
     +--v------v------v------v-------v------v--+
     |        Backend Dispatch                  |
     |  CPU (oneDNN)  |  NVIDIA (cuBLAS/Flash)  |
     |  INT8/INT4     |  INT8/INT4              |
     +--+-------------+-------------+----------+
        |                           |
     +--v--+                     +--v--+
     |CPU  |                     |GPU  |
     |Pool |                     |Pool |
     +-----+                     +-----+
                 |
          +------v------+
          | PagedKVCache |  Block allocator, block table
          +-------------+
```

## Abstraction Verdict: Keep / Simplify / Remove

### KEEP (with modifications)

| Abstraction | Location | Reason | Modification |
|-------------|----------|--------|-------------|
| `Tensor` | `include/backend/tensor/tensor.hpp` | Fundamental. view/slice/permute are correct and well-tested. | No changes to API. |
| `Storage` | `include/backend/core/storage/storage.hpp` | Clean memory block abstraction with device affinity. | None. |
| `Runtime` | `include/backend/core/runtime/runtime.hpp` | Device memory allocation via pool. Stream management. | Add pinned-memory allocation method for heterogeneous transfers. |
| `Context` | `include/backend/core/context/context.hpp` | Thread-local device context is the correct pattern. | None. |
| `BestFitMemoryPool` | `include/backend/core/memory/memory_pool.hpp` | Well-implemented pool with size-class optimization. | Keep for GPU device memory. Add thread-safety option. |
| `PooledAllocator` | `include/backend/core/memory/pooled_allocator.hpp` | Correct delegation to pool. | None. |
| `ZedinferRuntimeAPI` | `include/backend/device/runtime_api.hpp` | Clean function-pointer table for CPU/NVIDIA. | Add `malloc_pinned` / `free_pinned` for pinned host memory. |
| `Device` | `include/backend/device/device.hpp` | Simple value type. | None. |
| `SafeTensorsLoader` | `include/frontend/loader/safetensors.hpp` | Mmap-based, efficient. | Add quantized format support alongside existing path. |
| `HFTokenizer` | `include/frontend/tokenizer/hf_tokenizer.hpp` | Correct BPE implementation. | None. |
| `Sampler` | `include/frontend/sampler/sampler.hpp` | Argmax + General with temp/top-k/top-p. | None. |
| `Model` / `ModelConfig` / `ModelWeights` | `include/frontend/models/base.hpp` | Good base for multi-model support. | Add `forward()` to model subclasses. Add quantization config fields. |
| `InferenceSession` | `include/zedinfer/session.hpp` | Correct per-session state (KV cache, history). | Adapt to use paged KV cache. Move chat template to config. |
| `GenerationConfig` / `GenerationStats` | `include/zedinfer/generation_types.hpp` | Clean configuration and stats types. | None. |
| Operator dispatch (`ops::*`) | `include/backend/ops/ops.hpp` | Clean dispatch interface. | Add quantized variants. Batch-aware signatures. |

### SIMPLIFY

| Abstraction | Location | Change |
|-------------|----------|--------|
| `InferenceEngine` | `include/zedinfer/engine.hpp` | Remove mutable `last_stats_`. Make truly stateless. Stats returned per-call or per-session. `warmup()` and `profile()` stay but don't mutate engine state. |
| `ExecutorConfig` | `include/zedinfer/activation.hpp` | Rename to `RuntimeConfig` or `ModelRuntimeConfig`. Currently conflates model config with executor config. |

### REMOVE (for known models)

| Abstraction | Location | Reason | Replacement |
|-------------|----------|--------|-------------|
| `ComputeGraph` | `include/frontend/graph/graph.hpp` | Redundant for Qwen2/Qwen3. Graph walk adds per-node hash lookups, string matching, dynamic allocation overhead. All optimization passes are stubs. | Direct `Model::forward()` with pre-allocated scratch buffers. |
| `GraphBuilder` | `include/frontend/graph/builder.hpp` | Only exists to populate the graph. | Model subclass constructor binds weights directly. |
| `GraphExecutor` | `include/zedinfer/executor.hpp` | Walks the graph. | Replaced by `BatchedExecutor` that calls `Model::forward()`. |
| `ShapeTemplate` / `ShapeDim` | `include/frontend/graph/shape.hpp` | Only used for graph-level shape inference. | Shape computed directly in forward() from input dims and model config. |
| `PositionIDsCache` | `include/zedinfer/activation.hpp` | Graph executor artifact. | Position IDs generated inline in forward() or by executor. |

**Note**: The graph files should be preserved in the codebase (not deleted) but bypassed for production execution. They may be useful for debugging, visualization, or future dynamic model support.

## Direct Model Forward Design

### Rationale

For Qwen2/Qwen3, the model structure is fixed:
- Embedding -> N x TransformerLayer -> FinalNorm -> LMHead
- Each TransformerLayer: InputNorm -> Attention -> Residual -> PostNorm -> MLP -> Residual
- Qwen3 adds per-head QK norms

The graph adds overhead for these known structures:
1. Topological sort + cached order traversal
2. Hash map lookup per node for activations (`executor.cpp:139-145`)
3. String-based name matching for special cases (`executor.cpp:152-153`)
4. Dynamic `Tensor::create()` per node output (`executor.cpp:183-189`)
5. `std::any_cast` for node parameters (`graph.hpp:55`)

### Target Interface

```cpp
class Model {
public:
    // New: direct forward pass
    virtual tensor_t forward(
        const std::vector<int> &input_ids,
        int past_len,
        KVCache &kvcache) = 0;

    // New: batched forward for continuous batching
    virtual tensor_t forward_batch(
        const BatchContext &batch,
        PagedKVCache &kvcache) = 0;
};
```

### Scratch Buffer Management

Instead of allocating per-node per-forward:

```cpp
struct ScratchBuffers {
    tensor_t hidden;       // [max_batch_tokens, hidden_size]
    tensor_t residual;     // [max_batch_tokens, hidden_size]
    tensor_t q_proj;       // [max_batch_tokens, hidden_size]
    tensor_t k_proj;       // [max_batch_tokens, kv_dim]
    tensor_t v_proj;       // [max_batch_tokens, kv_dim]
    tensor_t attn_out;     // [max_batch_tokens, hidden_size]
    tensor_t gate;         // [max_batch_tokens, intermediate_size]
    tensor_t up;           // [max_batch_tokens, intermediate_size]
    tensor_t down;         // [max_batch_tokens, hidden_size]
    tensor_t logits;       // [max_batch_tokens, vocab_size]
    tensor_t norm_buf;     // [max_batch_tokens, hidden_size]
};
```

Allocated once at engine init for `max_batch_tokens` (configurable, e.g., 2048). Reused every forward call. Zero allocation overhead in the hot path.

## Threading Model

```
Main Thread (HTTP / CLI)
    |
    v
Scheduler Thread
    |-- maintains request queue
    |-- assembles batches
    |-- dispatches to worker
    |
    v
Worker Thread (1 per GPU, or 1 for CPU)
    |-- calls Model::forward_batch()
    |-- returns logits to scheduler
    |-- scheduler runs sampling, routes tokens back
```

Single GPU: one scheduler thread + one worker thread.
CPU-only: scheduler + worker can be same thread (synchronous).
Multi-GPU (future): one worker per GPU, scheduler distributes.

## Key Design Decisions

1. **Model owns its forward path**: `Qwen2Model::forward()` calls ops directly, manages scratch buffers, writes KV cache. No graph indirection.

2. **Executor becomes thin**: `BatchedExecutor` assembles batch metadata, calls `model->forward_batch()`, handles tokenization/sampling. Does not own the compute logic.

3. **KV cache owned by session, managed by scheduler**: Each session still owns a logical KV cache (sequence of block references). The block pool is global. The scheduler coordinates block allocation.

4. **Operator signatures gain batch awareness**: `ops::linear(out, in, weight, bias)` stays the same shape-wise because batch dimension is folded into the M dimension. But paged attention needs new signatures.

5. **Quantization is an operator concern**: Model forward calls `ops::linear()` with the same API. Dispatch inside `ops::linear()` checks weight dtype and routes to quantized kernel.
