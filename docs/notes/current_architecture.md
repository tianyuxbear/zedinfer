# Current Architecture

## High-Level Architecture

```
                     examples/
                   bench | chat | ping
                         |
                  +--------------+
                  | InferenceEngine |  (stateless, shared)
                  |  - model_        |  Model + weights
                  |  - graph_        |  ComputeGraph
                  |  - executor_     |  GraphExecutor
                  |  - tokenizer_    |  HFTokenizer
                  |  - sampler_      |  Sampler
                  +--------------+
                         |
              create_session()
                         |
                  +------------------+
                  | InferenceSession  |  (per-user state)
                  |  - engine_ (shared) |
                  |  - kvcache_         |  DynamicKVCache (owned)
                  |  - chat_history_    |
                  |  - past_len_        |
                  +------------------+
                         |
                     chat() / generate()
                         |
                  +------------------+
                  | GraphExecutor     |
                  |  forward()        |  Walks compute graph
                  +------------------+
                         |
                  +------------------+
                  | ComputeGraph      |  Topological order
                  |  GraphNode[]      |  OpType + weight + params
                  +------------------+
                         |
                  +------------------+
                  | ops::* dispatch   |  CPU or NVIDIA kernel
                  +------------------+
                         |
                  +------------------+
                  | Core Runtime      |
                  |  Context          |  Thread-local device mgmt
                  |  Runtime          |  Memory alloc, streams
                  |  Storage          |  Memory block with device
                  +------------------+
```

## Initialization Flow

**`InferenceEngine::create(model_path, device)`** (`src/zedinfer/engine.cpp:39-101`):

1. **Model loading** (`Model::parse()` at `src/frontend/models/base.cpp:21-49`):
   - Read `config.json` -> dispatch to `Qwen2Config` or `Qwen3Config`
   - Load weights via `SafeTensorsLoader` (mmap all `.safetensors` files)
   - For CPU: convert weights BF16/FP16 -> FP32
   - For GPU: transfer weights to device memory (H2D copy)

2. **Tokenizer loading** (`HFTokenizer::create()` at `include/frontend/tokenizer/hf_tokenizer.hpp:12`):
   - Parse `tokenizer.json` (BPE vocab, merges, regex pattern)
   - Parse `tokenizer_config.json` (special tokens, model_max_length)

3. **Graph construction** (`GraphBuilder::build()` at `src/frontend/graph/builder.cpp:17-523`):
   - Create INPUT nodes for input_ids, position_ids, per-layer k_cache/v_cache
   - Build embedding -> N transformer layers -> final norm -> lm_head
   - Each layer: input_norm -> attention(q/k/v_proj, rope, self_attn, o_proj) -> residual -> post_norm -> mlp(gate/up/swiglu/down) -> residual
   - Qwen3 adds per-head q_norm/k_norm after q/k projections
   - Set weight tensors and shape templates on each node

4. **Executor creation** (`GraphExecutor::create()` at `src/zedinfer/executor.cpp:36-51`):
   - Run `graph->optimize()` which performs shape inference and validation
   - Graph optimization stubs (fuse/fold/eliminate/optimize_memory) are `TO_BE_IMPLEMENTED()`
   - Pre-allocate `PositionIDsCache` [0..max_seq_len-1]

5. **Sampler creation**: Argmax sampler by default

6. **Warmup**: Run prefill (128 tokens) + decode (128 tokens) with dummy inputs

## Inference Flow

**`InferenceSession::chat(user_input)`** (`src/zedinfer/session.cpp:38-65`):

1. Build prompt: `<begin_of_sentence>` (first turn) + `<User>` + input + `<Assistant><think>\n`
2. Call `engine_->generate(*kvcache_, prompt, config_)`
3. Update past_len from KV cache

**`InferenceEngine::generate_tokens()`** (`src/zedinfer/engine.cpp:183-269`):

1. **Prefill phase**: `executor_->forward(kvcache, all_input_ids, past_len=0)`
   - Processes all prompt tokens in one pass
   - Sample first output token

2. **Decode loop**: For each subsequent token:
   - `executor_->forward(kvcache, {next_token}, past_len)`
   - Sample next token
   - Stream callback if enabled
   - Stop on EOS or max_seq_len

**`GraphExecutor::forward()`** (`src/zedinfer/executor.cpp:53-128`):

1. Prepare `input_ids` tensor on device
2. Slice `position_ids` from pre-allocated cache
3. Get KV cache slices for each layer (triggers DynamicKVCache growth if needed)
4. Execute all nodes in topological order:
   - Skip input nodes (already have activations)
   - For each compute node: gather inputs, resolve output shape, allocate output tensor
   - **Special cases**:
     - `v_proj` / `k_rope`: write directly into KV cache memory (zero-copy)
     - `q_norm` / `k_norm`: reshape output to `[seq_len * nhead, head_dim]` for per-head norm
     - `rope`: reshape input to 3D `[seq_len, nhead, head_dim]`
     - `self_attention`: flatten output back to `[seq_len, nhead * head_dim]`
   - Dispatch to `execute_op()` which calls `ops::*` functions
5. Update KV cache sequence length
6. Return logits activation

## Operator Dispatch

Each operator follows the pattern (`src/backend/ops/{op}/op.cpp`):

1. Check same device, same dtype, contiguity
2. Dispatch to `cpu::op()` or `nvidia::op()` based on device type
3. Set device context before GPU execution

**CPU linear** (`src/backend/ops/linear/cpu/linear_cpu.cpp:14-79`):
- FP32: Direct `matmul()` or `vecmul()` (M=1 optimization)
- BF16/FP16: Convert inputs/weights to FP32 -> compute -> convert back
- `matmul()` / `vecmul()` are handwritten with AVX512/AVX2 optimizations

**CPU self_attention** (`src/backend/ops/self_attention/cpu/self_attention_cpu.cpp:12-88`):
- Naive GQA implementation with causal mask
- OMP parallel over (token_position, head) with collapse(2)
- Per-token: compute Q*K^T scores -> causal softmax -> weighted V sum
- FP32 accumulation, down-cast for BF16/FP16 output

## Memory Management

### Tensor Lifecycle
- **Weight tensors**: Loaded at init, persist for engine lifetime (mmap or device memory)
- **Intermediate tensors**: Allocated fresh each `forward()` call via `Tensor::create()`
  - Goes through `core::context().runtime().allocateDeviceStorage()`
  - Backed by `BestFitMemoryPool` (size-class based, best-fit with coalescing)
  - Released when `tensor_t` (shared_ptr) refcount drops to 0
- **KV cache tensors**: Owned by DynamicKVCache, allocated/grown as needed

### Memory Pool (`include/backend/core/memory/memory_pool.hpp`)
- `BestFitMemoryPool`: Single-threaded pool with size-class buckets
- 64-byte alignment, min split size 4KB
- Default initial block: 64MB, max pool: 24GB
- Coalescing triggered every 128 deallocations
- Separate pools for device and host memory

### KV Cache (`src/backend/kvcache/dynamic.cpp`)
- Shape: `[capacity, num_kv_heads, head_dim]` per layer (K and V separate)
- Growth strategies: DOUBLE (2x), AGGRESSIVE (jump after N growths), CONSERVATIVE (linear)
- Growth requires allocating new larger tensors + full memcpy of existing data
- Capped at `model_max_seq_len`

## Device Abstraction

- `ZedinferRuntimeAPI` (`include/backend/device/runtime_api.hpp`): Function pointer table
  - Device management: get_device_count, set_device, device_synchronize
  - Stream management: create_stream, destroy_stream, stream_synchronize
  - Memory: malloc_device, free_device, malloc_host, free_host
  - Memcpy: sync and async variants
- CPU backend: `src/backend/device/cpu/cpu_runtime_api.cpp`
- NVIDIA backend: `src/backend/device/nvidia/nvidia_runtime_api.cu`
- `Context` (`include/backend/core/context/context.hpp`): Thread-local, holds runtime_map keyed by Device

## Compute Graph

- **Nodes**: `GraphNode` with OpType enum (INPUT, ADD, ARGMAX, EMBEDDING, LINEAR, RMS_NORM, ROPE, SELF_ATTENTION, SWIGLU)
- **Edges**: `node->add_input(other_node)` -> DAG structure
- **Execution order**: Kahn's algorithm topological sort, cached after first computation
- **Shape inference**: Static templates with `ShapeDim::SeqLen()` resolved at runtime
- **Optimization**: `optimize()` only runs shape inference/validation. Four optimization passes (fuse, fold, eliminate, optimize_memory) are declared but `TO_BE_IMPLEMENTED()`
- **Graph is rebuilt** on every `InferenceEngine::create()` call but is immutable after construction

## Current Limitations Summary

| Area | Current State | Impact |
|------|--------------|--------|
| Batching | batch_size=1 hardcoded | Cannot serve multiple users simultaneously |
| KV Cache | Contiguous, linearly growing | Memory waste, no sharing, expensive growth |
| Attention | Naive GQA, no tiling | Poor prefill performance at long sequences |
| Linear (CPU) | Handwritten matmul/vecmul | Suboptimal vs. oneDNN/MKL |
| Linear (GPU) | Custom PTX kernels | Suboptimal vs. cuBLAS |
| Graph exec | Dynamic alloc per forward | Memory allocation overhead per step |
| Graph opt | All stubs | No fusion, no memory reuse |
| Quantization | None | Cannot run INT8/INT4 |
| HTTP API | None | No serving capability |
| Thread safety | Not designed for concurrent sessions | Engine::last_stats_ is mutable shared state |
| Chat template | Hardcoded DeepSeek-R1 format | Not portable across models |
