# Gap Analysis: Current State vs Target Vision

## 1. Multi-User Support

### Current State
- `InferenceEngine` holds mutable `last_stats_` (`engine.hpp:92`) - not thread-safe for concurrent sessions
- `GraphExecutor::forward()` allocates intermediate tensors per call (`executor.cpp:53-128`) - no isolation between concurrent calls
- `InferenceSession` owns its own `kvcache_` and `chat_history_` - correctly session-scoped
- No request queue, no scheduler, no request-level isolation
- `core::Context` is thread-local (`context.hpp:16`) - good for multi-thread but no concurrency within a thread

### Gap
- Need thread-safe engine or per-request execution state
- Need a scheduler/request queue to serialize or parallelize access to shared resources (model weights, GPU)
- Need to decouple stats tracking from engine
- `ExecutionContext` (`shape.hpp:19-23`) hardcodes `batch_size=1` in all forward paths

### Key Files to Modify
- `include/zedinfer/engine.hpp` - Make stateless or add per-request state
- `src/zedinfer/engine.cpp:183-269` - `generate_tokens()` holds mutable state
- `src/zedinfer/executor.cpp:53-128` - `forward()` activation map is local but tensor allocation goes through global pool

---

## 2. Continuous Batching

### Current State
- `GraphExecutor::forward()` processes a single sequence: `const int seq_len = input_ids.size()` (`executor.cpp:58`)
- `ExecutionContext` has `batch_size` field but it's always set to 1 (`executor.cpp:107`)
- All operator kernels assume 2D tensors `[seq_len, dim]` - no batch dimension
- KV cache is per-session with shape `[capacity, num_kv_heads, head_dim]` - no batch stacking

### Gap
- Need batch dimension in tensor layouts: `[batch, seq_len, dim]` or flattened with offsets
- Need scheduler to group requests by phase (prefill vs decode)
- Need variable-length sequence support within a batch
- Need to coordinate KV cache access across batched sequences
- All operator kernels need batch-aware variants

### Key Files to Modify
- `include/frontend/graph/shape.hpp` - Add batch dimension to ShapeTemplate
- `src/zedinfer/executor.cpp` - Batched forward pass
- All `src/backend/ops/*/cpu/*.cpp` and `nvidia/*.cu` - Batch-aware kernels
- `src/backend/kvcache/dynamic.cpp` - Batch-aware cache management

---

## 3. Paged Attention / Paged KV Cache

### Current State
- `DynamicKVCache` allocates contiguous `[capacity, num_kv_heads, head_dim]` per layer (`dynamic.cpp:40-58`)
- Growth requires full reallocation + memcpy (`dynamic.cpp:61-97`)
- `get_k_cache_slice()` / `get_v_cache_slice()` return contiguous slices via `tensor->slice()` (`dynamic.cpp:179-225`)
- `self_attention` kernels expect contiguous K/V buffers
- No block table, no page management, no virtual memory mapping

### Gap
- Need `PagedKVCache` with fixed-size blocks (e.g., 16 or 32 tokens per block)
- Need block table mapping (sequence_id, layer, logical_block) -> physical_block_ptr
- Need paged attention kernels that read K/V from non-contiguous blocks via block table
- Need block allocator (pool of fixed-size blocks)
- Eliminates growth-copy overhead, enables memory sharing (e.g., prefix caching)

### Key Files to Modify
- `include/backend/kvcache/` - New `paged.hpp` with PagedKVCache
- `src/backend/ops/self_attention/` - New paged attention kernels (both CPU and NVIDIA)
- `src/zedinfer/executor.cpp:150-178` - KV cache slice logic needs to change

---

## 4. Direct Model Forward (Remove Redundant Graph)

### Current State
- `ComputeGraph` + `GraphNode` + `GraphBuilder` construct a full DAG at init time
- `GraphExecutor::forward()` walks the graph in topological order, dispatching ops one by one
- For each node: string name matching for special cases (`executor.cpp:152-153`), hash map lookups for activations, dynamic shape resolution
- Graph optimization passes are all `TO_BE_IMPLEMENTED()` (`graph.cpp:262-484`)
- For fixed model families (Qwen2/Qwen3), the graph adds overhead without flexibility benefit

### Gap
- For already-adapted models, replace graph execution with direct `Model::forward()` methods
- Model classes (`Qwen2Model`, `Qwen3Model`) should hold operator execution logic directly
- This eliminates: graph construction cost, per-node hash lookups, string matching, dynamic allocation overhead
- Keep graph infrastructure available for future dynamic/custom model support if needed

### Key Files to Modify
- `include/frontend/models/qwen2.hpp`, `qwen3.hpp` - Add `forward()` method
- `include/zedinfer/executor.hpp` - May become thin or split into graph-based vs direct
- `src/frontend/graph/builder.cpp` - Can be preserved but bypassed for known models

---

## 5. Operator Performance

### Current State

**CPU Linear** (`src/backend/ops/linear/cpu/linear_cpu.cpp`):
- BF16/FP16: Allocates temporary FP32 buffers, converts entire input+weight, computes in FP32, converts back
- `matmul()` / `vecmul()` are handwritten with AVX512/AVX2 (`include/backend/ops/linear/cpu/matmul.hpp`, `vecmul.hpp`)
- No oneDNN / MKL integration

**CPU Self-Attention** (`src/backend/ops/self_attention/cpu/self_attention_cpu.cpp`):
- Naive per-position, per-head loop with `std::vector<float> scores(total_len)` allocated per (i, h) iteration
- No tiling, no blocked computation, no cache-friendly memory access patterns
- OMP parallel over (token, head) but inner loops are sequential

**NVIDIA Linear** (`src/backend/ops/linear/nvidia/`):
- Custom PTX-based kernels (`helper_cuda_ptx.cuh`, `linear_fp16_kernel.cuh`, etc.)
- Separate kernel files for FP16, FP32, BF16
- No cuBLAS integration

**NVIDIA Self-Attention** (`src/backend/ops/self_attention/nvidia/`):
- Custom CUDA kernels
- No FlashAttention, no FlashDecoding

### Gap
- CPU linear: Integrate oneDNN for GEMM - significant perf gain expected
- CPU attention: Tiled/blocked implementation or eventually oneDNN attention primitives
- NVIDIA linear: Integrate cuBLAS - well-tuned for all shapes/dtypes
- NVIDIA attention: Integrate FlashAttention-2/3 for prefill, FlashDecoding for decode
- Attention redesign required anyway for paged attention support

### Key Files to Modify
- `src/backend/ops/linear/cpu/linear_cpu.cpp` - Replace with oneDNN calls
- `src/backend/ops/linear/nvidia/linear_nvidia.cu` - Replace with cuBLAS calls
- `src/backend/ops/self_attention/` - Full redesign for both backends

---

## 6. Quantization (INT8 / INT4)

### Current State
- `zedinferDataType_t` enum includes `ZEDINFER_DTYPE_I8` but no INT4 type (`zedinfer.h:42`)
- `Tensor::to(zedinferDataType_t)` only supports BF16/FP16 -> FP32 on CPU (`tensor.cpp:592-628`)
- No quantized operator implementations
- No quantization metadata (scales, zero-points)
- Weight loading assumes FP16/BF16/FP32 SafeTensors format

### Gap
- Define INT8/INT4 weight format with per-channel or per-group quantization parameters
- Add quantized weight loading path (GPTQ, AWQ, or custom format)
- Implement quantized linear kernels (INT8 GEMM, INT4 dequant-on-the-fly)
- For CPU: oneDNN INT8 GEMM or custom AVX-VNNI kernels
- For GPU: cuBLAS INT8, CUTLASS INT4, or custom kernels
- Need quantization-aware model config and weight mapping

### Key Files to Modify
- `include/zedinfer.h` - Add INT4 dtype
- `include/frontend/models/base.hpp` - Quantization config fields
- `src/frontend/models/base.cpp` - Quantized weight loading
- `include/backend/ops/linear/` - Quantized linear kernels
- New: quantization utility for scale/zero-point management

---

## 7. HTTP / OpenAPI Interface

### Current State
- No HTTP server code exists
- No request/response types
- No API endpoints
- `chat.cpp` uses linenoise for interactive CLI

### Gap
- Need HTTP server (e.g., cpp-httplib, Boost.Beast, or similar lightweight C++ HTTP lib)
- OpenAI-compatible API endpoints: `/v1/chat/completions`, `/v1/models`
- Request/response JSON serialization
- Server-Sent Events (SSE) for streaming
- Need scheduler integration for request queuing
- Need session/connection management
- No Python runtime dependency (constraint from CLAUDE.md)

### Key Files to Create
- `include/zedinfer/http_server.hpp` - Server configuration and lifecycle
- `src/zedinfer/http_server.cpp` - HTTP endpoint handlers
- `include/zedinfer/api_types.hpp` - OpenAI-compatible request/response types

---

## 8. Heterogeneous CPU/GPU Inference

### Current State
- Entire model placed on a single device at init time (`base.cpp:29-31`, `142-148`)
- CPU: weights converted to FP32; GPU: weights transferred to device
- No mixed-device execution, no layer-level device placement
- No pinned host memory for async transfers (except mmap-backed tensors)

### Gap
- Need per-layer or per-expert device placement configuration
- Need efficient CPU<->GPU data transfer with pinned memory
- Need async prefetch/transfer overlapping with computation
- Need scheduler awareness of device placement for latency estimation

### Key Files to Modify
- `src/frontend/models/base.cpp:106-160` - Per-tensor device placement during loading
- `include/backend/core/runtime/runtime.hpp` - Pinned memory allocation support
- `src/zedinfer/executor.cpp` - Cross-device data movement during execution

---

## 9. MoE Expert Offloading

### Current State
- No MoE support at all
- Model classes assume dense transformer layers (all layers identical)
- No expert routing, no expert weight management, no sparse execution

### Gap
- Need MoE model config (num_experts, top_k, expert routing)
- Need expert weight storage with device placement policy
- Need expert routing/gating implementation
- Need dynamic expert loading/offloading between CPU and GPU
- Need expert hotness tracking for placement decisions
- Need async prefetch of predicted-hot experts
- Target: Run Qwen-30B-A3B on 24GB GPU with INT8/INT4

### Key Files to Create
- `include/frontend/models/moe_base.hpp` - MoE model abstraction
- `include/backend/ops/moe/` - Expert routing and sparse computation
- Expert placement/scheduling policy

---

## 10. Graph Optimization (Memory Reuse)

### Current State
- `GraphExecutor::forward()` allocates output tensors for every non-input node (`executor.cpp:183-189`)
- `Tensor::create()` goes through `BestFitMemoryPool` but tensors are freed when `tensor_t` (shared_ptr) drops
- No tensor lifetime analysis, no in-place operations
- `ComputeGraph::optimize_memory()` is `TO_BE_IMPLEMENTED()` (`graph.cpp:482-484`)

### Gap
- If graph execution is retained, implement memory reuse based on tensor lifetime analysis
- If replaced by direct forward, can use pre-allocated scratch buffers sized once
- Direct forward approach is simpler: allocate fixed set of activation buffers at init

### Decision Needed
- Priority of graph optimization vs direct forward replacement
- Direct forward eliminates this gap entirely for known models

---

## Summary Priority Matrix

| Gap | Impact | Complexity | Dependency | Priority |
|-----|--------|-----------|-----------|----------|
| Direct model forward | Medium (perf) | Low | None | 1 - Foundation |
| Operator performance | High (user-visible) | Medium | None | 2 - Quick wins |
| Multi-user / session | High (functionality) | Medium | Direct forward | 3 - Foundation |
| Scheduler | High (functionality) | High | Multi-user | 4 - Core |
| Continuous batching | High (throughput) | High | Scheduler | 5 - Core |
| Paged KV cache | High (memory) | High | Continuous batching | 6 - Core |
| Paged attention | High (perf+memory) | High | Paged KV cache | 7 - Core |
| HTTP API | High (functionality) | Medium | Scheduler | 8 - Integration |
| INT8 quantization | High (memory/perf) | Medium | Operator perf | 9 - Extension |
| INT4 quantization | Medium | High | INT8 | 10 - Extension |
| Heterogeneous inference | Medium | High | Quantization | 11 - Advanced |
| MoE offloading | Medium | Very High | Heterogeneous | 12 - Advanced |
