# ZedInfer Multi-GPU Inference Implementation Plan

Last updated: 2026-06-17

This document is written against the current zedinfer codebase. It uses the local `../llama.cpp`, local `../vllm`, and recent LLM serving work as references, then maps them into concrete implementation steps for zedinfer. The intent is not to cover every distributed serving mode at once. The intent is to define the first path that can be implemented, verified, and extended without forcing broad rewrites.

## Summary

The recommended first milestone is single-process, single-node, multi-GPU dense Tensor Parallel inference. Start with Qwen2/Qwen3 dense BF16/FP16 inference, with `tensor_parallel_size = 2/4/8`, `pipeline_parallel_size = 1`, and `data_parallel_size = 1`. Do not include Qwen3.5 hybrid, GPTQ-Int4, or MoE Expert Parallel in the first milestone.

This is the best fit for the current architecture:

- `core::Context` can already manage multiple `Runtime` instances keyed by `(device_type, device_id)`, and `Tensor` already carries a device id.
- The serving path is a single C++ process, with no Python executor. Single-process multi-GPU support avoids introducing RPC, rank launch infrastructure, or cross-process batch coordination in the first step.
- Dense TP has a clear communication pattern: column-parallel linears run locally, row-parallel linears use `all_reduce(sum)`, and attention assigns local heads plus local KV cache to each rank.
- The current hybrid path depends on per-request SSM state and GatedDeltaNet. The current MoE path depends on an `ExpertPool` designed around one compute GPU. These should come after dense TP is stable.

First acceptance target:

- `--devices 0,1 --tp-size 2` can load a dense BF16/FP16 Qwen2/Qwen3 model.
- Single-request prefill/decode and continuous batching both return logits, with sampling still owned by the main rank.
- A small model or sliced-weight fixture matches `tp_size=1` logits within an acceptable tolerance.
- KV cache is sharded per rank, and scheduler admission uses the minimum available block count across rank-local pools.

## Current Code State

### Existing Foundations

- The device abstraction already includes a device id: `Device::cuda(int id = 0)` in `include/backend/device/device.hpp`, and `Device` is hashable.
- `ExecutorConfig` already carries `device_type`, `device_id`, `data_type`, and `max_seq_len`: `include/zedinfer/activation.hpp`.
- `core::Context` owns multiple runtimes: `runtime_map_` in `include/backend/core/context/context.hpp` is keyed by `Device`.
- `Runtime` has a compute stream and a transfer stream: `include/backend/core/runtime/runtime.hpp`.
- Most CUDA ops call `setDevice(out->deviceType(), out->deviceId())` at entry. Switching devices within one thread is already an established pattern.
- Paged KV cache is already represented by `BlockPool`, `BlockAllocator`, and `SequenceBlockTable`; `BlockPool` takes `device_type` and `device_id` in its constructor.

### Single-GPU Assumptions

These should be treated as prerequisites for multi-GPU work:

- `Model::parse(model_path, target_device, ...)` receives only a device type, not a device id. `InferenceEngine::create()` receives a `device::Device`, but currently calls `model::Model::parse(model_path, device.type(), ...)`.
- `Model::load_weights()` loads weights for one target device. After CPU mmap, there is no shard spec and no per-rank slicing policy.
- `InferenceEngine` owns one `device_`, one `ExecutorConfig`, one `BlockPool`, one `BlockAllocator`, and one `DecodeScratch`.
- `ServingLoop::step()` builds one `PagedForwardContext`, calls either `transformer_forward()` or `hybrid_transformer_forward()`, and returns one logits tensor to the scheduler.
- `Scheduler` owns one block allocator. KV admission is based on one pool's `available_blocks()`.
- `SequenceBlockTable::fi` is a single-device FlashInfer runtime cache. It stores `cache_device_type/cache_device_id` and single-device page-table tensors. It cannot be reused directly as a multi-rank cache.
- `ops::linear()` and many other ops use `CHECK_SAME_DEVICE`. This is the right guard for single-device ops, but TP must not pass cross-device tensors into existing ops. TP needs rank-local tensors plus explicit collectives.
- The non-quantized cuBLASLt wrapper supports a stream argument, but `src/backend/ops/linear/nvidia/linear_nvidia.cu` currently calls `cublas::linear(..., K)` without passing `runtime.stream()`, so it uses the default `nullptr` stream. With NCCL in the path, GEMM and collectives should use the rank's compute stream, or explicit events, so ordering is controlled.
- Qwen3.5 hybrid scheduling explicitly routes to `schedule_hybrid_single()`, one request per step, because there is no fused multi-sequence linear-attention kernel yet.
- MoE `ExpertPool` currently supports `ALL_GPU` and `PINNED_LRU` for one compute GPU. Expert Parallel needs token dispatch/combine with all-to-all and should be a later phase.

## Reference Designs

### llama.cpp

Local source: `../llama.cpp`.

Useful ideas:

- The public API shape is clear: `llama_split_mode` has `NONE`, `LAYER`, `ROW`, and `TENSOR`; `llama_model_params` has `devices`, `n_gpu_layers`, `main_gpu`, and `tensor_split`.
- `llama_prepare_model_devices()` builds the participating device list first. `split_mode=NONE` keeps only `main_gpu`; `split_mode=TENSOR` uses a meta device to express tensor splitting.
- `llama-model.cpp` computes normalized split ratios from free GPU memory when `tensor_split` is not explicitly provided, then uses them for layer/row assignment.

Do not copy directly:

- ggml meta devices and split buffer types are part of llama.cpp's graph backend. zedinfer does not have the same graph execution model.
- zedinfer uses explicit tensor op calls. TP fits better as explicit `ShardedModelWeights + ParallelContext + CollectiveOps`, not as a fake device.

Transferable decisions for zedinfer:

- Keep `devices`, `main_device`, and `tensor_split` concepts in CLI/API.
- Reuse llama.cpp's free-memory-based allocation idea later for layer split / PP.
- The first TP version does not need llama.cpp row split buffers; it needs a sharded loader and a TP forward path.

### vLLM

Local source: `../vllm`.

Useful abstractions:

- `ParallelConfig` keeps `tensor_parallel_size`, `pipeline_parallel_size`, `data_parallel_size`, and `enable_expert_parallel` together. `world_size = TP * PP`, with DP as the replica dimension.
- `communication_op.py` exposes a thin model-facing collective surface: `tensor_model_parallel_all_reduce`, `all_gather`, and `reduce_scatter`.
- `parallel_state.py` uses `GroupCoordinator` to own rank groups, device communicators, and host communicators, so model code does not manage the low-level communicator directly.
- `ColumnParallelLinear` shards the output dimension and optionally all-gathers; `RowParallelLinear` shards the input dimension and all-reduces the output.
- `QKVParallelLinear` has the GQA behavior zedinfer should mirror: Q heads are sharded by TP; KV heads are sharded when `tp_size < total_kv_heads` and replicated when `tp_size >= total_kv_heads`.
- `VocabParallelEmbedding` / `ParallelLMHead` shard the vocab dimension and combine with all-reduce or gather when needed.

Transferable decisions for zedinfer:

- Add a thin distributed surface first. Do not let model layers call NCCL directly.
- Use vLLM's column/row/QKV/vocab rules as the dense TP weight sharding policy.
- In the first version, replicate embedding/lm_head or keep lm_head on rank 0 for correctness and a smaller implementation surface. Add vocab parallel later to reduce memory.

### Recent Systems And Research

- Megatron-LM defines the standard Transformer TP split: MLP gate/up are column-parallel, down is row-parallel; attention QKV is column-parallel and O is row-parallel.
- vLLM/PagedAttention shows that paged KV cache and continuous batching are core serving abstractions. Multi-GPU support should not regress the scheduler to static batching.
- Sarathi/Sarathi-Serve shows why chunked prefill and decode-maximal batching still matter after TP. zedinfer should keep the current chunked prefill strategy.
- DistServe-style prefill/decode disaggregation is useful for later cluster serving. It should not be introduced in the first single-node TP milestone.
- DeepEP-like communication libraries are relevant for MoE all-to-all dispatch/combine. They are references for later Qwen3/Qwen3.5 MoE EP, not dependencies for dense TP.
- FlashInfer's paged KV API is aligned with zedinfer's current `PagedForwardContext`. Under TP, each rank can continue using FlashInfer with rank-local KV cache and rank-local page metadata.

## Target Architecture

### Parallel Configuration

Add a dedicated parallel config instead of expanding `ExecutorConfig` with all parallel fields:

```cpp
struct ParallelConfig {
    int tensor_parallel_size = 1;
    int pipeline_parallel_size = 1;
    int data_parallel_size = 1;
    bool enable_expert_parallel = false;
    std::vector<int> devices;        // local CUDA device ids, e.g. {0, 1}
    int main_device = 0;             // logits/sampling owner
    std::vector<float> tensor_split; // optional capacity weights
};
```

Suggested locations:

- `include/zedinfer/parallel_config.hpp`
- Add CLI flags in `examples/*` or the existing option parser: `--devices 0,1`, `--tp-size 2`, `--pp-size 1`, `--main-device 0`.
- Add an `InferenceEngine::create()` overload or extend the existing signature: `create(model_path, device, sched_config, parallel_config)`.

Validation rules:

- First version: require `device.type() == ZEDINFER_DEVICE_NVIDIA` for `tp_size > 1`.
- `devices.size() == tensor_parallel_size`.
- `pipeline_parallel_size == 1`, `data_parallel_size == 1`, `enable_expert_parallel == false`.
- `num_attention_heads % tp_size == 0`.
- Use the vLLM KV-head rule: `local_kv_heads = total_kv_heads / tp_size` when `tp_size < total_kv_heads`; otherwise `local_kv_heads = 1` and KV heads are replicated across ranks.

### Runtime And Collective Surface

Add a distributed backend layer:

- `include/backend/distributed/parallel_context.hpp`
- `include/backend/distributed/collective_ops.hpp`
- `src/backend/distributed/nccl_collective_ops.cu`
- `src/backend/distributed/dummy_collective_ops.cpp`

Minimal interface:

```cpp
struct RankInfo {
    int global_rank = 0;
    int tp_rank = 0;
    int tp_size = 1;
    int device_id = 0;
};

class CollectiveOps {
public:
    virtual void all_reduce_sum(tensor_t inout) = 0;
    virtual void all_gather(tensor_t out, tensor_t local, int dim) = 0;
    virtual void reduce_scatter_sum(tensor_t out, tensor_t in, int dim) = 0;
    virtual void broadcast(tensor_t tensor, int src_rank) = 0;
    virtual void barrier() = 0;
};
```

Only `all_reduce_sum()` and `broadcast()` are required for the first version. Keep `all_gather()` and `reduce_scatter()` in the interface for later vocab parallel or context parallel work.

NCCL initialization:

- Single process, multiple GPUs. Create one `ncclUniqueId`, then have each rank worker call `setDevice(device_id)` and `ncclCommInitRank(comm, tp_size, id, tp_rank)` in its own thread.
- Each rank should use its own thread-local `core::Context`. This avoids sharing one `Runtime` across multiple threads.
- Each collective should use the current rank runtime's compute stream. If a separate communication stream is added later, synchronize with events.
- Use `ncclGroupStart()` / `ncclGroupEnd()` for multi-GPU initialization and teardown to reduce deadlock risk.
- Add an `--nccl=y` xmake option and link `nccl`. If NCCL is not enabled, `tp_size > 1` should fail with a clear error.

Why single process first:

- The current engine, scheduler, and serving loop are in-process objects. Single-process TP can reuse the existing request queue, prefix cache, and sampler.
- NCCL supports single-threaded control, one-thread-per-GPU control, and multi-process control. zedinfer should start with one thread per GPU because launches can run concurrently and the existing thread-local `core::Context` design fits it.
- Multi-process TP can come later, but it will require cross-process batch broadcast, request id mapping, error propagation, and IPC tensor handles.

### Rank Workers

Add `ParallelExecutor` or `TPExecutor`:

- `include/zedinfer/parallel_executor.hpp`
- `src/zedinfer/parallel_executor.cpp`

Responsibilities:

- Own `ParallelContext`, rank worker threads, and one `ExecutorConfig` per rank.
- During engine initialization, load each rank's sharded weights and create each rank-local KV pool.
- In `ServingLoop::step()`, receive a `BatchContext`, broadcast the same batch metadata to every rank, and trigger rank-local forward.
- Rank 0 returns logits to the scheduler. Other ranks return only status/errors.

Execution model:

1. The main thread schedules a batch and builds `BatchContext`.
2. `ParallelExecutor::forward(batch_ctx)` submits a task to every rank worker.
3. Each rank worker builds its rank-local `PagedForwardContext` and local scratch, then calls `tp_transformer_forward_rank()`.
4. Row-parallel projections call `CollectiveOps::all_reduce_sum()`.
5. Rank 0 computes or collects logits and returns them to the main thread.

Avoid a production path that uses one host thread to switch devices and launch kernels sequentially. That is useful as a debug mode but wastes TP concurrency. A `ZEDINFER_TP_SERIAL_LAUNCH=1` debug switch is fine; the default should be one worker thread per GPU.

## Implementation Phases

### M0: Single-GPU Device-Id Cleanup

Goal: make the existing single-GPU path correct on non-zero GPUs.

Changes:

- Change `Model::parse()` from `(model_path, target_device, gpu_memory_utilization)` to accept either `device::Device` or `(target_device, target_device_id, ...)`.
- Make `Model::load_weights()` accept `target_device_id`; replace all `tensor->to(target_device, 0)` call sites with the actual target id.
- Remove hardcoded `ExecutorConfig(target_device, 0, ...)` from Qwen3.5 / vision / MTP / ExpertPool initialization paths and pass the engine exec config instead.
- In `src/backend/ops/linear/nvidia/linear_nvidia.cu`, pass `runtime.stream()` to the non-quantized `cublas::linear()` path.
- Add `test-device-id-nonzero`: when at least two GPUs exist, load a small fixture or run a core op smoke test on `Device::cuda(1)`; otherwise skip.

Acceptance:

- `tp_size=1 --device 1` can run a dense model end to end.
- New tests skip cleanly on one-GPU machines.

### M1: ParallelConfig And NCCL Backend

Goal: add parallel configuration and collective infrastructure without changing model forward yet.

Changes:

- Add `ParallelConfig` and store it in `InferenceEngine`.
- Add `ParallelContext` with `tp_size`, `tp_rank`, `device_id`, and rank group metadata.
- Add an `--nccl` xmake option in `xmake/device/nvidia.lua` or the relevant backend xmake file, then link `nccl`.
- Add the `CollectiveOps` abstraction and NCCL implementation.
- Add a collective unit test:
  - Each rank creates a `[4]` BF16/FP32 tensor and fills it with `rank + 1`.
  - After `all_reduce_sum()`, every rank sees `tp_size * (tp_size + 1) / 2`.
  - Test `broadcast(src=0)`.
  - Run repeated collectives to verify stream ordering.

Acceptance:

- `xmake f --nv-gpu=y --nccl=y` builds.
- `xmake run test-nccl-collectives --devices 0,1` passes.
- `--tp-size 2` without `--nccl=y` reports a clear error.

### M2: ShardedModelWeights And Sharded Loading

Goal: slice weights while loading, instead of loading the full model on every GPU and slicing afterward.

New structures:

```cpp
enum class ShardKind {
    Replicated,
    Column,      // shard output dim, weight [out, in]
    Row,         // shard input dim, weight [out, in]
    QKV,         // Q/K/V GQA-aware column shard
    Vocab,       // shard vocab dim
    Expert       // later
};

struct WeightShardSpec {
    ShardKind kind;
    int axis;
    int tp_rank;
    int tp_size;
    int64_t start;
    int64_t length;
};
```

Suggested locations:

- `include/frontend/models/sharded_weights.hpp`
- `src/frontend/models/sharded_weights.cpp`
- Call the shard resolver from `src/frontend/models/base.cpp`.

First dense-model sharding table:

| Weight | First Strategy | Notes |
| --- | --- | --- |
| `embed_tokens.weight` | Replicated | Correctness first; later use vocab parallel |
| `lm_head.weight` | Replicated on rank 0 or all ranks | First version can compute logits only on rank 0 |
| `*.input_layernorm.weight` | Replicated | Small tensor |
| `*.post_attention_layernorm.weight` | Replicated | Small tensor |
| `*.self_attn.q_proj.weight` | Column | Local Q heads |
| `*.self_attn.k_proj.weight` | QKV | GQA-aware KV heads |
| `*.self_attn.v_proj.weight` | QKV | GQA-aware KV heads |
| `*.self_attn.o_proj.weight` | Row | Input dim is local attention heads |
| `*.mlp.gate_proj.weight` | Column | Local intermediate |
| `*.mlp.up_proj.weight` | Column | Local intermediate |
| `*.mlp.down_proj.weight` | Row | Local intermediate to hidden, then all-reduce |
| `*.mlp.gate/up/down bias` | Follow weight | Qwen usually has no bias, but define the rule |
| `norm.weight` | Replicated | Final norm |
| MoE router/expert | Deferred | Disable TP+MoE first |

Slicing details:

- zedinfer linear weights are row-major `[out_features, in_features]`.
- Column parallel slices `out_features`, so local weight shape is `[out/tp, in]`.
- Row parallel slices `in_features`, so local weight shape is `[out, in/tp]`; the input tensor must also be a local shard.
- Qwen2/Qwen3 usually use separate `q_proj/k_proj/v_proj` weights, so fused QKV handling is not required first. Keep the resolver extensible for fused QKV.
- GQA KV rule:
  - If `tp_size <= num_kv_heads` and divisible, each rank owns `num_kv_heads/tp_size` KV heads.
  - If `tp_size > num_kv_heads`, replicate KV heads; implement the rank-to-KV-head mapping with vLLM semantics.
- Defer GPTQ-Int4. Packed `qweight`, scale, and `g_idx` slicing must respect format-specific packing and cannot reuse the BF16 slicing rule blindly.
- SafeTensors loading should support creating a CPU view/slice from the mmap source and then copying only that slice to the target rank. Do not copy the full tensor to every GPU first.

Acceptance:

- Unit tests cover shape and offset for every shard kind.
- Random CPU tensor slicing matches a hand-written reference slice.
- Loading a small model reports roughly balanced weight memory per rank.

### M3: Dense TP Forward

Goal: run Qwen2/Qwen3 dense TP forward.

Suggested additions:

- `include/frontend/models/tp_forward_config.hpp`
- `src/frontend/models/tp_transformer_forward.cpp`
- `include/frontend/models/parallel_tensor.hpp`, if a multi-rank tensor handle is useful.

Do not contort the existing `transformer_forward()` into a distributed path. Add a TP path and keep the single-GPU path unchanged:

```cpp
if (parallel_config.tensor_parallel_size > 1) {
    logits = parallel_executor_->forward(batch_ctx);
} else {
    logits = model::transformer_forward(...);
}
```

Rank-local data flow:

1. Replicate input token ids on every rank.
2. In the first version, each rank may compute a full hidden tensor `[N, hidden]` from replicated embeddings. Rank-0 embedding plus broadcast is also possible, but replicated embedding is simpler.
3. For every layer:
   - RMSNorm uses replicated weights and replicated hidden input.
   - Q/K/V column-parallel projections produce local `q_local/k_local/v_local`.
   - RoPE operates only on local heads.
   - KV scatter writes into rank-local KV cache.
   - Attention computes only local Q heads against local or replicated KV heads, producing `attn_local [N, local_q_heads * head_dim]`.
   - O projection is row-parallel: `linear(local_out_partial [N, hidden], attn_local, o_proj_local)`, then `all_reduce_sum(local_out_partial)` to produce replicated attention output.
   - Residual add produces replicated hidden.
   - MLP gate/up column-parallel projections produce local intermediate tensors.
   - `silu_mul` runs on local intermediate tensors.
   - Down projection is row-parallel, producing a `[N, hidden]` partial, then `all_reduce_sum()`.
   - Residual add produces replicated hidden for the next layer.
4. Final norm is replicated.
5. First-version logits are computed only on rank 0:
   - Rank 0 owns full `lm_head.weight`.
   - Other ranks do not compute logits.
   - The scheduler consumes rank-0 logits as before.

Memory tradeoff:

- Replicating `[N, hidden]` on every rank follows the standard Megatron/vLLM TP shape.
- Replicating embedding/lm_head costs extra vocab memory. This is acceptable for the first version; add vocab parallel once correctness is stable.

Important constraints:

- Existing `ops::linear()` requires same-device tensors. TP forward must pass only rank-local tensors to rank-local ops.
- `DecodeScratch` is currently single-device. TP needs one scratch instance per rank, with local-dimension shapes.
- `PagedForwardContext` currently binds directly to one `BlockAllocator`. TP needs rank-local contexts, and `SequenceBlockTable` runtime caches cannot be shared across ranks.
- The result of all-reduce should remain a tensor on the current rank's device. Do not move it to host just because sampling happens on rank 0 later.

Acceptance:

- Small-batch prefill and decode logits match `tp=1`.
- When a batch contains both decode and prefill rows, output row order matches `ScheduledBatch::build_context()`.
- If one rank worker fails, the main thread fails the batch instead of hanging clients.

### M4: Rank-Local KV Cache

Goal: use separate KV cache per TP rank while keeping global scheduling and prefix cache behavior.

Suggested data structures:

```cpp
struct ParallelBlockPools {
    std::vector<std::unique_ptr<kvcache::BlockPool>> pools;
    std::vector<std::unique_ptr<kvcache::BlockAllocator>> allocators;
};

struct ParallelSequenceBlockTable {
    std::vector<kvcache::SequenceBlockTable> per_rank;
};
```

The first version does not have to expose `ParallelSequenceBlockTable` in the public request API. It can be optional state inside `InferenceRequest`. The scheduler can still operate on a logical table, while rank-local forward reads the per-rank tables.

KV head count:

- `BlockConfig::num_kv_heads` should be local KV heads per rank, not global `num_key_value_heads`.
- When KV heads are replicated, each rank should store the corresponding replicated local heads, and page-pool sizing must use the local head count.

Admission:

- For TP, `Scheduler::can_admit()` should use the minimum available block count across rank allocators: `min(allocators[i]->available_blocks())`.
- `ensure_blocks()` must run for every rank table. If any rank fails, rollback already-extended ranks, or do a dry-run capacity check before committing allocation.

Prefix cache:

- Current `PrefixCache` stores one `SequenceBlockTable`. Under TP it must store all rank logical page ids, or store prefix key to per-rank tables.
- `SequenceBlockTable::fi` is runtime cache and should not be copied into prefix cache. The current copy semantics already copy only logical pages, which is the right direction.
- Prefix block refcounts must be incremented/released on every rank pool. Rank 0 alone is not enough.

FlashInfer cache:

- Current `SequenceBlockTable::fi` is one device-bound cache. Multi-rank TP needs one `fi` per rank.
- If `std::vector<SequenceBlockTable> per_rank` is used, this falls out naturally.
- If only one logical table is kept, change `fi` into `std::vector<FlashInferSeqCache> per_rank_fi`.

Acceptance:

- A long prompt that spans multiple pages has consistent page counts across ranks.
- On prefix hit, every rank's refcount is incremented correctly.
- After request release/free, every rank has correct `free_blocks/evictable_count` values.

### M5: Scheduler And Serving Integration

Goal: integrate TP into the existing serving loop with minimal disruption.

Changes:

- Add to `InferenceEngine`:
  - `ParallelConfig parallel_config_`
  - `std::unique_ptr<ParallelExecutor> parallel_executor_`
  - `std::vector<std::unique_ptr<BlockPool>> block_pools_`
  - `std::vector<std::unique_ptr<BlockAllocator>> block_allocators_`
  - `std::vector<std::unique_ptr<DecodeScratch>> decode_scratch_per_rank_`
- In `ServingLoop::step()`:
  - `tp_size == 1` uses the current path.
  - `tp_size > 1` and non-hybrid model calls `parallel_executor_->forward(batch_ctx, scratch_policy)`.
  - Hybrid model plus `tp_size > 1` fails with a clear first-version error.
- In `Scheduler`:
  - Add an `IBlockAdmission` or `BlockAllocatorView` abstraction to hide single-pool versus multi-pool admission.
  - Keep result processing on rank-0 logits.
- In `Profiler`:
  - Either disable profiler under TP first, or route it through `parallel_executor`. Do not let profiler silently use the single-GPU `block_pool()`.

Sampling:

- First version: sampling stays on rank 0. `Sampler`, `ArgmaxSampler`, and `GeneralSampler` consume rank-0 logits.
- If vocab-parallel logits are added later, sampling needs distributed top-k/top-p: local top-k then gather candidates to rank 0, or all-gather logits. Do not implement this in the first TP milestone.

Acceptance:

- CLI `chat` and HTTP serving both run dense models under TP.
- Continuous batching with multiple requests preserves request-to-output ownership.
- Error paths release KV blocks on every rank.

### M6: Performance And Memory Improvements

After first-version correctness:

- Add vocab-parallel embedding/lm_head to reduce memory for large-vocab models.
- Reuse workspace for repeated small all-reduces and avoid per-step allocation.
- Keep all-reduce and following kernels ordered on one compute stream. If overlap is needed, add a communication stream plus events.
- Choose rank order from NVLink/PCIe topology.
- For single-token decode, evaluate custom all-reduce or NCCL LL/LL128 protocol tuning.
- Before CUDA Graph support, verify whether NCCL collectives can be captured and define shape buckets for dynamic batches.

## Later Parallel Modes

### Pipeline / Layer Split

This is useful when the main goal is fitting a model that does not fit on one GPU. On slower interconnects it may be more practical than TP.

Implementation path:

- Reuse the llama.cpp-style `n_gpu_layers` and free-memory-based `tensor_split` idea to assign contiguous layer ranges to GPUs.
- Transfer hidden activations at layer-stage boundaries. For decode, this sends one activation per token per boundary, often less communication than per-layer TP all-reduce.
- Add pipeline scheduling/microbatching to improve throughput; otherwise single-request latency remains stage-serial.
- This can be M7 after dense TP, or earlier if the main user goal is fitting larger models on consumer multi-GPU systems.

### Data Parallel

This improves concurrent throughput but does not solve single-model memory.

Implementation path:

- Each DP replica is a full `InferenceEngine` or `ParallelExecutor`.
- The HTTP layer or an upper router assigns requests to replicas.
- Prefix cache and KV cache are not shared across replicas.

### MoE Expert Parallel

Qwen3-MoE/Qwen3.5-MoE EP should not be mixed into the first dense TP milestone. It requires:

- Dispatching tokens to expert-owner ranks based on router top-k.
- Combining expert outputs back to the original token owner.
- all-to-all or grouped send/recv; high-performance implementation can reference DeepEP.
- Current `ExpertPool::PINNED_LRU` can remain a per-rank local expert offload policy, but EP changes expert placement and migration granularity.

### Qwen3.5 Hybrid

Defer until dense TP and multi-rank KV cache are stable. Blockers:

- `schedule_hybrid_single()` currently runs one request per step. TP can speed up one request step, but it does not solve hybrid multi-request throughput.
- `SSMStatePool` is per-request recurrent state. TP requires splitting linear-attention state per rank and handling the MTP verify temp slot.
- GatedDeltaNet kernels currently do not have a fused multi-sequence path. Adding TP before this is stable makes communication and recurrent-state sharding land at the same time.

### Prefill/Decode Disaggregation

DistServe-style disaggregation is a later cluster-serving direction:

- Prefill GPUs compute prompt KV, then migrate KV pages to decode GPUs.
- The scheduler must allocate resources based on TTFT/TPOT.
- zedinfer's paged KV cache and prefix cache are good foundations, but it still needs cross-GPU/cross-process KV page transfer and an ownership protocol.

## File-Level Change List

| Phase | Files |
| --- | --- |
| M0 device id | `include/frontend/models/base.hpp`, `src/frontend/models/base.cpp`, `src/zedinfer/engine.cpp`, `src/frontend/models/qwen3_5.cpp`, `src/backend/ops/linear/nvidia/linear_nvidia.cu` |
| config | `include/zedinfer/parallel_config.hpp`, `include/zedinfer/engine.hpp`, `src/zedinfer/engine.cpp`, CLI examples |
| NCCL | `include/backend/distributed/*`, `src/backend/distributed/*`, `xmake/device/nvidia.lua`, `xmake/backend.lua` |
| sharded weights | `include/frontend/models/sharded_weights.hpp`, `src/frontend/models/sharded_weights.cpp`, `src/frontend/models/base.cpp`, safetensors loader helpers |
| TP forward | `include/frontend/models/tp_forward_config.hpp`, `src/frontend/models/tp_transformer_forward.cpp`, `include/frontend/models/parallel_tensor.hpp` |
| KV cache | `include/backend/kvcache/block_pool.hpp`, `src/backend/kvcache/block_pool.cpp`, `include/frontend/models/paged_forward_context.hpp`, `src/frontend/models/paged_forward_context.cpp`, `include/zedinfer/request.hpp` |
| serving | `include/zedinfer/parallel_executor.hpp`, `src/zedinfer/parallel_executor.cpp`, `src/zedinfer/serving_loop.cpp`, `include/zedinfer/scheduler.hpp`, `src/zedinfer/scheduler.cpp` |
| tests | `tests/test-nccl-collectives.cpp`, `tests/test-sharded-weights.cpp`, `tests/test-tp-forward-parity.cpp`, `tests/test-tp-kvcache.cpp` |

## Test Matrix

Unit tests:

- Non-zero device id smoke test.
- NCCL all-reduce/broadcast.
- Shard spec shapes and slice offsets.
- GQA KV-head allocation: `kv_heads < tp_size`, `kv_heads == tp_size`, `kv_heads > tp_size`.
- TP BlockPool admission and release.

Numerical tests:

- Use a tiny random model fixture and compare `tp=1` with `tp=2`:
  - Single-token decode logits.
  - Multi-token prefill logits.
  - Prefill plus two decode steps.
  - Multi-request row order inside a batch.
- Start BF16 tolerance at `rtol=1e-2, atol=1e-2`, then tighten based on actual kernel error.

Integration tests:

- Dense-model TP smoke via `chat`.
- Dense-model TP smoke via HTTP `/v1/chat/completions`.
- Matching output after prefix cache hit/miss.
- Admission under KV cache pressure does not deadlock.

Performance tests:

- `tp=1/2/4`, prefill tokens: 128, 1024, 4096.
- Decode batch size: 1, 8, 32.
- Record TTFT, TPOT, tokens/s, per-rank memory, and NCCL all-reduce time.
- Keep PCIe and NVLink results separate. They should not be summarized as one performance conclusion.

## Main Risks

- TP is interconnect-sensitive. On PCIe systems, dense TP may solve memory without improving decode latency.
- Replicated lm_head can become a first-version memory bottleneck. Large-vocab models or small GPUs may require vocab parallel earlier.
- GPTQ packed slicing is error-prone. Wait until BF16/FP16 TP parity is stable before adding GPTQ.
- Multi-rank KV rollback is tricky. Prefer a dry-run capacity check before commit-style allocation.
- With multiple worker threads, all ranks must enter NCCL collectives in the same order. On errors, mark the executor fatal and make all ranks exit the loop, instead of leaving one rank blocked inside all-reduce.

## References

Local source:

- `../llama.cpp/include/llama.h`
- `../llama.cpp/src/llama.cpp`
- `../llama.cpp/src/llama-model.cpp`
- `../vllm/vllm/config/parallel.py`
- `../vllm/vllm/distributed/parallel_state.py`
- `../vllm/vllm/distributed/communication_op.py`
- `../vllm/vllm/model_executor/layers/linear.py`
- `../vllm/vllm/model_executor/layers/vocab_parallel_embedding.py`

External references:

- vLLM Parallelism and Scaling: https://docs.vllm.ai/en/latest/serving/parallelism_scaling.html
- NVIDIA NCCL overview and collectives: https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/overview.html
- NCCL collective operations: https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html
- Megatron-LM tensor parallel paper: https://arxiv.org/abs/1909.08053
- vLLM / PagedAttention paper: https://arxiv.org/abs/2309.06180
- vLLM Paged Attention design notes: https://docs.vllm.ai/en/latest/design/paged_attention/
- FlashInfer paged KV cache API: https://docs.flashinfer.ai/api/page.html
- Sarathi-Serve: https://arxiv.org/abs/2403.02310
- DistServe: https://arxiv.org/abs/2401.09670
- DeepEP: https://github.com/deepseek-ai/DeepEP
