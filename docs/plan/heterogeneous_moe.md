# Heterogeneous MoE Design

## Goal

Enable running large Mixture-of-Experts (MoE) models like Qwen-30B-A3B on consumer GPUs (24GB VRAM) by:
1. Keeping hot experts on GPU
2. Offloading cold experts to CPU (pinned host memory)
3. Dynamically prefetching experts based on routing predictions
4. Combining with INT8/INT4 quantization for maximum memory savings

## Target Model: Qwen-30B-A3B

| Parameter | Value |
|-----------|-------|
| Total parameters | ~30B |
| Architecture | MoE Transformer |
| Experts per layer | 128 |
| Top-K active experts | 8 per token |
| Expert FFN hidden | varies per layer |
| Shared attention | Yes (non-expert) |
| Total expert parameters | ~24B (80% of model) |
| Attention + embedding + norm | ~6B (20%) |

## Expert Placement

### Memory Budget (24GB GPU, INT4)

```
Attention + embedding + norm (non-expert): ~6B params * 0.5 bytes (INT4) = ~3 GB
Expert weights: ~24B params * 0.5 bytes (INT4) = ~12 GB
KV cache budget: ~4-8 GB
Scratch buffers: ~1 GB

Total expert capacity on GPU: ~12 GB for experts
Expert capacity at INT4: ~24B params -> 12 GB -> all fit on GPU

With INT8: 24B * 1 byte = 24 GB -> exceeds budget
Non-expert INT8: 6B * 1 = 6 GB
Expert INT8: need ~18 GB -> only ~12 GB available -> offload ~6GB worth
```

### Placement Strategy

```cpp
// include/backend/moe/expert_placement.hpp

enum class ExpertLocation {
    GPU,        // resident on GPU memory
    CPU_PINNED, // in pinned host memory (fast PCIe transfer)
    CPU_PAGED,  // in regular host memory (slower transfer)
    DISK        // not loaded (extreme memory pressure)
};

struct ExpertPlacementPolicy {
    enum class Strategy {
        ALL_GPU,       // Keep all experts on GPU (enough VRAM)
        STATIC_SPLIT,  // Fixed partition: top-N on GPU, rest on CPU
        DYNAMIC_LRU,   // LRU-based: evict least-recently-used to CPU
        PREDICTIVE     // Prefetch based on routing prediction
    };
    Strategy strategy = Strategy::DYNAMIC_LRU;

    int gpu_expert_budget;       // max experts resident on GPU simultaneously
    int prefetch_lookahead = 1;  // layers ahead to prefetch
    float hot_threshold = 0.1;   // frequency threshold for "hot" expert
};

struct ExpertState {
    int expert_id;
    int layer_idx;
    ExpertLocation location;
    tensor_t weights_gate;   // gate_proj weight
    tensor_t weights_up;     // up_proj weight
    tensor_t weights_down;   // down_proj weight
    uint64_t last_access;    // monotonic counter
    uint64_t access_count;   // total access count
};
```

## CPU/GPU Memory Ownership

### Architecture

```
GPU Memory:
  +-- Non-expert weights (always resident)
  +-- Expert GPU Cache (fixed-size pool of expert weight slots)
  +-- KV Cache (paged blocks)
  +-- Scratch buffers

CPU Pinned Memory:
  +-- Expert CPU Pool (offloaded expert weights)
  +-- Prefetch staging area

CPU Regular Memory:
  +-- Model metadata
  +-- Tokenizer
```

### Expert Weight Pool

```cpp
// include/backend/moe/expert_pool.hpp

class ExpertPool {
public:
    ExpertPool(int num_gpu_slots,       // max experts on GPU at once
               int num_cpu_slots,       // max experts in pinned memory
               size_t expert_size_bytes, // bytes per expert (all 3 weight matrices)
               zedinferDeviceType_t gpu_device,
               int gpu_device_id);

    // Ensure expert is on GPU, transfer if needed
    // Returns GPU pointers to gate/up/down weights
    ExpertWeightPtrs ensure_on_gpu(int layer_idx, int expert_id);

    // Async prefetch: start H2D transfer without blocking
    void prefetch_to_gpu(int layer_idx, int expert_id, zedinferStream_t stream);

    // Evict least-recently-used expert from GPU to CPU
    void evict_lru();

    // Stats
    int gpu_residents() const;
    int transfers_this_step() const;
    double avg_transfer_time_ms() const;

private:
    struct Slot {
        int layer_idx = -1;
        int expert_id = -1;
        bool occupied = false;
        uint64_t last_access = 0;
        tensor_t gpu_buffer;     // pre-allocated GPU memory
        tensor_t cpu_buffer;     // pinned host memory
        bool transfer_pending = false;
    };

    std::vector<Slot> gpu_slots_;
    std::vector<Slot> cpu_slots_;
    std::unordered_map<std::pair<int,int>, int> location_map_; // (layer, expert) -> slot index
};
```

## Transfer Policy

### When to Transfer

1. **On-demand**: When router selects an expert not on GPU, transfer it synchronously. Simple but adds latency.
2. **Predictive prefetch**: After computing attention for layer L, predict which experts will be needed for layer L+1 based on routing scores, and start async transfers. Hides latency behind attention computation.
3. **Static placement with LRU**: Keep the N most-frequently-accessed experts on GPU. When a cold expert is needed, evict LRU and transfer.

Recommended: **Static placement with LRU + opportunistic prefetch**.

### Transfer Mechanics

```
PCIe 4.0 x16: ~25 GB/s (practical ~20 GB/s)

Expert size (INT4, Qwen-30B-A3B, estimated):
  gate_proj: ~2M params * 0.5 bytes = ~1 MB
  up_proj: ~2M params * 0.5 bytes = ~1 MB
  down_proj: ~2M params * 0.5 bytes = ~1 MB
  Total per expert: ~3 MB

Transfer time: 3 MB / 20 GB/s = ~0.15 ms

With INT8:
  Total per expert: ~6 MB
  Transfer time: 6 MB / 20 GB/s = ~0.3 ms
```

This is fast enough to hide behind compute for most cases, especially with prefetching.

### Async Transfer Pipeline

```
Step 1: Compute attention for layer L
        Meanwhile: prefetch experts for layer L (if not on GPU)

Step 2: Router computes expert assignments for layer L
        Start async transfer of assigned experts not on GPU

Step 3: Wait for transfers to complete
        Execute expert FFN computation

Step 4: Compute attention for layer L+1
        Meanwhile: prefetch experts for layer L+1 based on L's routing pattern
```

Using CUDA streams:
- **Compute stream**: attention, linear, norm operations
- **Transfer stream**: H2D copies for expert weights
- Events synchronize between streams when expert weights are needed

## MoE Forward Path

```cpp
// In MoEModel::forward() (layer-level):

tensor_t moe_layer_forward(
    tensor_t hidden,           // [batch_tokens, hidden_size]
    int layer_idx,
    MoEWeights &moe_weights,
    ExpertPool &expert_pool,
    ScratchBuffers &scratch) {

    // 1. Router: compute expert assignments
    // gate_logits = hidden @ router_weight  -> [batch_tokens, num_experts]
    auto gate_logits = scratch.gate_logits;
    ops::linear(gate_logits, hidden, moe_weights.router_weight[layer_idx], nullptr);

    // top-k selection: [batch_tokens, top_k] expert IDs + weights
    auto [expert_ids, expert_weights] = ops::topk_softmax(gate_logits, top_k);

    // 2. Prefetch experts that aren't on GPU
    std::set<int> needed_experts;
    for (int i = 0; i < batch_tokens * top_k; ++i) {
        needed_experts.insert(expert_ids_host[i]);
    }
    for (int eid : needed_experts) {
        expert_pool.prefetch_to_gpu(layer_idx, eid, transfer_stream);
    }

    // 3. Group tokens by expert for batched execution
    // token_to_expert[expert_id] = list of (token_idx, weight)
    auto groups = group_by_expert(expert_ids, expert_weights, batch_tokens, top_k);

    // 4. Execute each expert on its token group
    auto output = zeros_like(hidden);
    for (auto &[eid, token_group] : groups) {
        auto expert_ptrs = expert_pool.ensure_on_gpu(layer_idx, eid);

        // Gather tokens for this expert
        auto expert_input = gather(hidden, token_group.indices);  // [group_size, hidden_size]

        // Expert FFN: gate + up -> swiglu -> down
        auto gate_out = ops::linear(expert_input, expert_ptrs.gate);
        auto up_out = ops::linear(expert_input, expert_ptrs.up);
        ops::swiglu(gate_out, gate_out, up_out);
        auto down_out = ops::linear(gate_out, expert_ptrs.down);

        // Scatter weighted output back
        scatter_add(output, token_group.indices, down_out, token_group.weights);
    }

    return output;
}
```

## Runtime Scheduling Interaction

### Expert-Aware Batch Scheduling

The scheduler must account for expert transfer overhead when planning batches:

1. **Budget accounting**: Each request's token may trigger expert transfers. The scheduler estimates transfer cost and limits batch size to stay within latency budget.

2. **Locality-aware batching**: Prefer batching requests that use similar experts (reduces total unique experts needed per step). The router's softmax distribution can be cached from the previous token to predict next-step experts.

3. **Expert residency signal**: The scheduler queries `ExpertPool::gpu_residents()` to estimate how many transfers will be needed. If many cold experts are expected, reduce batch size to absorb transfer latency.

### Interaction with Continuous Batching

```
schedule():
    // Existing logic for batch assembly
    batch = assemble_decode_and_prefill_batch()

    // New: estimate expert transfer budget
    estimated_unique_experts = estimate_experts_needed(batch)
    cold_experts = estimated_unique_experts - expert_pool.gpu_residents()
    estimated_transfer_ms = cold_experts * avg_expert_transfer_time

    // If transfer overhead too high, reduce batch
    while estimated_transfer_ms > max_transfer_budget_ms:
        remove lowest-priority request from batch
        recalculate
```

## Expected Latency / Throughput Tradeoffs

### Scenario: Qwen-30B-A3B on RTX 4090 (24GB)

**All experts on GPU (INT4, fits in 24GB)**:
- Decode latency: ~15-25 ms/token (compute-bound)
- No transfer overhead
- Best case scenario

**80% experts on GPU, 20% on CPU (INT8, doesn't fully fit)**:
- Decode latency: ~20-30 ms/token
- ~20% of tokens need 1-2 cold expert transfers
- Cold expert transfer: ~0.3 ms each (INT8)
- Average overhead per token: ~0.3 * 0.2 * 2 = ~0.12 ms (negligible)
- With prefetching: overhead hidden behind attention compute

**50% experts on GPU (FP16, doesn't fit)**:
- Decode latency: ~30-50 ms/token
- ~50% of tokens need cold expert transfers
- Multiple transfers per step, harder to hide
- Recommendation: Use INT8 or INT4 instead

### Key Insight

With INT4 quantization, Qwen-30B-A3B likely fits entirely on a 24GB GPU. Heterogeneous execution becomes relevant for:
1. INT8 models that don't fully fit
2. Even larger models (70B+ MoE)
3. GPUs with less VRAM (16GB, 12GB)

## Files to Create

| File | Purpose |
|------|---------|
| `include/frontend/models/moe_base.hpp` | MoE model base class |
| `include/backend/moe/expert_pool.hpp` | Expert weight pool (GPU + CPU) |
| `include/backend/moe/expert_placement.hpp` | Placement policy |
| `include/backend/moe/router.hpp` | Expert routing (top-k selection) |
| `src/backend/moe/expert_pool.cpp` | Pool implementation |
| `src/backend/ops/moe/topk.cpp` | Top-K expert selection |
| `src/backend/ops/moe/scatter_gather.cpp` | Token gather/scatter by expert |

## Dependencies

MoE support requires:
1. **Quantization** (Stage 9-10): INT4 to fit model on GPU
2. **Paged KV cache** (Stage 6): Memory efficiency for multi-user
3. **Continuous batching** (Stage 5): Efficient multi-request serving
4. **Pinned memory** (Runtime API extension): For fast CPU<->GPU transfer
