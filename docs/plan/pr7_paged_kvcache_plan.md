# PR-7: Paged KV Cache and Block Pool (Single-Request) - Detailed Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the dynamically-growing contiguous KV cache with a statically-allocated block pool, where K/V data is stored in fixed-size blocks managed by a block allocator. The model forward path is unchanged (contiguous gather fallback). This establishes the memory foundation for paged attention (PR-8) and continuous batching (PR-9).

**Architecture:** A `BlockPool` pre-allocates all KV cache memory at engine init (sized from free VRAM). A `BlockAllocator` manages block assignment per-request. `PagedKVCache` implements the existing `KVCache` interface by writing new K/V into blocks and gathering them into contiguous buffers for the existing attention kernel. The model forward code is untouched.

**Tech Stack:** C++17, existing `KVCache` base class, existing `Tensor` / `Storage` infrastructure, `cudaMemGetInfo` for VRAM query.

---

## 1. Current State Analysis

### 1.1 How the Model Uses KV Cache

In `qwen2.cpp:49-148` (and similarly `qwen3.cpp`), each transformer layer does:

```cpp
// Write new V into cache (line 102-103)
auto v_slot = kvcache.get_v_cache_slice(L, past_len, sl);
ops::linear(v_slot->view({sl, kv_dim}), normed, W("v_proj.weight"), W("v_proj.bias"));

// Write new K into cache after RoPE (line 108-109)
auto k_slot = kvcache.get_k_cache_slice(L, past_len, sl);
ops::rope(k_slot->view({...}), k_tmp->view({...}), pos_ids, cfg.rope_theta);

// Read full K/V history for attention (line 112)
ops::self_attention(attn, q_rope,
    kvcache.get_k_cache_slice(L, past_len + sl),
    kvcache.get_v_cache_slice(L, past_len + sl), scale);

// After all layers (line 146)
kvcache.update_seq_len(sl);
```

**Critical observation:** The model writes new K/V via `get_{k,v}_cache_slice(layer, past_len, seq_len)` — which returns a *writable tensor view*. The model reads full history via `get_{k,v}_cache_slice(layer, total_len)` — which returns a *contiguous tensor* `[total_len, nkvhead, head_dim]`.

### 1.2 DynamicKVCache Limitations

- **Per-layer contiguous allocation**: Each layer's K and V are stored as `[capacity, num_kv_heads, head_dim]` tensors.
- **Growth via realloc+memcpy**: `ensure_capacity()` -> `grow_cache()` -> allocate new tensor + copy all data. Expensive and causes memory fragmentation.
- **No global memory budget**: Multiple sessions can over-allocate independently.
- **Per-session ownership**: Each session owns a `DynamicKVCache`. No sharing or pooling.

### 1.3 KVCache Interface (`include/backend/kvcache/base.hpp`)

```cpp
class KVCache {
public:
    int current_length() const;
    virtual int allocated_capacity() const = 0;
    virtual tensor_t get_k_cache(int layer_idx) = 0;
    virtual tensor_t get_v_cache(int layer_idx) = 0;
    virtual tensor_t get_k_cache_slice(int layer_idx, int total_len) = 0;
    virtual tensor_t get_v_cache_slice(int layer_idx, int total_len) = 0;
    virtual tensor_t get_k_cache_slice(int layer_idx, int past_len, int seq_len) = 0;
    virtual tensor_t get_v_cache_slice(int layer_idx, int past_len, int seq_len) = 0;
    void update_seq_len(int new_tokens);
    virtual void reset();
    virtual size_t memory_usage() const = 0;
    virtual float utilization() const = 0;
protected:
    KVCacheConfig config_;
    int current_length_;
    std::vector<tensor_t> k_caches_;
    std::vector<tensor_t> v_caches_;
};
```

The `PagedKVCache` must implement this same interface so the model forward code is unchanged.

---

## 2. Design

### 2.1 Block Layout

Each block holds `block_size` tokens of KV data for **one layer**, **one type** (K or V):

```
Block memory layout:
  [block_size, num_kv_heads, head_dim]  (contiguous)

Block bytes = block_size * num_kv_heads * head_dim * dtype_size
```

The pool contains `num_blocks` such blocks. K and V use separate blocks from the same pool.

A request with `seq_len` tokens at `num_layers` layers needs:
```
blocks_per_layer = ceil(seq_len / block_size)
total_blocks = blocks_per_layer * num_layers * 2 (K + V)
```

### 2.2 Block Pool

```cpp
// include/backend/kvcache/block_pool.hpp

struct BlockConfig {
    int block_size = 16;          // tokens per block
    int num_kv_heads;
    int head_dim;
    zedinferDataType_t dtype;
};

class BlockPool {
public:
    // Allocate a contiguous memory region for num_blocks blocks.
    // On GPU: one large cudaMalloc. On CPU: one large malloc.
    BlockPool(BlockConfig config, int num_blocks,
              zedinferDeviceType_t device_type, int device_id);

    int allocate();               // returns block_id (0..num_blocks-1), -1 if full
    void free(int block_id);
    void* block_data(int block_id) const;  // pointer to block's memory
    int total_blocks() const;
    int free_blocks() const;
    int used_blocks() const;
    const BlockConfig& config() const;
    size_t memory_usage() const;  // total pool bytes

private:
    BlockConfig config_;
    zedinferDeviceType_t device_type_;
    int device_id_;
    int num_blocks_;
    size_t block_bytes_;          // bytes per block

    void* pool_memory_ = nullptr; // one contiguous allocation
    std::vector<bool> allocated_; // bitmap: true = in use
    int free_count_;
    int next_free_ = 0;          // hint for next allocation scan
};
```

**Design decisions:**

- **Single contiguous allocation:** The pool allocates one large chunk (`num_blocks * block_bytes`) and indexes into it. No per-block malloc/free.
- **Bitmap allocator:** Simple `vector<bool>` tracks free/used. Sufficient for single-request mode. PR-9 may replace with a free-list for O(1) allocation.
- **Separate K/V blocks:** K and V blocks are the same size and come from the same pool. A request allocates `2 * num_layers * blocks_per_layer` blocks.

### 2.3 Block Allocator

```cpp
// include/backend/kvcache/block_pool.hpp (same header)

struct SequenceBlockTable {
    // block_table[layer][block_idx] = physical block ID in pool
    std::vector<std::vector<int>> k_blocks;  // [num_layers][blocks_per_layer]
    std::vector<std::vector<int>> v_blocks;  // [num_layers][blocks_per_layer]
    int seq_len = 0;                         // current token count
    int num_layers = 0;
};

class BlockAllocator {
public:
    BlockAllocator(BlockPool &pool, int num_layers);

    // Allocate initial blocks for a sequence with estimated token count.
    SequenceBlockTable allocate_sequence(int estimated_tokens);

    // Allocate one more block for a specific layer+type when current blocks are full.
    // Returns the new block ID. Throws if pool is full.
    int extend_sequence(SequenceBlockTable &table, int layer, bool is_k);

    // Free all blocks held by a sequence.
    void free_sequence(SequenceBlockTable &table);

    int available_blocks() const;
    int block_size() const;

private:
    BlockPool &pool_;
    int num_layers_;
};
```

### 2.4 PagedKVCache

The core challenge: implement the `KVCache` interface using block storage, while
returning contiguous tensors that the existing attention kernel can consume.

```cpp
// include/backend/kvcache/paged.hpp

class PagedKVCache : public KVCache {
public:
    PagedKVCache(const KVCacheConfig &config, BlockAllocator &allocator);
    ~PagedKVCache() override;

    // KVCache interface
    int allocated_capacity() const override;
    tensor_t get_k_cache(int layer_idx) override;
    tensor_t get_v_cache(int layer_idx) override;

    // Write slot: returns a tensor backed by block memory.
    // For multi-token writes (prefill), may span multiple blocks.
    tensor_t get_k_cache_slice(int layer_idx, int past_len, int seq_len) override;
    tensor_t get_v_cache_slice(int layer_idx, int past_len, int seq_len) override;

    // Read full history: gathers all blocks into a contiguous temp buffer.
    tensor_t get_k_cache_slice(int layer_idx, int total_len) override;
    tensor_t get_v_cache_slice(int layer_idx, int total_len) override;

    void reset() override;
    size_t memory_usage() const override;
    float utilization() const override;

    // Access block table (for future paged attention kernels in PR-8)
    const SequenceBlockTable& block_table() const { return block_table_; }

private:
    BlockAllocator &allocator_;
    SequenceBlockTable block_table_;

    // Temporary contiguous buffers for gather (reused across calls)
    std::vector<tensor_t> gather_k_bufs_;  // [num_layers]
    std::vector<tensor_t> gather_v_bufs_;  // [num_layers]
    int gather_capacity_ = 0;              // current capacity of gather buffers

    // Ensure gather buffers are large enough
    void ensure_gather_capacity(int required_len);

    // Gather blocks into contiguous buffer
    tensor_t gather_contiguous(int layer_idx, int total_len, bool is_k);

    // Get write slot within block(s)
    tensor_t get_write_slot(int layer_idx, int past_len, int seq_len, bool is_k);
};
```

**Key implementation details:**

#### Write path (`get_{k,v}_cache_slice(layer, past_len, seq_len)`)

The model writes new K/V tokens into the cache. For decode (`seq_len=1`), this always fits within one block. For prefill (`seq_len > 1`), it may span multiple blocks.

```
Single token (decode):
  block_idx = past_len / block_size
  offset    = past_len % block_size
  -> return view into block[block_idx] at offset, length 1

Multi-token (prefill, seq_len > 1):
  Tokens may span block boundary. Two approaches:

  Approach A (simple): Allocate a small temp buffer, let model write into it,
  then copy to blocks in update_seq_len(). This is the simplest and what we use.

  Approach B (zero-copy): Return a special tensor that maps to block memory.
  Complex because token range may cross block boundary. Defer to PR-8.
```

**Recommendation: Approach A for PR-7.** Allocate a small contiguous write buffer per layer. The model writes into it. When `update_seq_len()` is called, scatter the buffer contents into the appropriate blocks. The copy cost is trivial (one token per decode step, or `seq_len` tokens per prefill — much smaller than the model forward compute).

#### Read path (`get_{k,v}_cache_slice(layer, total_len)`)

The model reads full K/V history for attention. Since the existing `self_attention` kernel expects contiguous input, we gather all blocks into a contiguous buffer:

```
gather_contiguous(layer, total_len, is_k):
  for block_idx in 0..ceil(total_len/block_size)-1:
    src = pool.block_data(block_table[layer][block_idx])
    tokens_in_block = min(block_size, total_len - block_idx * block_size)
    memcpy(gather_buf + block_idx * block_size * token_bytes,
           src, tokens_in_block * token_bytes)
  return gather_buf.slice(0, 0, total_len)
```

The gather buffers are pre-allocated and reused. They grow if `total_len` exceeds current capacity (like DynamicKVCache, but only the gather buffer grows — block data stays in place).

### 2.5 VRAM Budget and Pool Sizing

Add a memory query function to the runtime API:

```cpp
// runtime_api.hpp - new function pointer
typedef void (*get_memory_info_api)(size_t *free, size_t *total);

// In ZedinferRuntimeAPI struct:
get_memory_info_api get_memory_info;
```

NVIDIA implementation: `cudaMemGetInfo(free, total)`.
CPU implementation: return configured max or system RAM.

At engine init:

```cpp
// engine.cpp, after model + scratch are allocated:
size_t free_bytes, total_bytes;
runtime_api->get_memory_info(&free_bytes, &total_bytes);

size_t kv_budget = static_cast<size_t>(free_bytes * gpu_memory_utilization);
// gpu_memory_utilization from SchedulerConfig, default 0.9

int block_bytes = block_config.block_size * block_config.num_kv_heads
                * block_config.head_dim * dtype_size;
int num_blocks = kv_budget / block_bytes;
```

### 2.6 Engine Integration

The engine creates a `BlockPool` and `BlockAllocator` at init. When creating a session, it creates a `PagedKVCache` backed by the shared allocator instead of a standalone `DynamicKVCache`.

```cpp
// engine.hpp - new members
std::unique_ptr<kvcache::BlockPool> block_pool_;
std::unique_ptr<kvcache::BlockAllocator> block_allocator_;

// engine.cpp - create_session():
auto kv_cache = std::make_unique<kvcache::PagedKVCache>(kv_config, *block_allocator_);
// (replaces DynamicKVCache::create_dynamic_kvcache)
```

The `DynamicKVCache` code is kept for fallback/testing. A config flag selects which to use.

### 2.7 Scheduler Integration

The scheduler's `run_one()` currently receives a `KVCache&` from the engine. This interface is unchanged — `PagedKVCache` implements `KVCache`. The scheduler doesn't need to know about blocks yet.

In PR-9 (continuous batching), the scheduler will use `block_allocator_->available_blocks()` for admission control.

---

## 3. File Structure

### New Files

| File | Responsibility |
|------|---------------|
| `include/backend/kvcache/block_pool.hpp` | `BlockConfig`, `BlockPool`, `SequenceBlockTable`, `BlockAllocator` |
| `src/backend/kvcache/block_pool.cpp` | Pool allocation, bitmap management, block allocator logic |
| `include/backend/kvcache/paged.hpp` | `PagedKVCache` class declaration |
| `src/backend/kvcache/paged.cpp` | `PagedKVCache` implementation (write, gather, lifecycle) |

### Modified Files

| File | What Changes |
|------|-------------|
| `include/backend/device/runtime_api.hpp` | Add `get_memory_info_api` function pointer |
| `src/backend/device/nvidia/nvidia_runtime_api.cu` | Implement via `cudaMemGetInfo` |
| `src/backend/device/cpu/cpu_runtime_api.cpp` | Implement (return configured max or system info) |
| `include/zedinfer/engine.hpp` | Add `block_pool_`, `block_allocator_` members |
| `src/zedinfer/engine.cpp` | Create block pool at init, use `PagedKVCache` in `create_session()` |
| `include/zedinfer/scheduler.hpp` | Add `gpu_memory_utilization` to `SchedulerConfig` |

### Unchanged Files

| File | Why Unchanged |
|------|--------------|
| `src/frontend/models/qwen2.cpp` | Uses `KVCache` interface — `PagedKVCache` implements it |
| `src/frontend/models/qwen3.cpp` | Same |
| `include/backend/ops/ops.hpp` | `self_attention` signature unchanged |
| `src/backend/ops/self_attention/` | Kernel unchanged — receives contiguous tensors |
| `include/backend/kvcache/base.hpp` | Base class unchanged |
| `include/backend/kvcache/dynamic.hpp` | Kept as fallback |

---

## 4. Interface Changes Summary

| Symbol | Before | After |
|--------|--------|-------|
| `BlockPool` | Does not exist | **New** in `block_pool.hpp` |
| `BlockConfig` | Does not exist | **New** in `block_pool.hpp` |
| `BlockAllocator` | Does not exist | **New** in `block_pool.hpp` |
| `SequenceBlockTable` | Does not exist | **New** in `block_pool.hpp` |
| `PagedKVCache` | Does not exist | **New** in `paged.hpp` |
| `get_memory_info_api` | Does not exist | **New** in `runtime_api.hpp` |
| `ZedinferRuntimeAPI` | 12 function pointers | 13 function pointers (+ `get_memory_info`) |
| `SchedulerConfig` | No memory config | **Add** `gpu_memory_utilization` field |
| `InferenceEngine` | No block pool | **Add** `block_pool_`, `block_allocator_` members |
| `InferenceEngine::create_session()` | Creates `DynamicKVCache` | Creates `PagedKVCache` (configurable) |

**Public API impact:** None. `create_session()`, `generate()`, `chat()` signatures unchanged. Model forward code unchanged.

---

## 5. Implementation Tasks

### Task 1: Add `get_memory_info` to Runtime API

**Files:**
- Modify: `include/backend/device/runtime_api.hpp`
- Modify: `src/backend/device/nvidia/nvidia_runtime_api.cu`
- Modify: `src/backend/device/cpu/cpu_runtime_api.cpp`

- [ ] **Step 1:** Add `get_memory_info_api` typedef and field to `ZedinferRuntimeAPI`
- [ ] **Step 2:** Implement NVIDIA variant: call `cudaMemGetInfo(&free, &total)`
- [ ] **Step 3:** Implement CPU variant: return system memory info or a configured max
- [ ] **Step 4:** Build to verify compilation
- [ ] **Step 5:** Commit: `feat(runtime): add get_memory_info to runtime API`

---

### Task 2: Implement BlockPool

**Files:**
- Create: `include/backend/kvcache/block_pool.hpp`
- Create: `src/backend/kvcache/block_pool.cpp`

- [ ] **Step 1:** Define `BlockConfig` and `SequenceBlockTable` structs
- [ ] **Step 2:** Implement `BlockPool` constructor — single contiguous allocation (`num_blocks * block_bytes`)
- [ ] **Step 3:** Implement `allocate()` — scan bitmap for free block, mark used, return ID
- [ ] **Step 4:** Implement `free()` — mark block as unused in bitmap
- [ ] **Step 5:** Implement `block_data()` — return `pool_memory_ + block_id * block_bytes_`
- [ ] **Step 6:** Implement query methods: `total_blocks()`, `free_blocks()`, `used_blocks()`, `memory_usage()`
- [ ] **Step 7:** Implement destructor — free the contiguous allocation via runtime API
- [ ] **Step 8:** Build to verify compilation
- [ ] **Step 9:** Commit: `feat(kvcache): implement BlockPool with bitmap allocator`

---

### Task 3: Implement BlockAllocator

**Files:**
- Modify: `include/backend/kvcache/block_pool.hpp` (add BlockAllocator)
- Modify: `src/backend/kvcache/block_pool.cpp` (add BlockAllocator implementation)

- [ ] **Step 1:** Implement `BlockAllocator` constructor
- [ ] **Step 2:** Implement `allocate_sequence()` — allocate `ceil(est_tokens / block_size) * num_layers * 2` blocks, populate `SequenceBlockTable`
- [ ] **Step 3:** Implement `extend_sequence()` — allocate one block, append to layer's block list
- [ ] **Step 4:** Implement `free_sequence()` — free all blocks in the table, clear vectors
- [ ] **Step 5:** Implement `available_blocks()`, `block_size()`
- [ ] **Step 6:** Build to verify compilation
- [ ] **Step 7:** Commit: `feat(kvcache): implement BlockAllocator for sequence block management`

---

### Task 4: Implement PagedKVCache

**Files:**
- Create: `include/backend/kvcache/paged.hpp`
- Create: `src/backend/kvcache/paged.cpp`

- [ ] **Step 1:** Declare `PagedKVCache` class implementing `KVCache` interface
- [ ] **Step 2:** Implement constructor — allocate initial blocks via `BlockAllocator`
- [ ] **Step 3:** Implement write path: `get_k_cache_slice(layer, past_len, seq_len)` / `get_v_cache_slice` — return write buffer, track pending writes
- [ ] **Step 4:** Implement `update_seq_len()` override — scatter pending write buffer into blocks
- [ ] **Step 5:** Implement gather path: `get_k_cache_slice(layer, total_len)` / `get_v_cache_slice` — gather blocks into contiguous buffer
- [ ] **Step 6:** Implement `ensure_gather_capacity()` — grow gather buffers when needed
- [ ] **Step 7:** Implement `reset()` — free all blocks, clear state
- [ ] **Step 8:** Implement `allocated_capacity()`, `memory_usage()`, `utilization()`
- [ ] **Step 9:** Build to verify compilation
- [ ] **Step 10:** Commit: `feat(kvcache): implement PagedKVCache with contiguous gather fallback`

---

### Task 5: Wire into Engine

**Files:**
- Modify: `include/zedinfer/engine.hpp`
- Modify: `src/zedinfer/engine.cpp`
- Modify: `include/zedinfer/scheduler.hpp`

- [ ] **Step 1:** Add `gpu_memory_utilization` (default 0.9) to `SchedulerConfig`
- [ ] **Step 2:** Add `block_pool_` and `block_allocator_` members to `InferenceEngine`
- [ ] **Step 3:** In `InferenceEngine::create()`, after model loading: query free memory, compute num_blocks, create `BlockPool` and `BlockAllocator`
- [ ] **Step 4:** In `create_session()`, create `PagedKVCache` instead of `DynamicKVCache`
- [ ] **Step 5:** Add config flag `use_paged_kvcache` (default true) to allow fallback to `DynamicKVCache`
- [ ] **Step 6:** Build to verify compilation
- [ ] **Step 7:** Commit: `feat(engine): create BlockPool at init, use PagedKVCache for sessions`

---

### Task 6: Build and correctness verification

- [ ] **Step 1:** Run `bash auto-build-test/scripts/run_build_check.sh` — verify clean build
- [ ] **Step 2:** Run `ping` — verify output is identical to before (bit-identical token sequence)
- [ ] **Step 3:** Run `chat` — verify multi-turn conversation works
- [ ] **Step 4:** Run `bench` — verify no performance regression beyond gather overhead
- [ ] **Step 5:** Log block pool stats: total blocks, used blocks, gather count
- [ ] **Step 6:** If failures, fix and retry
- [ ] **Step 7:** Commit any fixes

---

## 6. Correctness Validation

1. **Bit-identical output:** The model forward code is unchanged. `PagedKVCache` gathers blocks into the same contiguous format that `DynamicKVCache` provides. Output tokens must be identical.

2. **Block lifecycle:** Allocate sequence -> write tokens -> read back via gather -> verify data matches what was written. Free sequence -> verify blocks returned to pool.

3. **Capacity growth:** Prefill with 512 tokens on block_size=16 -> verify 32 blocks allocated per layer per type. Decode 100 more tokens -> verify additional blocks allocated as blocks fill up.

4. **Pool exhaustion:** Allocate blocks until pool is full. Verify `allocate()` returns -1. Free some blocks. Verify re-allocation succeeds.

5. **Multi-session:** Create 2 sessions, each allocating from the shared pool. Verify independent operation and correct output.

---

## 7. Benchmark Plan

Run config B (Qwen2-1.5B, prefill=128, decode=128, GPU) before and after.

**Expected overhead:** The contiguous gather adds a memcpy per layer per decode step. For Qwen2-1.5B (28 layers, 2 KV heads, 64 head_dim, BF16):
```
Per gather: seq_len * 2 * 64 * 2 bytes = seq_len * 256 bytes
At seq_len=256: 256 * 256 = 64 KB per layer, 28 layers * 2 (K+V) = 3.5 MB total
```
This is negligible vs model forward compute. Expected regression: < 1ms.

Also measure: peak memory usage vs DynamicKVCache. Expected: less peak memory because no over-allocation during growth.

---

## 8. Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|-----------|
| Gather copy overhead | Low | Low | Copy is small vs forward compute. PR-8 eliminates it. |
| Block bitmap scan is O(n) | Low | Very Low | n=num_blocks, typically ~1000s. PR-9 replaces with free-list if needed. |
| Write buffer scatter correctness | Medium | High | Test: write token, gather, compare against DynamicKVCache output. |
| `cudaMemGetInfo` returns inaccurate free memory | Low | Medium | Add safety margin (default 256MB). Allow manual override via config. |
| Model writes multi-token across block boundary (prefill) | Medium | Medium | Write buffer approach handles this transparently. |
| Pool memory allocation fails (not enough VRAM) | Low | High | Log available memory, fail gracefully with clear error. |

---

## 9. Rollback

Add `use_paged_kvcache = false` config flag in engine. When false, `create_session()` falls back to `DynamicKVCache::create_dynamic_kvcache()` — the current behavior. All new code (block_pool, paged) is simply unused.

---

## 10. Open Questions

1. **Should `BlockPool` support CPU and GPU from a single pool?**
   No. One pool per device. For PR-13 (heterogeneous), a separate CPU block pool can be added.

2. **What block_size to use?**
   Default 16. Tradeoff: smaller = less waste, more blocks to manage. 16 is standard in vLLM. Configurable via `BlockConfig`.

3. **Should the gather buffers be allocated from the block pool?**
   No. Gather buffers are temporary workspace, not KV storage. Allocate them from the normal tensor allocator (memory pool). They are reused across calls.

4. **How should `warmup()` and `profile()` work with paged KV?**
   They currently create a temporary `DynamicKVCache`. This is fine — they bypass the scheduler and don't need paged memory. No change needed.

5. **Should `gpu_memory_utilization` default to 0.9 or something lower for testing?**
   Default 0.9 for production. For testing on B200 with a 1.5B model, user can set 0.1 to limit to ~18GB. Expose via `SchedulerConfig`.
