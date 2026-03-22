# R2: Unify KV Management - Detailed Refactoring Plan

**Goal:** Converge the two KV cache management paths (Session/PagedKVCache vs Request/block_table) into a single path where per-request block table is the source of truth, eliminating duplicated scatter logic and simplifying the KVCache base class.

---

## 1. Problem Analysis

### 1.1 Current Dual Paths

**Path A: Session Mode** (`run_one` → `SingleForwardContext`)

```
Session owns PagedKVCache
  → PagedKVCache owns SequenceBlockTable + write buffers + gather buffers
  → SingleForwardContext::write_kv()
     → kvcache.get_v_cache_slice(layer, past_len, sl)  // returns write buffer or direct-to-block
     → memcpy(v_slot, v_tensor)                          // copy into KVCache-managed memory
  → SingleForwardContext::attend()
     → if paged decode: kvcache.k_pool_data(), kvcache.k_block_ids(layer)
     → if paged prefill: kvcache.scatter_layer_to_blocks(layer) + paged kernel
     → if non-paged: kvcache.get_k_cache_slice(layer, total_len)
  → SingleForwardContext::finalize()
     → kvcache.update_seq_len(sl)  // scatter write buffer to blocks (prefill path)
```

**Path B: Batch Mode** (`forward_batch` → `BatchedForwardContext`)

```
Request owns SequenceBlockTable (allocated by scheduler)
  → BatchedForwardContext::write_kv()
     → scatter_token_to_block() per slot per token  // direct scatter, no write buffer
  → BatchedForwardContext::attend()
     → batched decode: build GPU block table, paged_attention_decode_batched
     → per-request prefill: paged_attention_prefill
  → BatchedForwardContext::finalize()
     → noop (scheduler updates block_table.seq_len)
```

### 1.2 What's Duplicated

| Component | Path A (Session/PagedKVCache) | Path B (Batch/Request) |
|-----------|-------------------------------|------------------------|
| Block table | `PagedKVCache::block_table_` | `InferenceRequest::block_table` |
| Block allocation | `PagedKVCache` constructor | `Scheduler::schedule()` |
| Block freeing | `PagedKVCache` destructor | `Scheduler::complete_request()` |
| Token scatter | `PagedKVCache::scatter_to_blocks()` + `scatter_layer_to_blocks()` | `BatchedForwardContext::scatter_token_to_block()` |
| Direct-to-block decode | `PagedKVCache::get_k_cache_slice(sl==1)` → `Tensor::create(is_mmap)` | `scatter_token_to_block()` with 1 token |
| Attend dispatch | `SingleForwardContext::attend()` 3-way if/else | `BatchedForwardContext::attend()` decode+prefill |

### 1.3 PagedKVCache's Internal Complexity

PagedKVCache has accumulated significant machinery that's only partially used:

```
PagedKVCache internals:
  write_k_bufs_[num_layers]       // only used for prefill scatter
  write_v_bufs_[num_layers]       // only used for prefill scatter
  gather_k_bufs_[num_layers]      // only used for non-paged fallback reads
  gather_v_bufs_[num_layers]      // only used for non-paged fallback reads
  write_buf_capacity_
  gather_capacity_
  pending_write_past_len_         // complex state tracking
  pending_write_seq_len_
  direct_write_to_blocks_         // decode vs prefill flag
  scattered_layers_               // per-layer scatter counter
```

This complexity exists because PagedKVCache implements the `KVCache` interface
(designed for contiguous access) while internally using paged blocks. The
`get_k_cache_slice` → write buffer → scatter → gather dance is all glue code
to bridge two incompatible abstractions.

### 1.4 KVCache Base Class Pollution

The base class has 7 paged-specific methods:
```cpp
virtual bool is_paged() const { return false; }
virtual void *k_pool_data() const { return nullptr; }
virtual void *v_pool_data() const { return nullptr; }
virtual const int *k_block_ids(int) const { return nullptr; }
virtual const int *v_block_ids(int) const { return nullptr; }
virtual int block_size() const { return 0; }
virtual void scatter_layer_to_blocks(int) {}
```

`DynamicKVCache` inherits all of these as dead code.

---

## 2. Target Design

### 2.1 Core Principle

**Block table is the universal KV handle.** Both session mode and batch mode
use `SequenceBlockTable` as the per-request KV representation. `PagedKVCache`
as a full KVCache subclass is eliminated. The KV write and attention dispatch
logic lives in `ForwardContext` (from R1) — there is only ONE context
implementation for paged mode.

### 2.2 Session Holds Block Table, Not KVCache

```cpp
// session.hpp — after R2
class InferenceSession {
    // ...
    kvcache::SequenceBlockTable block_table_;  // replaces kvcache_
    int past_len_;
    // ...
};
```

`Session::chat()` builds an `InferenceRequest` with a reference to the
session's block table, submits it, and gets back the result. The session's
block table persists across turns (multi-turn KV reuse).

### 2.3 Unified PagedForwardContext

Replace both `SingleForwardContext` (paged path) and `BatchedForwardContext`
with a single `PagedForwardContext` that can handle both single and batched:

```cpp
class PagedForwardContext : public ForwardContext {
public:
    // Single-request mode
    PagedForwardContext(
        const std::vector<int> &input_ids,
        int past_len,
        kvcache::SequenceBlockTable &block_table,
        kvcache::BlockPool &pool);

    // Batch mode
    PagedForwardContext(
        const BatchContext &batch,
        kvcache::BlockAllocator &allocator);

    // ForwardContext interface
    int num_tokens() const override;
    void prepare_inputs(...) override;
    void write_kv(int layer, tensor_t k, tensor_t v) override;
    tensor_t attend(int layer, tensor_t q_rope, ...) override;
    void finalize() override;

private:
    // Common: list of slots (single mode has 1 slot, batch has N)
    struct Slot {
        kvcache::SequenceBlockTable *block_table;
        int token_offset;
        int num_tokens;
        int past_len;
        bool is_prefill;
    };
    std::vector<Slot> slots_;
    kvcache::BlockPool &pool_;
    int total_tokens_;
};
```

**Key insight:** A single request is just a batch of size 1. The write_kv and
attend logic is identical — scatter tokens to blocks per slot, then run paged
attention per slot.

### 2.4 DynamicKVCache Retained for Warmup/Profile

`DynamicKVCache` stays as-is for `warmup()` and `profile()`. These utility
methods don't use the scheduler or block pool. `ContiguousForwardContext`
(renamed from the non-paged path of `SingleForwardContext`) handles this:

```cpp
class ContiguousForwardContext : public ForwardContext {
    // Uses DynamicKVCache directly via get_k_cache_slice, self_attention
    // Only used by warmup() and profile()
};
```

### 2.5 Cleaned KVCache Base Class

After removing paged methods, the base class returns to its original design:

```cpp
class KVCache {
public:
    virtual int allocated_capacity() const = 0;
    virtual tensor_t get_k_cache(int layer_idx) = 0;
    virtual tensor_t get_v_cache(int layer_idx) = 0;
    virtual tensor_t get_k_cache_slice(int layer_idx, int total_len) = 0;
    virtual tensor_t get_v_cache_slice(int layer_idx, int total_len) = 0;
    virtual tensor_t get_k_cache_slice(int layer_idx, int past_len, int seq_len) = 0;
    virtual tensor_t get_v_cache_slice(int layer_idx, int past_len, int seq_len) = 0;
    virtual void update_seq_len(int new_tokens);
    virtual void reset();
    virtual size_t memory_usage() const = 0;
    virtual float utilization() const = 0;
    // NO paged methods
};
```

Only `DynamicKVCache` implements this interface.

---

## 3. Changes Summary

### 3.1 Files to Create

| File | Purpose |
|------|---------|
| `include/frontend/models/paged_forward_context.hpp` | `PagedForwardContext` declaration |
| `src/frontend/models/paged_forward_context.cpp` | Unified paged write_kv + attend for single and batch |

### 3.2 Files to Modify

| File | Change |
|------|--------|
| `include/zedinfer/session.hpp` | Replace `kvcache_t kvcache_` with `SequenceBlockTable block_table_` |
| `src/zedinfer/session.cpp` | `chat()` builds request with block_table_, calls engine |
| `include/zedinfer/engine.hpp` | `create_session()` no longer creates KVCache, `generate_tokens()` signature changes |
| `src/zedinfer/engine.cpp` | `create_session()` allocates block table instead of PagedKVCache; `generate_tokens()` creates `PagedForwardContext` |
| `include/frontend/models/forward_context.hpp` | Remove `SingleForwardContext` and `BatchedForwardContext`, keep `ContiguousForwardContext` for warmup |
| `src/frontend/models/forward_context.cpp` | Same — remove dual context, keep contiguous-only |
| `include/backend/kvcache/base.hpp` | Remove 7 paged methods |

### 3.3 Files to Delete

| File | Reason |
|------|--------|
| `include/backend/kvcache/paged.hpp` | Replaced by `PagedForwardContext` |
| `src/backend/kvcache/paged.cpp` | Same |

### 3.4 Files Unchanged

| File | Why |
|------|-----|
| `include/backend/kvcache/block_pool.hpp` | `BlockPool`, `BlockAllocator`, `SequenceBlockTable` stay as-is |
| `src/backend/kvcache/block_pool.cpp` | Same |
| `include/backend/kvcache/dynamic.hpp` | Kept for warmup/profile |
| `src/backend/kvcache/dynamic.cpp` | Same |
| `src/frontend/models/transformer_forward.cpp` | Shared loop unchanged — calls ctx.write_kv() and ctx.attend() |
| Model files (qwen2.cpp, qwen3.cpp) | Only change: use `PagedForwardContext` instead of `SingleForwardContext` |
| Attention kernels | Unchanged — paged_attention_decode, prefill, batched |
| Scheduler | Unchanged — already manages per-request block tables |

---

## 4. Data Flow After R2

### 4.1 Session Mode (chat/ping)

```
Session::chat(user_input)
  → encode prompt
  → engine.generate_tokens(session.block_table_, input_ids, config)
     → scheduler.submit(request with block_table ref)
     → scheduler.run_one(model, block_table, pool, ...)
        → PagedForwardContext ctx(input_ids, past_len, block_table, pool)
        → transformer_forward(model_cfg, ctx, exec_config)
           → ctx.write_kv()   // scatter to blocks
           → ctx.attend()     // paged attention
        → ctx.finalize()      // update block_table.seq_len
     → return result
  → update session.past_len_
```

### 4.2 Batch Mode (submit_async + run_loop)

```
engine.submit_async(request)
  → scheduler.submit(request with own block_table)

engine.step()
  → scheduler.schedule()  → ScheduledBatch
  → batch.build_context() → BatchContext
  → PagedForwardContext ctx(batch, allocator)
  → transformer_forward(model_cfg, ctx, exec_config)
     → ctx.write_kv()   // scatter per slot
     → ctx.attend()     // batched decode + per-request prefill
  → scheduler.process_results()
```

### 4.3 Warmup/Profile

```
engine.warmup()
  → DynamicKVCache tmp_kv(...)
  → ContiguousForwardContext ctx(dummy_ids, 0, tmp_kv)
  → transformer_forward(model_cfg, ctx, exec_config)
     → ctx.write_kv()   // direct to contiguous DynamicKVCache
     → ctx.attend()     // ops::self_attention (contiguous)
```

---

## 5. Implementation Tasks

### Task 1: Create PagedForwardContext

- [ ] Define `PagedForwardContext` header with Slot struct
- [ ] Implement single-request constructor (1 slot)
- [ ] Implement batch constructor (N slots from BatchContext)
- [ ] Implement `write_kv()` — scatter tokens to blocks per slot (unified logic)
- [ ] Implement `attend()` — paged decode (single or batched) + paged prefill
- [ ] Implement `prepare_inputs()` and `finalize()`
- [ ] Build to verify

### Task 2: Rename SingleForwardContext to ContiguousForwardContext

- [ ] Rename class in header and implementation
- [ ] Remove all paged paths (is_paged checks, scatter, paged attention)
- [ ] Keep only the `ops::self_attention` contiguous path
- [ ] Remove BatchedForwardContext entirely
- [ ] Build to verify

### Task 3: Clean KVCache base class

- [ ] Remove 7 paged virtual methods from `base.hpp`
- [ ] Update `DynamicKVCache` if needed (should be no-op since defaults are gone)
- [ ] Build to verify

### Task 4: Modify Session to hold block table

- [ ] Replace `kvcache_t kvcache_` with `SequenceBlockTable block_table_` in session
- [ ] Add `BlockAllocator*` reference to session (for block management)
- [ ] Update constructor, `chat()`, `reset()` to use block_table_
- [ ] Build to verify

### Task 5: Update Engine to create PagedForwardContext

- [ ] `create_session()` allocates initial blocks via allocator, no PagedKVCache
- [ ] `generate_tokens()` creates `PagedForwardContext` with session's block_table
- [ ] `warmup()` / `profile()` use `ContiguousForwardContext` with DynamicKVCache
- [ ] `step()` creates `PagedForwardContext` with BatchContext (replaces BatchedForwardContext)
- [ ] Build to verify

### Task 6: Delete PagedKVCache

- [ ] Delete `include/backend/kvcache/paged.hpp`
- [ ] Delete `src/backend/kvcache/paged.cpp`
- [ ] Remove includes/references in engine, session, etc.
- [ ] Build to verify
- [ ] Run `ping`, `chat`, `batch_bench` — verify correctness

---

## 6. Risks

| Risk | Mitigation |
|------|-----------|
| Session multi-turn KV reuse breaks | Session holds block_table across turns, past_len tracks position — same semantics as before |
| Block lifetime management | Session destructor frees blocks via allocator. Same as PagedKVCache destructor did. |
| Warmup/profile regression | ContiguousForwardContext preserves exact DynamicKVCache path — zero change |
| Virtual dispatch overhead in PagedForwardContext | Same as current ForwardContext — one call per layer, negligible |
| run_one() backward compat | run_one() constructs PagedForwardContext with 1 slot — functionally identical |

---

## 7. Impact

| Before | After |
|--------|-------|
| 2 KV management paths | 1 path (block table everywhere) |
| 3 scatter implementations | 1 (in PagedForwardContext) |
| 7 paged methods on KVCache base | 0 |
| PagedKVCache (87 lines header, 300 lines impl) | Deleted |
| Adding prefix caching: implement in 2 paths | Implement in 1 path |
| Adding KV quantization: implement in 2 paths | Implement in 1 path |
