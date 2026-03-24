# Post-Refactoring Analysis (After R1-R6)

> Checkpoint analysis of remaining issues after the R1-R6 refactoring cycle.
> Focus: code quality, performance bottlenecks, and obstacles for upcoming features.

---

## 1. Current Architecture (Post R1-R6)

```
InferenceEngine (resource container)
  ├── Model (config + weights + forward_config())
  ├── Tokenizer (HF BPE)
  ├── Sampler (Argmax)
  ├── BlockPool (static VRAM allocation)
  ├── BlockAllocator (per-request block management)
  ├── ServingLoop (generate, batch, scheduler)
  └── Profiler (warmup, profile)

transformer_forward() (shared loop, parameterized by):
  ├── ModelForwardConfig (bias, QK norm flags)
  ├── ForwardContext (contiguous or paged)
  └── ExecutorConfig (device, dtype)

Attention dispatch:
  └── ops::attention(AttentionParams) → dispatch by params fields
```

**What R1-R6 achieved:**
- 4 forward files → 1 shared loop
- 2 KV management paths → 1 (block table everywhere)
- KVCache base class → deleted (DynamicKVCache standalone)
- Model::forward/forward_batch → deleted (Model is data holder with forward_config())
- 4 attention functions → 1 parameterized dispatch
- Engine 380 lines → 140 lines (split into Engine + ServingLoop + Profiler)
- Graph execution dead code → already deleted

---

## 2. Remaining Issues

### 2.1 Dual Execution Paths (run_one vs schedule+step)

**Problem:**

`Scheduler::run_one()` and `Scheduler::schedule() + ServingLoop::step()` are two independent execution paths for the same logical operation (generate tokens from a request):

```
Path A — run_one() (session-based, chat/ping):
  Pop request from queue
  → manual prefill: create PagedForwardContext, call transformer_forward
  → manual decode loop: for each token {
      ensure blocks (inline block extension)
      create PagedForwardContext with 1 token
      call transformer_forward
      sample
      stream callback
      stop check
    }
  → return GenerationResult

Path B — schedule() + step() (batch mode, batch_bench):
  schedule(): assemble ScheduledBatch from active + waiting
  → build_context(): flatten tokens into BatchContext
  → step(): create PagedForwardContext, call transformer_forward
  → process_results(): per-request sample, advance state, complete
```

**Impact:**
- Every new feature (prefix caching, CUDA graph, KV quantization) must be implemented in both paths.
- Bug fixes may apply to one path but not the other.
- run_one() has inline block extension logic that duplicates what BlockAllocator already does.

**Recommendation:**

Eliminate `run_one()` entirely. Make session-based generation go through `schedule() + step()`:

```
generate_tokens(block_table, input_ids, config):
  → build request
  → scheduler.submit(request)
  → while (!request.complete):
      scheduler.schedule()   // returns batch with this 1 request
      step()                 // forward + process_results
  → return result from request.result_promise
```

Single request is just batch_size=1. No special path needed. The scheduler's decode-first policy naturally handles it.

**Priority:** Medium. Do before adding CUDA graph or prefix caching.

---

### 2.2 Per-Layer Block Table GPU Upload

**Problem:**

In `PagedForwardContext::attend()`, every transformer layer re-creates GPU tensors for block tables and uploads them from CPU:

```cpp
// Called 28 times per forward (once per layer)
auto k_bt_t = make_typed({num_decode * max_blocks}, I32);
k_bt_t->load(k_bt.data());   // CPU → GPU memcpy
auto v_bt_t = make_typed({num_decode * max_blocks}, I32);
v_bt_t->load(v_bt.data());   // CPU → GPU memcpy
auto sl_t = make_typed({num_decode}, I32);
sl_t->load(sl.data());       // CPU → GPU memcpy
```

Block tables are identical across layers (K and V block IDs don't change between layers for the same request). Yet we:
1. Allocate 3 new GPU tensors × 28 layers = 84 GPU malloc calls
2. Upload 3 × 28 = 84 CPU→GPU memcpy calls

**Impact:**
- ~84 × 10us = ~840us overhead per forward pass (non-trivial for decode)
- **Directly blocks CUDA Graph capture**: CUDA Graphs require fixed memory addresses. Dynamic allocation in the loop prevents capture.

**Recommendation:**

Cache block table GPU tensors across layers. Two options:

Option A: Upload once in `attend()` first call, reuse for subsequent layers:
```cpp
// First layer: upload
if (layer == 0) {
    cached_k_bt_ = make_typed(...); cached_k_bt_->load(...);
    cached_v_bt_ = make_typed(...); cached_v_bt_->load(...);
    cached_sl_ = make_typed(...); cached_sl_->load(...);
}
// All layers: use cached
ops::attention(params with cached tensors);
```

Option B: Upload in `PagedForwardContext` constructor, before any layer runs. Store as member.

**Priority:** High. Prerequisite for CUDA Graph.

---

### 2.3 Per-Token Scatter in write_kv

**Problem:**

`PagedForwardContext::write_kv()` scatters K/V tokens to blocks one token at a time:

```cpp
for (slot : slots) {
    for (t = 0; t < slot.num_tokens; t++) {
        copy_to_block(src + t * token_bytes, token_bytes, dst);
    }
}
```

For batch decode with 128 requests × 28 layers:
- 128 tokens × 28 layers × 2 (K+V) = 7168 individual `cudaMemcpy` calls per forward
- Each `cudaMemcpy` has ~5us launch overhead on GPU

**Impact:**
- 7168 × 5us = ~35ms overhead (significant for decode which should be ~30-50ms total)
- Note: for decode, the actual write happens via `Tensor::create(is_mmap=true)` in the single-request paged path, but in the batch path (through PagedForwardContext), it goes through scatter.

**Recommendation:**

Write a custom CUDA scatter kernel that processes all tokens in one launch:

```cpp
// One kernel launch for all slots, all tokens
__global__ void scatter_kv_kernel(
    const T* src,           // [total_tokens, nkvhead, head_dim] flattened K or V
    T* pool_base,
    const int* block_tables, // [num_slots, max_blocks]
    const int* offsets,      // per-slot: token_offset, past_len, num_tokens
    int block_size, int token_bytes, int num_slots);
```

**Priority:** Low-Medium. Measurable impact at large batch sizes but not a correctness issue.

---

### 2.4 Warmup Doesn't Cover Paged Path

**Problem:**

`Profiler::warmup()` uses `DynamicKVCache + ContiguousForwardContext`, which exercises:
- `ops::self_attention` (contiguous kernel, possibly cuDNN)
- Contiguous KV cache write/read

But actual serving uses:
- `ops::attention` (paged decode/prefill kernels)
- Block pool scatter/read

The first real request after warmup triggers cold-start for paged attention kernels (CUDA JIT compilation, cache misses).

**Impact:**
- First request may have higher latency (~10-50ms extra)
- Observed earlier: cuDNN warmup path appeared to affect GPU state differently than non-cuDNN path

**Recommendation:**

After block pool creation, run a short paged warmup:

```cpp
// In engine create(), after init_block_pool():
auto tmp_table = block_allocator_->allocate_sequence(128);
PagedForwardContext ctx(dummy_tokens, 0, tmp_table, *block_pool_);
transformer_forward(model_->forward_config(), ctx, exec_config_);
block_allocator_->free_sequence(tmp_table);
```

**Priority:** Low. Affects only first-request latency.

---

### 2.5 ServingLoop Lacks Thread Synchronization

**Problem:**

`ServingLoop::run_loop()` is a busy-polling loop:

```cpp
void run_loop() {
    while (scheduler_.has_work()) {
        step();
    }
}
```

For HTTP API (PR-10), the engine thread needs to:
- Sleep when no requests are pending (save CPU)
- Wake immediately when a new request is submitted

Currently there's no mechanism for this. `submit_async()` adds to the queue but doesn't signal the engine thread.

**Impact:**
- **Blocks HTTP API implementation (PR-10).** Without signaling, the engine thread either busy-waits (wastes CPU) or polls with sleep (adds latency).

**Recommendation:**

Add `std::condition_variable` to coordinate submit and engine loop:

```cpp
class ServingLoop {
    std::condition_variable work_cv_;
    std::mutex work_mutex_;

    void submit_async(request) {
        scheduler_.submit(request);
        work_cv_.notify_one();  // wake engine thread
    }

    void run_loop() {
        while (running_) {
            {
                std::unique_lock lock(work_mutex_);
                work_cv_.wait(lock, [&] { return scheduler_.has_work() || !running_; });
            }
            while (scheduler_.has_work()) {
                step();
            }
        }
    }
};
```

**Priority:** High. Must be done before PR-10 (HTTP API).

---

### 2.6 build_context() Depends on schedule() Side Effects

**Problem:**

`ScheduledBatch::build_context()` computes prefill chunk offsets using `req->prefill_progress` which was already advanced by `schedule()`:

```cpp
// In build_context():
int chunk_start = req->prefill_progress - chunk_size;  // depends on schedule() having updated progress
```

This implicit ordering dependency already caused a bug (out-of-bounds access on `input_ids`) that was fixed.

**Impact:**
- Fragile code: if `schedule()` or `build_context()` ordering changes, silent data corruption.
- Violates principle of least surprise.

**Recommendation:**

Store `chunk_start` explicitly in `ScheduledBatch` at schedule time:

```cpp
struct ScheduledBatch {
    // ...
    std::vector<int> prefill_chunk_starts;  // explicit, no back-computation needed
};
```

Then `build_context()` uses `prefill_chunk_starts[i]` directly.

**Priority:** Low. Already fixed, but the fix is fragile.

---

## 3. Priority Summary for Upcoming Features

### For CUDA Graph (next performance optimization):

| Prerequisite | Status |
|-------------|--------|
| Block table GPU caching (2.2) | **Must fix** |
| Eliminate run_one dual path (2.1) | Recommended |
| Fixed memory addresses in forward | Partially met (tensor pool helps) |

### For HTTP API (PR-10):

| Prerequisite | Status |
|-------------|--------|
| Condition variable in ServingLoop (2.5) | **Must fix** |
| Thread-safe submit (already done) | Done |
| Session management | Done (create_session) |

### For flash-attn varlen:

| Prerequisite | Status |
|-------------|--------|
| cu_seqlens in AttentionParams | Need to add field |
| Merge prefill attention loop | Need to restructure attend() |
| Library integration | External dependency |

### For Prefix Caching:

| Prerequisite | Status |
|-------------|--------|
| Block reference counting | Need to add to BlockPool |
| Hash table for prefix lookup | Need new component |
| Copy-on-write for shared blocks | Need new logic |
| Unified KV path (R2 done) | Done |

---

## 4. Code Health Metrics (Post R1-R6)

| Metric | Before R1-R6 | After R1-R6 |
|--------|-------------|-------------|
| Forward loop copies | 4 files, 557 lines | 1 file, ~100 lines |
| KV management paths | 2 (PagedKVCache + Request block_table) | 1 (block table) |
| Attention functions in ops.hpp | 4 | 1 (parameterized) |
| KVCache class hierarchy | 2 classes (base + dynamic) | 1 standalone class |
| Model virtual methods | 10 (forward, forward_batch, 4 weight names, 4 accessors) | 5 (config, weights, model_type, num_params, forward_config) |
| Engine lines | ~380 | ~140 |
| Dead code (graph execution) | ~500 lines | 0 |
| Total serving code duplication | High | Minimal |

The architecture is in good shape for the next phase of feature development.
