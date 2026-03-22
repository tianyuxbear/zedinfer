# PR-9: Continuous Batching - Detailed Implementation Plan

**Goal:** Enable multiple requests to be processed in a single forward pass, with requests entering and leaving the batch independently at every decode step.

**Architecture:** The scheduler assembles a `ScheduledBatch` of decode + prefill slots each iteration. Token IDs from all slots are concatenated (no padding). Per-token operators (linear, embedding, RoPE, norm) process the flattened batch. Attention uses per-request block tables to read each request's KV cache independently. After forward, logits are routed back to per-request samplers.

---

## 1. Current State vs Target

### Current (Single-Request)

```
Scheduler::run_one()
  → model.forward(input_ids, past_len, kvcache)  // one request
  → sampler.sample(logits)                        // one token
  → loop until complete
```

- One request occupies the GPU at a time
- Decode: M=1 GEMM (GEMV) — low GPU utilization
- Other requests wait in queue

### Target (Continuous Batching)

```
Scheduler::schedule()          → ScheduledBatch with N requests
  → build BatchContext         → concat tokens, collect block tables
  → model.forward_batch(batch) → one forward pass for all requests
  → process_results()          → sample per-request, advance state, trigger callbacks
  → repeat every iteration
```

- Multiple decode requests batched: M=N GEMM — high GPU utilization
- New requests admitted at every iteration (prefill interleaved with decode)
- Completed requests exit immediately, freeing blocks for new requests

---

## 2. Key Design Decisions

### 2.1 No Padding (Concat)

All tokens from all requests are concatenated into one flat sequence. Only attention needs to know request boundaries.

```
Request A: 1 decode token  (past_len=500)
Request B: 1 decode token  (past_len=200)
Request C: 64 prefill tokens (past_len=0)

token_ids = [a0, b0, c0..c63]   total_tokens = 66
```

Per-token operators see `[66, hidden_size]` — no change from single-request.

### 2.2 Decode-First Scheduling

Decode slots are scheduled before prefill slots. This ensures:
- Active requests maintain consistent Time-Between-Tokens (TBT)
- New prefill doesn't spike decode latency
- Prefill is bounded by `max_prefill_tokens` per iteration

### 2.3 Chunked Prefill

Long prompts are split across iterations:
```
Prompt = 2048 tokens, max_prefill_tokens = 512
  Iteration 1: prefill tokens 0-511
  Iteration 2: prefill tokens 512-1023
  Iteration 3: prefill tokens 1024-1535
  Iteration 4: prefill tokens 1536-2047 → move to decode
```

### 2.4 Per-Request KV Cache (Paged)

Each request owns a `SequenceBlockTable` (not a full `PagedKVCache` object). The shared `BlockPool` holds all physical memory. The scheduler manages block allocation/deallocation.

### 2.5 Engine Loop Redesign

Current: `generate_tokens()` is synchronous — blocks until one request completes.

Target: Engine runs an **iteration loop** that:
1. Calls `scheduler.schedule()` to get next batch
2. Calls `model.forward_batch(batch)` for one forward pass
3. Calls `scheduler.process_results()` to advance state
4. Repeats until all requests complete (or new ones arrive)

For the HTTP API (PR-10), `submit()` is called from HTTP threads, the iteration loop runs on the engine thread.

---

## 3. Data Structures

### 3.1 InferenceRequest Extensions

```cpp
struct InferenceRequest {
    // ... existing fields ...

    // New for continuous batching:
    int prefill_progress = 0;          // tokens already prefilled (for chunked prefill)
    kvcache::SequenceBlockTable block_table;  // per-request block table (moved from PagedKVCache)
    std::promise<GenerationResult> result_promise;  // async result delivery
};
```

Key change: **block table moves from `PagedKVCache` to `InferenceRequest`**. Each request owns its block table. The scheduler allocates/frees blocks through `BlockAllocator`. `PagedKVCache` as a per-session object is no longer the primary KV management path — the scheduler manages KV directly per-request.

### 3.2 BatchContext

```cpp
struct BatchContext {
    // Flattened tokens from all requests (no padding)
    std::vector<int> token_ids;

    // Per-slot metadata
    struct Slot {
        InferenceRequest *request;
        int token_offset;     // start position in token_ids
        int num_tokens;       // tokens in this slot (1 for decode, N for prefill chunk)
        int past_len;         // tokens already in KV cache
        bool is_prefill;
    };
    std::vector<Slot> slots;

    // Position IDs (global position for each token)
    std::vector<int64_t> position_ids;

    // Paged attention metadata (per-slot block tables)
    // k_block_tables[slot_idx] = physical block IDs for K
    // v_block_tables[slot_idx] = physical block IDs for V
    std::vector<const int *> k_block_tables;
    std::vector<const int *> v_block_tables;
    std::vector<int> seq_lens;  // KV cache length per slot (for attention)

    int total_tokens() const;
    int num_decode_slots() const;
    int num_prefill_slots() const;
};
```

### 3.3 ScheduledBatch

```cpp
struct ScheduledBatch {
    std::vector<InferenceRequest *> decode_requests;  // each contributes 1 token
    std::vector<InferenceRequest *> prefill_requests; // each contributes N tokens (chunked)
    std::vector<int> prefill_chunk_sizes;             // tokens to process per prefill request

    int total_tokens() const;
    BatchContext build_context() const;  // assemble into BatchContext
};
```

---

## 4. Operator Changes

### 4.1 Per-Token Operators: No Change

| Operator | Single-Request | Batched | Change |
|----------|---------------|---------|--------|
| embedding | `[sl, vocab]` → `[sl, hidden]` | `[total, vocab]` → `[total, hidden]` | None (M increases) |
| linear | `[sl, K] × [K, N]` | `[total, K] × [K, N]` | None (M increases) |
| rms_norm | `[sl, hidden]` | `[total, hidden]` | None (per-row) |
| rope | `[sl, nhead, dim]` | `[total, nhead, dim]` | None (per-token, uses position_ids) |
| swiglu | `[sl, inter]` | `[total, inter]` | None (per-element) |
| add | `[sl, hidden]` | `[total, hidden]` | None (per-element) |

The M=total_tokens in GEMM is the main throughput win: cuBLAS with M=32 is much faster than 32× M=1.

### 4.2 Attention: Major Change

Attention is the **only** operator that must know request boundaries.

**Batched Decode Attention:**
```
Grid = (num_decode_requests, nhead)
Each CUDA block: one (request, head) pair
  → reads Q for this request's token
  → reads K/V via this request's block table
  → writes output for this request's token
```

This extends the current `paged_attention_decode_kernel` from Grid=(nhead) to Grid=(num_reqs, nhead). Each block uses `block_tables[req_idx]` and `seq_lens[req_idx]`.

**Batched Prefill Attention:**
```
For each prefill request in the batch:
  → call paged_attention_prefill with this request's Q chunk, block table, past_len
```

Prefill requests can be processed sequentially (one kernel launch per prefill request) or with a varlen kernel. Sequential is simpler and sufficient for the initial implementation since prefill is bounded by `max_prefill_tokens`.

### 4.3 K/V Write Path

For batched forward, each request's K/V projection outputs need to be written to that request's blocks:

```
linear produces: K_all = [total_tokens, kv_dim]  (concatenated)
scatter per slot:
  for each slot s:
    K_slot = K_all[s.token_offset : s.token_offset + s.num_tokens]
    write K_slot to request's blocks at positions [s.past_len .. s.past_len + s.num_tokens)
```

This is a per-slot scatter operation after K/V projection and RoPE.

### 4.4 Logits Routing

```
model.forward_batch() returns: logits = [total_tokens, vocab_size]
For decode slots: sample logits[slot.token_offset]
For prefill slots: sample logits[slot.token_offset + slot.num_tokens - 1] (last token)
```

---

## 5. Model Forward Batch

### 5.1 New Method: `Model::forward_batch()`

```cpp
// include/frontend/models/base.hpp
virtual tensor_t forward_batch(
    const BatchContext &batch,
    kvcache::BlockAllocator &allocator,
    kvcache::BlockPool &pool,
    const ExecutorConfig &exec_config) = 0;
```

### 5.2 Implementation Sketch

```cpp
tensor_t Qwen2Model::forward_batch(const BatchContext &batch, ...) {
    int total = batch.total_tokens();

    // Prepare flattened inputs
    auto ids = make({total}, I32);  ids->load(batch.token_ids);
    auto pos = make({total}, I64);  pos->load(batch.position_ids);

    // Embedding [total, hidden]
    auto hidden = make({total, hidden_size});
    ops::embedding(hidden, ids, W("embed_tokens.weight"));

    for (layer L) {
        // RMSNorm, Q/K/V projections: [total, dim] — identical to single-request
        ops::rms_norm(normed, hidden, ...);
        ops::linear(q_all, normed, W("q_proj"), ...);  // [total, hidden]
        ops::linear(k_all, normed, W("k_proj"), ...);  // [total, kv_dim]
        ops::linear(v_all, normed, W("v_proj"), ...);  // [total, kv_dim]

        // RoPE: [total, nhead, head_dim] — uses position_ids
        ops::rope(q_rope, q_all, pos, theta);
        ops::rope(k_rope, k_all, pos, theta);

        // === Per-slot K/V scatter to blocks ===
        for (slot s in batch.slots) {
            // Extract this slot's K/V from the flattened output
            auto k_slot = k_rope->slice(0, s.token_offset, s.token_offset + s.num_tokens);
            auto v_slot = v_all->slice(0, s.token_offset, s.token_offset + s.num_tokens);
            // Scatter to this request's blocks
            scatter_kv_to_blocks(k_slot, v_slot, s.request->block_table, L, s.past_len, pool);
        }

        // === Batched Attention ===
        auto attn = make({total, nhead, head_dim});

        // Decode slots: batched paged decode kernel
        if (batch.num_decode_slots() > 0) {
            ops::paged_attention_decode_batched(
                attn, q_rope, pool,
                batch.k_block_tables, batch.v_block_tables,
                batch.seq_lens, ...);
        }
        // Prefill slots: per-request paged prefill kernel
        for (prefill_slot s in batch) {
            ops::paged_attention_prefill(
                attn_slice, q_slice, pool,
                s.k_block_table, s.v_block_table,
                s.num_tokens, s.past_len, ...);
        }

        // O projection, residual, MLP: [total, dim] — identical to single-request
        ops::linear(o, attn, W("o_proj"), ...);
        ops::add(h1, hidden, o);
        // ... MLP ...
    }

    // Output head: [total, vocab_size]
    auto logits = make({total, vocab_size});
    ops::linear(logits, normed, W("lm_head.weight"), ...);
    return logits;
}
```

---

## 6. Scheduler Redesign

### 6.1 New Scheduler Interface

```cpp
class Scheduler {
public:
    Scheduler(SchedulerConfig config, kvcache::BlockAllocator &allocator);

    // Thread-safe submit (called from HTTP threads or session threads)
    void submit(std::unique_ptr<InferenceRequest> request);

    // Schedule next batch (called from engine loop)
    ScheduledBatch schedule();

    // Process results after forward (advance state, sample, trigger callbacks)
    void process_results(
        ScheduledBatch &batch,
        tensor_t logits,
        sampler::Sampler &sampler,
        tokenizer::Tokenizer &tokenizer,
        const std::vector<int> &stop_token_ids);

    bool has_work() const;

    // Single-request convenience (kept for bench/ping compatibility)
    GenerationResult run_one(
        model::Model &model,
        kvcache::KVCache &kvcache,
        const ExecutorConfig &exec_config,
        sampler::Sampler &sampler,
        tokenizer::Tokenizer &tokenizer,
        const std::vector<int> &stop_token_ids);

private:
    SchedulerConfig config_;
    kvcache::BlockAllocator &allocator_;
    std::mutex submit_mutex_;  // protects waiting_queue_ for concurrent submit

    std::deque<std::unique_ptr<InferenceRequest>> waiting_queue_;
    std::vector<std::unique_ptr<InferenceRequest>> active_requests_;  // in decode phase

    void schedule_decodes(ScheduledBatch &batch);
    void schedule_prefills(ScheduledBatch &batch);
    bool can_admit(const InferenceRequest &req) const;
    void complete_request(InferenceRequest &req);
};
```

### 6.2 Scheduling Algorithm

```
schedule():
    batch = empty

    // 1. Decode-first: all active decode requests (1 token each)
    for req in active_requests_:
        if batch.total_tokens() + 1 > max_batch_tokens: break
        batch.decode_requests.push_back(req)

    // 2. Admit new prefill requests (with token budget and block check)
    remaining = min(max_batch_tokens - batch.total_tokens(), max_prefill_tokens)
    while waiting_queue_ not empty AND remaining > 0:
        req = waiting_queue_.front()
        blocks_needed = estimate_blocks(req)
        if allocator_.available_blocks() < blocks_needed: break
        if active_requests_.size() + 1 > max_batch_requests: break

        // Allocate blocks for new request
        req.block_table = allocator_.allocate_sequence(req.input_ids.size())

        // Chunked prefill
        chunk = min(req.input_ids.size() - req.prefill_progress, remaining)
        batch.prefill_requests.push_back(req)
        batch.prefill_chunk_sizes.push_back(chunk)
        remaining -= chunk

        req.prefill_progress += chunk
        if req.prefill_progress >= req.input_ids.size():
            req.phase = DECODE
            active_requests_.push_back(req)
        waiting_queue_.pop_front()

    return batch
```

### 6.3 Result Processing

```
process_results(batch, logits, sampler, tokenizer, stop_ids):
    for decode_req in batch.decode_requests:
        token = sampler.sample(logits at decode_req's offset)
        decode_req.output_ids.push_back(token)
        decode_req.last_token = token
        decode_req.generated_count++

        if is_stop(token) or reached max_tokens:
            complete_request(decode_req)  // free blocks, set promise, remove from active
        else:
            if decode_req.stream_callback:
                decode_req.stream_callback(tokenizer.decode({token}))

    for prefill_req in batch.prefill_requests:
        if prefill_req.phase == DECODE:
            // Prefill just completed, sample first token
            token = sampler.sample(logits at last prefill token's offset)
            prefill_req.output_ids.push_back(token)
            prefill_req.last_token = token
            // Will be in decode batch next iteration
```

---

## 7. Engine Loop Redesign

### 7.1 Current: Synchronous Per-Request

```cpp
GenerationResult generate_tokens(kvcache, input_ids, config) {
    scheduler_.submit(request);
    return scheduler_.run_one(model, kvcache, ...);  // blocks until complete
}
```

### 7.2 Target: Iteration Loop

```cpp
// New engine method for serving mode
void InferenceEngine::run_loop() {
    while (true) {
        auto batch = scheduler_.schedule();
        if (batch.empty()) {
            // No work — wait for submit (condition variable)
            continue;
        }

        auto batch_ctx = batch.build_context();
        tensor_t logits = model_->forward_batch(batch_ctx, *block_allocator_, *block_pool_, exec_config_);
        scheduler_.process_results(batch, logits, *sampler_, *tokenizer_, stop_token_ids_);
    }
}
```

### 7.3 Backward Compatibility

`run_one()` is kept for `bench`/`chat`/`ping`. It wraps submit + blocking wait:

```cpp
GenerationResult run_one(...) {
    // Same as current implementation — submit, run prefill+decode loop for one request
    // Uses the old single-request path with PagedKVCache
}
```

The serving loop (`run_loop`) and single-request mode coexist. `run_one` is used by CLI tools, `run_loop` is used by the HTTP server (PR-10).

---

## 8. Attention Kernel Changes

### 8.1 Batched Paged Decode

Extend current `paged_attention_decode_kernel`:

```
Current:  Grid = (nhead), block_table is for ONE request
Batched:  Grid = (num_requests, nhead)
          block_tables[req_idx] = this request's block IDs
          seq_lens[req_idx] = this request's KV cache length
```

New op:
```cpp
void paged_attention_decode_batched(
    tensor_t out,              // [num_reqs, nhead, head_dim]
    tensor_t q,                // [num_reqs, nhead, head_dim]
    const void *pool_base,
    tensor_t k_block_tables,   // [num_reqs, max_blocks] int32
    tensor_t v_block_tables,   // [num_reqs, max_blocks] int32
    tensor_t seq_lens,         // [num_reqs] int32
    float scale,
    int nhead, int nkvhead, int head_dim, int block_size);
```

### 8.2 Prefill: Per-Request Sequential

For initial implementation, prefill requests call the existing `paged_attention_prefill` kernel once per request. No new kernel needed.

Future optimization: flash-attn varlen API with `cu_seqlens` to batch multiple prefills.

---

## 9. Thread Safety

| Component | Current | PR-9 |
|-----------|---------|------|
| `Scheduler::submit()` | No locking | `std::mutex` on `waiting_queue_` |
| `Scheduler::schedule()` | N/A | Called from engine thread only — no lock needed |
| `BlockAllocator` | No locking | Called from engine thread only — no lock needed |
| `InferenceRequest` | Owned by scheduler | Owned by scheduler, accessed from engine thread |

Only `submit()` needs a mutex since it's called from HTTP handler threads. All other scheduler methods run on the engine thread.

---

## 10. Implementation Tasks

### Task 1: BatchContext and ScheduledBatch data structures

**Files:**
- Create: `include/zedinfer/batch_context.hpp`
- Create: `src/zedinfer/batch_context.cpp`

- [ ] Define `BatchContext` with token_ids, slots, position_ids, block tables, seq_lens
- [ ] Define `ScheduledBatch` with decode/prefill request lists
- [ ] Implement `ScheduledBatch::build_context()` — assemble flattened batch
- [ ] Implement `BatchContext::total_tokens()`, `num_decode_slots()`, `num_prefill_slots()`
- [ ] Build to verify

---

### Task 2: Extend InferenceRequest for batched mode

**Files:**
- Modify: `include/zedinfer/request.hpp`

- [ ] Add `prefill_progress` field
- [ ] Add `SequenceBlockTable block_table` field (per-request KV ownership)
- [ ] Add `std::promise<GenerationResult> result_promise` for async result
- [ ] Build to verify

---

### Task 3: Batched paged decode attention kernel

**Files:**
- Modify: `src/backend/ops/self_attention/nvidia/paged_attention_nvidia.cu`
- Modify: `include/backend/ops/self_attention/nvidia/paged_attention_nvidia.cuh`
- Modify: `include/backend/ops/ops.hpp`
- Modify: `src/backend/ops/self_attention/op.cpp`

- [ ] Add `paged_attention_decode_batched` kernel: Grid=(num_reqs, nhead), per-request block table
- [ ] Add dispatch in ops.hpp and op.cpp
- [ ] Build to verify

---

### Task 4: Redesign Scheduler for batch scheduling

**Files:**
- Modify: `include/zedinfer/scheduler.hpp`
- Modify: `src/zedinfer/scheduler.cpp`

- [ ] Add `schedule()` method returning `ScheduledBatch`
- [ ] Add `process_results()` method
- [ ] Add `std::mutex` on `submit()` for thread safety
- [ ] Add `active_requests_` list (decode-phase requests)
- [ ] Implement decode-first scheduling with token budget
- [ ] Implement prefill admission with block availability check and chunking
- [ ] Implement `complete_request()` — free blocks, set promise
- [ ] Keep `run_one()` for backward compatibility
- [ ] Build to verify

---

### Task 5: Implement Model::forward_batch()

**Files:**
- Modify: `include/frontend/models/base.hpp`
- Create: `src/frontend/models/qwen2_forward_batch.cpp`
- Create: `src/frontend/models/qwen3_forward_batch.cpp`

- [ ] Add `virtual forward_batch()` to `Model` base class
- [ ] Implement for Qwen2: concat tokens → per-token ops → per-slot KV scatter → batched attention → logits
- [ ] Implement for Qwen3: same with per-head Q/K norm
- [ ] Build to verify

---

### Task 6: KV scatter helper for batched forward

**Files:**
- Create: `include/zedinfer/kv_scatter.hpp`
- Create: `src/zedinfer/kv_scatter.cpp`

- [ ] Implement `scatter_kv_to_blocks()`: given K/V tensor slice + block table + past_len, scatter tokens into blocks
- [ ] Handle decode (1 token, direct to block) and prefill (multi-token, per-token scatter)
- [ ] GPU variant (cudaMemcpy per token or custom scatter kernel)
- [ ] Build to verify

---

### Task 7: Engine iteration loop

**Files:**
- Modify: `include/zedinfer/engine.hpp`
- Modify: `src/zedinfer/engine.cpp`

- [ ] Add `run_loop()` method for serving mode (batch iteration loop)
- [ ] Modify scheduler construction to take `BlockAllocator&`
- [ ] Keep `generate_tokens()` / `run_one()` for CLI tools
- [ ] Build to verify

---

### Task 8: Build and correctness verification

- [ ] Build all targets (zedinfer_ops, ping, chat, bench)
- [ ] Run `ping` with single request — verify identical output
- [ ] Test 2 concurrent requests (via programmatic submit) — verify independent correct output
- [ ] Test 4 concurrent requests — verify throughput improvement
- [ ] Benchmark: single-request decode latency should not regress
- [ ] Benchmark: multi-request throughput (4, 8, 16 requests)

---

## 11. Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|-----------|
| Batched attention correctness (cross-request leakage) | Medium | High | Per-request block table isolation. Compare each request's output against single-request reference. |
| K/V scatter ordering in batched forward | Medium | High | Scatter must complete before attention for each layer. Synchronize or use same stream. |
| Memory pressure with many concurrent requests | Medium | Medium | Admission control via `available_blocks()`. Reject when pool is exhausted. |
| Chunked prefill state management | Medium | Medium | `prefill_progress` tracks position. Each chunk is an independent forward call. |
| Thread safety of submit | Low | Medium | Simple mutex on deque. Low contention (submit is fast). |
| `run_one()` regression | Low | Medium | Keep it as-is, only used by CLI tools. Separate code path from batched loop. |

---

## 12. Open Questions

1. **Should prefill requests in a batch share the forward pass with decode requests?**
   Phase 1: No. Process prefill and decode as separate forward calls within an iteration.
   Phase 2: Yes. Concat all tokens, use per-slot attention. Requires varlen attention kernel.

2. **When should blocks be allocated for new requests?**
   At admission time in `schedule()`. If allocation fails, request stays in queue.

3. **Should `run_one()` use the new batched infrastructure or stay as-is?**
   Stay as-is for simplicity. `run_one()` is for CLI tools where single-request latency matters.

4. **How to handle the HTTP server (PR-10) integration?**
   HTTP handlers call `submit()` with a `promise`. Engine `run_loop()` runs on a dedicated thread.
   Completed requests have their promises fulfilled, waking up the HTTP handler.

5. **Should we implement preemption in this PR?**
   No. Preemption (swap to CPU) is a separate concern. For now, if blocks run out, new requests wait in queue.
