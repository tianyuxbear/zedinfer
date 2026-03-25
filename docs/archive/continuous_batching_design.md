# Continuous Batching Design

## Concepts

**Continuous batching** (also called "iteration-level scheduling") allows new requests to join a running batch at every decode step, rather than waiting for the entire batch to finish. This maximizes GPU utilization and minimizes waiting time.

Key differences from static batching:
- Requests enter and leave the batch independently
- Prefill and decode requests can be mixed (or prefill is chunked)
- Each request tracks its own sequence length and KV cache state
- The scheduler re-evaluates the batch composition every iteration

## Request Abstraction

```cpp
// include/zedinfer/request.hpp

enum class RequestPhase {
    QUEUED,     // waiting for scheduler admission
    PREFILL,    // processing prompt tokens
    DECODE,     // generating tokens one-by-one
    COMPLETE,   // finished (EOS, max_tokens, or error)
    PREEMPTED   // temporarily removed from batch (for fairness)
};

struct InferenceRequest {
    // Identity
    uint64_t request_id;
    std::string session_id;

    // Input
    std::vector<int> prompt_ids;       // full tokenized prompt
    GenerationConfig config;

    // State
    RequestPhase phase = RequestPhase::QUEUED;
    int prefill_progress = 0;          // how many prompt tokens processed so far (for chunked prefill)
    int generated_count = 0;
    int last_token = -1;

    // KV cache reference
    // For paged KV cache: list of block IDs per layer
    // For contiguous KV cache: pointer to session's cache
    struct KVCacheRef {
        std::vector<std::vector<int>> block_ids;  // [layer][block_index] -> physical block id
        int seq_len = 0;                           // current cached token count
    } kv_ref;

    // Output
    std::vector<int> output_ids;
    std::function<void(const std::string &)> stream_callback;
    std::promise<GenerationResult> result_promise;

    // Timing
    std::chrono::steady_clock::time_point arrival_time;
    std::chrono::steady_clock::time_point first_token_time;
    GenerationStats stats;

    // Priority (lower = higher priority)
    int priority = 0;
};
```

## Scheduler Abstraction

```cpp
// include/zedinfer/scheduler.hpp

struct SchedulerConfig {
    int max_batch_tokens = 2048;     // max total tokens in a batch (prefill + decode)
    int max_batch_requests = 64;     // max concurrent requests in a batch
    int max_prefill_tokens = 512;    // max tokens to prefill per iteration (chunking)
    int max_queue_size = 256;        // max pending requests
    float preemption_threshold = 0.9; // block utilization threshold to trigger preemption
};

struct BatchSlot {
    InferenceRequest *request;
    int seq_offset;   // offset into request's token sequence (for chunked prefill)
    int seq_len;      // number of tokens in this slot for this iteration
};

struct ScheduledBatch {
    std::vector<BatchSlot> prefill_slots;  // requests doing prefill
    std::vector<BatchSlot> decode_slots;   // requests doing decode
    int total_tokens() const;
};

class Scheduler {
public:
    Scheduler(SchedulerConfig config, BlockAllocator &block_alloc);

    // Submit a new request
    void submit(std::unique_ptr<InferenceRequest> request);

    // Schedule next batch
    ScheduledBatch schedule();

    // Process results: advance request state, trigger callbacks
    void process_results(
        ScheduledBatch &batch,
        tensor_t logits,
        Sampler &sampler,
        Tokenizer &tokenizer);

    // Query state
    int pending_count() const;
    int active_count() const;
    bool has_work() const;

private:
    SchedulerConfig config_;
    BlockAllocator &block_alloc_;

    // Request pools
    std::deque<std::unique_ptr<InferenceRequest>> waiting_queue_;
    std::vector<InferenceRequest*> active_requests_;  // currently in decode phase
    std::vector<InferenceRequest*> preempted_requests_;

    // Internal scheduling policies
    void admit_requests(ScheduledBatch &batch);
    void schedule_decodes(ScheduledBatch &batch);
    void schedule_prefills(ScheduledBatch &batch);
    bool can_admit(const InferenceRequest &req) const;
    void preempt_if_needed();
    void reactivate_preempted();
};
```

## Admission Policy

The scheduler decides which waiting requests to admit based on:

1. **Token budget**: `batch.total_tokens() + new_request_prefill_tokens <= max_batch_tokens`
2. **KV block availability**: `block_alloc.available_blocks() >= estimated_blocks_needed(request)`
3. **Request limit**: `active_requests_.size() + prefill_slots.size() < max_batch_requests`
4. **Queue order**: FIFO within same priority level

Estimated blocks for a request:
```
estimated_blocks = ceil(prompt_tokens / block_size) + ceil(estimated_output_tokens / block_size)
```
Where `estimated_output_tokens` defaults to `min(max_new_tokens, 256)` as a conservative estimate.

## Token-Level Scheduling (Per-Iteration)

Each `schedule()` call produces one batch for one forward pass:

```
schedule():
    batch = empty

    // 1. Schedule all active decode requests (one token each)
    for req in active_requests_:
        if batch.total_tokens() + 1 > max_batch_tokens:
            break  // batch full
        batch.decode_slots.push_back({req, 0, 1})

    // 2. Try to admit new prefill requests
    remaining_budget = max_batch_tokens - batch.total_tokens()
    remaining_budget = min(remaining_budget, max_prefill_tokens)

    while waiting_queue_ not empty AND remaining_budget > 0:
        req = waiting_queue_.front()
        if not can_admit(req):
            break

        // Chunked prefill: process up to remaining_budget tokens
        tokens_to_process = min(req.prompt_ids.size() - req.prefill_progress, remaining_budget)
        batch.prefill_slots.push_back({req, req.prefill_progress, tokens_to_process})

        remaining_budget -= tokens_to_process
        req.prefill_progress += tokens_to_process

        if req.prefill_progress >= req.prompt_ids.size():
            // Prefill complete -> move to decode
            req.phase = RequestPhase::DECODE
            active_requests_.push_back(req)
        waiting_queue_.pop_front()

    return batch
```

## Fairness and Starvation Handling

### Problem
Long-running decode requests can monopolize the batch, starving new prefill requests. Conversely, large prefills can delay decode steps for existing requests.

### Policies

1. **Prefill token cap** (`max_prefill_tokens`): Limits tokens consumed by prefill per iteration. Ensures decode requests always get processed. Default: 512 tokens, so prefill of a 2048-token prompt takes 4 iterations.

2. **Priority aging**: Requests in `waiting_queue_` have their priority decreased (numerically) by 1 every N scheduler iterations they wait. Prevents indefinite starvation.

3. **Preemption**: When KV block utilization exceeds `preemption_threshold`:
   - Select the youngest (most recently admitted) lowest-priority request
   - Save its KV cache state (block IDs retained but marked reclaimable)
   - Move to `preempted_requests_`
   - Free blocks
   - Re-admit when blocks become available (preempted requests get higher priority)

4. **Max generation limit**: Requests exceeding `max_new_tokens` are forcibly completed. Prevents single request from monopolizing indefinitely.

5. **Decode-first policy**: Decode slots are always scheduled before prefill slots. This ensures Time-To-First-Token (TTFT) for new requests doesn't degrade Time-Between-Tokens (TBT) for active requests.

## Batch Assembly for Model Forward

The `ScheduledBatch` is converted to a `BatchContext` for `Model::forward_batch()`:

```cpp
struct BatchContext {
    // Flattened token IDs from all slots
    std::vector<int> token_ids;

    // Per-slot metadata
    struct SlotInfo {
        int request_idx;      // index into original batch
        int start_pos;        // starting position in the concatenated token_ids
        int seq_len;          // tokens in this slot
        int past_len;         // tokens already in KV cache for this request
        bool is_prefill;
    };
    std::vector<SlotInfo> slots;

    // For paged attention: block tables for all requests
    // block_tables[request_idx][layer_idx] = vector of physical block IDs
    std::vector<std::vector<std::vector<int>>> block_tables;

    int total_tokens() const;
};
```

For non-paged (contiguous) KV cache (initial implementation), `block_tables` is empty and attention uses direct slice access.

## Interaction with KV Cache

- **Submit**: Scheduler allocates initial KV blocks for new request
- **Prefill**: KV cache grows as prompt tokens are processed (new blocks allocated)
- **Decode**: Each token consumes 1 slot in the current block; new block allocated when current is full
- **Complete**: All blocks for the request are freed
- **Preempt**: Blocks are freed (data lost, request must re-prefill when resumed) — or optionally swapped to CPU for resume without re-prefill

## Implementation Order

1. **Single-request scheduler** (no batching): Wrap current generate loop in scheduler interface. One request at a time.
2. **Batched decode**: Multiple decode requests in one forward pass. Requires batch-aware operators.
3. **Prefill scheduling**: Admit prefill requests alongside decode.
4. **Chunked prefill**: Split large prefills across iterations.
5. **Preemption**: Block-level memory management + request swap.
