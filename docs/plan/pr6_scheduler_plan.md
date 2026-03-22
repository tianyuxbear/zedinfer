# PR-6: Scheduler (Single-Request Mode) - Detailed Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Introduce a `Scheduler` class that encapsulates the generate loop, processing one request at a time, to lay the foundation for continuous batching (PR-7).

**Architecture:** The scheduler receives `InferenceRequest` objects, manages a FIFO queue, and runs prefill+decode for a single request via `run_one()`. The engine's `generate_tokens()` delegates to the scheduler instead of running the generate loop inline. This establishes the submit-schedule-execute-process_results lifecycle that PR-7 extends to batched execution.

**Tech Stack:** C++17, existing `InferenceRequest`/`GenerationResult` types from PR-4, existing `Model::forward()` from PR-2.

---

## 1. Goals and Scope

### In Scope

| Item | Description |
|------|-------------|
| `Scheduler` class | New class with `submit()`, `has_work()`, `run_one()` methods |
| `SchedulerConfig` | Configuration struct (queue size, token limits) |
| Engine integration | `generate_tokens()` delegates to scheduler |
| `InferenceRequest` activation | Populate and use the `InferenceRequest` struct defined in PR-4 |
| Request lifecycle | QUEUED -> PREFILL -> DECODE -> COMPLETE state transitions |
| Sequential processing | One request at a time, FIFO order |

### Out of Scope

| Item | Belongs to |
|------|-----------|
| Batched forward pass | PR-7 (Continuous Batching) |
| `ScheduledBatch` / `BatchContext` | PR-7 |
| `forward_batch()` on Model | PR-7 |
| Block allocator / paged KV cache | PR-8 |
| Preemption | PR-7 |
| Thread safety / locking | PR-7 (needed only when requests arrive concurrently) |
| HTTP API | PR-10 |

---

## 2. Dependencies and Assumptions

### Hard Dependencies

| PR | What it provides | Confirmed present |
|----|-----------------|-------------------|
| PR-2 (Direct Forward) | `Model::forward()` at `include/frontend/models/base.hpp:101-105` | Yes |
| PR-4 (Stateless Engine) | `GenerationResult` at `include/zedinfer/request.hpp:15-18`, `InferenceRequest` at `request.hpp:35-46`, `RequestPhase` at `request.hpp:24-29`, stateless `generate_tokens()` at `engine.cpp:147-215` | Yes |
| PR-5 (Chat Template) | `ChatTemplate` at `include/zedinfer/chat_template.hpp`, `build_stop_token_ids()` at `engine.cpp:217-245` | Yes |

### Assumptions

1. The engine remains single-threaded for PR-6. Multiple concurrent callers are a PR-7 concern.
2. `InferenceRequest` is extended in-place (adding fields) rather than replaced.
3. The scheduler does not own the KV cache or model. These are passed to `run_one()` by the caller.
4. The scheduler does not own the tokenizer or sampler. These are also passed in.

---

## 3. Current Generate Flow (Baseline)

The current flow in `engine.cpp:147-215` (`generate_tokens`) is:

```
InferenceEngine::generate_tokens(kvcache, input_ids, config)
  |
  |-- prefill: model_->forward(input_ids, past_len, kvcache, exec_config_)
  |-- sample first token
  |-- if stop -> return
  |-- stream callback
  |
  |-- decode loop (i = 1 .. max_new_tokens):
  |     |-- model_->forward({next_token}, past_len, kvcache, exec_config_)
  |     |-- sample
  |     |-- stream callback
  |     |-- stop check (EOS, max_seq_len)
  |
  |-- return GenerationResult { output_ids, stats }
```

Key observations:
- The generate loop, timing, sampling, stop-checking, and streaming are all inline in `generate_tokens()`.
- `generate_tokens()` takes a raw `kvcache` reference and `input_ids` vector - not an `InferenceRequest`.
- `InferenceRequest` from PR-4 is defined but unused.

---

## 4. Target Generate Flow (After PR-6)

```
InferenceEngine::generate_tokens(kvcache, input_ids, config)
  |
  |-- build InferenceRequest from input_ids + config
  |-- scheduler_.submit(request)
  |-- result = scheduler_.run_one(model_, kvcache, exec_config_,
  |                                sampler_, tokenizer_, stop_token_ids_)
  |-- return result
```

Inside `Scheduler::run_one()`:
```
run_one(model, kvcache, exec_config, sampler, tokenizer, stop_ids)
  |
  |-- pop request from waiting_queue_
  |-- request.phase = PREFILL
  |
  |-- prefill: logits = model.forward(request.input_ids, past_len, kvcache, exec_config)
  |-- next_token = sampler.sample(logits)
  |-- update request state (phase, generated_count, last_token, output_ids)
  |-- timing
  |-- stop check -> if done, request.phase = COMPLETE, return result
  |-- stream callback
  |
  |-- request.phase = DECODE
  |-- decode loop:
  |     |-- logits = model.forward({request.last_token}, past_len, kvcache, exec_config)
  |     |-- next_token = sampler.sample(logits)
  |     |-- update request state
  |     |-- timing
  |     |-- stream callback
  |     |-- stop check (EOS, max_seq_len) -> if done, COMPLETE, break
  |
  |-- request.phase = COMPLETE
  |-- return GenerationResult { request.output_ids, stats }
```

The logic is identical to the current `generate_tokens()`, but:
1. It operates on an `InferenceRequest` object rather than raw parameters.
2. It tracks request phase transitions (QUEUED -> PREFILL -> DECODE -> COMPLETE).
3. It lives in `Scheduler` rather than `InferenceEngine`.
4. The scheduler holds a request queue (trivially one-deep for now).

---

## 5. Detailed Design

### 5.1 `SchedulerConfig`

```cpp
// include/zedinfer/scheduler.hpp

struct SchedulerConfig {
    int max_batch_tokens = 2048;     // Max total tokens in a batch (future PR-7)
    int max_batch_requests = 64;     // Max concurrent requests (future PR-7)
    int max_prefill_tokens = 512;    // Max prefill tokens per iteration (future PR-7)
    int max_queue_size = 256;        // Max pending requests in queue
};
```

In single-request mode, only `max_queue_size` is actively enforced. The other
fields are defined now so the struct doesn't change in PR-7.

### 5.2 `InferenceRequest` Extensions

The existing `InferenceRequest` at `include/zedinfer/request.hpp:35-46` needs
additional fields for the scheduler to operate:

```cpp
struct InferenceRequest {
    // --- Existing fields (from PR-4) ---
    uint64_t request_id = 0;
    std::string session_id;
    std::vector<int> input_ids;
    GenerationConfig config;
    RequestPhase phase = RequestPhase::QUEUED;
    int generated_count = 0;
    int last_token = -1;
    std::vector<int> output_ids;

    // --- New fields for PR-6 ---
    // Stream callback forwarded from GenerationConfig
    std::function<void(const std::string &)> stream_callback;

    // Timing
    std::chrono::steady_clock::time_point arrival_time;
    GenerationStats stats;
};
```

**Design decision:** `stream_callback` and `stats` are moved into the request
so the scheduler can operate on self-contained request objects. The callback is
copied from `GenerationConfig` at request construction time.

**Alternative considered:** Keep `stream_callback` on `GenerationConfig` and
pass config alongside the request. Rejected because the scheduler should operate
on requests, not on (request, config) pairs. The config fields that matter for
generation (`max_new_tokens`, `stream`, `verbose`) are already accessible via
`request.config`.

### 5.3 `Scheduler` Class

```cpp
// include/zedinfer/scheduler.hpp

class Scheduler {
public:
    explicit Scheduler(SchedulerConfig config);

    // Submit a new request to the queue.
    // Throws if queue is full (size >= max_queue_size).
    void submit(std::unique_ptr<InferenceRequest> request);

    // True if there are pending or active requests.
    bool has_work() const;

    // Number of requests in waiting queue.
    int pending_count() const;

    // Number of currently active requests (0 or 1 in single-request mode).
    int active_count() const;

    // Run one request to completion.
    // Pops the front request from the queue, runs prefill + decode loop,
    // returns the result.
    //
    // In PR-7 this evolves into schedule() + process_results() for batched
    // execution. The run_one() method is retained as a convenience for
    // single-request callers.
    GenerationResult run_one(
        model::Model &model,
        kvcache::KVCache &kvcache,
        const ExecutorConfig &exec_config,
        sampler::Sampler &sampler,
        tokenizer::Tokenizer &tokenizer,
        const std::vector<int> &stop_token_ids);

private:
    SchedulerConfig config_;

    // Request queue (FIFO). In single-request mode, typically 0 or 1 deep.
    std::deque<std::unique_ptr<InferenceRequest>> waiting_queue_;

    // Currently executing request (nullptr when idle).
    InferenceRequest *active_request_ = nullptr;

    // Request ID counter.
    uint64_t next_request_id_ = 1;
};
```

**Design decisions:**

1. **`run_one()` takes references, not ownership.** The scheduler does not own
   the model, KV cache, sampler, or tokenizer. These are engine resources. The
   scheduler owns request lifecycle only.

2. **`stop_token_ids` passed as parameter.** The scheduler needs to know when to
   stop generation. This is currently built in `engine.cpp:217-245`
   (`build_stop_token_ids`). Passing it as a parameter avoids the scheduler
   needing access to the engine's internals.

3. **`active_request_` is a raw pointer** (non-owning). The `unique_ptr` stays
   in the scheduler's internal state. In single-request mode, the request is
   popped from `waiting_queue_` and pointed to by `active_request_` during
   execution. After completion, the request is destroyed.

4. **No mutex.** Single-request mode is single-threaded. PR-7 adds a mutex when
   `submit()` can be called from a different thread than `run_one()`.

### 5.4 Engine Integration

**Changes to `include/zedinfer/engine.hpp`:**

Add a `Scheduler` member and a helper to build a request from raw parameters:

```cpp
class InferenceEngine : public std::enable_shared_from_this<InferenceEngine> {
    // ... existing public interface unchanged ...

private:
    // ... existing members ...
    Scheduler scheduler_;  // New: scheduler instance

    // Build an InferenceRequest from raw generate_tokens() parameters
    std::unique_ptr<InferenceRequest> build_request(
        const std::vector<int> &input_ids,
        const GenerationConfig &config);
};
```

**Changes to `src/zedinfer/engine.cpp`:**

`generate_tokens()` (currently at lines 147-215) is refactored to delegate:

```cpp
GenerationResult InferenceEngine::generate_tokens(
    kvcache::KVCache &kvcache,
    const std::vector<int> &input_ids,
    const GenerationConfig &config) {

    auto request = build_request(input_ids, config);
    scheduler_.submit(std::move(request));
    return scheduler_.run_one(
        *model_, kvcache, exec_config_,
        *sampler_, *tokenizer_, stop_token_ids_);
}
```

The current prefill + decode loop in `generate_tokens()` moves entirely into
`Scheduler::run_one()`. The engine method becomes a thin wrapper.

**`build_request()` implementation:**

```cpp
std::unique_ptr<InferenceRequest> InferenceEngine::build_request(
    const std::vector<int> &input_ids,
    const GenerationConfig &config) {

    auto req = std::make_unique<InferenceRequest>();
    req->input_ids = input_ids;
    req->config = config;
    req->stream_callback = config.stream ? config.stream_callback : nullptr;
    req->arrival_time = std::chrono::steady_clock::now();
    return req;
}
```

The `request_id` is assigned by the scheduler in `submit()`.

### 5.5 `Scheduler::run_one()` Implementation

This method contains the exact same logic as the current `generate_tokens()`
(engine.cpp:147-215), but operating on an `InferenceRequest`:

```
run_one():
    1. Pop front request from waiting_queue_
    2. Set active_request_ = request.get()
    3. request.phase = PREFILL
    4. Prefill:
       - logits = model.forward(request.input_ids, past_len, kvcache, exec_config)
       - next_token = sampler.sample(logits)
       - record prefill timing in request.stats
       - request.output_ids.push_back(next_token)
       - request.last_token = next_token
       - request.generated_count = 1
       - if stop(next_token): request.phase = COMPLETE, return
       - if stream_callback: callback(tokenizer.decode({next_token}))
    5. request.phase = DECODE
    6. past_len += input_ids.size()
    7. Decode loop (i = 1 .. max_new_tokens):
       - logits = model.forward({request.last_token}, past_len, kvcache, exec_config)
       - next_token = sampler.sample(logits)
       - record decode step timing in request.stats
       - request.output_ids.push_back(next_token)
       - request.last_token = next_token
       - request.generated_count++
       - past_len++
       - if stream_callback: callback(tokenizer.decode({next_token}))
       - if stop(next_token): break
       - if past_len >= max_seq_len: break
    8. request.phase = COMPLETE
    9. Finalize stats (generated_tokens, total_tokens)
   10. active_request_ = nullptr
   11. Return GenerationResult { request.output_ids, request.stats }
```

### 5.6 Stop Token Checking

The stop check logic currently in `engine.cpp:247-252` (`should_stop()`) is
simple: iterate `stop_token_ids_` and check for match. Rather than moving this
method to the scheduler, it is passed as a `const std::vector<int>&` parameter
and checked inline. This keeps the scheduler decoupled from the engine.

Alternatively, a free function or lambda can be used:

```cpp
// In scheduler.cpp
static bool is_stop_token(int token_id, const std::vector<int> &stop_ids) {
    for (int stop_id : stop_ids) {
        if (token_id == stop_id) return true;
    }
    return false;
}
```

---

## 6. File Structure

### New Files

| File | Responsibility |
|------|---------------|
| `include/zedinfer/scheduler.hpp` | `Scheduler` class declaration, `SchedulerConfig` struct |
| `src/zedinfer/scheduler.cpp` | `Scheduler` method implementations (`submit`, `run_one`, `has_work`, etc.) |

### Modified Files

| File | What Changes |
|------|-------------|
| `include/zedinfer/request.hpp` | Add `stream_callback`, `arrival_time`, `stats` fields to `InferenceRequest` |
| `include/zedinfer/engine.hpp` | Add `Scheduler scheduler_` member, add `build_request()` private method |
| `src/zedinfer/engine.cpp` | `generate_tokens()` delegates to scheduler; add `build_request()` implementation; remove inline prefill+decode loop |
| Build config (xmake) | Add `scheduler.cpp` to zedinfer target sources |

### Unchanged Files

| File | Why Unchanged |
|------|--------------|
| `include/zedinfer/session.hpp` | Session calls `engine_->generate()` which calls `generate_tokens()` - the session API is unaffected |
| `src/zedinfer/session.cpp` | No change - session calls `engine_->generate()` as before |
| `examples/chat.cpp` | Calls `session->chat()` - unaffected |
| `examples/ping.cpp` | Calls `session->chat()` - unaffected |
| `examples/bench.cpp` | Uses `engine->profile()` / `engine->warmup()` which bypass generate_tokens() - unaffected |
| Model / operator code | No changes to forward path or operator kernels |

---

## 7. Interface Changes Summary

| Symbol | Before | After |
|--------|--------|-------|
| `Scheduler` | Does not exist | **New** class in `scheduler.hpp` |
| `SchedulerConfig` | Does not exist | **New** struct in `scheduler.hpp` |
| `InferenceRequest::stream_callback` | Does not exist | **New** field |
| `InferenceRequest::arrival_time` | Does not exist | **New** field |
| `InferenceRequest::stats` | Does not exist | **New** field |
| `InferenceEngine::scheduler_` | Does not exist | **New** private member |
| `InferenceEngine::build_request()` | Does not exist | **New** private method |
| `InferenceEngine::generate_tokens()` | Contains inline prefill+decode loop | Delegates to `scheduler_.run_one()` |

**Public API impact:** None. `InferenceEngine::generate_tokens()` signature is unchanged.
`InferenceEngine::generate()` signature is unchanged. `InferenceSession::chat()` is unchanged.
All examples compile and behave identically.

---

## 8. State Transitions and Request Lifecycle

```
                submit()
                  |
                  v
             +---------+
             | QUEUED  |  (in waiting_queue_)
             +---------+
                  |
             run_one() pops from queue
                  |
                  v
             +---------+
             | PREFILL |  (active_request_ set, processing prompt)
             +---------+
                  |
             prefill complete
                  |
            +-----+------+
            |            |
       (stop token)  (continue)
            |            |
            v            v
       +---------+  +---------+
       | COMPLETE|  | DECODE  |  (token-by-token generation)
       +---------+  +---------+
                         |
                    (stop/max_tokens/max_seq_len)
                         |
                         v
                    +---------+
                    | COMPLETE|
                    +---------+
                         |
                    return GenerationResult
                    active_request_ = nullptr
                    request destroyed
```

In single-request mode:
- At most one request is in PREFILL or DECODE at any time.
- The waiting queue is typically 0 or 1 deep.
- `run_one()` blocks until the request completes.

---

## 9. Relationship to PR-7 (Continuous Batching)

PR-6 establishes the interface and request lifecycle. PR-7 extends it:

| PR-6 (Single-Request) | PR-7 (Batched) |
|-----------------------|-----------------|
| `run_one()` processes one request synchronously | `schedule()` assembles a `ScheduledBatch` of multiple requests |
| No batch assembly | `BatchContext` built from `ScheduledBatch` |
| `model.forward()` (single sequence) | `model.forward_batch()` (multiple sequences) |
| No preemption | Preemption when KV blocks run low |
| No concurrency | `submit()` called from HTTP thread, `schedule()` from engine thread |
| `waiting_queue_` is 0-1 deep | `waiting_queue_` can be deep |
| No `active_requests_` list (just `active_request_` pointer) | `active_requests_` vector tracks all in-flight decode requests |

The key interfaces that PR-7 adds to `Scheduler`:
- `ScheduledBatch schedule()` - replaces the "pop and run" pattern
- `void process_results(ScheduledBatch&, tensor_t logits, ...)` - post-forward result routing
- Thread-safe `submit()` with mutex
- `active_requests_` vector

PR-6's `run_one()` is retained as a convenience method for single-request callers
(e.g., bench, ping) even after PR-7 adds batched execution.

---

## 10. Implementation Tasks

### Task 1: Extend InferenceRequest

**Files:**
- Modify: `include/zedinfer/request.hpp:35-46`

- [ ] **Step 1:** Read the current `InferenceRequest` definition
- [ ] **Step 2:** Add `stream_callback`, `arrival_time`, `stats` fields
- [ ] **Step 3:** Add necessary includes (`<functional>`, `<chrono>`, for `GenerationStats`)
- [ ] **Step 4:** Build to verify compilation
- [ ] **Step 5:** Commit: `feat(request): add stream_callback, arrival_time, stats to InferenceRequest`

---

### Task 2: Create Scheduler header

**Files:**
- Create: `include/zedinfer/scheduler.hpp`

- [ ] **Step 1:** Create `scheduler.hpp` with `SchedulerConfig` struct and `Scheduler` class declaration
- [ ] **Step 2:** Include necessary headers (`request.hpp`, `generation_types.hpp`, forward declarations for Model, KVCache, etc.)
- [ ] **Step 3:** Build to verify header compiles when included
- [ ] **Step 4:** Commit: `feat(scheduler): add Scheduler class declaration and SchedulerConfig`

---

### Task 3: Implement Scheduler

**Files:**
- Create: `src/zedinfer/scheduler.cpp`
- Modify: Build config (xmake) to add `scheduler.cpp`

- [ ] **Step 1:** Implement `Scheduler::Scheduler(config)` constructor
- [ ] **Step 2:** Implement `submit()` - assign request_id, push to waiting_queue_, validate queue size
- [ ] **Step 3:** Implement `has_work()`, `pending_count()`, `active_count()` query methods
- [ ] **Step 4:** Implement `run_one()` - move the prefill+decode logic from `engine.cpp:147-215` into this method, operating on `InferenceRequest`
- [ ] **Step 5:** Add `scheduler.cpp` to the build target
- [ ] **Step 6:** Build to verify compilation
- [ ] **Step 7:** Commit: `feat(scheduler): implement single-request Scheduler with run_one()`

---

### Task 4: Wire Scheduler into Engine

**Files:**
- Modify: `include/zedinfer/engine.hpp`
- Modify: `src/zedinfer/engine.cpp`

- [ ] **Step 1:** Add `#include "zedinfer/scheduler.hpp"` and `Scheduler scheduler_` member to `engine.hpp`
- [ ] **Step 2:** Add `build_request()` private method declaration to `engine.hpp`
- [ ] **Step 3:** Initialize `scheduler_` in engine constructor (with default `SchedulerConfig`)
- [ ] **Step 4:** Implement `build_request()` in `engine.cpp`
- [ ] **Step 5:** Replace the inline prefill+decode loop in `generate_tokens()` with delegation to `scheduler_.submit()` + `scheduler_.run_one()`
- [ ] **Step 6:** Build to verify compilation
- [ ] **Step 7:** Commit: `refactor(engine): delegate generate_tokens() to Scheduler::run_one()`

---

### Task 5: Build verification and functional test

- [ ] **Step 1:** Run `bash auto-build-test/scripts/run_build_check.sh` - verify clean build
- [ ] **Step 2:** Verify `bench`, `chat`, `ping` all compile and link
- [ ] **Step 3:** Run `ping` example - verify output is identical to pre-PR-6
- [ ] **Step 4:** Run `bench` example - verify no performance regression (< 0.5ms overhead)
- [ ] **Step 5:** Run existing test suite (`test-tokenizer`, `test-loader`, etc.)
- [ ] **Step 6:** If any failures, fix and retry
- [ ] **Step 7:** Commit any fixes; final commit: `test: verify scheduler integration passes all checks`

---

## 11. Correctness Validation

1. **Bit-identical output:** The scheduler's `run_one()` must produce the exact
   same token sequence as the current `generate_tokens()` for any given prompt.
   The logic is transplanted, not rewritten.

2. **Stats accuracy:** `GenerationResult.stats` must report the same timing
   values (prefill_time_ms, decode_time_ms, total_time_ms) as before. The
   timing instrumentation is preserved identically.

3. **Stream callback:** Token streaming must fire at the same points with the
   same decoded text as before.

4. **Stop token behavior:** Generation must stop on the same tokens as before.
   The stop logic is unchanged, just relocated.

5. **Sequential requests:** Submit 3 requests sequentially, verify each produces
   independent correct output and stats.

---

## 12. Benchmark Plan

Run config B (Qwen2-1.5B, prefill=128, decode=128, GPU) before and after.

**Expected:** Less than 0.5ms overhead from scheduler indirection. The scheduler
adds one `deque::push_back`, one `deque::pop_front`, and a few field assignments
per request. This is negligible compared to model forward times (~10-100ms).

If overhead exceeds 0.5ms, investigate whether `InferenceRequest` construction
(copying `input_ids` vector) is the cause and consider a move-only path.

---

## 13. Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|-----------|
| Logic divergence during transplant | Low | High (wrong output) | Bit-identical E2E test. Minimal refactoring - move code, don't rewrite. |
| `InferenceRequest` copy overhead | Very Low | Low | `input_ids` is moved into request via `build_request()`. Reserve `output_ids`. |
| Over-abstraction for single-request | Medium | Low | Keep `run_one()` as a simple sequential method. No threading, no callbacks, no batch assembly. |
| `std::function` in request causes allocation | Low | Very Low | Only one copy per request. Not in hot loop. |
| Build system fails to pick up new `.cpp` | Low | Low | Add to xmake target explicitly. |
| `warmup()` / `profile()` bypass scheduler | None | None | By design: these methods don't use `generate_tokens()`. They call `model_->forward()` directly. No change needed. |

---

## 14. Rollback

Remove `scheduler.hpp`, `scheduler.cpp`. Revert `engine.hpp` and `engine.cpp`
to restore the inline generate loop. All changes are localized to 4 files
(2 new, 2 modified). The `InferenceRequest` field additions are harmless
and can be left in place.

---

## 15. Open Questions

1. **Should `SchedulerConfig` be passed to `InferenceEngine::create()` or
   hard-defaulted?**
   Recommendation: Hard-default for now. The config values only matter in
   PR-7 (batching). Expose as a parameter when there's a real use case.

2. **Should `run_one()` be blocking or return a future?**
   Recommendation: Blocking. Single-request mode is synchronous. PR-7/PR-10
   may introduce async patterns if HTTP serving requires it.

3. **Should `stream_callback` live on `InferenceRequest` or remain on
   `GenerationConfig`?**
   Recommendation: Copy to `InferenceRequest` at construction time. The
   scheduler should operate on self-contained requests. This is consistent
   with the continuous_batching_design.md which places `stream_callback` on
   the request (line 51).

4. **Should `warmup()` and `profile()` also go through the scheduler?**
   Recommendation: No. These are engine-level utilities that directly call
   `model_->forward()` with dummy data. They don't produce `InferenceRequest`
   objects and don't need scheduling.

---

## 16. Checklist for Reviewer

- [ ] `Scheduler` class exists with `submit()`, `run_one()`, `has_work()`, `pending_count()`, `active_count()`
- [ ] `SchedulerConfig` defines batch/queue limits (used by PR-7, defined now)
- [ ] `InferenceRequest` has `stream_callback`, `arrival_time`, `stats` fields
- [ ] `run_one()` logic is identical to the previous `generate_tokens()` loop
- [ ] `generate_tokens()` is now a thin wrapper that builds a request and delegates
- [ ] Request lifecycle: QUEUED -> PREFILL -> DECODE -> COMPLETE
- [ ] `bench`, `chat`, `ping` compile and produce correct output
- [ ] No mutable state added to `InferenceEngine` (scheduler is the only new member, and it's encapsulated)
- [ ] No threading, no mutex (single-request mode)
- [ ] No performance regression on config B
- [ ] No operator or model code changes
