# R5: Split Engine Responsibilities - Detailed Refactoring Plan

**Goal:** Decompose the monolithic `InferenceEngine` into focused components with single responsibilities, preparing for HTTP API (PR-10) integration.

---

## 1. Current State Analysis (Post R1-R4)

### 1.1 InferenceEngine Responsibilities

`InferenceEngine` currently handles **5 distinct concerns**:

```
InferenceEngine (~380 lines)
  ├── Lifecycle: create(), constructor, init_block_pool(), build_stop_token_ids()
  ├── Session:   create_session()
  ├── Sync Gen:  generate(), generate_tokens(), build_request()
  ├── Batch:     submit_async(), step(), run_loop()
  └── Profile:   warmup(), profile()
```

### 1.2 Member Variables

```cpp
std::shared_ptr<model::Model> model_;
std::shared_ptr<tokenizer::Tokenizer> tokenizer_;
std::shared_ptr<sampler::Sampler> sampler_;
device::Device device_;
ExecutorConfig exec_config_;
ChatTemplate chat_template_;
std::vector<int> stop_token_ids_;
Scheduler scheduler_;
SchedulerConfig scheduler_config_;
std::unique_ptr<kvcache::BlockPool> block_pool_;
std::unique_ptr<kvcache::BlockAllocator> block_allocator_;
```

11 member variables spanning model runtime, tokenization, sampling, scheduling, memory management, and configuration.

### 1.3 Why Split Now?

PR-10 (HTTP API) will add:
- HTTP server lifecycle (start/stop)
- Request parsing (JSON → InferenceRequest)
- Response formatting (GenerationResult → JSON)
- SSE streaming
- Concurrency management (HTTP threads → engine thread)

Adding all of this to `InferenceEngine` would push it to 600+ lines with 7+ responsibilities. The split should happen **before** PR-10.

### 1.4 What's Already Clean

After R1-R4, the engine is already well-structured internally:
- `model_->forward_config()` + `transformer_forward()` handle all compute
- `scheduler_` handles batching and request lifecycle
- `PagedForwardContext` / `ContiguousForwardContext` handle KV management
- `block_pool_` / `block_allocator_` handle memory

The issue isn't internal spaghetti — it's that one class is the entry point for everything.

---

## 2. Target Design

### 2.1 Three Components

```
┌──────────────────────────────────────────────────┐
│  InferenceEngine (lifecycle + resource ownership)  │
│  - create(): load model, tokenizer, sampler       │
│  - init_block_pool(): allocate VRAM               │
│  - create_session(): allocate block table          │
│  - Owns: model_, tokenizer_, sampler_, pool_       │
│  - Provides accessors for shared resources         │
└──────────────┬──────────────┬────────────────────┘
               │              │
    ┌──────────▼──┐   ┌──────▼────────────┐
    │  Profiler    │   │  ServingLoop       │
    │  warmup()    │   │  submit_async()    │
    │  profile()   │   │  step()            │
    │              │   │  run_loop()        │
    │  Uses:       │   │  generate()        │
    │  DynamicKV   │   │  generate_tokens() │
    │  Contiguous  │   │                    │
    │  Context     │   │  Uses: Scheduler   │
    └─────────────┘   │  PagedForward      │
                      │  Context           │
                      └────────────────────┘
```

### 2.2 InferenceEngine (Slimmed)

Becomes a **resource container and factory**. No generate logic.

```cpp
class InferenceEngine : public std::enable_shared_from_this<InferenceEngine> {
public:
    static std::shared_ptr<InferenceEngine> create(
        const std::string &model_path, device::Device device);

    std::unique_ptr<InferenceSession> create_session(const GenerationConfig &config);

    // Resource accessors (used by ServingLoop and Profiler)
    model::Model &model() { return *model_; }
    tokenizer::Tokenizer &tokenizer() { return *tokenizer_; }
    sampler::Sampler &sampler() { return *sampler_; }
    const ExecutorConfig &exec_config() const { return exec_config_; }
    const ChatTemplate &chat_template() const { return chat_template_; }
    const std::vector<int> &stop_token_ids() const { return stop_token_ids_; }
    kvcache::BlockPool *block_pool() { return block_pool_.get(); }
    kvcache::BlockAllocator *block_allocator() { return block_allocator_.get(); }

private:
    // ... same members, but no generate/batch methods
};
```

### 2.3 ServingLoop (New)

Owns the Scheduler. Handles synchronous generation and batched serving.

```cpp
class ServingLoop {
public:
    ServingLoop(std::shared_ptr<InferenceEngine> engine);

    // Synchronous generation (session-based, used by chat/ping)
    std::string generate(kvcache::SequenceBlockTable &block_table,
                         const std::string &prompt,
                         const GenerationConfig &config);

    GenerationResult generate_tokens(kvcache::SequenceBlockTable &block_table,
                                     const std::vector<int> &input_ids,
                                     const GenerationConfig &config);

    // Async batch mode (used by HTTP API and batch_bench)
    std::future<GenerationResult> submit_async(std::unique_ptr<InferenceRequest> request);
    bool step();
    void run_loop();

private:
    std::shared_ptr<InferenceEngine> engine_;
    Scheduler scheduler_;
};
```

### 2.4 Profiler (New, optional)

Extracts warmup/profile into a standalone utility.

```cpp
class Profiler {
public:
    Profiler(std::shared_ptr<InferenceEngine> engine);

    void warmup(size_t prefill_len = 128, size_t decode_steps = 128);
    std::pair<double, double> profile(size_t prefill_len = 128, size_t decode_steps = 128);

private:
    std::shared_ptr<InferenceEngine> engine_;
};
```

### 2.5 Session Updated

Session holds a reference to `ServingLoop` (for `generate()`) instead of `InferenceEngine`:

```cpp
class InferenceSession {
    std::shared_ptr<InferenceEngine> engine_;  // for tokenizer, chat_template
    ServingLoop &serving_;                      // for generate()
};
```

Or simpler: session still holds engine, and engine provides access to the serving loop.

---

## 3. Migration Approach

### 3.1 Phase 1: Extract ServingLoop (Minimal Change)

Move generate/batch methods from engine to `ServingLoop`. Engine creates and owns a `ServingLoop`. Public API unchanged — engine delegates to serving loop.

```cpp
// Engine delegates:
std::string InferenceEngine::generate(...) {
    return serving_loop_->generate(...);
}
```

This is non-breaking. All existing code (chat, ping, bench, batch_bench) works unchanged.

### 3.2 Phase 2: Extract Profiler

Move warmup/profile to `Profiler`. Engine creates a temporary profiler during init.

### 3.3 Phase 3: Expose ServingLoop Directly (PR-10)

When HTTP API is added, it creates a `ServingLoop` and calls `submit_async()` + `run_loop()` directly, without going through engine delegation.

---

## 4. Files Affected

### New Files

| File | Purpose |
|------|---------|
| `include/zedinfer/serving_loop.hpp` | `ServingLoop` class |
| `src/zedinfer/serving_loop.cpp` | generate, generate_tokens, submit_async, step, run_loop |
| `include/zedinfer/profiler.hpp` | `Profiler` class |
| `src/zedinfer/profiler.cpp` | warmup, profile |

### Modified Files

| File | Change |
|------|--------|
| `include/zedinfer/engine.hpp` | Remove generate/batch/profile methods. Add resource accessors. Add `serving_loop_` member. |
| `src/zedinfer/engine.cpp` | Remove ~200 lines of generate/batch/profile code. Delegate to serving loop. |
| `include/zedinfer/session.hpp` | Minor: access serving loop via engine |
| `src/zedinfer/session.cpp` | Minor: call serving_loop->generate() |

### Unchanged Files

| File | Why |
|------|-----|
| Scheduler, ForwardContext, Model, Ops, Kernels | Internal to serving loop, not affected |
| Examples (chat, ping, bench, batch_bench) | Use engine API which delegates transparently |

---

## 5. Implementation Tasks

### Task 1: Create ServingLoop

- [ ] Create `include/zedinfer/serving_loop.hpp` and `src/zedinfer/serving_loop.cpp`
- [ ] Move `generate()`, `generate_tokens()`, `build_request()`, `submit_async()`, `step()`, `run_loop()` from engine
- [ ] ServingLoop constructor takes `shared_ptr<InferenceEngine>`, creates Scheduler
- [ ] Build to verify

### Task 2: Create Profiler

- [ ] Create `include/zedinfer/profiler.hpp` and `src/zedinfer/profiler.cpp`
- [ ] Move `warmup()` and `profile()` from engine
- [ ] Build to verify

### Task 3: Slim down Engine

- [ ] Remove moved methods from `engine.hpp/cpp`
- [ ] Add resource accessors to engine
- [ ] Engine creates `ServingLoop` and `Profiler` in `create()`
- [ ] Add delegation methods for backward compat (engine.generate → serving_loop.generate)
- [ ] Build to verify

### Task 4: Verify all entry points

- [ ] `ping` — uses session.chat() → engine.generate() → serving_loop
- [ ] `chat` — same
- [ ] `bench` — uses engine.profile() → profiler
- [ ] `batch_bench` — uses engine.submit_async() → serving_loop
- [ ] All produce correct output

---

## 6. Line Count Impact

```
engine.cpp before:  ~380 lines
engine.cpp after:   ~130 lines (create, init, accessors, delegation)
serving_loop.cpp:   ~180 lines (generate, batch, scheduler management)
profiler.cpp:       ~80 lines (warmup, profile)

Total: same, but split across 3 focused files.
```

---

## 7. Priority Assessment

**Low priority but recommended before PR-10.** The current engine works fine for all existing use cases. But PR-10 (HTTP API) will add server lifecycle, request parsing, and concurrency management. Doing it on the current monolithic engine would make the class unmanageable.

**If PR-10 is imminent:** do R5 first.
**If PR-10 is deferred:** R5 can wait.

---

## 8. Impact on PR-10 (HTTP API)

Without R5:
```
HttpServer → InferenceEngine::submit_async()  (engine = God class)
```

With R5:
```
HttpServer → ServingLoop::submit_async()  (clean separation)
HttpServer knows about: ServingLoop, InferenceRequest, GenerationResult
HttpServer does NOT know about: Model, Tokenizer, BlockPool, ForwardContext
```

The HTTP server only depends on the serving interface, not the inference internals.
