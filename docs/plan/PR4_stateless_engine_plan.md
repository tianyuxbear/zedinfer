# PR-4: Stateless Engine and Request Types - Detailed Implementation Plan

## Goal

Make `InferenceEngine` truly stateless by removing the mutable `last_stats_` member
and returning generation results (including stats) per-call. Introduce `InferenceRequest`
and `GenerationResult` types that will later plug into the scheduler (PR-6).

This is a pure API-reshaping PR. No new compute logic, no kernel changes, no performance
impact. The primary value is eliminating shared mutable state from the engine, which is
a prerequisite for safe concurrent multi-session access.

---

## Problem Statement

### Thread Safety Issue

`InferenceEngine` is designed to be shared across sessions (`std::shared_ptr<InferenceEngine>`
held by each `InferenceSession`). However, `generate_tokens()` mutates `last_stats_`
(`engine.hpp:61`) on every call:

- `engine.cpp:145` - resets `last_stats_`
- `engine.cpp:159` - calls `update_stats_prefill()`
- `engine.cpp:185` - calls `update_stats_decode()`
- `engine.cpp:205-206` - writes `generated_tokens` and `total_tokens`

If two sessions call `generate()` concurrently on the same engine, `last_stats_` is a
data race (R2 in `docs/notes/risk_register.md`).

### Leaky Abstraction

`last_stats()` returns a reference to engine-internal state that is overwritten on
the next `generate_tokens()` call. The caller must read it before any other session
generates, or the data is lost. This is fragile and undocumented.

### Missing Request Abstraction

The scheduler (PR-6) needs a `InferenceRequest` type to track per-request state
(phase, generated count, KV cache reference, output buffer, etc.). Defining this
type now avoids a larger API change later.

---

## Scope

| In Scope | Out of Scope |
|----------|-------------|
| Remove `last_stats_` from `InferenceEngine` | Scheduler (PR-6) |
| Return `GenerationResult` from `generate_tokens()` | `SessionRegistry` (PR-6+) |
| Define `InferenceRequest` struct | Chat template extraction (PR-5) |
| Adapt `generate()` to use returned result | Batched forward (PR-7) |
| Adapt `session.cpp`, `bench.cpp`, `ping.cpp` | Multi-session concurrency test |
| Define `RequestPhase` enum | |

---

## Dependencies

- **PR-2** (Direct Model Forward): Must be merged. `engine.cpp` already calls
  `model_->forward()` directly (confirmed at `engine.cpp:155,181`).
- **PR-3a/3b** (cuBLAS/oneDNN): Independent. PR-4 does not touch operator code.

---

## Files Affected

| Action | File | What Changes |
|--------|------|-------------|
| **New** | `include/zedinfer/request.hpp` | `GenerationResult`, `InferenceRequest`, `RequestPhase` |
| **Modify** | `include/zedinfer/engine.hpp` | Remove `last_stats_`, `last_stats()`, `update_stats_*`; change return types |
| **Modify** | `src/zedinfer/engine.cpp` | `generate_tokens()` returns `GenerationResult`; `generate()` uses local stats |
| **Modify** | `include/zedinfer/session.hpp` | Add `last_stats_` to session (moved from engine) |
| **Modify** | `src/zedinfer/session.cpp` | Read stats from `GenerationResult` returned by `generate()` |
| **Modify** | `examples/bench.cpp` | No change needed (uses `profile()`, not `last_stats()`) |
| **Modify** | `examples/ping.cpp` | No change needed (uses `session->chat()` which handles stats internally) |

---

## Detailed Design

### 1. New File: `include/zedinfer/request.hpp`

```cpp
#pragma once

#include "zedinfer/generation_types.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace zedinfer {

/**
 * Result of a single generation call.
 * Returned by InferenceEngine::generate_tokens() and generate().
 */
struct GenerationResult {
    std::vector<int> output_ids;
    GenerationStats stats;
};

/**
 * Phase of an inference request lifecycle.
 * Used by the scheduler (PR-6+) to track request progress.
 */
enum class RequestPhase {
    QUEUED,     // waiting for scheduler admission
    PREFILL,    // processing prompt tokens
    DECODE,     // generating tokens one-by-one
    COMPLETE    // finished (EOS, max_tokens, or error)
};

/**
 * A single inference request.
 * Currently a data definition for future scheduler integration (PR-6).
 * Not consumed by engine or session in this PR.
 */
struct InferenceRequest {
    uint64_t request_id = 0;
    std::string session_id;
    std::vector<int> input_ids;
    GenerationConfig config;

    RequestPhase phase = RequestPhase::QUEUED;
    int generated_count = 0;
    int last_token = -1;

    std::vector<int> output_ids;
};

} // namespace zedinfer
```

**Design decisions:**

- `GenerationResult` bundles output IDs + stats. This is the return type for
  `generate_tokens()`, replacing the bare `std::vector<int>`.
- `InferenceRequest` is defined now but not used by engine/session in this PR.
  It is a forward declaration for PR-6 (scheduler). Defining it here avoids
  a second API-breaking change later.
- `stream_callback` and `result_promise` are intentionally omitted from
  `InferenceRequest` at this stage. They will be added in PR-6 when the
  scheduler needs async result delivery.
- `RequestPhase::PREEMPTED` (from `docs/design/continuous_batching_design.md`)
  is omitted for now - it belongs to the preemption logic in PR-7.

### 2. Changes to `include/zedinfer/engine.hpp`

**Remove:**
- Line 43: `const GenerationStats &last_stats() const { return last_stats_; }`
- Line 61: `GenerationStats last_stats_;`
- Line 64: `void update_stats_prefill(double time_ms, int num_tokens);`
- Line 65: `void update_stats_decode(double time_ms);`

**Change return types:**
- `generate_tokens()` (line 38-41): `std::vector<int>` -> `GenerationResult`
- `generate()` (line 33-36): Return type stays `std::string`, but internally
  uses the `GenerationResult` from `generate_tokens()` instead of `last_stats_`.

**New include:**
- `#include "zedinfer/request.hpp"` (for `GenerationResult`)

**Result:**

```cpp
class InferenceEngine : public std::enable_shared_from_this<InferenceEngine> {
public:
    static std::shared_ptr<InferenceEngine> create(
        const std::string &model_path,
        device::Device device);

    std::unique_ptr<InferenceSession> create_session(const GenerationConfig &gen_config);

    std::string generate(
        kvcache::KVCache &kvcache,
        const std::string &prompt,
        const GenerationConfig &config);

    GenerationResult generate_tokens(
        kvcache::KVCache &kvcache,
        const std::vector<int> &input_ids,
        const GenerationConfig &config);

    // last_stats() removed — stats are in GenerationResult

    void warmup(size_t prefill_len = 128, size_t decode_steps = 128);
    std::pair<double, double> profile(size_t prefill_len = 128, size_t decode_steps = 128);

private:
    InferenceEngine(
        std::shared_ptr<model::Model> model,
        std::shared_ptr<tokenizer::Tokenizer> tokenizer,
        std::shared_ptr<sampler::Sampler> sampler,
        device::Device device,
        ExecutorConfig exec_config);

    std::shared_ptr<model::Model> model_;
    std::shared_ptr<tokenizer::Tokenizer> tokenizer_;
    std::shared_ptr<sampler::Sampler> sampler_;
    device::Device device_;
    ExecutorConfig exec_config_;
    // last_stats_ removed
    // update_stats_prefill() removed
    // update_stats_decode() removed

    bool should_stop(int token_id) const;
};
```

### 3. Changes to `src/zedinfer/engine.cpp`

#### `generate_tokens()` (lines 140-208)

Stats become local to the function and are returned as part of `GenerationResult`.

**Before (current):**
```cpp
std::vector<int> InferenceEngine::generate_tokens(...) {
    last_stats_ = GenerationStats();
    last_stats_.prompt_tokens = input_ids.size();
    // ... generate loop writing to last_stats_ ...
    return generated_ids;
}
```

**After:**
```cpp
GenerationResult InferenceEngine::generate_tokens(
    kvcache::KVCache &kvcache,
    const std::vector<int> &input_ids,
    const GenerationConfig &config) {

    GenerationResult result;
    GenerationStats &stats = result.stats;
    stats.prompt_tokens = input_ids.size();

    result.output_ids.reserve(config.max_new_tokens);

    // Prefill
    auto t0 = std::chrono::high_resolution_clock::now();

    int past_len = kvcache.current_length();
    tensor_t logits = model_->forward(input_ids, past_len, kvcache, exec_config_);
    int next_token = sampler_->sample(logits);

    auto t1 = std::chrono::high_resolution_clock::now();
    stats.prefill_time_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    stats.total_time_ms += stats.prefill_time_ms;

    result.output_ids.push_back(next_token);

    if (config.stream && config.stream_callback) {
        config.stream_callback(tokenizer_->decode({next_token}));
    }

    if (should_stop(next_token)) {
        stats.generated_tokens = result.output_ids.size();
        stats.total_tokens = stats.prompt_tokens + stats.generated_tokens;
        return result;
    }

    // Decode
    past_len += input_ids.size();

    for (int i = 1; i < config.max_new_tokens; ++i) {
        auto s0 = std::chrono::high_resolution_clock::now();

        logits = model_->forward({next_token}, past_len, kvcache, exec_config_);
        next_token = sampler_->sample(logits);

        auto s1 = std::chrono::high_resolution_clock::now();
        double step_ms = std::chrono::duration<double, std::milli>(s1 - s0).count();
        stats.decode_time_ms += step_ms;
        stats.total_time_ms += step_ms;

        result.output_ids.push_back(next_token);
        past_len++;

        if (config.stream && config.stream_callback) {
            config.stream_callback(tokenizer_->decode({next_token}));
        }

        if (should_stop(next_token)) break;

        if (past_len >= static_cast<int>(tokenizer_->get_config().model_max_length)) {
            if (config.verbose) {
                LOGI << "[Inference] Reached max sequence length";
            }
            break;
        }
    }

    stats.generated_tokens = result.output_ids.size();
    stats.total_tokens = stats.prompt_tokens + stats.generated_tokens;
    return result;
}
```

**Key changes:**
- `last_stats_` replaced by local `stats` reference into `result.stats`
- `update_stats_prefill()` / `update_stats_decode()` inlined (they were trivial wrappers)
- `generated_ids` replaced by `result.output_ids`
- Returns `GenerationResult` (moved, not copied - RVO applies)

#### `generate()` (lines 98-138)

**Before:**
```cpp
auto output_ids = generate_tokens(kvcache, input_ids, config);
std::string output = tokenizer_->decode(output_ids);
// ...
if (config.print_stats) {
    LOGI << last_stats_.summary();
}
```

**After:**
```cpp
auto result = generate_tokens(kvcache, input_ids, config);
std::string output = tokenizer_->decode(result.output_ids);
// ...
if (config.print_stats) {
    LOGI << result.stats.summary();
}
```

#### Remove helper methods

Delete:
- `update_stats_prefill()` (lines 214-218)
- `update_stats_decode()` (lines 220-222)

These are replaced by direct assignment to the local `stats` reference.

### 4. Changes to `include/zedinfer/session.hpp` and `src/zedinfer/session.cpp`

The session's `chat()` method calls `engine_->generate()`, which returns a `std::string`.
Internally, `generate()` now uses `GenerationResult` but still returns the decoded string.
So the session code needs minimal change.

However, the session currently has no way to access generation stats after `chat()`.
To preserve the stat-printing behavior (which was previously driven by `config.print_stats`
inside `engine_->generate()`), **no session change is needed** - the stats printing
happens inside `generate()` before returning the string.

If we want to expose stats to the session for future use (e.g., returning stats
from `chat()`), we can optionally:

**Option A (minimal, recommended for this PR):**
No change to session. Stats are printed by `generate()` internally.

**Option B (for future use):**
Change `generate()` to return `std::pair<std::string, GenerationStats>` or a new
`GenerateStringResult` struct. This would require changing session and all callers.

**Recommendation: Option A.** Keep the session change minimal. The scheduler (PR-6)
will restructure the generate path anyway. Changing `generate()` return type now
would be churn.

### 5. Changes to Examples

#### `examples/bench.cpp`
**No changes needed.** `bench.cpp` uses `engine->profile()` and `engine->warmup()`,
neither of which touch `last_stats_` or `generate_tokens()`.

#### `examples/ping.cpp`
**No changes needed.** `ping.cpp` calls `session->chat(prompt)` which calls
`engine_->generate()`. The stats are printed inside `generate()` via
`config.print_stats`, which is unaffected by this refactor.

#### `examples/chat.cpp`
**No changes needed.** Same reasoning as `ping.cpp`.

---

## Implementation Steps

```
Step 1: Create include/zedinfer/request.hpp
        Define GenerationResult, RequestPhase, InferenceRequest.

Step 2: Modify include/zedinfer/engine.hpp
        - Add #include "zedinfer/request.hpp"
        - Change generate_tokens() return type to GenerationResult
        - Remove last_stats_ member
        - Remove last_stats() accessor
        - Remove update_stats_prefill() declaration
        - Remove update_stats_decode() declaration

Step 3: Modify src/zedinfer/engine.cpp
        - Rewrite generate_tokens() to use local GenerationResult
        - Rewrite generate() to use result from generate_tokens()
        - Delete update_stats_prefill() definition
        - Delete update_stats_decode() definition

Step 4: Build and verify
        - Run auto-build-test to confirm compilation
        - Verify bench, chat, ping compile and link

Step 5: Test
        - Run existing tests (test-tokenizer, test-loader, etc.)
        - Run bench/chat/ping to verify functional correctness
        - Verify stats output is identical to before
```

---

## Interface Changes Summary

| Symbol | Before | After |
|--------|--------|-------|
| `InferenceEngine::generate_tokens()` | Returns `std::vector<int>` | Returns `GenerationResult` |
| `InferenceEngine::last_stats()` | Public accessor | **Removed** |
| `InferenceEngine::last_stats_` | Private member | **Removed** |
| `InferenceEngine::update_stats_prefill()` | Private method | **Removed** |
| `InferenceEngine::update_stats_decode()` | Private method | **Removed** |
| `GenerationResult` | Does not exist | **New** in `request.hpp` |
| `RequestPhase` | Does not exist | **New** in `request.hpp` |
| `InferenceRequest` | Does not exist | **New** in `request.hpp` (unused in this PR) |

---

## Build System Changes

None. No new `.cpp` files (only a new header). No new dependencies.
The new `request.hpp` header is included by `engine.hpp`, which is already
in the include path.

---

## Correctness Validation

1. **E2E snapshot test** (if available from PR-1): Output token IDs must be
   **bit-identical** to before. This PR changes only how stats are stored
   and returned, not any compute logic.

2. **Stats verification**: Run `ping` with `print_stats = true`. Compare
   stats output format and values against a run from before this PR.
   Expected: identical.

3. **All existing GTest targets** must pass unchanged:
   - `test-tokenizer`
   - `test-loader`
   - `test-memorypool`
   - `test-storage`
   - `test-tensor`

4. **All example binaries** must compile and run:
   - `bench` (uses `profile()` / `warmup()` only)
   - `chat` (uses `session->chat()`)
   - `ping` (uses `session->chat()`)

---

## Benchmark Plan

Run config B (Qwen2-1.5B, prefill=128, decode=128, GPU) before and after.
Expected: **no measurable change**. This PR is pure API reshaping with no
compute or allocation changes.

---

## Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|-----------|
| External code calls `engine->last_stats()` | Low | Build break | Grep for `last_stats` across entire codebase before removing. Only internal usage found in `engine.cpp:134`. |
| `GenerationResult` move semantics suboptimal | Very Low | Minor perf | RVO/NRVO applies. `std::vector` has move constructor. Verified: single return path in each branch. |
| `InferenceRequest` defined but unused generates compiler warnings | Low | Noise | All fields are initialized with defaults. No `-Wunused` for types. |
| `generate()` return type change proposal creep | Medium | Scope creep | Explicitly out of scope. `generate()` still returns `std::string`. |

---

## Rollback

Re-add `last_stats_` member and `last_stats()` accessor to `engine.hpp`.
Change `generate_tokens()` return type back to `std::vector<int>`.
Re-add `update_stats_*` helpers. All changes are localized to 3 files
(engine.hpp, engine.cpp, request.hpp).

---

## Relationship to Subsequent PRs

| PR | How this PR helps |
|----|------------------|
| **PR-5 (Chat Template)** | Clean separation: session owns conversation state, engine is stateless. Template extraction doesn't need to worry about engine-level state. |
| **PR-6 (Scheduler)** | `InferenceRequest` and `GenerationResult` are the types the scheduler will produce and consume. `generate_tokens()` returning `GenerationResult` maps directly to `Scheduler::run_one()` return type. |
| **PR-7 (Continuous Batching)** | Stateless engine means the batched executor can call `generate_tokens()` (or its batched successor) without worrying about shared mutable state across concurrent requests. |
| **PR-10 (HTTP API)** | `GenerationResult.stats` maps directly to the `usage` field in the OpenAI-compatible response (`docs/design/http_api_design.md`). |

---

## Checklist for Reviewer

- [ ] `last_stats_` no longer exists in `engine.hpp`
- [ ] `generate_tokens()` returns `GenerationResult`
- [ ] No mutable state remains in `InferenceEngine` (check: all members are either
      `const`, `shared_ptr` to immutable objects, or the immutable `exec_config_`)
- [ ] `bench`, `chat`, `ping` compile and produce correct output
- [ ] Stats output from `ping` matches pre-PR output
- [ ] `InferenceRequest` type is defined but not yet consumed by engine/session
- [ ] No new `.cpp` files (header-only addition)
- [ ] No performance regression on config B
