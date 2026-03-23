# PR-10: HTTP / OpenAPI Server — Detailed Implementation Plan

> **HISTORICAL DOCUMENT.** This was the pre-implementation spec written before coding began.
> The actual implementation evolved significantly — notably switching from **stateless** to **stateful sessions** with KV cache reuse.
> For the current design, see `docs/design/http_api_design.md`.

**Goal:** Add an OpenAI-compatible HTTP server with SSE streaming and an embedded web chat UI. Wire HTTP handlers to the existing `ServingLoop` infrastructure for multi-user concurrent inference.

**Architecture:** cpp-httplib (header-only, MIT) runs an HTTP thread pool. Handlers parse OpenAI-format JSON, build `InferenceRequest` objects, and submit them via `ServingLoop::submit_async()`. The engine thread runs `ServingLoop::run_serving()`, waking on new submissions. Results flow back through `std::future` (non-streaming) or `stream_callback` → thread-safe queue → SSE (streaming). A bundled single-page web UI connects to the same API.

**Tech Stack:** C++17, cpp-httplib (yhirose/cpp-httplib), nlohmann/json (already present), argparse (already present). Same library choices as llama.cpp.

---

## 1. Current State Analysis

### 1.1 ServingLoop (Ready for HTTP)

`ServingLoop` (`include/zedinfer/serving_loop.hpp`, `src/zedinfer/serving_loop.cpp`) already provides:

| Method | Purpose | Thread Safety |
|--------|---------|---------------|
| `submit_async(request)` | Submit request, return `std::future<GenerationResult>` | Thread-safe (mutex + cv notify) |
| `run_serving()` | Engine loop: sleep when idle, wake on submit, drain work | Engine thread only |
| `stop()` | Signal graceful shutdown | Any thread |
| `step()` | One schedule+forward+process_results iteration | Engine thread only |

The HTTP server only needs to call `submit_async()` from handler threads and `run_serving()` from the engine thread.

**Required additions to ServingLoop/Engine:** The `/health` endpoint needs `pending_count()` and `active_count()` from the `Scheduler`, which is private to `ServingLoop`. Add thin delegation methods to `ServingLoop` and `InferenceEngine`.

### 1.2 InferenceRequest (Ready)

`InferenceRequest` (`include/zedinfer/request.hpp:39-71`) has all fields needed:

- `input_ids` — tokenized prompt (HTTP handler tokenizes the formatted messages)
- `config` — `GenerationConfig` with `max_new_tokens`, `stream`
- `stream_callback` — called from engine thread on each generated token
- `result_promise` — fulfilled when generation completes

### 1.3 ChatTemplate (Needs Extension)

`ChatTemplate` (`include/zedinfer/chat_template.hpp`) currently formats one user turn at a time in `InferenceSession::chat()` (`session.cpp:63-89`). The HTTP handler receives a full `messages` array and needs to format the entire conversation into a single prompt string.

**Required addition:** `ChatTemplate::apply(messages)` method.

### 1.4 Dependencies Already Present

- `nlohmann/json` — `third_party/include/nlohmann/json.hpp` (used by model config, tokenizer, chat template)
- `argparse` — `third_party/include/argparse/argparse.hpp` (used by batch_bench, bench)
- Build pattern — `xmake/examples.lua` shows how to add new binary targets

### 1.5 Missing Dependency

- `cpp-httplib` — must be vendored to `third_party/include/httplib.h`

### 1.6 Known Caveats

**`GenerationConfig::validate()` and streaming:** `validate()` (`generation_types.cpp:18`) throws if `stream == true && !stream_callback`. The HTTP handler must set `req->config.stream = true` (because the scheduler checks `config.stream && req->stream_callback` at `scheduler.cpp:193,221`). This is safe because HTTP requests go through `submit_async()` directly, which does NOT call `validate()`. The `validate()` check is only hit in the `ServingLoop::generate()` path (used by CLI chat/ping). Same pattern as `batch_bench.cpp`.

**`Scheduler::submit()` throws on queue full:** `submit()` (`scheduler.cpp:29`) throws `std::runtime_error` when the queue is full. Since `submit_async()` calls `submit()` directly, this exception propagates to the HTTP handler thread. All handler code must wrap `submit_async()` in try/catch and return HTTP 503 on queue-full.

**`SchedulerConfig` not configurable from outside:** `InferenceEngine::create()` uses a default `SchedulerConfig`. There is no public API to pass custom config. PR10 must extend `create()` to accept an optional `SchedulerConfig` parameter for CLI-configurable batch/memory settings.

**No model name accessor:** `InferenceEngine` exposes `model().model_type()` (returns "qwen2"/"qwen3") but not a human-friendly model name. PR10 will derive the model name from the model directory basename, or accept `--model-name` CLI flag. Need to store and expose `model_name_` on the engine.

---

## 2. Design

### 2.1 Threading Model

```
                    ┌──────────────────────────┐
                    │  main thread              │
                    │  1. create engine          │
                    │  2. create HttpServer      │
                    │  3. start engine thread    │
                    │  4. server.start() [block] │
                    └──────────────────────────┘
                              │
            ┌─────────────────┼─────────────────┐
            v                                   v
  ┌──────────────────┐              ┌──────────────────┐
  │  HTTP thread pool │              │  engine thread    │
  │  (cpp-httplib)    │              │  run_serving()    │
  │                   │   submit_    │                   │
  │  handler() ──────────async()──> │  schedule()       │
  │                   │              │  step()           │
  │  <──── future ──────────────── │  process_results() │
  │  <──── callback ───────────── │  stream_callback()  │
  └──────────────────┘              └──────────────────┘
```

- **Main thread**: creates engine, starts engine thread, then blocks on `httplib::Server::listen()`.
- **HTTP thread pool**: cpp-httplib manages a thread pool (default: hardware concurrency). Handlers run on these threads.
- **Engine thread**: runs `ServingLoop::run_serving()`. Single thread processes all inference.

Signal handling: SIGINT → `server.stop()` + `engine->stop_serving()`.

### 2.2 HttpServer Class

```cpp
// include/zedinfer/http_server.hpp

struct ServerConfig {
    std::string host = "0.0.0.0";
    int port = 8080;
    int request_timeout_sec = 300;  // 5 minutes
};

class HttpServer {
public:
    HttpServer(ServerConfig config, std::shared_ptr<InferenceEngine> engine);

    void start();  // blocking — calls httplib::Server::listen()
    void stop();   // non-blocking — calls httplib::Server::stop()

private:
    ServerConfig config_;
    std::shared_ptr<InferenceEngine> engine_;
    httplib::Server server_;

    // Route handlers
    void handle_chat_completions(const httplib::Request &req, httplib::Response &res);
    void handle_models(const httplib::Request &req, httplib::Response &res);
    void handle_health(const httplib::Request &req, httplib::Response &res);
    void handle_index(const httplib::Request &req, httplib::Response &res);

    // Helpers
    std::string generate_request_id();
    void send_error(httplib::Response &res, int status, const std::string &message,
                    const std::string &type);
};
```

### 2.3 API Endpoints

#### POST /v1/chat/completions

**Request:**
```json
{
    "model": "deepseek-r1-qwen3-8b",
    "messages": [
        {"role": "user", "content": "Hello"}
    ],
    "max_tokens": 512,
    "stream": false
}
```

**Non-streaming response:**
```json
{
    "id": "chatcmpl-abc123",
    "object": "chat.completion",
    "created": 1711234567,
    "model": "deepseek-r1-qwen3-8b",
    "choices": [{
        "index": 0,
        "message": {"role": "assistant", "content": "..."},
        "finish_reason": "stop"
    }],
    "usage": {
        "prompt_tokens": 5,
        "completion_tokens": 42,
        "total_tokens": 47
    }
}
```

**Streaming response (SSE):**
```
data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","created":1711234567,"model":"...","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}

data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","created":1711234567,"model":"...","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}

...

data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","created":1711234567,"model":"...","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":42,"total_tokens":47}}

data: [DONE]
```

#### GET /v1/models

```json
{
    "object": "list",
    "data": [{
        "id": "deepseek-r1-qwen3-8b",
        "object": "model",
        "created": 1711234567
    }]
}
```

#### GET /health

```json
{
    "status": "ok",
    "model": "deepseek-r1-qwen3-8b",
    "active_requests": 3,
    "pending_requests": 12,
    "block_pool": {
        "total_blocks": 10000,
        "free_blocks": 8500,
        "utilization": 0.15
    }
}
```

#### GET /

Serves the embedded web UI (`web/index.html`).

### 2.4 Non-Streaming Handler Flow

```
handle_chat_completions(req, res):
    1. Parse JSON body → on failure: 400 invalid_json
    2. Validate required fields (messages array) → on failure: 400 missing_field
    3. Format messages → prompt string via ChatTemplate::apply()
    4. Tokenize prompt via engine->tokenizer().encode()
    5. Build InferenceRequest:
       - input_ids = tokenized prompt
       - config.max_new_tokens = body["max_tokens"] or default
       - config.stream = false
       - (do NOT set config.stream=true here — request goes via submit_async, not generate)
    6. try { future = engine->submit_async(request) }
       catch (runtime_error) → 503 queue_full
    7. result = future.get()  // blocks until complete
    8. Decode output: engine->tokenizer().decode(result.output_ids)
    9. Format OpenAI response JSON
   10. res.set_content(json, "application/json")
```

### 2.5 Streaming Handler Flow

For SSE streaming, we use cpp-httplib's chunked content provider. The key challenge is bridging the engine thread (which calls `stream_callback`) with the HTTP thread (which writes SSE chunks).

**Thread-safe token queue:**

```cpp
// In http_server.cpp (internal)
template<typename T>
class TokenQueue {
public:
    void push(T item);
    bool try_pop(T &item, std::chrono::milliseconds timeout);
    void close();  // signals no more items
    bool is_closed() const;
private:
    std::queue<T> queue_;
    std::mutex mutex_;
    std::condition_variable cv_;
    bool closed_ = false;
};
```

**Streaming handler:**

```
handle_chat_completions(req, res):  [stream=true]
    1. Parse JSON, validate, format prompt, tokenize (same as non-streaming)
    2. Create shared TokenQueue<std::string>
    3. Build InferenceRequest:
       - req->config.stream = true  (required: scheduler checks config.stream && stream_callback
         at scheduler.cpp:193,221. Safe because submit_async() does not call validate())
       - req->stream_callback = [queue](token) { queue->push(token); }
    4. try { future = engine->submit_async(request) }
       catch (runtime_error) → 503 queue_full
    5. Set SSE headers:
       - Content-Type: text/event-stream
       - Cache-Control: no-cache
       - Connection: keep-alive
    6. Use res.set_chunked_content_provider():
       Provider lambda (called by cpp-httplib on HTTP thread):
         a. Send initial chunk: {"delta":{"role":"assistant"}}
         b. Loop: try_pop from queue with timeout
            - If token received: send SSE data chunk
            - If timeout with no token: check if future is ready
              - If not ready: continue (keep-alive)
              - If ready: drain remaining tokens from queue, then send final chunk
         c. Drain phase: while (queue->try_pop(token, 0ms)) { send_chunk(token); }
         d. Send usage stats + [DONE]
         e. Return false to signal done
```

**Important:** The scheduler calls `stream_callback` for each non-stop token (`scheduler.cpp:193-194`), then calls `complete_request()` which sets the promise *without* calling `stream_callback` for the stop token. This means when the future becomes ready, there may be tokens still in the queue from the last batch. The drain phase (step 6c) ensures all tokens are sent before `[DONE]`.

The `stream_callback` is called from the engine thread inside `Scheduler::process_results()`. The `TokenQueue` safely transfers tokens to the HTTP content provider thread.

**`TokenQueue::close()` usage:** Not needed in normal flow (future signals completion). Reserved for client-disconnect cleanup: if the content provider detects a broken pipe and returns false, the queue is abandoned. The engine callback may push into a dead queue — this is harmless (push succeeds, nobody reads).

### 2.6 ChatTemplate Extension

Add a method to format a full OpenAI-style `messages` array into a prompt string:

```cpp
// In chat_template.hpp
struct ChatTemplate {
    // ... existing fields ...

    // New fields for system message formatting
    std::string system_prefix;
    std::string system_suffix;

    // Format an OpenAI-style messages array into a prompt string.
    // messages: vector of {role, content} pairs
    // add_generation_prompt: if true, append generation_prompt at the end
    std::string apply(
        const std::vector<std::pair<std::string, std::string>> &messages,
        bool add_generation_prompt = true) const;
};
```

The `system_prefix`/`system_suffix` fields must be populated in the default factories:

- **DeepSeek-R1:** No standard system format. Use empty prefix/suffix (system content prepended as plain text before first user turn).
- **Qwen ChatML:** `system_prefix = "<|im_start|>system\n"`, `system_suffix = "<|im_end|>\n"`.

Implementation in `chat_template.cpp`:

```
apply(messages, add_generation_prompt):
    result = bos_token

    for each (role, content) in messages:
        if role == "system":
            if system_prefix is not empty:
                result += system_prefix + content + system_suffix
            else:
                result += content + "\n"
        elif role == "user":
            result += user_prefix + content + user_suffix
        elif role == "assistant":
            result += assistant_prefix + content + assistant_suffix

    if add_generation_prompt:
        result += generation_prompt

    return result
```

This replaces the inline formatting in `InferenceSession::chat()` with a reusable method. The session's `chat()` can be refactored to use it as well, but that's optional for PR10.

### 2.7 Error Handling

All errors follow OpenAI format:

```json
{"error": {"message": "...", "type": "...", "code": "..."}}
```

| Condition | HTTP Status | Type | Code |
|-----------|------------|------|------|
| Malformed JSON body | 400 | `invalid_request_error` | `invalid_json` |
| Missing `messages` field | 400 | `invalid_request_error` | `missing_field` |
| Empty messages array | 400 | `invalid_request_error` | `invalid_messages` |
| Scheduler queue full | 503 | `server_error` | `queue_full` |
| Request timeout (future wait) | 408 | `timeout_error` | `request_timeout` |
| Internal exception | 500 | `internal_error` | `internal` |

### 2.8 Web UI

A single-page chat application served at `GET /`. The file lives at `web/index.html` in the source tree.

**Features:**
- Chat bubble layout (user messages right-aligned, assistant left-aligned)
- Streaming token display via `EventSource` API (SSE)
- Client-side conversation history (messages array) — sent in full with each request
- Auto-scroll to bottom on new tokens
- Input textarea with Enter to send (Shift+Enter for newline)
- Model name display from `GET /v1/models`
- New conversation button (clears local history)
- Clean dark theme, responsive layout
- Markdown rendering for code blocks (lightweight: `<pre><code>` detection, no external lib dependency)
- Loading indicator during generation

**Architecture:**
- Pure HTML + CSS + JS, no build tools, no framework
- All code in a single `index.html` file (embedded `<style>` and `<script>`)
- Connects to same-origin `/v1/chat/completions` with `stream: true`
- Uses `fetch()` with `ReadableStream` for SSE parsing (more control than `EventSource`)

**Serving:**
- `HttpServer` reads `web/index.html` at startup and serves it at `GET /`
- File path resolved relative to executable or configurable via `--static-dir`

### 2.9 serve.cpp Entry Point

```cpp
// Global pointers for signal handler (C++ signal() requires C-linkage function pointer,
// cannot capture locals — standard pattern used by llama.cpp and similar projects)
static HttpServer *g_server = nullptr;
static std::shared_ptr<InferenceEngine> g_engine = nullptr;

static void signal_handler(int) {
    if (g_server) g_server->stop();
    if (g_engine) g_engine->stop_serving();
}

int main(int argc, char *argv[]) {
    // Parse args
    argparse::ArgumentParser program("ZedInfer Server");
    program.add_argument("model_path").help("Path to model directory");
    program.add_argument("--host").default_value(std::string("0.0.0.0"));
    program.add_argument("--port").default_value(8080).scan<'i', int>();
    program.add_argument("--nvidia").default_value(false).implicit_value(true);
    program.add_argument("--max-batch-tokens").default_value(2048).scan<'i', int>();
    program.add_argument("--max-batch-requests").default_value(64).scan<'i', int>();
    program.add_argument("--gpu-memory-utilization").default_value(0.9f).scan<'g', float>();
    program.add_argument("--model-name").default_value(std::string(""));
    program.parse_args(argc, argv);

    // Build SchedulerConfig from CLI args
    SchedulerConfig sched_config;
    sched_config.max_batch_tokens = program.get<int>("--max-batch-tokens");
    sched_config.max_batch_requests = program.get<int>("--max-batch-requests");
    sched_config.gpu_memory_utilization = program.get<float>("--gpu-memory-utilization");

    // Create engine (with custom scheduler config)
    auto device = Device(use_nvidia ? NVIDIA : CPU, 0);
    auto engine = InferenceEngine::create(model_path, device, sched_config);
    g_engine = engine;

    // Start engine thread
    std::thread engine_thread([&engine] { engine->run_serving(); });

    // Create and start HTTP server
    ServerConfig config{host, port, 300};
    HttpServer server(config, engine);
    g_server = &server;

    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);

    printf("Server listening on http://%s:%d\n", host.c_str(), port);
    server.start();  // blocks until stop()

    // Cleanup
    engine->stop_serving();
    engine_thread.join();
    g_server = nullptr;
    g_engine = nullptr;
}
```

---

## 3. File Structure

### New Files

| File | Purpose |
|------|---------|
| `include/zedinfer/http_server.hpp` | `ServerConfig`, `HttpServer` class declaration |
| `src/zedinfer/http_server.cpp` | HTTP handlers, JSON parsing, SSE streaming, TokenQueue |
| `examples/serve.cpp` | HTTP serving entry point |
| `web/index.html` | Embedded web chat UI |
| `third_party/include/httplib.h` | cpp-httplib header (vendored, MIT license) |

### Modified Files

| File | What Changes |
|------|-------------|
| `include/zedinfer/chat_template.hpp` | Add `system_prefix`, `system_suffix` fields; add `apply(messages)` method |
| `src/zedinfer/chat_template.cpp` | Implement `apply(messages)`, populate system fields in default factories |
| `include/zedinfer/serving_loop.hpp` | Add `pending_count()`, `active_count()` delegation methods |
| `include/zedinfer/engine.hpp` | Add `pending_count()`, `active_count()` delegation; add `model_name()` accessor; add `model_name_` member; extend `create()` to accept optional `SchedulerConfig` |
| `src/zedinfer/engine.cpp` | Store model name (derived from path or config); accept `SchedulerConfig` in `create()` |
| `xmake/examples.lua` | Add `serve` target |

### Unchanged Files

| File | Why Unchanged |
|------|--------------|
| `src/zedinfer/serving_loop.cpp` | No logic changes (new methods are inline in header) |
| `include/zedinfer/scheduler.hpp` | Already has `pending_count()`, `active_count()` |
| `src/zedinfer/scheduler.cpp` | No changes |
| `include/zedinfer/request.hpp` | Already has stream_callback, result_promise |
| Examples: bench, chat, ping, batch_bench | Unaffected |

---

## 4. Interface Changes Summary

| Symbol | Before | After |
|--------|--------|-------|
| `HttpServer` | Does not exist | **New** class |
| `ServerConfig` | Does not exist | **New** struct |
| `ChatTemplate::apply()` | Does not exist | **New** method |
| `ChatTemplate::system_prefix/suffix` | Does not exist | **New** fields (empty default for backward compat) |
| `InferenceEngine::create()` | Takes `(model_path, device)` | **Add** optional `SchedulerConfig` overload |
| `InferenceEngine::model_name()` | Does not exist | **New** accessor |
| `InferenceEngine::pending_count()` | Does not exist | **New** delegation to scheduler |
| `InferenceEngine::active_count()` | Does not exist | **New** delegation to scheduler |
| `ServingLoop::pending_count()` | Does not exist | **New** delegation to scheduler |
| `ServingLoop::active_count()` | Does not exist | **New** delegation to scheduler |
| `serve` binary | Does not exist | **New** example target |

**Public API impact:** Additive only. New fields on `ChatTemplate` default to empty strings — existing code unaffected. `InferenceEngine::create()` gets a new overload; existing 2-arg call still works. `bench`, `chat`, `ping`, `batch_bench` are unaffected.

---

## 5. Implementation Tasks

### Task 1: Vendor cpp-httplib

**Files:**
- Create: `third_party/include/httplib.h`

- [ ] Download latest stable release of yhirose/cpp-httplib
- [ ] Place single header file in `third_party/include/httplib.h`
- [ ] Verify it compiles with C++17 and our warning flags (may need pragmas to suppress warnings in third-party header)
- [ ] Build to verify

---

### Task 2: Add `ChatTemplate::apply()` for messages formatting

**Files:**
- Modify: `include/zedinfer/chat_template.hpp`
- Modify: `src/zedinfer/chat_template.cpp`

- [ ] Add `system_prefix`, `system_suffix` fields to `ChatTemplate`
- [ ] Populate in `default_deepseek_r1()`: both empty (system as plain text)
- [ ] Populate in `default_qwen_chatml()`: `"<|im_start|>system\n"` / `"<|im_end|>\n"`
- [ ] Add `apply()` method declaration
- [ ] Implement: iterate messages, format with bos + role-specific prefix/suffix
- [ ] Handle `system` role with `system_prefix`/`system_suffix` (fall back to plain text if empty)
- [ ] Handle `add_bos_first_turn_only` correctly
- [ ] Append `generation_prompt` when `add_generation_prompt=true`
- [ ] Build to verify

---

### Task 3: Extend Engine and ServingLoop accessors

**Files:**
- Modify: `include/zedinfer/serving_loop.hpp`
- Modify: `include/zedinfer/engine.hpp`
- Modify: `src/zedinfer/engine.cpp`

- [ ] Add `pending_count()`, `active_count()` inline methods to `ServingLoop` (delegate to `scheduler_`)
- [ ] Add `pending_count()`, `active_count()` inline methods to `InferenceEngine` (delegate to `serving_loop_`)
- [ ] Add `model_name_` member to `InferenceEngine`; derive from model directory basename in `create()`
- [ ] Add `model_name()` accessor
- [ ] Add `create()` overload accepting `SchedulerConfig` parameter (or make it an optional arg with default)
- [ ] Build to verify — existing examples must still compile with the old 2-arg `create()` call

---

### Task 4: Create HttpServer header

**Files:**
- Create: `include/zedinfer/http_server.hpp`

- [ ] Define `ServerConfig` struct (host, port, timeout)
- [ ] Declare `HttpServer` class with constructor, `start()`, `stop()`
- [ ] Declare private handler methods
- [ ] Include necessary headers (forward-declare httplib types or include)
- [ ] Build to verify

---

### Task 5: Implement HttpServer — non-streaming endpoint

**Files:**
- Create: `src/zedinfer/http_server.cpp`

- [ ] Implement constructor: set up routes on `httplib::Server`
- [ ] Implement `handle_chat_completions()` non-streaming path:
    - Parse JSON body
    - Extract messages array
    - Call `ChatTemplate::apply()` to format prompt
    - Tokenize with `engine->tokenizer().encode()`
    - Build `InferenceRequest` (stream=false)
    - `submit_async()` → `future.get()`
    - Decode output tokens
    - Format OpenAI-compatible JSON response
- [ ] Implement `handle_models()`: return model info
- [ ] Implement `handle_health()`: return scheduler stats + block pool stats
- [ ] Implement `send_error()` helper
- [ ] Implement `generate_request_id()` helper
- [ ] Implement `start()` and `stop()`
- [ ] Build to verify

---

### Task 6: Implement SSE streaming

**Files:**
- Modify: `src/zedinfer/http_server.cpp`

- [ ] Implement `TokenQueue<std::string>` (thread-safe queue with close signal)
- [ ] Implement streaming path in `handle_chat_completions()`:
    - Create shared `TokenQueue`
    - Set `stream_callback` to push tokens into queue
    - `submit_async()` to get future
    - Use `res.set_chunked_content_provider()`:
      - Send initial SSE chunk (role delta)
      - Loop: pop tokens from queue, format as SSE data chunks
      - On completion: send final chunk with usage stats + `[DONE]`
- [ ] Handle edge cases: client disconnect, timeout
- [ ] Build to verify

---

### Task 7: Create serve.cpp entry point

**Files:**
- Create: `examples/serve.cpp`
- Modify: `xmake/examples.lua` — add `serve` target

- [ ] Parse CLI arguments (model_path, --host, --port, --nvidia, --gpu-memory-utilization, --max-batch-tokens)
- [ ] Initialize logger
- [ ] Create `InferenceEngine`
- [ ] Start engine thread (`run_serving()`)
- [ ] Create `HttpServer` with config
- [ ] Register SIGINT handler for graceful shutdown
- [ ] Call `server.start()` (blocking)
- [ ] On shutdown: `stop()` + join engine thread
- [ ] Add `serve` target to `xmake/examples.lua` (depends on `zedinfer`)
- [ ] Build to verify

---

### Task 8: Create Web UI

**Files:**
- Create: `web/index.html`
- Modify: `src/zedinfer/http_server.cpp` — add `GET /` handler

- [ ] Design chat layout: sidebar (optional) + main chat area + input box
- [ ] Implement message rendering: user bubbles (right), assistant bubbles (left)
- [ ] Implement SSE streaming via `fetch()` + `ReadableStream`:
    - Parse `data: {...}` lines
    - Extract `choices[0].delta.content`
    - Append to current assistant message
- [ ] Implement conversation management:
    - Client-side `messages` array
    - Each send appends user message, calls API with full array
    - On response: append assistant message to array
    - "New chat" button clears array
- [ ] Style: clean dark theme, responsive, modern look
- [ ] Add `GET /` route in HttpServer to serve `index.html`:
    - Load file content at startup (or embed as string constant)
    - Serve with `Content-Type: text/html`
- [ ] Build and test in browser

---

### Task 9: Build verification and functional testing

- [ ] Run `bash auto-build-test/scripts/run_build_check.sh` — verify clean build
- [ ] Verify `bench`, `chat`, `ping`, `batch_bench` all still compile and work
- [ ] Start `serve` with a model, test:
    - `curl POST /v1/chat/completions` (non-streaming) — verify correct JSON response
    - `curl POST /v1/chat/completions` with `stream: true` — verify SSE chunks + `[DONE]`
    - `curl GET /v1/models` — verify model listing
    - `curl GET /health` — verify health status
    - Open `http://localhost:8080/` in browser — verify web UI loads and works
    - Multi-turn conversation in web UI — verify history accumulates correctly
    - Multiple concurrent curl requests — verify all complete correctly
- [ ] If any failures, fix and retry

---

## 6. Correctness Validation

1. **Non-streaming output matches CLI:** Same prompt via HTTP and via `ping` should produce identical token output (same model, same greedy sampling).

2. **Streaming completeness:** Concatenated SSE token chunks must equal the non-streaming response content. No tokens lost or duplicated.

3. **Multi-user isolation:** 4 concurrent requests with different prompts must produce independent correct results (same as `batch_bench` validation).

4. **Chat template correctness:** `ChatTemplate::apply(messages)` output must produce the same token sequence as `InferenceSession::chat()` for equivalent conversation turns.

5. **Error handling:** Invalid requests return proper HTTP status codes and OpenAI-format error JSON.

---

## 7. Benchmark Plan

### HTTP Load Test

After implementation, measure serving performance:

```bash
# Single request latency
curl -w "@curl-format.txt" -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Hello"}],"max_tokens":64}'

# Concurrent load (requires 'hey' or 'wrk')
hey -n 100 -c 10 -m POST \
    -H "Content-Type: application/json" \
    -d '{"messages":[{"role":"user","content":"Hi"}],"max_tokens":64}' \
    http://localhost:8080/v1/chat/completions
```

**Metrics:** requests/sec, P50/P90/P99 latency, error rate.

**Expected:** HTTP overhead < 1ms per request. Throughput limited by GPU inference, not HTTP layer.

---

## 8. Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|-----------|
| cpp-httplib compiler warnings | Medium | Low | Wrap include with `#pragma` to suppress in third-party header |
| SSE streaming with content provider correctness | Medium | Medium | Test with curl `--no-buffer` and browser EventSource |
| Client disconnect during streaming | Medium | Low | cpp-httplib handles broken pipe; TokenQueue::close() prevents engine-side hang |
| Thread safety of submit_async | Low | High | Already mutex-protected. Verified in batch_bench. |
| ChatTemplate::apply() mismatch with session::chat() | Low | Medium | Test: same messages → same tokenized output |
| CORS issues with web UI | Low | Low | Same-origin (served from same server), no CORS needed |
| Large prompt tokenization blocking HTTP thread | Low | Medium | Tokenization is fast (~1ms for 2K tokens). Acceptable. |
| cpp-httplib thread pool exhaustion with many streaming connections | Low | Medium | Default pool = hardware concurrency. Long-lived SSE connections consume a thread each. Bounded by max_batch_requests (64). |
| No request cancellation on client disconnect | Medium | Low | Abandoned streaming requests continue consuming GPU until completion. Out of scope for PR10 — track as follow-up. Could add `std::atomic<bool> cancelled` to `InferenceRequest` checked in `process_results()`. |
| Large request body DoS | Low | Low | Use `server.set_payload_max_length()` (e.g., 10MB). Reject oversized requests with 413. |

---

## 9. Rollback

Delete new files (`http_server.hpp`, `http_server.cpp`, `serve.cpp`, `web/index.html`, `httplib.h`). Revert `xmake/examples.lua` and `chat_template.hpp/cpp`. No other code depends on the HTTP server.

---

## 10. Open Questions

1. **Should `serve` accept scheduler config via CLI args?**
   **Resolved: Yes.** Expose `--max-batch-tokens`, `--max-batch-requests`, `--gpu-memory-utilization`, `--model-name`. Add `SchedulerConfig` parameter to `InferenceEngine::create()`.

2. **Should the web UI be a separate file or embedded as a C++ string constant?**
   Recommendation: Separate file (`web/index.html`) loaded at startup. Easier to iterate on the UI without recompiling. Fall back to a minimal "no UI found" message if the file is missing. A compile-time embed option (via `xxd -i`) could be added later as a build flag for single-binary deployment.

3. **Should we support `system` role in messages?**
   **Resolved: Yes.** Add `system_prefix`/`system_suffix` to `ChatTemplate`. Qwen ChatML uses `<|im_start|>system\n...<|im_end|>\n`. DeepSeek-R1 uses plain text (empty prefix/suffix).

4. **Should we add CORS headers for cross-origin clients?**
   Recommendation: Not in PR10 (same-origin web UI). Add `--cors` flag later if needed for external frontends like Open WebUI.

5. **Should the `model` field in requests be validated?**
   Recommendation: Accept any value (or ignore it). We serve one model. Log a warning if it doesn't match. Don't reject — this is what llama.cpp does.

6. **Should abandoned streaming requests be cancelled?**
   Out of scope for PR10. Track as follow-up. Could add `std::atomic<bool> cancelled` flag on `InferenceRequest`, checked in `process_results()`.

7. **Should `usage` be included in streaming responses unconditionally?**
   Yes, include in final SSE chunk unconditionally (matches llama.cpp behavior). OpenAI only sends when `stream_options.include_usage=true`, but unconditional is simpler and informative.
