# PR10: HTTP API Server — Implementation Plan

> **HISTORICAL DOCUMENT.** This was the step-by-step implementation plan used during development.
> The implementation evolved beyond this plan (added stateful sessions, error recovery, cancellation, web UI persistence).
> For the current design, see `docs/design/http_api_design.md`.

**Goal:** Add an OpenAI-compatible HTTP server with SSE streaming and embedded web chat UI to zedinfer inference engine.

**Architecture:** cpp-httplib (header-only) HTTP thread pool communicates with the engine thread via the existing `ServingLoop::submit_async()` / `std::future` bridge. SSE streaming uses a thread-safe `TokenQueue` to relay tokens from engine callbacks to HTTP content providers. A single-page web UI (HTML+CSS+JS) is served at `GET /`.

**Tech Stack:** C++17, cpp-httplib (yhirose/cpp-httplib, MIT), nlohmann/json (already present), argparse (already present).

**Spec:** `docs/plan/PR10_http_api_plan.md`

---

## File Structure

### New Files

| File | Responsibility |
|------|---------------|
| `third_party/include/httplib.h` | cpp-httplib header-only HTTP library (vendored) |
| `include/zedinfer/http_server.hpp` | `ServerConfig` struct, `HttpServer` class declaration |
| `src/zedinfer/http_server.cpp` | HTTP endpoint handlers, JSON formatting, `TokenQueue`, SSE streaming |
| `examples/serve.cpp` | HTTP serving entry point with CLI args and signal handling |
| `web/index.html` | Single-page chat UI (HTML + embedded CSS + JS) |

### Modified Files

| File | What Changes |
|------|-------------|
| `include/zedinfer/chat_template.hpp` | Add `system_prefix`, `system_suffix` fields; add `apply()` method |
| `src/zedinfer/chat_template.cpp` | Implement `apply()`; populate `system_*` in default factories |
| `include/zedinfer/serving_loop.hpp` | Add inline `pending_count()`, `active_count()` |
| `include/zedinfer/engine.hpp` | Add `model_name_` member + accessor; add `pending_count()`, `active_count()` delegation; add `create()` overload with `SchedulerConfig` |
| `src/zedinfer/engine.cpp` | Store model name from path; implement `create()` overload |
| `xmake/examples.lua` | Add `serve` target |

---

## Task 1: Vendor cpp-httplib

**Files:**
- Create: `third_party/include/httplib.h`

- [ ] **Step 1:** Download cpp-httplib v0.18.3 (or latest stable) single header from https://github.com/yhirose/cpp-httplib/releases

```bash
cd /home/scratch.tianyux_coreai/workspace/zedinfer
curl -L -o third_party/include/httplib.h \
  https://github.com/yhirose/cpp-httplib/releases/download/v0.18.3/httplib.h
```

- [ ] **Step 2:** Verify the download is valid (should be ~800KB single header)

```bash
wc -l third_party/include/httplib.h
# Expected: ~10000-12000 lines
head -5 third_party/include/httplib.h
# Expected: comment header with "httplib.h" and version
```

- [ ] **Step 3:** Create a minimal compile test — add a temporary include in any existing `.cpp` to verify it compiles with our flags

```bash
# Quick compile check — just verify the header parses
echo '#include "httplib.h"' > /tmp/httplib_test.cpp
echo 'int main() { httplib::Server s; return 0; }' >> /tmp/httplib_test.cpp
g++ -std=c++17 -I third_party/include -c /tmp/httplib_test.cpp -o /dev/null
```

If compiler warnings flood from the third-party header, we will wrap the include in our `http_server.hpp` with pragma pushes:
```cpp
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#include "httplib.h"
#pragma GCC diagnostic pop
```

- [ ] **Step 4:** Run full build to verify no regressions

```bash
bash auto-build-test/scripts/run_build_check.sh
```

- [ ] **Step 5:** Commit

```bash
git add third_party/include/httplib.h
git commit -m "chore: vendor cpp-httplib v0.18.3 (header-only HTTP library)"
```

---

## Task 2: Extend ChatTemplate with `apply()` and system role support

**Files:**
- Modify: `include/zedinfer/chat_template.hpp`
- Modify: `src/zedinfer/chat_template.cpp`

- [ ] **Step 1:** Add new fields and method declaration to `include/zedinfer/chat_template.hpp`

Add after the existing `output_prefix` field:

```cpp
    // System message formatting (empty = plain text fallback)
    std::string system_prefix;
    std::string system_suffix;
```

Add method declaration after `default_qwen_chatml()`:

```cpp
    // Format an OpenAI-style messages array into a prompt string.
    // Each pair is (role, content) where role is "system", "user", or "assistant".
    // If add_generation_prompt is true, appends generation_prompt at the end.
    std::string apply(
        const std::vector<std::pair<std::string, std::string>> &messages,
        bool add_generation_prompt = true) const;
```

Add the necessary include at the top:
```cpp
#include <utility>
#include <vector>
```

- [ ] **Step 2:** Populate `system_prefix`/`system_suffix` in default factories in `src/zedinfer/chat_template.cpp`

In `default_deepseek_r1()` — add before `return t;`:
```cpp
    t.system_prefix        = "";
    t.system_suffix        = "";
```

In `default_qwen_chatml()` — add before `return t;`:
```cpp
    t.system_prefix        = "<|im_start|>system\n";
    t.system_suffix        = "<|im_end|>\n";
```

Also add the fields to the JSON override loader (inside the `try` block in `load()`):
```cpp
    t.system_prefix = j.value("system_prefix", "");
    t.system_suffix = j.value("system_suffix", "");
```

- [ ] **Step 3:** Implement `ChatTemplate::apply()` in `src/zedinfer/chat_template.cpp`

Add at the end of the file, before the closing `} // namespace zedinfer`:

```cpp
std::string ChatTemplate::apply(
    const std::vector<std::pair<std::string, std::string>> &messages,
    bool add_generation_prompt) const {

    std::string result;

    // BOS token: always prepend once. add_bos_first_turn_only is irrelevant here
    // because apply() formats a complete conversation in one call (always "first turn").
    // The flag only matters for incremental per-turn formatting (InferenceSession::chat).
    if (!bos_token.empty()) {
        result += bos_token;
    }

    for (const auto &[role, content] : messages) {
        if (role == "system") {
            if (!system_prefix.empty()) {
                result += system_prefix + content + system_suffix;
            } else {
                result += content + "\n";
            }
        } else if (role == "user") {
            result += user_prefix + content + user_suffix;
        } else if (role == "assistant") {
            result += assistant_prefix + content + assistant_suffix;
        }
    }

    if (add_generation_prompt && !generation_prompt.empty()) {
        result += generation_prompt;
    }

    return result;
}
```

- [ ] **Step 4:** Build to verify

```bash
bash auto-build-test/scripts/run_build_check.sh
```

- [ ] **Step 5:** Verify existing examples still work (ChatTemplate fields default to empty, backward compatible)

```bash
# Existing chat/ping should still compile and run since new fields default to ""
xmake build ping chat
```

- [ ] **Step 6:** Commit

```bash
git add include/zedinfer/chat_template.hpp src/zedinfer/chat_template.cpp
git commit -m "feat(chat-template): add apply() for OpenAI messages formatting and system role support"
```

---

## Task 3: Extend Engine and ServingLoop accessors

**Files:**
- Modify: `include/zedinfer/serving_loop.hpp`
- Modify: `include/zedinfer/engine.hpp`
- Modify: `src/zedinfer/engine.cpp`

- [ ] **Step 1:** Add delegation methods and `SchedulerConfig` support to `ServingLoop`

In `include/zedinfer/serving_loop.hpp`, change the constructor to accept config:

```cpp
    explicit ServingLoop(std::shared_ptr<InferenceEngine> engine,
                         SchedulerConfig sched_config = {});
```

Add in the public section after `void stop();`:

```cpp
    // Scheduler status (for health endpoint)
    int pending_count() const { return scheduler_.pending_count(); }
    int active_count() const { return scheduler_.active_count(); }
```

In `src/zedinfer/serving_loop.cpp`, update the constructor to pass config to scheduler:

```cpp
ServingLoop::ServingLoop(std::shared_ptr<InferenceEngine> engine,
                         SchedulerConfig sched_config)
    : engine_(std::move(engine)),
      scheduler_(std::move(sched_config)) {
    if (engine_->block_allocator()) {
        scheduler_.set_block_allocator(engine_->block_allocator());
    }
}
```

- [ ] **Step 2:** Add `model_name_` and delegation methods to `InferenceEngine`

In `include/zedinfer/engine.hpp`, add to public accessors (after `block_allocator()` accessor):

```cpp
    const std::string &model_name() const { return model_name_; }
    int pending_count() const { return serving_loop_ ? serving_loop_->pending_count() : 0; }
    int active_count() const { return serving_loop_ ? serving_loop_->active_count() : 0; }
```

Add `model_name_` to private members (after `scheduler_config_`):

```cpp
    std::string model_name_;
```

Add `create()` overload declaration (alongside existing `create()`):

```cpp
    static std::shared_ptr<InferenceEngine> create(
        const std::string &model_path, device::Device device,
        const SchedulerConfig &sched_config);
```

- [ ] **Step 3:** Implement changes in `src/zedinfer/engine.cpp`

Add at the top of existing `create()`, after the opening brace, add model name derivation:

```cpp
    // Derive model name from directory basename
    std::string model_name = model_path;
    auto last_slash = model_name.find_last_of("/\\");
    if (last_slash != std::string::npos) {
        model_name = model_name.substr(last_slash + 1);
    }
```

Store the model name after constructing the engine (after `build_stop_token_ids()`):

```cpp
    engine->model_name_ = model_name;
```

Modify existing `create()` to store `scheduler_config_` before `init_block_pool()` and pass it to `ServingLoop`. The cleanest approach: change the existing `create()` to accept an optional `SchedulerConfig` with a default value.

In `engine.hpp`, change the declaration:

```cpp
    static std::shared_ptr<InferenceEngine> create(
        const std::string &model_path, device::Device device,
        SchedulerConfig sched_config = {});
```

In `engine.cpp`, update `create()` signature and wire the config:

```cpp
std::shared_ptr<InferenceEngine> InferenceEngine::create(
    const std::string &model_path,
    device::Device device,
    SchedulerConfig sched_config) {

    // ... existing code ...

    // Store scheduler config BEFORE init_block_pool (which reads gpu_memory_utilization)
    engine->scheduler_config_ = sched_config;

    // ... existing build_stop_token_ids(), profiler warmup ...

    engine->init_block_pool();

    // Pass scheduler config to ServingLoop so the Scheduler uses it
    engine->serving_loop_ = std::make_unique<ServingLoop>(engine, engine->scheduler_config_);

    return engine;
}
```

This is a signature change, not an overload. Existing 2-arg callers still compile because the third arg has a default value.

- [ ] **Step 4:** Build to verify — existing callers must still compile

```bash
bash auto-build-test/scripts/run_build_check.sh
```

- [ ] **Step 5:** Commit

```bash
git add include/zedinfer/serving_loop.hpp include/zedinfer/engine.hpp src/zedinfer/engine.cpp
git commit -m "feat(engine): add model_name, pending/active count accessors, SchedulerConfig overload"
```

---

## Task 4: Create HttpServer header

**Files:**
- Create: `include/zedinfer/http_server.hpp`

- [ ] **Step 1:** Create `include/zedinfer/http_server.hpp`

```cpp
#pragma once

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#include "httplib.h"
#pragma GCC diagnostic pop

#include <atomic>
#include <memory>
#include <string>

namespace zedinfer {

class InferenceEngine;

struct ServerConfig {
    std::string host = "0.0.0.0";
    int port = 8080;
    int request_timeout_sec = 300;
};

class HttpServer {
public:
    HttpServer(ServerConfig config, std::shared_ptr<InferenceEngine> engine);

    void start();  // blocking
    void stop();   // non-blocking, safe from signal handler

private:
    ServerConfig config_;
    std::shared_ptr<InferenceEngine> engine_;
    httplib::Server server_;
    std::atomic<uint64_t> request_counter_{0};
    std::string web_ui_html_;  // cached web UI content

    // Route handlers
    void handle_chat_completions(const httplib::Request &req, httplib::Response &res);
    void handle_models(const httplib::Request &req, httplib::Response &res);
    void handle_health(const httplib::Request &req, httplib::Response &res);

    // Helpers
    std::string generate_request_id();
    void send_error(httplib::Response &res, int status,
                    const std::string &message, const std::string &type,
                    const std::string &code = "");
    void load_web_ui();
};

} // namespace zedinfer
```

- [ ] **Step 2:** Build to verify (header compiles when included by other zedinfer sources via wildcard glob)

```bash
bash auto-build-test/scripts/run_build_check.sh
```

- [ ] **Step 3:** Commit

```bash
git add include/zedinfer/http_server.hpp
git commit -m "feat(http): add HttpServer header with ServerConfig and class declaration"
```

---

## Task 5: Implement HttpServer — non-streaming endpoints

**Files:**
- Create: `src/zedinfer/http_server.cpp`

This is the largest task. It implements the constructor (route setup), all non-streaming handlers, and helpers.

- [ ] **Step 1:** Create `src/zedinfer/http_server.cpp` with includes and namespace

```cpp
#include "zedinfer/http_server.hpp"
#include "zedinfer/engine.hpp"
#include "zedinfer/request.hpp"

#include <nlohmann/json.hpp>
#include <plog/Log.h>

#include <chrono>
#include <condition_variable>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <queue>
#include <sstream>

using json = nlohmann::json;
namespace fs = std::filesystem;

namespace zedinfer {
```

- [ ] **Step 2:** Implement helpers: `generate_request_id()`, `send_error()`, `load_web_ui()`

```cpp
std::string HttpServer::generate_request_id() {
    uint64_t id = request_counter_.fetch_add(1, std::memory_order_relaxed);
    return "chatcmpl-" + std::to_string(id);
}

void HttpServer::send_error(httplib::Response &res, int status,
                            const std::string &message, const std::string &type,
                            const std::string &code) {
    json err;
    err["error"]["message"] = message;
    err["error"]["type"] = type;
    err["error"]["code"] = code;
    res.status = status;
    res.set_content(err.dump(), "application/json");
}

void HttpServer::load_web_ui() {
    // Try paths relative to executable, then working directory
    std::vector<std::string> search_paths = {
        "web/index.html",
        "../web/index.html",
    };

    for (const auto &path : search_paths) {
        if (fs::exists(path)) {
            std::ifstream file(path);
            if (file.is_open()) {
                std::ostringstream ss;
                ss << file.rdbuf();
                web_ui_html_ = ss.str();
                LOGI << "[HttpServer] Loaded web UI from " << path;
                return;
            }
        }
    }
    web_ui_html_ = "<html><body><h1>ZedInfer</h1><p>Web UI not found. "
                    "Place web/index.html in the project directory.</p></body></html>";
    LOGW << "[HttpServer] Web UI not found, using fallback";
}
```

- [ ] **Step 3:** Implement constructor with route setup

```cpp
HttpServer::HttpServer(ServerConfig config, std::shared_ptr<InferenceEngine> engine)
    : config_(std::move(config)), engine_(std::move(engine)) {

    load_web_ui();

    server_.set_payload_max_length(10 * 1024 * 1024);  // 10MB max request body

    server_.Post("/v1/chat/completions",
        [this](const httplib::Request &req, httplib::Response &res) {
            handle_chat_completions(req, res);
        });

    server_.Get("/v1/models",
        [this](const httplib::Request &req, httplib::Response &res) {
            handle_models(req, res);
        });

    server_.Get("/health",
        [this](const httplib::Request &req, httplib::Response &res) {
            handle_health(req, res);
        });

    server_.Get("/",
        [this](const httplib::Request &, httplib::Response &res) {
            res.set_content(web_ui_html_, "text/html");
        });
}
```

- [ ] **Step 4:** Implement `handle_models()` and `handle_health()`

```cpp
void HttpServer::handle_models(const httplib::Request &, httplib::Response &res) {
    auto now = std::chrono::system_clock::now();
    auto epoch = std::chrono::duration_cast<std::chrono::seconds>(
        now.time_since_epoch()).count();

    json response;
    response["object"] = "list";
    response["data"] = json::array({
        {{"id", engine_->model_name()},
         {"object", "model"},
         {"created", epoch}}
    });
    res.set_content(response.dump(), "application/json");
}

void HttpServer::handle_health(const httplib::Request &, httplib::Response &res) {
    json response;
    response["status"] = "ok";
    response["model"] = engine_->model_name();
    response["active_requests"] = engine_->active_count();
    response["pending_requests"] = engine_->pending_count();

    auto *pool = engine_->block_pool();
    if (pool) {
        response["block_pool"]["total_blocks"] = pool->total_blocks();
        response["block_pool"]["free_blocks"] = pool->free_blocks();
        response["block_pool"]["utilization"] =
            1.0 - static_cast<double>(pool->free_blocks()) / pool->total_blocks();
    }
    res.set_content(response.dump(), "application/json");
}
```

- [ ] **Step 5:** Implement `handle_chat_completions()` — non-streaming path

```cpp
void HttpServer::handle_chat_completions(const httplib::Request &req,
                                          httplib::Response &res) {
    // 1. Parse JSON
    json body;
    try {
        body = json::parse(req.body);
    } catch (const json::parse_error &e) {
        send_error(res, 400, std::string("Invalid JSON: ") + e.what(),
                   "invalid_request_error");
        return;
    }

    // 2. Validate messages
    if (!body.contains("messages") || !body["messages"].is_array() ||
        body["messages"].empty()) {
        send_error(res, 400, "Missing or empty 'messages' array",
                   "invalid_request_error");
        return;
    }

    // 3. Extract parameters
    bool stream = body.value("stream", false);
    int max_tokens = body.value("max_tokens", 512);

    // 4. Format messages to prompt
    std::vector<std::pair<std::string, std::string>> messages;
    for (const auto &msg : body["messages"]) {
        std::string role = msg.value("role", "");
        std::string content = msg.value("content", "");
        messages.emplace_back(role, content);
    }
    std::string prompt = engine_->chat_template().apply(messages);

    // 5. Tokenize
    auto input_ids = engine_->tokenizer().encode(prompt);

    // 6. Branch: streaming vs non-streaming
    if (stream) {
        handle_chat_completions_stream(req, res, input_ids, max_tokens);
        return;
    }

    // --- Non-streaming path ---
    std::string request_id = generate_request_id();

    auto inference_req = std::make_unique<InferenceRequest>();
    inference_req->input_ids = std::move(input_ids);
    inference_req->config.max_new_tokens = max_tokens;
    inference_req->arrival_time = std::chrono::steady_clock::now();

    // 7. Submit
    std::future<GenerationResult> future;
    try {
        future = engine_->submit_async(std::move(inference_req));
    } catch (const std::runtime_error &) {
        send_error(res, 503, "Server overloaded, queue full", "server_error");
        return;
    }

    // 8. Wait for result with timeout
    if (future.wait_for(std::chrono::seconds(config_.request_timeout_sec)) ==
        std::future_status::timeout) {
        send_error(res, 408, "Request timed out", "timeout_error", "request_timeout");
        return;
    }
    auto result = future.get();

    // 9. Decode output
    std::string output_text = engine_->tokenizer().decode(result.output_ids);

    // 10. Format response
    auto now = std::chrono::system_clock::now();
    auto epoch = std::chrono::duration_cast<std::chrono::seconds>(
        now.time_since_epoch()).count();

    json response;
    response["id"] = request_id;
    response["object"] = "chat.completion";
    response["created"] = epoch;
    response["model"] = engine_->model_name();
    response["choices"] = json::array({
        {{"index", 0},
         {"message", {{"role", "assistant"}, {"content", output_text}}},
         {"finish_reason", "stop"}}
    });
    response["usage"] = {
        {"prompt_tokens", result.stats.prompt_tokens},
        {"completion_tokens", result.stats.generated_tokens},
        {"total_tokens", result.stats.total_tokens}
    };

    res.set_content(response.dump(), "application/json");
}
```

Note: This references `handle_chat_completions_stream()` which is implemented in Task 6. For now, add a stub forward declaration at the top of the file:

```cpp
// Forward declare — implemented in streaming section
void handle_chat_completions_stream(const httplib::Request &req, httplib::Response &res,
                                     std::vector<int> input_ids, int max_tokens);
```

Actually, make it a private method declaration in the header instead. Add to `include/zedinfer/http_server.hpp` in the private section:

```cpp
    void handle_chat_completions_stream(const httplib::Request &req, httplib::Response &res,
                                         std::vector<int> input_ids, int max_tokens);
```

- [ ] **Step 6:** Implement `start()` and `stop()`

```cpp
void HttpServer::start() {
    server_.listen(config_.host, config_.port);
}

void HttpServer::stop() {
    server_.stop();
}

} // namespace zedinfer
```

- [ ] **Step 7:** Build to verify (streaming stub can be empty for now)

Add a temporary stub for the streaming method:
```cpp
void HttpServer::handle_chat_completions_stream(
    const httplib::Request &, httplib::Response &res,
    std::vector<int>, int) {
    send_error(res, 501, "Streaming not yet implemented", "server_error");
}
```

```bash
bash auto-build-test/scripts/run_build_check.sh
```

- [ ] **Step 8:** Commit

```bash
git add include/zedinfer/http_server.hpp src/zedinfer/http_server.cpp
git commit -m "feat(http): implement non-streaming chat completions, models, and health endpoints"
```

---

## Task 6: Implement SSE streaming

**Files:**
- Modify: `src/zedinfer/http_server.cpp`

- [ ] **Step 1:** Add `TokenQueue` class at the top of `http_server.cpp` (inside anonymous namespace)

```cpp
namespace {

template <typename T>
class TokenQueue {
public:
    void push(T item) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            queue_.push(std::move(item));
        }
        cv_.notify_one();
    }

    bool try_pop(T &item, std::chrono::milliseconds timeout) {
        std::unique_lock<std::mutex> lock(mutex_);
        if (!cv_.wait_for(lock, timeout, [this] { return !queue_.empty(); })) {
            return false;
        }
        item = std::move(queue_.front());
        queue_.pop();
        return true;
    }

private:
    std::queue<T> queue_;
    std::mutex mutex_;
    std::condition_variable cv_;
};

} // anonymous namespace
```

- [ ] **Step 2:** Replace the streaming stub with the full implementation

```cpp
void HttpServer::handle_chat_completions_stream(
    const httplib::Request &, httplib::Response &res,
    std::vector<int> input_ids, int max_tokens) {

    std::string request_id = generate_request_id();
    std::string model_name = engine_->model_name();

    auto now = std::chrono::system_clock::now();
    auto epoch = std::chrono::duration_cast<std::chrono::seconds>(
        now.time_since_epoch()).count();

    auto token_queue = std::make_shared<TokenQueue<std::string>>();

    // Build request with stream callback
    auto inference_req = std::make_unique<InferenceRequest>();
    inference_req->input_ids = std::move(input_ids);
    inference_req->config.max_new_tokens = max_tokens;
    // IMPORTANT: config.stream MUST be true — the scheduler checks
    // (config.stream && stream_callback) at scheduler.cpp:193,221.
    // This is safe because submit_async() does not call validate(),
    // so the validate() check for stream==true && !config.stream_callback is never hit.
    inference_req->config.stream = true;
    inference_req->arrival_time = std::chrono::steady_clock::now();
    inference_req->stream_callback = [token_queue](const std::string &token) {
        token_queue->push(token);
    };

    // Submit
    std::future<GenerationResult> future;
    try {
        future = engine_->submit_async(std::move(inference_req));
    } catch (const std::runtime_error &) {
        send_error(res, 503, "Server overloaded, queue full", "server_error");
        return;
    }

    // Set SSE headers and stream via chunked content provider
    res.set_header("Cache-Control", "no-cache");
    res.set_header("Connection", "keep-alive");

    // Shared state for the content provider lambda
    auto state = std::make_shared<struct StreamState>();
    state->future = std::move(future);
    state->token_queue = token_queue;
    state->request_id = request_id;
    state->model_name = model_name;
    state->epoch = epoch;
    state->sent_role = false;
    state->done = false;

    res.set_chunked_content_provider(
        "text/event-stream",
        [state](size_t /*offset*/, httplib::DataSink &sink) -> bool {

            // Send initial role chunk once
            if (!state->sent_role) {
                state->sent_role = true;
                json chunk;
                chunk["id"] = state->request_id;
                chunk["object"] = "chat.completion.chunk";
                chunk["created"] = state->epoch;
                chunk["model"] = state->model_name;
                chunk["choices"] = json::array({
                    {{"index", 0},
                     {"delta", {{"role", "assistant"}}},
                     {"finish_reason", nullptr}}
                });
                std::string data = "data: " + chunk.dump() + "\n\n";
                sink.write(data.data(), data.size());
            }

            // Try to pop tokens from queue
            std::string token;
            while (state->token_queue->try_pop(token, std::chrono::milliseconds(50))) {
                json chunk;
                chunk["id"] = state->request_id;
                chunk["object"] = "chat.completion.chunk";
                chunk["created"] = state->epoch;
                chunk["model"] = state->model_name;
                chunk["choices"] = json::array({
                    {{"index", 0},
                     {"delta", {{"content", token}}},
                     {"finish_reason", nullptr}}
                });
                std::string data = "data: " + chunk.dump() + "\n\n";
                sink.write(data.data(), data.size());
            }

            // Check if generation is complete
            if (state->future.wait_for(std::chrono::milliseconds(0)) ==
                std::future_status::ready) {

                // Drain any remaining tokens
                while (state->token_queue->try_pop(token, std::chrono::milliseconds(0))) {
                    json chunk;
                    chunk["id"] = state->request_id;
                    chunk["object"] = "chat.completion.chunk";
                    chunk["created"] = state->epoch;
                    chunk["model"] = state->model_name;
                    chunk["choices"] = json::array({
                        {{"index", 0},
                         {"delta", {{"content", token}}},
                         {"finish_reason", nullptr}}
                    });
                    std::string data = "data: " + chunk.dump() + "\n\n";
                    sink.write(data.data(), data.size());
                }

                // Send final chunk with finish_reason and usage
                auto result = state->future.get();

                json final_chunk;
                final_chunk["id"] = state->request_id;
                final_chunk["object"] = "chat.completion.chunk";
                final_chunk["created"] = state->epoch;
                final_chunk["model"] = state->model_name;
                final_chunk["choices"] = json::array({
                    {{"index", 0},
                     {"delta", json::object()},
                     {"finish_reason", "stop"}}
                });
                final_chunk["usage"] = {
                    {"prompt_tokens", result.stats.prompt_tokens},
                    {"completion_tokens", result.stats.generated_tokens},
                    {"total_tokens", result.stats.total_tokens}
                };
                std::string data = "data: " + final_chunk.dump() + "\n\n";
                sink.write(data.data(), data.size());

                // Send [DONE]
                std::string done = "data: [DONE]\n\n";
                sink.write(done.data(), done.size());

                sink.done();
                return false;  // signal completion
            }

            return true;  // continue streaming
        });
}
```

- [ ] **Step 3:** Ensure `StreamState` struct is defined BEFORE `handle_chat_completions_stream` in the file (inside the anonymous namespace, after `TokenQueue`)

```cpp
struct StreamState {
    std::future<GenerationResult> future;
    std::shared_ptr<TokenQueue<std::string>> token_queue;
    std::string request_id;
    std::string model_name;
    int64_t epoch;
    bool sent_role;
};
```

**Important:** This struct must appear above `handle_chat_completions_stream()` in the file, since it is referenced by `std::make_shared<StreamState>()`. Place it right after the `TokenQueue` template class in the anonymous namespace.

- [ ] **Step 4:** Build to verify

```bash
bash auto-build-test/scripts/run_build_check.sh
```

- [ ] **Step 5:** Commit

```bash
git add src/zedinfer/http_server.cpp
git commit -m "feat(http): implement SSE streaming for chat completions"
```

---

## Task 7: Create serve.cpp entry point and build target

**Files:**
- Create: `examples/serve.cpp`
- Modify: `xmake/examples.lua`

- [ ] **Step 1:** Add `serve` target to `xmake/examples.lua`

Append before the final line:

```lua
target("serve")
    set_kind("binary")
    add_deps("zedinfer")
    add_files("../examples/serve.cpp")
    set_rundir("$(projectdir)")
    on_install(function (target) end)
target_end()
```

- [ ] **Step 2:** Create the entry point

```cpp
#include "backend/device/device.hpp"
#include "utils/logging.hpp"
#include "utils/system_info.hpp"
#include "zedinfer.h"
#include "zedinfer/engine.hpp"
#include "zedinfer/http_server.hpp"
#include "zedinfer/scheduler.hpp"

#include <argparse/argparse.hpp>
#include <csignal>
#include <iostream>
#include <memory>
#include <string>
#include <thread>

using namespace zedinfer;

// Global pointers for signal handler
static HttpServer *g_server = nullptr;
static std::shared_ptr<InferenceEngine> g_engine = nullptr;

static void signal_handler(int) {
    if (g_server) g_server->stop();
    if (g_engine) g_engine->stop_serving();
}

int main(int argc, char *argv[]) {
    utils::initLoggerWithOverwrite(plog::info, "logs/serve.log");
    LOG_VERBOSE_(utils::BOTH) << utils::get_runtime_info();

    argparse::ArgumentParser program("ZedInfer Server");

    program.add_argument("model_path")
        .help("Path to the model directory");

    program.add_argument("--host")
        .help("Host to bind to")
        .default_value(std::string("0.0.0.0"));

    program.add_argument("--port")
        .help("Port to listen on")
        .default_value(8080)
        .scan<'i', int>();

    program.add_argument("--nvidia")
        .help("Use NVIDIA GPU backend")
        .default_value(false)
        .implicit_value(true);

    program.add_argument("--max-batch-tokens")
        .help("Maximum tokens per batch")
        .default_value(2048)
        .scan<'i', int>();

    program.add_argument("--max-batch-requests")
        .help("Maximum concurrent requests")
        .default_value(64)
        .scan<'i', int>();

    program.add_argument("--gpu-memory-utilization")
        .help("Fraction of GPU memory for KV cache (0.0-1.0)")
        .default_value(0.9f)
        .scan<'g', float>();

    try {
        program.parse_args(argc, argv);
    } catch (const std::exception &err) {
        std::cerr << err.what() << std::endl;
        std::cerr << program;
        return 1;
    }

    auto model_path = program.get<std::string>("model_path");
    auto host = program.get<std::string>("--host");
    int port = program.get<int>("--port");
    bool use_nvidia = program.get<bool>("--nvidia");

    zedinferDeviceType_t device_type =
        use_nvidia ? ZEDINFER_DEVICE_NVIDIA : ZEDINFER_DEVICE_CPU;
    device::Device device(device_type, 0);

    // Build scheduler config from CLI args
    SchedulerConfig sched_config;
    sched_config.max_batch_tokens = program.get<int>("--max-batch-tokens");
    sched_config.max_batch_requests = program.get<int>("--max-batch-requests");
    sched_config.gpu_memory_utilization = program.get<float>("--gpu-memory-utilization");

    // Create engine
    auto engine = InferenceEngine::create(model_path, device, sched_config);
    g_engine = engine;

    // Start engine serving loop on a dedicated thread
    std::thread engine_thread([&engine] {
        engine->run_serving();
    });

    // Create HTTP server
    ServerConfig server_config;
    server_config.host = host;
    server_config.port = port;

    HttpServer server(server_config, engine);
    g_server = &server;

    // Register signal handlers
    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);

    printf("\n========================================\n");
    printf("  ZedInfer Server\n");
    printf("  Model: %s\n", engine->model_name().c_str());
    printf("  Listening: http://%s:%d\n", host.c_str(), port);
    printf("  Web UI: http://%s:%d/\n", host.c_str(), port);
    printf("  API: http://%s:%d/v1/chat/completions\n", host.c_str(), port);
    printf("========================================\n\n");

    server.start();  // blocks until stop()

    // Cleanup
    engine->stop_serving();
    engine_thread.join();
    g_server = nullptr;
    g_engine = nullptr;

    printf("Server stopped.\n");
    return 0;
}
```

- [ ] **Step 3:** Build to verify

```bash
bash auto-build-test/scripts/run_build_check.sh
```

- [ ] **Step 4:** Verify all existing targets still build

```bash
xmake build ping chat bench batch_bench serve
```

- [ ] **Step 5:** Commit

```bash
git add examples/serve.cpp xmake/examples.lua
git commit -m "feat(serve): add HTTP serving entry point with CLI args and signal handling"
```

---

## Task 8: Create Web UI

**Files:**
- Create: `web/index.html`

- [ ] **Step 1:** Create `web/index.html`

This is a self-contained single-page chat application. Full HTML with embedded CSS and JS. Key features:
- Dark theme, modern chat layout
- User messages right-aligned (blue), assistant messages left-aligned (gray)
- Streaming via `fetch()` + `ReadableStream` (not EventSource, for better control)
- Client-side conversation `messages` array — sent in full with each request
- Auto-scroll, Enter-to-send, Shift+Enter for newline
- "New Chat" button to clear history
- Model name fetched from `GET /v1/models`
- Loading spinner during generation
- Basic markdown rendering: code blocks (``` ... ```) rendered as `<pre><code>`

The file will be ~300-500 lines of HTML/CSS/JS. Create it as a complete, working file.

Key JS architecture:
```javascript
let messages = [];  // client-side conversation history

async function sendMessage() {
    const content = input.value.trim();
    if (!content) return;

    messages.push({role: "user", content});
    renderMessage("user", content);
    input.value = "";

    // Create assistant placeholder
    const assistantDiv = renderMessage("assistant", "");

    // Stream response
    const response = await fetch("/v1/chat/completions", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({messages, stream: true, max_tokens: 2048})
    });

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let assistantContent = "";

    while (true) {
        const {done, value} = await reader.read();
        if (done) break;

        const text = decoder.decode(value);
        // Parse SSE lines: "data: {...}\n\n"
        for (const line of text.split("\n")) {
            if (!line.startsWith("data: ")) continue;
            const data = line.slice(6);
            if (data === "[DONE]") break;
            const chunk = JSON.parse(data);
            const delta = chunk.choices?.[0]?.delta?.content;
            if (delta) {
                assistantContent += delta;
                assistantDiv.innerHTML = renderMarkdown(assistantContent);
                scrollToBottom();
            }
        }
    }

    messages.push({role: "assistant", content: assistantContent});
}
```

- [ ] **Step 2:** Build (no build needed for HTML, but verify serve can load it)

```bash
xmake build serve
```

- [ ] **Step 3:** Commit

```bash
git add web/index.html
git commit -m "feat(web): add embedded chat UI with streaming support"
```

---

## Task 9: Integration testing and verification

- [ ] **Step 1:** Run full build

```bash
bash auto-build-test/scripts/run_build_check.sh
```

- [ ] **Step 2:** Verify all existing targets still work

```bash
xmake build ping chat bench batch_bench serve
```

- [ ] **Step 3:** Start the server (requires model files — skip if not available)

```bash
xmake run serve /path/to/model --nvidia --port 8080
```

- [ ] **Step 4:** Test non-streaming endpoint

```bash
curl -s http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"What is 2+2?"}],"max_tokens":64}' | python3 -m json.tool
```

Expected: Valid JSON with `choices[0].message.content` containing the answer.

- [ ] **Step 5:** Test streaming endpoint

```bash
curl -N http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Hello"}],"max_tokens":64,"stream":true}'
```

Expected: SSE chunks (`data: {...}`) streaming in, ending with `data: [DONE]`.

- [ ] **Step 6:** Test utility endpoints

```bash
curl -s http://localhost:8080/v1/models | python3 -m json.tool
curl -s http://localhost:8080/health | python3 -m json.tool
```

- [ ] **Step 7:** Test web UI

Open `http://localhost:8080/` in a browser. Verify:
- Page loads with chat interface
- Can type and send a message
- Tokens stream in character-by-character
- Multi-turn works (send follow-up message)
- "New Chat" clears history

- [ ] **Step 8:** Test concurrent requests

```bash
# 4 concurrent requests
for i in 1 2 3 4; do
  curl -s http://localhost:8080/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d "{\"messages\":[{\"role\":\"user\",\"content\":\"Count to $i\"}],\"max_tokens\":32}" &
done
wait
```

Expected: All 4 complete independently with correct output.

- [ ] **Step 9:** Test error handling

```bash
# Invalid JSON
curl -s http://localhost:8080/v1/chat/completions -d "not json"
# Expected: 400

# Missing messages
curl -s http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" -d '{"max_tokens":10}'
# Expected: 400
```

- [ ] **Step 10:** Final commit if any fixes were needed

```bash
git add -A
git commit -m "fix(http): integration test fixes"
```
