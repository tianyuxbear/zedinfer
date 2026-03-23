# HTTP API Design

> Updated to reflect the implemented state (PR-10). Supersedes the original pre-implementation design.

## Overview

ZedInfer provides an OpenAI-compatible HTTP API with SSE streaming and an embedded web chat UI. The server uses **stateful sessions** — each conversation maintains server-side KV cache across turns, so only new tokens are prefilled (not the full history). A stateless fallback is available when no `session_id` is provided.

## Architecture

```
HTTP Clients (curl, OpenAI SDK, Web UI)
        |
        v
┌──────────────────────────┐
│  HttpServer               │  cpp-httplib thread pool
│  POST /v1/chat/completions│
│  GET  /v1/models          │
│  GET  /health             │
│  DELETE /v1/sessions/:id  │
│  GET  / (Web UI)          │
└───────────┬──────────────┘
            │ submit_async(request)    ← thread-safe
            v
┌──────────────────────────┐
│  ServingLoop              │  engine thread
│  run_serving()            │  wakes on condition_variable
│  schedule() → step()     │
└───────────┬──────────────┘
            │ result_promise / stream_callback
            v
    HTTP handler returns response / SSE stream
```

**Two threads:**
- **HTTP thread pool** (cpp-httplib): handles HTTP I/O, JSON parsing, SSE streaming
- **Engine thread** (`run_serving()`): runs inference (schedule → forward → process_results)

Communication: `submit_async()` (thread-safe, returns `std::future`) and `stream_callback` (pushes tokens to a `TokenQueue`).

## Library

**cpp-httplib v0.38.0** (header-only, MIT license), vendored at `third_party/cpp-httplib-0.38.0/httplib.h`. Same choice as llama.cpp.

## Endpoints

### POST /v1/chat/completions

OpenAI-compatible chat completion endpoint.

**Request:**
```json
{
    "model": "deepseek-r1-qwen3-8b",
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, who are you?"}
    ],
    "max_tokens": 1024,
    "stream": false,
    "session_id": "optional-session-id-for-multi-turn"
}
```

**Behavior with `session_id`:**
- First request with a new `session_id`: creates an `InferenceSession` with KV cache blocks
- Subsequent requests with same `session_id`: only the latest user message is prefilled (KV cache reused from previous turns)
- If the server session expired (idle timeout): auto-fallback to full messages prefill

**Behavior without `session_id`:**
- Stateless mode: full messages array formatted and prefilled every time

**Response (non-streaming):**
```json
{
    "id": "chatcmpl-0",
    "object": "chat.completion",
    "created": 1711234567,
    "model": "DeepSeek-R1-Distill-Qwen-1.5B",
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

**Response (streaming, SSE):**
```
data: {"id":"chatcmpl-0","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}

data: {"id":"chatcmpl-0","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}

...

data: {"id":"chatcmpl-0","object":"chat.completion.chunk","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{...}}

data: [DONE]
```

### GET /v1/models

```json
{
    "object": "list",
    "data": [{
        "id": "DeepSeek-R1-Distill-Qwen-1.5B",
        "object": "model",
        "created": 1711234567
    }]
}
```

### GET /health

```json
{
    "status": "ok",
    "model": "DeepSeek-R1-Distill-Qwen-1.5B",
    "active_requests": 1,
    "pending_requests": 0,
    "active_sessions": 3,
    "block_pool": {
        "total_blocks": 50000,
        "free_blocks": 48000,
        "utilization": 0.04
    }
}
```

### DELETE /v1/sessions/:id

Immediately frees KV cache blocks for a session. Called by the web UI on "New Chat".

```json
{"deleted": true}
```

### GET /

Serves the embedded web chat UI from `web/index.html`.

### GET /images/*

Serves static files from `web/images/` (icons, logos).

## Session Management

```
InferenceSession (server-side, per session_id)
  ├── block_table_     KV cache blocks (persistent across turns)
  ├── past_len_        tokens already in KV cache
  ├── is_first_turn_   BOS token handling
  ├── chat_history_    conversation record
  └── template_        chat formatting template

SessionEntry (in HttpServer)
  ├── session          unique_ptr<InferenceSession>
  ├── last_access      timestamp for idle timeout
  └── busy             atomic<bool> for concurrent access protection
```

**Lifecycle:**
- Created lazily on first request with a `session_id`
- Idle timeout: 30 minutes (configurable via `ServerConfig::session_idle_timeout`)
- Max sessions: 100 (configurable, LRU eviction when at capacity)
- Explicit deletion: `DELETE /v1/sessions/:id`
- Concurrent access: per-session `busy` flag, returns HTTP 409 if busy

**Session expiry fallback:**
- When a session is accessed after expiry (recreated fresh), the server detects `past_len == 0`
- Falls back to full messages prefill (same as stateless mode)
- Logged as "fresh start, full prefill"

## Streaming Implementation

SSE streaming uses a thread-safe `TokenQueue` to bridge the engine thread (producer) and HTTP thread (consumer):

```
Engine thread                    HTTP thread (content provider)
    │                                      │
    ├─ stream_callback(token) ──→ TokenQueue.push(token)
    │                                      │
    │                              TokenQueue.try_pop(50ms)
    │                                      │
    │                              UTF-8 validation ──→ SSE chunk
    │                                      │
    ├─ complete_request() ──────→ future ready
    │                                      │
    │                              drain queue ──→ final chunk + [DONE]
```

**UTF-8 safety:** Tokens from BPE tokenizers may contain partial multi-byte characters. A forward-scanning `valid_utf8_length()` function buffers incomplete sequences until the next token completes them.

**output_prefix:** For reasoning models (DeepSeek-R1), `<think> ` is sent as a separate SSE content delta before model tokens. Not stored in session history (consistent with CLI behavior).

## Error Handling

| Error | HTTP Status | Error Type | Code |
|-------|------------|------------|------|
| Invalid JSON body | 400 | `invalid_request_error` | `invalid_json` |
| Missing messages | 400 | `invalid_request_error` | `missing_field` |
| Prompt too long | 400 | `invalid_request_error` | `prompt_too_long` |
| Session busy | 409 | `conflict_error` | `session_busy` |
| Request timeout | 408 | `timeout_error` | `request_timeout` |
| Queue full | 503 | `server_error` | `queue_full` |
| Generation failed | 500 | `internal_error` | `generation_failed` |

All errors use OpenAI-compatible format: `{"error": {"message": "...", "type": "...", "code": "..."}}`.

**Error recovery:** If a forward pass fails mid-generation, `InferenceSession::abort_turn()` syncs `past_len_` with the actual `block_table_.seq_len` so subsequent turns don't have state gaps.

## Request Cancellation

When a streaming client disconnects:
1. cpp-httplib's `on_close` callback fires → sets `cancelled` flag on the request
2. `Scheduler::process_results()` checks `cancelled` each decode step → completes request early
3. Session state is recovered via `abort_turn()`
4. KV cache blocks are freed normally

The web UI uses `AbortController` to abort the fetch on "Stop" button click, which triggers the same server-side cancellation.

## Web UI

Single-page chat application at `web/index.html`, served as static file.

**Features:**
- Clean light theme (ChatGPT-style), custom SVG icons
- SSE streaming via `fetch()` + `ReadableStream`
- Client-side `localStorage` persistence (survives page refresh)
- Conversation history sidebar (switch, rename via double-click, delete)
- Stop generation button (AbortController → server-side cancellation)
- Adjustable `max_tokens` in settings panel
- Markdown rendering: headers, tables, links, code blocks with copy button
- Model name from `GET /v1/models`

**Session management:**
- `sessionId` generated on page load, stored in localStorage
- "New Chat" saves current conversation to history, creates new sessionId, calls `DELETE /v1/sessions/:id`
- Page refresh restores conversation from localStorage (same sessionId → server session may still be alive)
- Switching to old conversation: if server session alive → continues. If expired → next message triggers full prefill fallback.

## Entry Point

```
examples/serve.cpp
  --host            (default: 0.0.0.0)
  --port            (default: 8080)
  --nvidia          (GPU backend)
  --gpu-memory-utilization (default: 0.9)
  --max-batch-tokens      (default: 2048)
  --max-batch-requests    (default: 64)
```

Signal handling: SIGINT/SIGTERM → `server.stop()` + `engine->stop_serving()` → graceful shutdown.

## Files

| File | Purpose |
|------|---------|
| `include/zedinfer/http_server.hpp` | `ServerConfig`, `HttpServer`, `SessionLock` |
| `src/zedinfer/http_server.cpp` | All endpoint handlers, session management, SSE streaming |
| `examples/serve.cpp` | HTTP serving entry point with CLI args |
| `web/index.html` | Single-page chat UI |
| `web/images/*.svg` | favicon, logo, user/bot avatars |
| `third_party/cpp-httplib-0.38.0/httplib.h` | HTTP library (vendored) |

## Known Limitations

| Limitation | Reason | Future Direction |
|-----------|--------|-----------------|
| No temperature/sampling params | Sampler is per-engine (ArgmaxSampler), not per-request. Requires scheduler architecture change. | Per-request `GeneralSampler` in scheduler |
| No prefix caching | Stateful sessions don't share KV blocks across sessions | Block reference counting + hash table |
| No function calling | API format only, model capability dependent | Add tool/function_call message types |
| No CORS | Same-origin web UI only | Add `--cors` flag |
| Static file cache not hot-reloadable | Files cached at startup | Development convenience, not production issue |
