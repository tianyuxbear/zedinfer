# HTTP API Design

## Constraints

- No Python runtime in the serving path (per CLAUDE.md)
- Lightweight C++ HTTP library
- OpenAI-compatible endpoints for ecosystem compatibility
- Server-Sent Events (SSE) for streaming
- Connection to scheduler for request routing

## Library Choice

**cpp-httplib** (header-only, MIT license):
- Single header file, zero build complexity
- Supports HTTP/1.1, SSE via chunked transfer encoding
- Thread-per-connection model (sufficient for moderate concurrency)
- Well-maintained, widely used

Add to `third_party/include/httplib.h` or via xmake package.

## Endpoints

### POST /v1/chat/completions

OpenAI-compatible chat completion endpoint.

**Request**:
```json
{
    "model": "deepseek-r1-qwen3-8b",
    "messages": [
        {"role": "user", "content": "Hello, who are you?"}
    ],
    "max_tokens": 512,
    "temperature": 1.0,
    "top_p": 1.0,
    "top_k": 0,
    "stream": false,
    "session_id": "optional-session-id-for-multi-turn"
}
```

**Response (non-streaming)**:
```json
{
    "id": "chatcmpl-abc123",
    "object": "chat.completion",
    "created": 1711234567,
    "model": "deepseek-r1-qwen3-8b",
    "choices": [{
        "index": 0,
        "message": {"role": "assistant", "content": "I am DeepSeek-R1..."},
        "finish_reason": "stop"
    }],
    "usage": {
        "prompt_tokens": 5,
        "completion_tokens": 42,
        "total_tokens": 47
    }
}
```

**Response (streaming, SSE)**:
```
data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"I"},"finish_reason":null}]}

data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":" am"},"finish_reason":null}]}

...

data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":42,"total_tokens":47}}

data: [DONE]
```

### GET /v1/models

List available models.

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

### GET /health

Health check endpoint.

```json
{
    "status": "ok",
    "active_requests": 3,
    "pending_requests": 12,
    "gpu_memory_used_mb": 8192,
    "kv_block_utilization": 0.45
}
```

## Server Architecture

```cpp
// include/zedinfer/http_server.hpp

struct ServerConfig {
    std::string host = "0.0.0.0";
    int port = 8080;
    int max_connections = 128;
    int request_timeout_ms = 300000;  // 5 minutes
};

class HttpServer {
public:
    HttpServer(ServerConfig config,
               std::shared_ptr<Scheduler> scheduler,
               std::shared_ptr<InferenceEngine> engine);

    void start();  // blocking
    void stop();

private:
    ServerConfig config_;
    std::shared_ptr<Scheduler> scheduler_;
    std::shared_ptr<InferenceEngine> engine_;
    std::unique_ptr<httplib::Server> server_;

    // Handlers
    void handle_chat_completions(const httplib::Request &req, httplib::Response &res);
    void handle_chat_completions_stream(const httplib::Request &req, httplib::Response &res);
    void handle_models(const httplib::Request &req, httplib::Response &res);
    void handle_health(const httplib::Request &req, httplib::Response &res);

    // Request conversion
    std::unique_ptr<InferenceRequest> parse_chat_request(const nlohmann::json &body);
    nlohmann::json format_response(const GenerationResult &result, const std::string &request_id);
    std::string format_stream_chunk(const std::string &request_id, const std::string &token, bool is_done);
};
```

## Streaming Implementation

For SSE streaming, the handler uses cpp-httplib's content provider:

```cpp
void HttpServer::handle_chat_completions_stream(
    const httplib::Request &req, httplib::Response &res) {

    auto body = nlohmann::json::parse(req.body);
    auto inference_req = parse_chat_request(body);

    // Set up streaming callback
    std::string request_id = generate_request_id();
    auto token_queue = std::make_shared<ThreadSafeQueue<std::string>>();

    inference_req->stream_callback = [token_queue](const std::string &token) {
        token_queue->push(token);
    };

    // Submit to scheduler
    auto future = inference_req->result_promise.get_future();
    scheduler_->submit(std::move(inference_req));

    // Stream response via SSE
    res.set_header("Content-Type", "text/event-stream");
    res.set_header("Cache-Control", "no-cache");
    res.set_header("Connection", "keep-alive");

    res.set_content_provider(
        "text/event-stream",
        [token_queue, request_id, &future](size_t offset, httplib::DataSink &sink) {
            std::string token;
            while (token_queue->try_pop(token, std::chrono::milliseconds(100))) {
                std::string chunk = format_stream_chunk(request_id, token, false);
                sink.write(chunk.data(), chunk.size());
            }
            if (future.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
                auto result = future.get();
                std::string done_chunk = format_stream_chunk(request_id, "", true);
                sink.write(done_chunk.data(), done_chunk.size());
                sink.done();
                return false;  // done
            }
            return true;  // continue
        });
}
```

## Request Flow

```
HTTP Request
    |
    v
HttpServer::handle_chat_completions()
    |-- parse JSON body
    |-- tokenize prompt (engine->tokenizer_->encode())
    |-- create InferenceRequest
    |-- scheduler_->submit(request)
    |
    v
Scheduler picks up request
    |-- admission check
    |-- schedule into batch
    |-- model->forward_batch()
    |-- sample tokens
    |-- stream_callback(token_text)  // for streaming
    |-- complete when EOS / max_tokens
    |
    v
result_promise.set_value(result)
    |
    v
HttpServer formats JSON response / SSE [DONE]
```

## Error Handling

| Error | HTTP Status | Response |
|-------|------------|----------|
| Invalid JSON body | 400 | `{"error": {"message": "...", "type": "invalid_request_error"}}` |
| Queue full | 503 | `{"error": {"message": "Server overloaded", "type": "server_error"}}` |
| Request timeout | 408 | `{"error": {"message": "Request timed out", "type": "timeout_error"}}` |
| Internal error | 500 | `{"error": {"message": "...", "type": "internal_error"}}` |

## Example Entry Point

```cpp
// examples/serve.cpp
int main(int argc, char *argv[]) {
    auto model_path = parse_args(argc, argv);
    auto device = device::Device::cuda(0);
    auto engine = InferenceEngine::create(model_path, device);

    SchedulerConfig sched_config;
    auto scheduler = std::make_shared<Scheduler>(sched_config, engine->block_allocator());

    ServerConfig server_config;
    server_config.port = 8080;

    HttpServer server(server_config, scheduler, engine);
    server.start();  // blocking
}
```

## Files to Create

| File | Purpose |
|------|---------|
| `include/zedinfer/http_server.hpp` | Server class and config |
| `src/zedinfer/http_server.cpp` | Endpoint handlers |
| `include/zedinfer/api_types.hpp` | OpenAI-compatible request/response types |
| `examples/serve.cpp` | HTTP serving entry point |
| `third_party/include/httplib.h` | cpp-httplib header (or xmake dependency) |
