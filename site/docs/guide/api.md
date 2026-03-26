# API Reference

ZedInfer exposes an OpenAI-compatible HTTP API.

---

## POST /v1/chat/completions

Chat completions with optional streaming and multi-turn sessions.

### Request

```json
{
    "messages": [
        {"role": "user", "content": "Hello"}
    ],
    "max_tokens": 512,
    "stream": false,
    "session_id": ""
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `messages` | array | -- | Chat messages (required). Uses the last `user` message for generation. |
| `max_tokens` | int | `512` | Maximum tokens to generate |
| `stream` | bool | `false` | Enable SSE streaming |
| `session_id` | string | `""` | Session ID for multi-turn conversation. Empty = stateless. |

### Response (non-streaming)

```json
{
    "id": "chatcmpl-0",
    "object": "chat.completion",
    "model": "DeepSeek-R1-Distill-Qwen-1.5B",
    "choices": [{
        "index": 0,
        "message": {"role": "assistant", "content": "Hello! How can I assist you?"},
        "finish_reason": "stop"
    }],
    "usage": {"prompt_tokens": 7, "completion_tokens": 12, "total_tokens": 19}
}
```

### Response (streaming)

Server-Sent Events stream. Each event contains a partial response:

```
data: {"choices":[{"delta":{"content":"Hello"},"index":0}]}

data: {"choices":[{"delta":{"content":"!"},"index":0}]}

data: [DONE]
```

### Examples

```bash
# Basic request
curl http://localhost:8080/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"messages":[{"role":"user","content":"Hello"}]}'

# Streaming
curl http://localhost:8080/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"messages":[{"role":"user","content":"Hello"}],"stream":true}'

# Multi-turn session
curl http://localhost:8080/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"messages":[{"role":"user","content":"Hello"}],"session_id":"abc123"}'
```

---

## GET /v1/models

List loaded models.

```bash
curl http://localhost:8080/v1/models
```

```json
{
    "object": "list",
    "data": [{"id": "DeepSeek-R1-Distill-Qwen-1.5B", "object": "model", "created": 1774496680}]
}
```

---

## GET /health

Health check endpoint. Returns model status, request counts, and KV cache utilization.

```bash
curl http://localhost:8080/health
```

```json
{
    "status": "ok",
    "model": "DeepSeek-R1-Distill-Qwen-1.5B",
    "active_requests": 0,
    "pending_requests": 0,
    "active_sessions": 0,
    "block_pool": {
        "total_blocks": 20398686,
        "free_blocks": 20398686,
        "utilization": 0.0
    }
}
```

---

## DELETE /v1/sessions/:id

Delete a stateful session and release its KV cache blocks.

```bash
curl -X DELETE http://localhost:8080/v1/sessions/abc123
```

```json
{"deleted": true}
```
