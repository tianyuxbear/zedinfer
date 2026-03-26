# Docker Deployment Guide

---

## 1. Host Requirements

| Requirement | Minimum | How to Check |
|-------------|---------|-------------|
| NVIDIA Driver | ≥ 570 (for CUDA 13.x) | `nvidia-smi` |
| Docker Engine | ≥ 20.10 | `docker --version` |
| NVIDIA Container Toolkit | Latest | `nvidia-ctk --version` |
| Disk space (image) | ~4 GB | — |
| Disk space (model) | Depends on model | — |

Host does NOT need CUDA Toolkit installed — it ships inside the image.

### Install NVIDIA Container Toolkit (if not present)

```bash
# Ubuntu/Debian
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
    sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

---

## 2. Build Image

### Basic build

```bash
docker build -t zedinfer:latest .
```

### Embed git hash

Pass the current commit hash so `--version` shows it instead of "unknown":

```bash
docker build --build-arg GIT_HASH=$(git rev-parse --short HEAD) -t zedinfer:latest .
```

### Version tagging

```bash
# Tag with semantic version
docker build -t zedinfer:0.1.0 .

# Also tag as latest
docker tag zedinfer:0.1.0 zedinfer:latest

# Tag with git commit hash for traceability
docker build -t zedinfer:$(git rev-parse --short HEAD) .
```

### Multi-tag in one step

```bash
VERSION=0.1.0
GIT_HASH=$(git rev-parse --short HEAD)
docker build \
    -t zedinfer:${VERSION} \
    -t zedinfer:latest \
    -t zedinfer:${GIT_HASH} \
    .
```

### Build cache

Docker caches each layer. After code changes, only `COPY . .` and subsequent layers rebuild. To force full rebuild:

```bash
docker build --no-cache -t zedinfer:latest .
```

---

## 3. Run

### Start server

```bash
docker run --gpus all -p 8080:8080 \
    -v /path/to/models:/models \
    zedinfer:latest /models/DeepSeek-R1-Distill-Qwen-1.5B --nvidia
```

### `serve` CLI arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `model_path` (positional) | — | Path to model directory (required) |
| `--nvidia` | `false` | Use GPU backend |
| `--host` | `0.0.0.0` | Bind address |
| `--port` | `8080` | Listen port |
| `--max-batch-tokens` | `2048` | Max tokens per batch |
| `--max-batch-requests` | `64` | Max concurrent requests |
| `--gpu-memory-utilization` | `0.9` | Fraction of GPU memory for KV cache (0.0–1.0) |

### Examples

```bash
# Custom port
docker run --gpus all -p 9090:9090 \
    -v /path/to/models:/models \
    zedinfer:latest /models/Qwen3-8B --nvidia --port 9090

# Limit GPU memory usage
docker run --gpus all -p 8080:8080 \
    -v /path/to/models:/models \
    zedinfer:latest /models/Qwen3-8B --nvidia --gpu-memory-utilization 0.5

# Specify single GPU
docker run --gpus '"device=0"' -p 8080:8080 \
    -v /path/to/models:/models \
    zedinfer:latest /models/Qwen3-8B --nvidia

# Detached mode (background)
docker run -d --gpus all -p 8080:8080 \
    --name zedinfer-server \
    -v /path/to/models:/models \
    zedinfer:latest /models/Qwen3-8B --nvidia
```

---

## 4. API Endpoints

### `POST /v1/chat/completions`

OpenAI-compatible chat completions.

**Request:**

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
| `messages` | array | — | Chat messages (required). Uses last `user` message. |
| `max_tokens` | int | `512` | Max tokens to generate |
| `stream` | bool | `false` | Enable SSE streaming |
| `session_id` | string | `""` | Session ID for multi-turn conversation. Empty = stateless. |

**Response (non-streaming):**

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

**Response (streaming):** SSE stream of `data: {...}` chunks, terminated by `data: [DONE]`.

**Example:**

```bash
# Non-streaming
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

### `GET /v1/models`

List loaded models.

```bash
curl http://localhost:8080/v1/models
```

### `GET /health`

Health check. Returns model name, request counts, block pool utilization.

```bash
curl http://localhost:8080/health
```

### `DELETE /v1/sessions/:id`

Delete a stateful session and release its KV cache.

```bash
curl -X DELETE http://localhost:8080/v1/sessions/abc123
```

### `GET /`

Web UI (served from `web/index.html`).

```
http://localhost:8080/
```

---

## 5. Other Binaries

Override the entrypoint to run bench, chat, ping, or batch_bench.

### bench — Single-request profiling

```bash
docker run --gpus all -v /path/to/models:/models \
    --entrypoint /app/bench \
    zedinfer:latest /models/Qwen3-8B --nvidia -p 128 -d 128 -r 3
```

| Argument | Default | Description |
|----------|---------|-------------|
| `model_path` | — | Model directory (required) |
| `--nvidia` | `false` | Use GPU |
| `-p, --prefill-len` | `128` | Prefill token count |
| `-d, --decode-len` | `128` | Decode token count |
| `-r, --rounds` | `3` | Benchmark rounds |
| `--gpu-memory-utilization` | `0.9` | GPU memory fraction |

### batch_bench — Batched throughput profiling

```bash
docker run --gpus all -v /path/to/models:/models \
    --entrypoint /app/batch_bench \
    zedinfer:latest /models/Qwen3-8B --nvidia -b 32 -p 128 -d 128
```

| Argument | Default | Description |
|----------|---------|-------------|
| `model_path` | — | Model directory (required) |
| `--nvidia` | `false` | Use GPU |
| `-b, --batch-size` | `4` | Concurrent requests |
| `-p, --prefill-len` | `128` | Prefill token count per request |
| `-d, --decode-len` | `128` | Decode token count per request |
| `-r, --rounds` | `1` | Benchmark rounds |
| `--gpu-memory-utilization` | `0.9` | GPU memory fraction |

### chat — Interactive terminal chat

```bash
docker run --gpus all -it -v /path/to/models:/models \
    --entrypoint /app/chat \
    zedinfer:latest /models/Qwen3-8B --nvidia
```

Note: requires `-it` (interactive + tty) for terminal input.

| Argument | Default | Description |
|----------|---------|-------------|
| `model_path` | — | Model directory (required) |
| `--nvidia` | `false` | Use GPU |
| `--max-tokens` | `16384` | Max tokens per response |
| `--gpu-memory-utilization` | `0.9` | GPU memory fraction |

### ping — Quick single-inference test

```bash
docker run --gpus all -v /path/to/models:/models \
    --entrypoint /app/ping \
    zedinfer:latest /models/Qwen3-8B --nvidia --prompt "Who are you?"
```

| Argument | Default | Description |
|----------|---------|-------------|
| `model_path` | — | Model directory (required) |
| `--nvidia` | `false` | Use GPU |
| `--prompt` | `"Who are you?"` | Prompt text |
| `--gpu-memory-utilization` | `0.9` | GPU memory fraction |

---

## 6. Stop Service

```bash
# Foreground (Ctrl+C)
# Sends SIGINT → graceful shutdown (HTTP stop + engine drain)

# Detached container
docker stop zedinfer-server          # Sends SIGTERM → graceful shutdown (10s timeout)
docker stop -t 30 zedinfer-server    # 30s timeout for large models with in-flight requests

# Force kill (immediate, no cleanup)
docker kill zedinfer-server

# Remove stopped container
docker rm zedinfer-server
```

---

## 7. Logs

Logs are written to `/app/logs/` inside the container.

```bash
# View live logs (stdout)
docker logs -f zedinfer-server

# Copy log files out
docker cp zedinfer-server:/app/logs ./logs

# Mount a host directory for persistent logs
docker run --gpus all -p 8080:8080 \
    -v /path/to/models:/models \
    -v /path/to/logs:/app/logs \
    zedinfer:latest /models/Qwen3-8B --nvidia
```

---

## 8. Push to Registry

### Docker Hub

```bash
# Login
docker login

# Tag for your registry
docker tag zedinfer:0.1.0 yourorg/zedinfer:0.1.0
docker tag zedinfer:0.1.0 yourorg/zedinfer:latest

# Push
docker push yourorg/zedinfer:0.1.0
docker push yourorg/zedinfer:latest
```

### Private registry (e.g., Harbor, NGC, ACR)

```bash
# Tag
docker tag zedinfer:0.1.0 registry.example.com/zedinfer:0.1.0

# Push
docker push registry.example.com/zedinfer:0.1.0
```

### NVIDIA NGC

```bash
docker login nvcr.io -u '$oauthtoken' -p <NGC_API_KEY>
docker tag zedinfer:0.1.0 nvcr.io/yourorg/zedinfer:0.1.0
docker push nvcr.io/yourorg/zedinfer:0.1.0
```

---

## 9. Image Contents

| Path | Content |
|------|---------|
| `/app/serve` | HTTP server binary |
| `/app/bench` | Single-request profiler |
| `/app/batch_bench` | Batched throughput profiler |
| `/app/chat` | Interactive terminal chat |
| `/app/ping` | Quick single-inference test |
| `/app/web/` | Web UI static files |
| `/app/logs/` | Runtime logs (writable) |
| `/app/NOTICE` | Third-party license attributions |

No source code, build tools, test binaries, or model weights in the image.

---

## 10. Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `docker: Error response from daemon: could not select device driver` | NVIDIA Container Toolkit not installed | See Section 1 install steps |
| `nvidia-container-cli: initialization error` | Driver too old or not loaded | `nvidia-smi` to check; update driver |
| Container starts but GPU not detected | Missing `--gpus all` | Add `--gpus all` to `docker run` |
| `HEALTHCHECK` reports unhealthy | Model still loading (large models need 1-2 min) | Check `docker logs`; increase `--start-period` if needed |
| `libcublas.so: cannot open shared object` | Used `base` image instead of `runtime` | Rebuild with `runtime` base |
| Port already in use | Another process on 8080 | Change port: `--port 9090` + `-p 9090:9090` |
| OOM killed | GPU memory exhausted | Lower `--gpu-memory-utilization` or use smaller model |
