# Quick Start

Get ZedInfer running in under 2 minutes.

---

## Prerequisites

- NVIDIA GPU with driver >= 590 (required for CUDA 13.x)
- Docker with [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- Model weights (Safetensors format)

## 1. Pull the image

```bash
docker pull tianyuxbear/zedinfer:latest
```

## 2. Run the server

```bash
docker run --gpus all -p 8080:8080 --name zedinfer \
    -v /path/to/models:/models \
    zedinfer:latest /models/DeepSeek-R1-Distill-Qwen-1.5B --nvidia
```

The server starts on port 8080. Model weights are mounted via `-v`, not baked into the image.

## 3. Send a request

=== "curl"

    ```bash
    curl http://localhost:8080/v1/chat/completions \
        -H "Content-Type: application/json" \
        -d '{"messages":[{"role":"user","content":"Hello"}]}'
    ```

=== "Python"

    ```python
    import requests

    resp = requests.post("http://localhost:8080/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "Hello"}]
    })
    print(resp.json()["choices"][0]["message"]["content"])
    ```

=== "Web UI"

    Open [http://localhost:8080](http://localhost:8080) in your browser.

## Configuration

| Argument | Default | Description |
|----------|---------|-------------|
| `model_path` | -- | Path to model directory (required) |
| `--nvidia` | `false` | Use GPU backend |
| `--host` | `0.0.0.0` | Bind address (inside Docker) |
| `--port` | `8080` | Listen port |
| `--max-batch-tokens` | `2048` | Max tokens per batch |
| `--max-batch-requests` | `64` | Max concurrent requests |
| `--gpu-memory-utilization` | `0.9` | Fraction of GPU memory for KV cache (0.0--1.0) |
| `--max-tokens` | `0` | Default max response tokens when API requests omit `max_tokens` (`0` = unlimited) |

## Stop the server

```bash
# Foreground: Ctrl+C

# Background container
docker stop zedinfer
docker rm zedinfer
```

The server handles SIGTERM gracefully -- in-flight requests complete before shutdown.
