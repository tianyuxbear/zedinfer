# CLI Tools

ZedInfer ships 5 binaries in the Docker image. Override the entrypoint to run tools other than `serve`.

---

## Shared logging

All CLI tools support `-v`/`--version`, `--log-level`, `--log-file`, `--log-to-console`, `--no-log-to-console`, `--log-append`, and `--log-overwrite`. The build version is the current HEAD tag when available on a clean worktree, otherwise the short commit hash, with `-dirty` appended when uncommitted changes exist. Logs are mirrored to stdout/stderr by default, except for interactive `chat`, where console logs are disabled to avoid interleaving with prompts and streamed output.

See the [Logging Guide](logging.md) for routing, file behavior, color formatting, startup banner behavior, and developer rules.

---

## serve (default)

HTTP server with Web UI. This is the default entrypoint.

```bash
docker run --gpus all -p 8080:8080 --name zedinfer \
    -v /path/to/models:/models \
    zedinfer:latest /models/Qwen3-8B --nvidia
```

| Argument | Default | Description |
|----------|---------|-------------|
| `model_path` | -- | Model directory (required) |
| `--nvidia` | `false` | Use GPU backend |
| `--host` | `0.0.0.0` | Bind address (inside Docker) |
| `--port` | `8080` | Listen port |
| `--max-batch-tokens` | `2048` | Max tokens per batch |
| `--max-batch-requests` | `64` | Max concurrent requests |
| `--gpu-memory-utilization` | `0.9` | GPU memory fraction for KV cache |
| `--max-tokens` | `0` | Default max response tokens when API requests omit `max_tokens` (`0` = unlimited) |

---

## bench

Single-request latency profiler. Measures prefill and decode separately.

```bash
docker run --gpus all -v /path/to/models:/models \
    --entrypoint /app/bench \
    zedinfer:latest /models/Qwen3-8B --nvidia -p 128 -d 128 -r 3
```

| Argument | Default | Description |
|----------|---------|-------------|
| `-p, --prefill-len` | `128` | Prefill token count |
| `-d, --decode-len`, `--max-tokens` | `128` | Decode token count |
| `-r, --rounds` | `3` | Benchmark rounds |

---

## batch_bench

Batched throughput profiler. Measures total throughput with concurrent requests.

```bash
docker run --gpus all -v /path/to/models:/models \
    --entrypoint /app/batch_bench \
    zedinfer:latest /models/Qwen3-8B --nvidia -b 32 -p 128 -d 128
```

| Argument | Default | Description |
|----------|---------|-------------|
| `-b, --batch-size` | `4` | Concurrent requests |
| `-p, --prefill-len` | `128` | Prefill tokens per request |
| `-d, --decode-len`, `--max-tokens` | `128` | Decode tokens per request |
| `-r, --rounds` | `1` | Benchmark rounds |

---

## chat

Interactive terminal chat with conversation history.

```bash
docker run --gpus all -it -v /path/to/models:/models \
    --entrypoint /app/chat \
    zedinfer:latest /models/Qwen3-8B --nvidia
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--max-tokens` | `0` | Max tokens per response (`0` = unlimited) |

In-session commands: `/exit`/`/quit`/`/q` to exit, `/reset`/`/clear`/`/cls` to reset conversation, `/help` to show help.

!!! note
    Requires `-it` (interactive + tty) for terminal input.

---

## ping

Quick single-inference test. Useful for verifying model loading and basic generation.

```bash
docker run --gpus all -v /path/to/models:/models \
    --entrypoint /app/ping \
    zedinfer:latest /models/Qwen3-8B --nvidia --prompt "Who are you?"
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--prompt` | `"Who are you?"` | Prompt text |
| `--max-new-tokens`, `--max-tokens` | `0` | Max tokens to generate (`0` = unlimited) |

## Version

All binaries support `--version`:

```bash
docker run --rm zedinfer:latest --version
# zedinfer 0.1.0 (build 6f1385b, 2026-03-26)
```
