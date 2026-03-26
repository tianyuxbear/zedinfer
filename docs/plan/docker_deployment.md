# Docker Deployment Plan

> Closed-source distribution of ZedInfer via Docker images.
> No source code in the final image. All runtime dependencies bundled.

---

## 1. Distribution Strategy

**Multi-stage Docker build:**
- **Stage 1 (builder):** Full build environment with source code, XMake, CUDA dev tools
- **Stage 2 (runtime):** Only compiled binaries + runtime dependencies, no source code

**Base images:**
- Builder: `nvidia/cuda:12.6-devel-ubuntu22.04` (includes nvcc, CUDA headers, cuBLAS dev)
- Runtime: `nvidia/cuda:12.6-runtime-ubuntu22.04` (includes cuBLAS, cudart, no dev tools)

**User runs with:** `docker run --gpus all`

---

## 2. What Goes Into the Image

### Included (runtime image)

| Component | Source | Size (approx) |
|-----------|--------|---------------|
| `serve` binary | Compiled from source | ~50 MB |
| `bench` binary | Compiled from source | ~40 MB |
| `chat` binary | Compiled from source | ~40 MB |
| `ping` binary | Compiled from source | ~40 MB |
| `batch_bench` binary | Compiled from source | ~40 MB |
| `web/` directory | Static files (HTML, JS, SVG) | ~200 KB |
| `libdnnl.so` (oneDNN) | apt or compiled | ~50 MB |
| `libicuuc.so` etc (ICU4C) | apt | ~30 MB |
| cuBLAS/cuBLASLt | From base image (`nvidia/cuda:12.6-runtime`) | Included |
| CUDA runtime | From base image | Included |
| NOTICE file | License attributions | <1 KB |

### NOT included

| Component | Reason |
|-----------|--------|
| Source code (`include/`, `src/`) | Closed-source |
| Build tools (XMake, nvcc, gcc) | Not needed at runtime |
| Test binaries | Not needed in production |
| Python bindings | Not needed in serving |
| `third_party/` headers | Compiled into binaries |
| Model weights | User provides at runtime via volume mount |

---

## 3. Dockerfile

```dockerfile
# ==============================================================================
# Stage 1: Build
# ==============================================================================
FROM nvidia/cuda:12.6-devel-ubuntu22.04 AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl unzip build-essential libicu-dev \
    && rm -rf /var/lib/apt/lists/*

# Install XMake
RUN curl -fsSL https://xmake.io/shget.text | bash
ENV PATH="/root/.local/bin:$PATH"

# Install oneDNN (build dependency)
# XMake will handle this via add_requires("onednn")

# Copy source code
WORKDIR /build
COPY . .

# Configure and build (release mode, GPU + oneDNN)
# Use x86-64-v3 baseline for broad CPU compatibility
# oneDNN handles AVX-512 dispatch at runtime automatically
RUN xmake f -m release --nv-gpu=y --onednn=y \
    && xmake build serve bench chat ping batch_bench

# ==============================================================================
# Stage 2: Runtime
# ==============================================================================
FROM nvidia/cuda:12.6-runtime-ubuntu22.04

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libicu72 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy oneDNN shared library from builder
COPY --from=builder /root/.xmake/packages/o/onednn/*/lib/libdnnl.so* /usr/lib/x86_64-linux-gnu/

# Copy binaries
WORKDIR /app
COPY --from=builder /build/build/linux/x86_64/release/serve .
COPY --from=builder /build/build/linux/x86_64/release/bench .
COPY --from=builder /build/build/linux/x86_64/release/chat .
COPY --from=builder /build/build/linux/x86_64/release/ping .
COPY --from=builder /build/build/linux/x86_64/release/batch_bench .

# Copy web UI
COPY --from=builder /build/web ./web

# Copy license notices
COPY --from=builder /build/NOTICE ./NOTICE

# Default: run serve on port 8080
EXPOSE 8080
ENTRYPOINT ["/app/serve"]
CMD ["--port", "8080", "--host", "0.0.0.0"]
```

---

## 4. Build and Run

### Build the image

```bash
docker build -t zedinfer:latest .
```

### Run the server

```bash
# Model weights mounted as a volume
docker run --gpus all -p 8080:8080 \
    -v /path/to/models:/models \
    zedinfer:latest /models/DeepSeek-R1-Distill-Qwen-1.5B --nvidia
```

### Run benchmark

```bash
docker run --gpus all \
    -v /path/to/models:/models \
    --entrypoint /app/bench \
    zedinfer:latest /models/DeepSeek-R1-Distill-Qwen-1.5B --nvidia -p 128 -d 128 -r 3
```

### Interactive chat

```bash
docker run --gpus all -it \
    -v /path/to/models:/models \
    --entrypoint /app/chat \
    zedinfer:latest /models/DeepSeek-R1-Distill-Qwen-1.5B --nvidia
```

---

## 5. CPU Compatibility

### Problem
Custom CPU kernels (rms_norm, rope, add, swiglu, attention) may use AVX-512 instructions via compile-time `#ifdef __AVX512F__`. If compiled with `-march=native` on an AVX-512 machine, the binary crashes on AVX2-only machines.

### Solution
Build the Docker image with `-march=x86-64-v3` (AVX2 baseline):

```lua
-- xmake.lua: add for distribution builds
if is_mode("release") then
    add_cxflags("-march=x86-64-v3")  -- AVX2 baseline for broad compatibility
end
```

**oneDNN handles this automatically** — its internal kernels dispatch to the best ISA (AVX2/AVX-512/AMX) at runtime regardless of compile flags.

### Performance impact
Minimal. Custom CPU kernels (norm, rope, add, swiglu) are memory-bound. The difference between AVX2 and AVX-512 for these operations is <5%. The compute-heavy linear operation goes through oneDNN which auto-dispatches.

---

## 6. CUDA Architecture Coverage

The gencode list in `xmake/device/nvidia.lua` auto-detects CUDA version:

| CUDA Version | Architectures | GPUs Covered |
|-------------|---------------|-------------|
| 12.0 - 12.7 | sm_80, sm_86, sm_89, sm_90 | A100, A6000, RTX 3090, RTX 4090, H100 |
| 12.8+ | + sm_100 | + B100, B200 |

For Docker: the builder image's CUDA version determines which architectures are compiled. Use CUDA 12.8+ base image to cover Blackwell.

---

## 7. NOTICE File

All vendored dependencies require license attribution. Create `NOTICE` at project root:

```
ZedInfer - LLM Inference Engine
Copyright (c) 2025-2026 ZedInfer Contributors

This product includes software developed by third parties:

- nlohmann/json (MIT License) - https://github.com/nlohmann/json
  Copyright (c) 2013-2025 Niels Lohmann

- plog (MIT License) - https://github.com/SergiusTheBest/plog
  Copyright (c) 2016 Sergey Podobry

- argparse (MIT License) - https://github.com/p-ranav/argparse
  Copyright (c) 2018 Pranav Srinivas Kumar

- cpp-httplib (MIT License) - https://github.com/yhirose/cpp-httplib
  Copyright (c) 2017 yhirose

- linenoise (BSD-2-Clause License) - https://github.com/antirez/linenoise
  Copyright (c) 2010-2014 Salvatore Sanfilippo, Pieter Noordhuis

- dbg-macro (MIT License) - https://github.com/sharkdp/dbg-macro
  Copyright (c) 2019 David Peter

- oneDNN (Apache-2.0 License) - https://github.com/oneapi-src/oneDNN
  Copyright 2016-2025 Intel Corporation

- ICU4C (Unicode License) - https://icu.unicode.org/
  Copyright (c) 1995-2025 Unicode, Inc.
```

---

## 8. Security Hardening (Pre-Distribution)

| Item | Current | Target |
|------|---------|--------|
| HTTP bind address | `0.0.0.0` (all interfaces) | Default `127.0.0.1`, `--host 0.0.0.0` opt-in |
| Authentication | None | Optional `--api-key` flag |
| HTTPS | None | Recommend reverse proxy (nginx/caddy) |
| Rate limiting | None | Optional `--max-concurrent` flag |
| Version info | None | `--version` flag |

These are application-level changes, independent of Docker packaging.

---

## 9. Image Size Optimization

| Technique | Savings |
|-----------|---------|
| Multi-stage build (no source/build tools) | ~2 GB |
| `--no-install-recommends` for apt | ~200 MB |
| Strip binaries (`-s` flag in release mode) | ~50 MB |
| Only copy needed CUDA libs (runtime, not devel) | ~1 GB |
| Use `ubuntu:22.04` + manual CUDA install (advanced) | ~500 MB |

Expected final image size: **~3-4 GB** (dominated by CUDA runtime libraries).

---

## 10. Implementation Steps

```
Step 1: Create NOTICE file with all dependency attributions
Step 2: Add xmake install target (copies binaries + web/ to output dir)
Step 3: Write Dockerfile (multi-stage as above)
Step 4: Test: docker build + docker run with model volume
Step 5: Add --version flag to all binaries
Step 6: Security: default bind to 127.0.0.1, add --api-key option
Step 7: CI/CD: automated image build on tag push
```
