# ==============================================================================
# Stage 1: Build
# ==============================================================================
FROM nvidia/cuda:13.1.1-devel-ubuntu24.04 AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl unzip build-essential libicu-dev \
    && rm -rf /var/lib/apt/lists/*

# Install XMake
RUN curl -fsSL https://xmake.io/shget.text | bash
ENV PATH="/root/.local/bin:$PATH"

# Copy source code
WORKDIR /build
COPY . .

# Git hash passed from host (docker build --build-arg GIT_HASH=...)
# Read by xmake on_config as fallback when .git/ is unavailable.
ARG GIT_HASH=unknown
ENV ZEDINFER_GIT_HASH=${GIT_HASH}

# Build release binaries (portable AVX2 baseline + GPU + oneDNN)
ENV XMAKE_ROOT=y
RUN xmake f -m release --nv-gpu=y --onednn=y --portable=y -y \
    && xmake

# Locate the oneDNN shared library installed by XMake for later COPY
RUN mkdir -p /staging/lib \
    && find /root/.xmake -name "libdnnl.so*" -exec cp {} /staging/lib/ \;

# ==============================================================================
# Stage 2: Runtime
# ==============================================================================
FROM nvidia/cuda:13.1.1-runtime-ubuntu24.04

# Install runtime dependencies only
#   libicu74  — ICU4C for BPE tokenizer (Ubuntu 24.04 ships ICU 74)
#   libgomp1  — OpenMP runtime for CPU parallelism
#   curl      — for Docker HEALTHCHECK
RUN apt-get update && apt-get install -y --no-install-recommends \
    libicu74 libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

# Copy oneDNN shared library from staging
COPY --from=builder /staging/lib/libdnnl.so* /usr/lib/x86_64-linux-gnu/
RUN ldconfig

# Create non-root user for security
RUN useradd -r -s /bin/false zedinfer
WORKDIR /app

# Copy binaries
COPY --from=builder /build/build/linux/x86_64/release/serve  .
COPY --from=builder /build/build/linux/x86_64/release/bench  .
COPY --from=builder /build/build/linux/x86_64/release/chat   .
COPY --from=builder /build/build/linux/x86_64/release/ping   .
COPY --from=builder /build/build/linux/x86_64/release/batch_bench .

# Copy web UI (serve resolves "web/" relative to CWD)
COPY --from=builder /build/web ./web

# Copy license notices
COPY --from=builder /build/NOTICE ./NOTICE

# Logs directory (writable by non-root user)
RUN mkdir -p /app/logs && chown -R zedinfer:zedinfer /app/logs

USER zedinfer

EXPOSE 8080
STOPSIGNAL SIGTERM

HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD curl -sf http://localhost:8080/health || exit 1

ENTRYPOINT ["/app/serve"]
CMD ["--port", "8080", "--host", "0.0.0.0"]
