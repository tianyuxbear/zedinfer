# FlashInfer Backend

This document describes how FlashInfer is applied in ZedInfer today, what it accelerates, where the fallback boundaries are, and which code was changed to support it.

---

## Overview

FlashInfer is integrated as an optional NVIDIA paged-attention backend. It is enabled at build time with `--flashinfer=y`, but the runtime still keeps the original paged CUDA kernels as a fallback for unsupported shapes or for A/B comparisons.

Current scope:

- NVIDIA paged decode can use FlashInfer for both single-request and batched decode.
- NVIDIA paged prefill can use FlashInfer for both single-request and batched prefill.
- CPU attention is unchanged.
- KV scatter is still handled by ZedInfer's own paged KV write path.

The source is tracked as a git submodule in `third_party/flashinfer`.

---

## Enable It

Initialize submodules first:

```bash
git submodule update --init --recursive
```

Build with NVIDIA support and FlashInfer enabled:

```bash
xmake f -m release --nv-gpu=y --flashinfer=y
xmake build
```

Quick validation:

```bash
xmake run ping /path/to/model --nvidia
xmake run bench /path/to/model --nvidia -p 128 -d 128 -r 3
```

Runtime switches:

- `ZEDINFER_DISABLE_FLASHINFER=1`
  Forces the runtime back to the legacy paged-attention kernels without rebuilding.
- `ZEDINFER_FLASHINFER_DISABLE_FASTPATH=1`
  Disables the single-request decode fast path inside the FlashInfer wrapper so the planner path can be debugged or benchmarked directly.

---

## Current Dispatch Rules

FlashInfer is considered only when all of the following are true:

- Built with `--flashinfer=y`
- Built with `--nv-gpu=y`
- Runtime device is NVIDIA
- Attention dtype is `fp16` or `bf16`
- `head_dim` is `64`, `128`, or `256`
- `block_size > 0`
- `nhead % nkvhead == 0`

If any of these checks fail, `ops::attention()` falls back to the original paged-attention implementation.

There are two additional runtime decisions inside the supported set:

- Decode native kernel support currently depends on GQA group size `nhead / nkvhead` being one of `1`, `2`, `3`, `4`, or `8`.
- Batched prefill uses FlashInfer only when all prefill slices are contiguous in the flattened token buffer produced by `PagedForwardContext`.

---

## Decode Path

### Metadata layout

`PagedForwardContext` converts each decode request into FlashInfer's paged-KV CSR form:

- `kv_indptr`
- `kv_page_indices`
- `kv_last_page_len`

For single-request decode, the per-layer page table tensor and the fixed-size single-request metadata tensors are cached on the `SequenceBlockTable` and reused across layers. For batched decode, per-layer active page indices are concatenated and uploaded once per forward context.

### Native decode vs fallback decode

If the GQA group size is one of the currently supported values (`1`, `2`, `3`, `4`, `8`), the wrapper uses FlashInfer's decode kernel.

If the group size is not covered by the decode kernel, ZedInfer still keeps the request on the FlashInfer path, but routes it through the FlashInfer prefill kernel with `qo_indptr={0,1}` semantics. This preserves correctness while avoiding an immediate drop back to the legacy kernels for that decode shape.

### Single-request fast path

Single-request decode has a dedicated fast path inside `flashinfer_wrapper.cu`:

- It skips FlashInfer's decode planner.
- It reuses a trivial descriptor materialized by `PagedForwardContext`.
- It calls `BatchDecodeWithPagedKVCacheDispatched` directly.

That path can be disabled with `ZEDINFER_FLASHINFER_DISABLE_FASTPATH=1`.

---

## Prefill Path

### What gets merged

FlashInfer prefill is built over the prefill slices already assembled by `PagedForwardContext`. ZedInfer merges them into a single FlashInfer call only when those prefill slices are contiguous in the flattened query/output tensors.

If prefill slices are not contiguous, ZedInfer falls back to the legacy per-slot paged prefill path.

### Metadata layout

Prefill uses:

- `qo_indptr`
- `kv_indptr`
- `kv_last_page_len`

For single-request prefill, the existing per-layer page table cache is reused directly.

For batched prefill, ZedInfer concatenates the active page indices per layer and uploads that compact list before calling FlashInfer.

---

## What Changed In The Codebase

The FlashInfer integration is not a single wrapper drop-in. It required coordinated changes across build, metadata, dispatch, and tests.

### Build and dependency management

- `xmake.lua`
  Adds the `flashinfer` build option, includes local FlashInfer headers, and checks submodule include paths.
- `third_party/flashinfer`
  Tracks the upstream FlashInfer source as a git submodule.

### Attention dispatch and parameters

- `include/backend/ops/attention_params.hpp`
  Extends `AttentionParams` with FlashInfer-specific CSR metadata, plus host-side copies needed by FlashInfer planners.
- `src/backend/ops/self_attention/op.cpp`
  Makes unified attention dispatch prefer FlashInfer when `params.use_flashinfer` is set on an NVIDIA device.

### Runtime metadata and cache reuse

- `include/backend/kvcache/block_pool.hpp`
  Extends `SequenceBlockTable` with runtime-only GPU caches for full per-layer page tables and single-request decode metadata.
- `src/frontend/models/paged_forward_context.cpp`
  Detects when FlashInfer can be used, builds decode/prefill CSR metadata, caches reusable tensors, and prepares the correct `AttentionParams` for each layer.

### FlashInfer wrapper

- `src/backend/ops/self_attention/nvidia/flashinfer_wrapper.cu`
  Bridges `AttentionParams` to FlashInfer's decode and prefill APIs, performs planner calls, sizes temporary workspaces, retries on workspace overflow, and dispatches by dtype and head dimension.

### KV write path support

- `src/frontend/models/paged_forward_context.cpp`
  GPU KV scatter now reuses cached page-table tensors so the same page metadata can serve both the write path and FlashInfer attention setup.

### Correctness tests

- `tests/models/test_flashinfer_decode.cpp`
  Adds decode and prefill parity tests against the legacy paged-attention kernels for both fp16 and bf16, including cases that intentionally fall back from the FlashInfer decode kernel to the FlashInfer prefill kernel.

---

## Validation And Benchmarking

Recommended validation sequence:

```bash
xmake build test-models
xmake run test-models
xmake run ping /path/to/model --nvidia
```

Recommended benchmark comparison:

```bash
# FlashInfer enabled
xmake run bench /path/to/model --nvidia -p 128 -d 128 -r 3

# Same build, legacy kernels forced at runtime
ZEDINFER_DISABLE_FLASHINFER=1 xmake run bench /path/to/model --nvidia -p 128 -d 128 -r 3
```

When evaluating changes, cover both prefill and decode across multiple lengths. At minimum, benchmark:

- short prefill + short decode
- short prefill + long decode
- long prefill + short decode
- long prefill + long decode

The existing `bench` and `batch_bench` binaries are sufficient for A/B comparisons because the runtime kill switch avoids a rebuild between the two paths.
