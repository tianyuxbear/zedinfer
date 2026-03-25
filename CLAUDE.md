# CLAUDE.md

## Purpose

ZedInfer is a self-developed C++17 LLM inference framework. No Python in the serving path.

**New to this project?** Read these files in order:
1. `docs/architecture.md` — current system design
2. `docs/roadmap.md` — what's done, what's next, priorities
3. `docs/plan/` — design docs for upcoming features
4. `docs/prompt.md` — onboarding prompt for new agents

---

## Current State

**Completed features:** direct model forward, cuBLAS/oneDNN linear, paged KV cache, paged attention, continuous batching, prefix caching, HTTP API with Web UI, SSE streaming, DecodeScratch pre-allocation.

**Supported models:** Qwen2 (DeepSeek-R1-Distill-Qwen-1.5B), Qwen3 (DeepSeek-R1-0528-Qwen3-8B)

**Upcoming work (see `docs/plan/`):** FlashInfer integration, CUDA graph, INT8/INT4 quantization, heterogeneous CPU/GPU inference, MoE expert offloading.

---

## Working Rules

### 1. Code Evidence Rule
- Read relevant source code BEFORE proposing any modification
- Cite exact file path and line numbers as evidence
- Never guess or assume code structure

### 2. Document-first for major features
Before coding a major feature, create or update a design doc in `docs/plan/`. Wait for approval.

### 3. Small, staged changes
Break work into small, reviewable steps. For each step: scope, files affected, test plan, risks.

### 4. Performance awareness
This is a performance-sensitive C++ framework. Avoid:
- unnecessary abstraction layers
- unnecessary heap allocation in hot paths
- unnecessary runtime indirection
- virtual dispatch where a concrete type suffices

### 5. No Python runtime dependency
Do not introduce Python into the serving path. Python is acceptable only for offline tools (codegen, testing, benchmarking).

### 6. Quantization and heterogeneous inference must be designed, not hacked in
Design explicit interfaces for weight format, device placement, quantized dispatch, and offloading policy.

### 7. Be repository-specific
Tie recommendations to actual files, classes, and functions. Avoid generic advice.

---

## Code Style

- All code comments must be in English
- Do not add Chinese characters in code or comments

---

## Key Files

| File | Purpose |
|------|---------|
| `docs/architecture.md` | Current system architecture |
| `docs/roadmap.md` | What's done + what's planned |
| `docs/plan/` | Design documents for upcoming features |
| `include/zedinfer/engine.hpp` | Engine: resource container |
| `include/zedinfer/scheduler.hpp` | Scheduler: batching + admission |
| `include/zedinfer/request.hpp` | Request lifecycle |
| `include/backend/kvcache/block_pool.hpp` | BlockPool + PrefixCache types |
| `src/frontend/models/transformer_forward.cpp` | Shared forward loop |
| `src/frontend/models/paged_forward_context.cpp` | KV scatter + attention dispatch |
| `src/backend/ops/self_attention/nvidia/paged_attention_nvidia.cu` | CUDA paged attention kernels |

---

## Build & Test

```bash
# Build
xmake f -m release --nv-gpu=y --onednn=y
xmake build

# Run tests
xmake run test-blockpool
xmake run test-prefixcache
xmake run test-sampler
xmake run test-chattemplate
xmake run test-tensor
ZEDINFER_TEST_MODEL_PATH=/path/to/model xmake run test-tokenizer
ZEDINFER_TEST_MODEL_PATH=/path/to/model xmake run test-loader

# Benchmark
xmake run bench /path/to/model --nvidia -p 128 -d 128 -r 3 --gpu-memory-utilization 0.1

# Serve
xmake run serve /path/to/model --nvidia --port 8080
```