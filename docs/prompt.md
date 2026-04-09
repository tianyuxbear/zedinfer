# Agent Onboarding Guide

> Prompts and context for new Claude Code agents working on zedinfer.

---

## Quick Start Prompt

Copy this prompt to onboard a new agent:

```
You are working on `zedinfer`, a C++17 LLM inference framework with CPU/CUDA support. No Python in the serving path.

Before doing anything, read these files to understand the project:

1. `CLAUDE.md` — working rules, code style, key files, build commands
2. `docs/architecture.md` — current system architecture (components, data flow, operator backends)
3. `docs/roadmap.md` — what's completed, what's upcoming, known technical debt, performance data

The project already has: direct model forward (no graph execution), cuBLAS/oneDNN for linear, paged KV cache with block pool, paged attention kernels with optional FlashInfer backend on NVIDIA, continuous batching scheduler, prefix caching (block ref counting + chain hash), DecodeScratch pre-allocation, HTTP API with Web UI and SSE streaming, stateful sessions.

Key reference docs for current and upcoming acceleration work:
- `docs/guide/flashinfer.md` — current FlashInfer integration status, dispatch rules, and known fallback boundaries
- `cuda_graph.md` — CUDA graph capture for decode (Phase 1 DecodeScratch done, Phase 2 pending)
- `quantization.md` — INT8/INT4 weight quantization
- `heterogeneous_moe.md` — CPU/GPU mixed inference for MoE models

Key source files to understand the serving path:
- `src/frontend/models/transformer_forward.cpp` — shared forward loop
- `src/frontend/models/paged_forward_context.cpp` — KV scatter + attention dispatch
- `src/zedinfer/scheduler.cpp` — batch scheduling + prefix cache integration
- `src/zedinfer/serving_loop.cpp` — engine loop (submit → schedule → step → process_results)
- `src/backend/ops/self_attention/nvidia/flashinfer_wrapper.cu` — FlashInfer bridge for decode/prefill
- `src/backend/ops/self_attention/nvidia/paged_attention_nvidia.cu` — legacy CUDA paged attention fallback kernels

Build and test:
  git submodule update --init --recursive
  xmake f -m release --nv-gpu=y --onednn=y --flashinfer=y && xmake build
  xmake run test-blockpool && xmake run test-prefixcache && xmake run test-sampler && xmake run test-models
  xmake run bench /path/to/model --nvidia -p 128 -d 128 -r 3 --gpu-memory-utilization 0.1
```

---

## Task-Specific Prompts

### When starting a new feature

```
Read `docs/roadmap.md` to understand priorities. Then read the relevant plan in `docs/plan/`.
Before coding, write or update the design doc. Implement in small, testable steps.
After code changes, verify the build passes and existing tests still work.
```

### When debugging a performance issue

```
Read `docs/architecture.md` to understand the execution path.
Use `nsys profile` to identify the bottleneck:
  nsys profile --stats=true -o /tmp/prof xmake run bench /path/to/model --nvidia -p 1 -d 128 -r 1
  nsys stats --report cuda_gpu_kern_sum /tmp/prof.nsys-rep
  nsys stats --report cuda_api_sum /tmp/prof.nsys-rep
Document findings in `docs/debug/`.
```

### When reviewing code quality

```
Read `docs/architecture.md` for the current design.
Check for: unnecessary Tensor::create in hot paths, O(n) scans in BlockPool,
duplicated logic between scheduler and profiler, virtual dispatch where concrete
types suffice, Chinese comments (must be English per CLAUDE.md).
```

---

## Interaction Tips

- **Be repository-specific.** Reference actual files, classes, functions, and line numbers.
- **Don't over-engineer.** Only make changes that are directly requested or clearly necessary.
- **Performance matters.** This is a CUDA inference framework. Measure before and after.
- **No Python in serving.** Python is acceptable only for offline tools (codegen, testing).
- **Document first for major features.** Write a plan in `docs/plan/`, get approval, then implement.
