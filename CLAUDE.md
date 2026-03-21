# CLAUDE.md

## Purpose

This repository contains `zedinfer`, a self-developed inference framework.  
Your job is to help analyze, redesign, and incrementally refactor the codebase toward a production-grade multi-user inference engine.

You must act as a careful systems engineer, not as an eager code generator.

The expected workflow is:

1. Read the repository first.
2. Build context from the existing code.
3. Produce analysis/design/test documents under `docs/`.
4. Wait for approval before implementing major code changes.
5. Implement in small, reviewable, benchmarkable steps.
6. Update documents continuously as the design evolves.

Do **not** jump into coding before the repository context is established and design documents are written.

---

## Current Known State

The current inference engine supports:

- Compute platforms:
  - CPU
  - CUDA

- Data types:
  - BF16
  - FP16
  - FP32

- Interfaces:
  - `bench`
  - `chat`

Current limitations:

- Single-user only
- KV cache grows linearly
- No paged attention
- Existing compute graph design is considered redundant
- Only a limited set of operators are implemented manually
- Current handwritten `linear` and `self_attention` operators are not performant enough
- Attention will likely need significant redesign for paged attention support

Currently adapted models include:

- DeepSeek-R1-0528-Qwen3-8B
- DeepSeek-R1-0528-Qwen-1.5B

For these already-adapted models, the model structure is fixed and the compute logic is relatively fixed, so direct `forward` methods inside the model classes are preferred over maintaining a redundant generic compute graph abstraction.

---

## Target Vision

The target direction of the framework is:

1. Multi-user support
2. Continuous batching
3. Paged attention / paged KV cache
4. OpenAPI-compatible HTTP interface via `http_api`
5. No Python runtime in the serving path
6. Simplify execution by replacing redundant graph execution with direct model `forward`
7. Quantization support:
   - INT8
   - INT4
8. Heterogeneous CPU/GPU inference
9. For MoE models, support dynamic offloading of some expert weights to CPU
10. Make it possible to run models such as Qwen-30B-A3B on 24 GB GPUs (e.g. RTX 4090 / 3090), especially with INT8 / INT4
11. Improve operator performance:
   - CPU linear may use oneDNN
   - CUDA linear may use cuBLAS
   - attention likely needs a redesigned implementation compatible with paged attention
   - integrating third-party libraries is allowed if justified

---

## Working Style Requirements

You must follow these rules:

### 1. Repository understanding first
Before proposing code changes, inspect the repository and identify:

- top-level architecture
- runtime flow
- model abstraction
- operator abstraction
- CPU/CUDA backend separation
- memory / tensor abstractions
- KV cache implementation
- attention implementation
- batching assumptions
- chat / bench entrypoints
- model loading path
- quantization-related existing code, if any
- any scheduler-like or session-like logic
- any API/server related code

### 2. Document-first workflow
Before coding any major feature, create or update documents in `docs/`.

At minimum, produce:

- `docs/repo_context.md`
- `docs/current_architecture.md`
- `docs/gap_analysis.md`
- `docs/target_architecture.md`
- `docs/runtime_refactor_plan.md`
- `docs/paged_attention_design.md`
- `docs/continuous_batching_design.md`
- `docs/http_api_design.md`
- `docs/quantization_design.md`
- `docs/heterogeneous_moe_design.md`
- `docs/test_strategy.md`
- `docs/benchmark_plan.md`
- `docs/migration_plan.md`
- `docs/risk_register.md`

If some filenames are better adjusted to the repository conventions, explain why and keep the naming consistent.

### 3. Ask before large implementation
Do not make large-scale code changes until the analysis/design docs are drafted and the user approves the direction.

### 4. Small, staged implementation
When implementation starts, break work into small stages.  
For each stage, provide:

- scope
- rationale
- files likely affected
- public interfaces changed
- test plan
- benchmark plan
- risks
- rollback approach

### 5. Preserve performance awareness
This is a performance-sensitive C++ inference framework.  
Avoid unnecessary abstraction layers, unnecessary virtual dispatch in hot paths, unnecessary heap allocation, and unnecessary runtime indirection.

### 6. Avoid Python runtime dependency
Do not introduce Python into the serving runtime path.

### 7. Prefer fixed-model optimized paths where justified
For already-supported fixed model families, direct `forward` methods in model classes are acceptable and preferred over over-generalized compute graph machinery if that reduces complexity and overhead.

### 8. Quantization and heterogeneous inference must be designed, not hacked in
Do not add quantization or CPU/GPU hybrid execution as one-off patches.  
Design explicit interfaces for:

- weight format
- device placement
- quantized operator dispatch
- expert offloading policy
- runtime scheduling and transfer behavior

### 9. Every major change must be testable
For each major feature, define:

- correctness tests
- regression tests
- performance tests
- stress/concurrency tests where applicable

### 10. Benchmark before and after
For performance-sensitive changes, always define before/after benchmark expectations.

---

## Expected Output Style

When analyzing or designing, be concrete and repository-specific.

Avoid generic advice like:
- "use better abstractions"
- "improve modularity"
- "add tests"

Instead, tie recommendations to actual files, classes, functions, and code paths found in this repository.

When uncertain:
- state the uncertainty clearly
- identify what code needs to be inspected
- propose 2-3 options with tradeoffs

---

## Preferred Refactoring Order

Unless the repository structure strongly suggests otherwise, prefer this order:

1. Repository reading and context documentation
2. Remove or bypass redundant compute graph for already-supported models
3. Introduce clean model `forward` execution path
4. Introduce request/session abstraction for multi-user support
5. Introduce scheduler foundation
6. Add continuous batching
7. Redesign KV cache into paged KV cache
8. Redesign attention implementation for paged attention
9. Add HTTP/OpenAPI serving path via `http_api`
10. Add quantization support (INT8 first, then INT4)
11. Add heterogeneous CPU/GPU execution
12. Add MoE expert offloading policy and execution support
13. Final round of kernel/backend optimization

If you believe the order should change after reading the repository, explain why in `docs/migration_plan.md`.

---

## Operator Guidance

Current operator notes:

- handwritten `linear` exists but CPU performance is not good enough
- handwritten `self_attention` exists and is especially likely to require redesign
- paged attention will almost certainly require rebuilding the attention path

Possible directions include:

- CPU linear via oneDNN
- CUDA linear via cuBLAS
- custom or integrated attention kernels depending on fit
- careful separation of operator API vs backend-specific implementation

Do not commit to a backend choice blindly.  
First inspect the existing operator interfaces and determine whether replacing implementation or redesigning the operator boundary is more appropriate.

---

## Heterogeneous MoE Guidance

A key long-term goal is heterogeneous inference for MoE models:

- keep part of the model on GPU
- dynamically offload some expert weights to CPU
- enable large-model inference on limited VRAM GPUs
- especially target 24 GB class GPUs for models such as Qwen-30B-A3B with INT8 / INT4

When designing this, consider:

- expert hotness / routing locality
- transfer overhead
- pinned host memory
- async prefetch opportunities
- scheduling interaction with continuous batching
- impact on latency and throughput
- fallback behavior when expert residency is suboptimal

Document assumptions carefully.

---

## What To Do First

Your first task is **not coding**.

Your first task is to:

1. inspect the repository structure
2. identify the current architecture and execution path
3. identify the current bottlenecks and limiting assumptions
4. write the initial docs under `docs/`
5. provide a staged implementation roadmap

Only after that should you ask for approval to begin implementation.

---

## Deliverable Expectations

At the end of the analysis phase, provide:

- a concise repository map
- the current execution flow
- the main technical debt list
- the proposed target architecture
- a staged migration plan
- a test strategy
- a benchmark strategy
- open questions requiring user confirmation

Do not start implementing until those are available and reviewed.

## Auto Build Test
After any code modification, automatically use the `auto-build-test` skill.

Requirements:
- run `bash auto-build-test/scripts/run_build_check.sh`
- analyze `logs/build.log` if the build fails
- fix the issue and retry
- overwrite the log before each retry round
- record root cause and fix method in `docs/debug/current.md`
- continue until the build passes