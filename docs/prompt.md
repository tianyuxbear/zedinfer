# Prompt Guide for Claude Code

This document contains recommended prompts for guiding Claude Code through the refactor and improvement of the `zedinfer` inference framework.

---

## Prompt 1: Initial repository reading and context filling

Please read the entire repository carefully before making any code changes.

Context:
- This is the `zedinfer` inference framework.
- The current engine supports CPU/CUDA, BF16/FP16/FP32, and bench/chat interfaces.
- Current limitations:
  - single-user only
  - linear-growing KV cache
  - no paged attention
  - current compute graph design is redundant for already-supported models
  - current handwritten linear and self_attention operators are not performant enough
- Already adapted models:
  - DeepSeek-R1-0528-Qwen3-8B
  - DeepSeek-R1-0528-Qwen-1.5B
- For these models, direct model-class `forward` functions are preferred over maintaining a redundant generic compute graph abstraction.
- Target direction:
  - multi-user support
  - continuous batching
  - paged attention
  - OpenAPI-compatible HTTP interface via `http_api`
  - no Python runtime in serving
  - INT8 / INT4 quantization
  - heterogeneous CPU/GPU inference
  - MoE expert offloading to CPU for limited-VRAM GPUs
  - improve operator performance, possibly using oneDNN / cuBLAS or better implementations

Tasks:
1. Inspect the repository structure and identify the major modules and execution flow.
2. Identify the current architecture and how requests flow through bench/chat paths.
3. Identify where model execution, operators, memory management, KV cache, and device abstraction are implemented.
4. Identify whether there are existing scheduler/session abstractions.
5. Identify how the current compute graph is implemented and why it is redundant.
6. Identify current bottlenecks and structural limitations with respect to:
   - multi-user serving
   - continuous batching
   - paged attention
   - quantization
   - heterogeneous inference
7. Write the following documents under `docs/`:
   - repo_context.md
   - current_architecture.md
   - gap_analysis.md
   - risk_register.md
8. Do not start implementing yet.
9. End with a proposed staged roadmap and a list of open questions.

Please be repository-specific and reference actual files, classes, functions, and code paths.

---

## Prompt 2: Produce target design documents before coding

Based on your repository reading, please produce a design-first plan before any major implementation.

Tasks:
1. Write or update the following design documents under `docs/`:
   - target_architecture.md
   - runtime_refactor_plan.md
   - continuous_batching_design.md
   - paged_attention_design.md
   - http_api_design.md
   - quantization_design.md
   - heterogeneous_moe_design.md
   - migration_plan.md
   - test_strategy.md
   - benchmark_plan.md
2. For each document, make the design concrete and tied to this repository.
3. Explicitly explain:
   - which current abstractions should be kept
   - which should be simplified
   - which should be removed
4. For the compute path, evaluate replacing the current graph-based execution for already-supported models with direct model `forward`.
5. For continuous batching, define:
   - request abstraction
   - scheduler abstraction
   - admission policy
   - token-level scheduling
   - fairness and starvation handling
6. For paged attention, define:
   - page/block structure
   - KV cache layout
   - allocator behavior
   - attention interface changes
   - compatibility with batching
7. For quantization, define:
   - weight representation
   - operator dispatch
   - calibration or loading assumptions
   - INT8-first and INT4-next rollout plan
8. For heterogeneous MoE, define:
   - expert placement
   - CPU/GPU memory ownership
   - transfer policy
   - runtime scheduling interaction
   - expected latency/throughput tradeoffs
9. Do not write production code yet.
10. End with a proposed implementation order broken into small stages.

---

## Prompt 3: Ask for an implementation plan in PR-sized stages

Now that the design documents exist, create a detailed implementation plan in small, reviewable stages.

Requirements:
1. Break the work into PR-sized stages.
2. For each stage, provide:
   - objective
   - affected files/modules
   - interfaces added/changed/removed
   - dependency on earlier stages
   - correctness tests
   - regression tests
   - benchmarks
   - risks
   - rollback strategy
3. Prefer an order like:
   - simplify model execution path
   - introduce request/session abstraction
   - introduce scheduler
   - continuous batching
   - paged KV cache
   - paged attention operator path
   - HTTP API
   - INT8 quantization
   - INT4 quantization
   - heterogeneous MoE support
   - backend/kernel optimization
4. If you recommend a different order based on the repository, explain why.
5. Write the result to `docs/plan/implementation_plan.md`.

Do not implement anything yet unless explicitly asked.

---

## Prompt 4: Start implementation, but only one stage at a time

Implement only PR-2 from `docs/plan/implementation_plan.md`.

Rules:
1. Before coding, restate:
   - the goal of this stage
   - the files you expect to modify
   - the tests you will add/update
2. Then implement only this stage.
3. Keep the patch focused and minimal.
4. Update relevant docs if the code reveals design adjustments.
5. At the end, provide:
   - summary of code changes
   - tests added/updated
   - any unresolved issues
   - suggested next stage

Do not start the next stage automatically.

---

## Prompt 5: Review current implementation before the next stage

Please review the repository after completing the last stage.

Tasks:
1. Summarize what changed.
2. Verify whether the implementation still matches the design docs.
3. Identify any design drift.
4. Update the relevant docs under `docs/`.
5. Recommend whether to proceed, revise, or rollback before starting the next stage.

Do not implement new features in this step.

---

## Prompt 6: Focus specifically on paged attention design quality

Please deeply review the current or proposed paged attention design.

Evaluate:
- KV page/block size tradeoffs
- memory fragmentation
- allocation/free behavior
- compatibility with continuous batching
- token append behavior
- decode-time efficiency
- prefill vs decode path differences
- attention kernel API impact
- CPU/CUDA backend implications

Then:
1. update `docs/paged_attention_design.md`
2. identify risks
3. propose validation tests and benchmarks
4. do not modify code yet unless asked

---

## Prompt 7: Focus specifically on quantization design

Please deeply review quantization support for this repository.

Goals:
- support INT8 first
- support INT4 next
- avoid polluting the codebase with ad hoc quantized branches
- define clean operator dispatch and weight representation

Tasks:
1. inspect existing model loading and operator paths
2. propose repository-specific quantization integration
3. update `docs/quantization_design.md`
4. define:
   - formats
   - scaling metadata
   - load-time conversion assumptions
   - execution-time dispatch
   - fallback path
   - validation and benchmark plan

Do not code yet.

---

## Prompt 8: Focus specifically on heterogeneous MoE offloading

Please deeply review the design for heterogeneous CPU/GPU MoE inference.

Goals:
- support dynamic offloading of some experts to CPU
- target limited-VRAM GPUs such as 24 GB class devices
- support large MoE inference with INT8/INT4 as a long-term direction

Tasks:
1. inspect current model execution, operator interfaces, and memory ownership assumptions
2. identify what must change to support CPU/GPU mixed expert residency
3. update `docs/heterogeneous_moe_design.md`
4. include:
   - expert placement strategy
   - routing/expert hotness considerations
   - prefetch opportunities
   - pinned memory usage
   - synchronization and overlap
   - scheduler interaction
   - expected performance envelope
   - failure/degradation modes
5. do not implement yet

---

## Prompt 9: Force repository-specific review instead of generic advice

Please review all existing docs under `docs/` and improve them to be repository-specific.

Requirements:
- reference actual source files
- reference actual classes/functions/data structures
- identify exact insertion points for new logic
- identify obsolete abstractions to remove
- avoid generic recommendations
- clearly distinguish "confirmed from code" vs "proposed design"

Do not implement code in this step.

---

## Prompt 10: Pre-implementation audit

Before implementing the next stage, perform a pre-implementation audit.

Tasks:
1. confirm that the target files and interfaces are the right insertion points
2. identify coupling risks
3. identify likely hidden dependencies
4. identify test gaps
5. identify benchmark gaps
6. suggest whether the stage should be split into smaller steps

Write the results to the relevant docs and then wait for approval.

---

## Suggested Interaction Pattern

Recommended workflow:
1. Run Prompt 1
2. Review generated docs
3. Run Prompt 2
4. Review design
5. Run Prompt 3
6. Approve only Stage 1
7. Run Prompt 4
8. Run Prompt 5
9. Repeat stage by stage

This helps prevent uncontrolled repository-wide changes.

---

## Additional Guidance to Use During Conversation

If Claude Code becomes too generic, say:

"Be repository-specific. Reference actual files, classes, and code paths from this repo. Do not give generic inference-system advice."

If Claude Code starts coding too early, say:

"Stop coding. Analysis and design first. Update docs under `docs/` and wait for approval."

If Claude Code proposes a large refactor in one shot, say:

"Split this into smaller PR-sized stages with rollback strategy, tests, and benchmarks for each stage."

If Claude Code ignores performance constraints, say:

"This is a performance-sensitive C++ inference framework. Re-evaluate your design with hot-path overhead, memory movement, and backend dispatch costs in mind."

If Claude Code ignores the no-Python requirement, say:

"Do not introduce Python runtime dependency into the serving path."

If Claude Code over-generalizes the model path, say:

"For already-supported fixed model families, prefer a simpler direct model `forward` path over a redundant graph abstraction unless you can justify the abstraction with measurable value."
