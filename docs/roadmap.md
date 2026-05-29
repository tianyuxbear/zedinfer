# ZedInfer Roadmap

> Implementation status and future plans. Reference for continuing development.

---

## Completed Features

### Core Infrastructure
| Feature | PR/Commit | Key Files |
|---------|-----------|-----------|
| Direct model forward (no graph) | PR-2 | `transformer_forward.cpp` |
| cuBLAS for GPU linear | PR-3a | `linear_cublas.cu` |
| oneDNN for CPU linear | PR-3b | `linear_cpu.cpp` |
| Stateless engine + request types | PR-4 | `engine.hpp`, `request.hpp` |
| Configurable chat template | PR-5 | `chat_template.hpp/.cpp` |
| Scheduler (single-request) | PR-6 | `scheduler.hpp/.cpp` |
| Paged KV cache + block pool | PR-7 | `block_pool.hpp/.cpp` |
| Paged attention kernels | PR-8 | `paged_attention_nvidia.cu` |
| Continuous batching | PR-9 | `scheduler.cpp`, `batch_context.cpp` |
| HTTP API + Web UI + SSE | PR-10 | `http_server.cpp`, `web/index.html` |
| FlashInfer paged attention backend | post-PR-10 | `paged_forward_context.cpp`, `flashinfer_wrapper.cu` |

### Qwen3.5 / Qwen3.6 Family (v0.2.0 → present)

Full enablement of the Qwen3.5/3.6 hybrid-MoE-VL family (dense 27B + MoE 35B-A3B), plus the GPTQ-INT4, MoE-offload, and heterogeneous-inference items that were previously "upcoming".

| Area | Description | Key Files |
|------|-------------|-----------|
| Hybrid forward | Per-layer dispatch: GatedDeltaNet linear-attn + full-attn + MoE/dense FFN | `hybrid_transformer_forward.cpp` |
| Linear attention | GDN decode/prefill kernels, `causal_conv1d`, `qk_l2norm`, SSU; `SSMStatePool` + `SSMSnapshotCache` | `ops/mamba/`, `ssm_state_pool.cpp` |
| MoE + expert offload | `ExpertPool` + PINNED_LRU host staging (35B-A3B INT4 on 24GB), GPU top-k, fused 3-D expert load, auto GPU-slot sizing | `expert_pool.cpp`, `moe_forward.cpp` |
| GPTQ INT4 | int4 weight-only quantized linear / expert dispatch (w4a8 matvec) | `linear` quantized path |
| Vision / multimodal | Qwen3.5-VL vision tower + CUDA kernels, multimodal processor, `--image`, OpenAI multimodal HTTP | `vision_tower.cpp`, `multimodal_processor.cpp` |
| MTP speculative decode | MTP head (dense + MoE), per-request K/V, SSM temp-slot reject, true rejection sampling; opt-in `--mtp` | `mtp_module.cpp`, `scheduler.cpp` |
| Positional / gates | 3D mRoPE (interleaved), `attn_output_gate`, `shared_expert_gate` | `ops/...` |
| Chat / reasoning | minja Jinja templates, thinking-budget force-emit, open/closed-think | `chat_template_jinja.cpp` |

See `docs/debug/mtp_correctness_dense_and_deployment.md` for the MTP findings (when it nets a speedup) and `docs/perf/moe_offload_24g.md` for the 24GB offload validation.

### Refactoring (R1-R6)
| Refactor | Description |
|----------|-------------|
| R1: Unified forward | 4 forward files -> 1 `transformer_forward.cpp` |
| R2: Unified KV management | Session + batch mode -> single block table path |
| R3: Clean KVCache interface | Removed KVCache base class, DynamicKVCache standalone |
| R4: Unified attention dispatch | 4 attention functions -> 1 `ops::attention(AttentionParams)` |
| R5: Split engine | Engine -> Engine + ServingLoop + Profiler |
| R6: Dead code removal | Graph execution code deleted |

### Recent Optimizations
| Optimization | Impact | Key Change |
|-------------|--------|------------|
| ArgmaxSampler pinned buffer | Eliminated 570us/call cudaMallocHost | Pre-allocated host buffer |
| Paged decode kernel smem optimization | ~16% decode improvement | Precomputed physical addresses in shared memory |
| Prefix caching | Skip redundant prefill for shared prefixes | Block ref counting + chain hash |
| DecodeScratch pre-allocation | Eliminates ~500 Tensor::create/step | Fixed-address decode buffers |
| BlockPool O(1) counters | Fast admission control | Incremental free/evictable counters |
| FlashInfer paged attention path | NVIDIA decode/prefill can route to FlashInfer with legacy fallback | CSR metadata + wrapper dispatch |

### Code Health
| Cleanup | Description |
|---------|-------------|
| Removed cuDNN integration | Only used in warmup, not serving |
| Removed ContiguousForwardContext | Warmup now uses paged path |
| Removed DynamicKVCache + KVCache base | No longer needed |
| ForwardContext de-virtualization | PagedForwardContext used directly |
| Request ownership simplification | `borrow_block_table()` / `own_block_table()` API |
| Scheduler decomposition | Extracted `allocate_blocks_for_request()` |
| Engine slimming | Delegation methods -> `serving_loop()` / `profiler()` accessors |
| attend() decomposition | Split into decode_single/decode_batched/prefill |
| ensure_blocks extraction | Shared by scheduler and profiler |
| CPU memcpy kind fix | Accept all memcpy kinds on CPU |
| Dead-code removal (review pass) | `NaiveAllocator`, unused `full_attention_interval`, no-op `fixup_qwen3_5_rmsnorm_weights` |
| Scheduler concurrency fix (review pass) | Idle-wait CV moved into Scheduler under `submit_mutex_`; status queries locked — no data race / lost wakeup |
| Session UAF fix (review pass) | `delete_session` defers reclamation of a busy (mid-stream) session |
| `Tensor::to()` guard (review pass) | Rejects strided/gappy views that would silently mis-copy under preserved strides |
| Hot-path getenv hoist (review pass) | `ZEDINFER_DUMP_TOKEN_IDS` / `DUMP_TOP_LOGITS` / `MTP_*` resolved once, not per token |
| Expert-parse dedup (review pass) | Shared `parse_expert_tensor_name` / `assign_expert_tensor` in `expert_weights.cpp` |
| Docs refresh (review pass) | `architecture.md` updated for Qwen3.5/3.6 + MoE/MTP/vision/GPTQ; env-var table; FlashInfer prerequisite |
| `SequenceBlockTable` decoupling (review pass) | FlashInfer plan/index cache grouped into `FlashInferSeqCache fi`; `clear_runtime_caches()` is one assignment; copy carries only logical state. Validated byte-identical across single-decode / batched-decode (batch_bench seeded) / prefill (ppl). See `docs/plan/refactor_sequence_block_table.md` |

---

## Upcoming Work

### Priority 1: FlashInfer Optimization And Coverage

**Current status:** FlashInfer is already integrated as an optional NVIDIA paged-attention backend behind `--flashinfer=y`. The legacy paged CUDA kernels remain in-tree as the fallback path.

**Near-term focus:**
- Reduce planner and workspace overhead on common decode shapes
- Expand native decode-kernel coverage so fewer shapes fall back to the FlashInfer prefill kernel
- Reuse more per-context metadata across repeated bench/serve iterations
- Continue measuring against the legacy kernels with runtime A/B toggles

**Reference:** `docs/guide/flashinfer.md`

### Priority 2: CUDA Graph (Phase 2)

**Goal:** Capture decode forward pass as CUDA graph, replay with single launch. Saves ~2ms/step kernel launch overhead.

**Status:** Phase 1 (DecodeScratch) complete. Phase 2 (capture/replay) not started.

**Consideration:** Not compatible with MoE/heterogeneous inference. Keep implementation simple, don't over-invest.

**Plan:** `docs/plan/cuda_graph.md`

> **Done since v0.2.0** (was Priorities 4–6): GPTQ **INT4** quantization, **MoE expert offloading** (`ExpertPool` PINNED_LRU, 35B-A3B on 24GB), and **heterogeneous CPU/GPU** inference (pinned host experts + async prefetch overlap) all shipped with the Qwen3.5/3.6 enablement above. INT8 was skipped (INT4 supersedes it for the VRAM-limited target).

### Priority 3: Fused MoE GEMM (ALL-GPU only)

**Goal:** Replace the per-token, per-expert int4 matvec loop (launch-bound: ~990 tiny launches/token, GPU ~70% idle) with a grouped int4 expert GEMM. Raises the MoE decode baseline and lets MTP's n_q=2 verify amortize.

**Constraint:** Requires experts resident (ALL_GPU). **Mutually exclusive with PINNED_LRU offload** (it breaks the sliding-window prefetch/compute overlap), so it applies to ≥32GB GPUs, not the 24GB offload path. FlashInfer ships a usable W4-group-scaled cutlass fused MoE (`use_w4_group_scaling`, C++ runner) — integration = build kernels per arch + repack GPTQ → cutlass layout.

### Priority 4: Wider INT4 coverage

**Goal:** Quantize the still-bf16 parts (linear-attn / embed / lm_head — currently ~7GB on 35B-A3B, ~20GB on dense 27B) to fit the **dense 27B in 24GB ALL_GPU**, which would make MTP a net speedup on a 4090. Risk: quantizing GatedDeltaNet recurrent state may hurt quality.

### Priority 5: Better MTP draft head / offload H2D

**MTP draft quality:** accept rate is workload-dependent (~10–64%) and capped by the 1-layer head; a stronger draft head raises the dense-27B win beyond +11.5%. **Offload H2D:** the only lever for the 24GB-4090 35B-A3B path (transfer coalescing, prefetch depth, LRU hit rate) — orthogonal to MTP/fused-MoE.

---

## Known Technical Debt

| Issue | Severity | Blocked by |
|-------|----------|-----------|
| Some NVIDIA shapes still fall back to legacy paged attention | Medium | Unsupported FlashInfer runtime shape |
| Some decode shapes use FlashInfer prefill kernel fallback instead of native decode | Medium | Wider decode-kernel coverage |
| Non-contiguous mixed prefill slices still fall back to legacy per-slot prefill | Medium | Better prefill packing / metadata reuse |
| Operator correctness coverage is still incomplete | Medium | More parity tests across operators and serving flows |
| No scheduler/serving integration tests | Medium | — |
| build_decode_cache per-context rebuild | Low | More metadata/cache reuse across iterations |
| Two parallel forward loops (`transformer_forward` vs `hybrid_transformer_forward`) duplicate the skeleton | Medium | Design: `docs/plan/forward-unification.md` |
| `forward_config()` rebuilt every decode step (re-resolves `router_weights` etc. via string lookups) | Low (measured) | Measured 0.03% of a decode step on MoE, ~0% dense — deprioritized; GEMMs dominate. See `docs/plan/per_layer_weight_resolution.md` |
| Op dispatch keeps an unreachable `case ZEDINFER_DEVICE_CPU` after an early CPU return (~11 ops) | Low | Cosmetic; harmless dead branch |

---

## Performance Reference

**Machine:** B200 180GB VRAM
**Model:** DeepSeek-R1-Distill-Qwen-1.5B (BF16)

| Config | Metric | Value |
|--------|--------|-------|
| Single decode (paged, p=128 d=128) | Decode throughput | ~129 tok/s |
| Batch=4 | Total throughput | ~672 tok/s |
| Batch=32 | Total throughput | ~1,820 tok/s |
| Batch=128 | Total throughput | ~2,170 tok/s |

**Bottleneck:** NVIDIA paged attention remains the primary optimization target. FlashInfer is integrated, and further work is focused on planner overhead, coverage, and fallback reduction.
