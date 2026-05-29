# ZedInfer Architecture

> Single source of truth for the current system design. Updated as the codebase evolves.

---

## Overview

ZedInfer is a C++17 LLM inference engine supporting CPU (x86 AVX-512) and NVIDIA GPU (CUDA) backends. No Python in the serving path.

**Supported models:**
- Qwen2 (DeepSeek-R1-Distill-Qwen-1.5B)
- Qwen3 dense (Qwen3-8B, DeepSeek-R1-0528-Qwen3-8B) and Qwen3-MoE (35B-A3B, expert offloading)
- Qwen3.5 / 3.6 family — dense 27B + MoE 35B-A3B: hybrid linear-attention (GatedDeltaNet) + full-attention,
  3D-MRoPE, GPTQ-Int4, vision-language (Qwen3.5-VL), and MTP speculative decoding.

**Data types:** BF16, FP16, FP32; GPTQ-Int4 weights (packed) for quantized linears and MoE experts.
**Interfaces:** CLI (`bench` / `chat` / `ping` / `ppl` / `batch_bench`), HTTP API (OpenAI-compatible) with Web UI.

> **Hybrid models require FlashInfer.** The GatedDeltaNet scan (`ops::mamba::ssu`) is only built under
> `--flashinfer=y`; without it the op throws at runtime. So the Qwen3.5/3.6 hybrid family must be built with
> `--flashinfer=y` and there is no CPU path for the linear-attention layers. Non-hybrid models (Qwen2/Qwen3)
> run on either backend without FlashInfer.

---

## System Topology

```
HTTP Clients / CLI
       |
  HttpServer (cpp-httplib thread pool)        MultiModalProcessor + VisionTower (vision models)
       |  submit_async()                              |  image embeds
       v                                              v
  ServingLoop (engine thread)
    |-- Scheduler (batch assembly, admission, prefix cache, SSM-snapshot pairing, MTP verify batches)
    |-- transformer_forward()          (dense / Qwen2 / Qwen3 / Qwen3-MoE)
    |-- hybrid_transformer_forward()   (Qwen3.5/3.6: per-layer linear-attn vs full-attn)
    |-- MTPModule                      (speculative draft of t+2, optional)
    |-- PagedForwardContext (KV scatter, paged attention dispatch)
       |
  Operators (ops::*)
    linear      : cuBLAS/cuBLASLt (GPU), oneDNN (CPU); GPTQ-Int4 quantized kernels
    attention   : FlashInfer on supported NVIDIA paged paths, custom paged kernels as fallback
    mamba (ssu/gdn/causal_conv1d) : GatedDeltaNet linear attention (FlashInfer-backed)
    moe         : top-k softmax routing + per-expert FFN via ExpertPool
    rms_norm, rope, mrope (3D), add, swiglu, embedding, argmax, vision_attention
       |
  PagedKVCache                         ExpertPool (MoE weight residency: ALL_GPU / PINNED_LRU)
    BlockPool (static VRAM allocation)  SSMStatePool (per-request recurrent state, hybrid)
    BlockAllocator (per-request blocks) SSMSnapshotCache (prefix-coherent SSM state)
    PrefixCache (cross-request sharing)
       |
  Runtime: MemoryPool, Device API, Tensor
```

---

## Key Components

### InferenceEngine (`include/zedinfer/engine.hpp`)
Resource container and factory. Owns model, tokenizer, sampler, block pool, prefix cache, decode scratch. Exposes `serving_loop()` and `profiler()` for callers.

### ServingLoop (`include/zedinfer/serving_loop.hpp`)
Drives inference: `submit_async()` for HTTP, `generate()` for CLI sessions. Contains the Scheduler and runs `schedule() -> step() -> process_results()` loop.

### Scheduler (`include/zedinfer/scheduler.hpp`)
- Decode-first scheduling with chunked prefill
- Block-based admission control via `BlockAllocator::available_blocks()`
- Prefix cache integration: matches prompt prefix on admission, skips redundant prefill
- `allocate_blocks_for_request()` handles prefix match + allocation + extension

### PagedForwardContext (`include/frontend/models/paged_forward_context.hpp`)
Concrete execution context for `transformer_forward()`. No virtual dispatch (ForwardContext base class removed). Handles:
- KV scatter to blocks (`write_kv`)
- Paged attention dispatch: `attend_decode_single()`, `attend_decode_batched()`, `attend_prefill()`
- GPU block table caching across layers
- FlashInfer CSR metadata construction and cache reuse for decode/prefill

### transformer_forward (`src/frontend/models/transformer_forward.cpp`)
Shared forward loop for the dense / non-hybrid families (Qwen2, Qwen3, Qwen3-MoE), parameterized by
`ModelForwardConfig` (bias, Q/K norm flags, MoE routing). Takes `PagedForwardContext&` directly. When
`DecodeScratch*` is provided and N=1, uses pre-allocated buffers (zero Tensor::create per decode step).
The Qwen3.5/3.6 hybrid family uses `hybrid_transformer_forward` instead (see below).

### DecodeScratch (`include/frontend/models/decode_scratch.hpp`)
Pre-allocated GPU buffers for single-token decode (~400KB). Eliminates ~500 `Tensor::create`/`BestFitMemoryPool` round-trips per decode step.

### BlockPool (`include/backend/kvcache/block_pool.hpp`)
Fixed-size KV cache block pool. Features:
- O(1) stats via `free_count_` / `evictable_count_` counters
- Reference counting: `share()` / `release()` for prefix caching
- LRU eviction for memory pressure
- Content hash + immutable flags for cached prefix blocks

### PrefixCache (`include/backend/kvcache/prefix_cache.hpp`)
Chain-hashed prefix matching. When multiple requests share the same token prefix, KV blocks are shared (ref_count > 1) instead of recomputed.

### InferenceRequest (`include/zedinfer/request.hpp`)
Unified ownership model: `borrow_block_table()` (session mode) or `own_block_table()` (batch mode). Single `block_table()` accessor, no dual-path branching.

### Profiler (`include/zedinfer/profiler.hpp`)
Warmup and benchmarking using the paged path (same kernels as actual serving). Uses DecodeScratch for decode measurement.

### hybrid_transformer_forward (`src/frontend/models/hybrid_transformer_forward.cpp`)
Forward loop for the Qwen3.5/3.6 hybrid family. Per layer it dispatches to either a GatedDeltaNet
linear-attention layer (`forward_linear_attn_layer`, backed by `ops::mamba`) or a full softmax-attention
layer (`forward_full_attn_layer`), driven by `ModelConfig::layer_types`. Shares MoE/MLP dispatch with the
dense path via `moe_layer_forward`. (Note: this and `transformer_forward` currently duplicate the shared
skeleton — see `docs/plan/forward-unification.md`.)

### ExpertPool (`include/frontend/models/expert_pool.hpp`)
Manages GPU residency of MoE expert FFN weights. Two strategies: `ALL_GPU` (all experts resident) and
`PINNED_LRU` (N experts/layer on GPU, rest in CPU pinned memory, migrated on demand) so large MoE models
(35B-A3B) fit in 24 GB. Sized automatically from VRAM budget or forced via `ZEDINFER_MOE_GPU_SLOTS`.

### SSMStatePool / SSMSnapshotCache (`src/frontend/models/ssm_state_pool.cpp`, `include/backend/kvcache/ssm_snapshot_cache.hpp`)
Hybrid (Qwen3.5) only. `SSMStatePool` holds per-request GatedDeltaNet recurrent state + causal-conv window
(allocated per admitted request). `SSMSnapshotCache` persists per-prompt SSM state so PrefixCache reuse stays
coherent for linear-attention layers — the scheduler only honors a prefix hit when the SSM snapshot restores.

### MTPModule (`src/frontend/models/mtp_module.cpp`)
Optional Multi-Token-Prediction head for speculative decoding (Qwen3.5). After each main forward it drafts
t+2; the next scheduler step issues a 2-token verify batch. Per-request MTP K/V state lives on `InferenceRequest`.
Off by default; enabled via `--mtp` (or `ZEDINFER_MTP_SPEC` / `ZEDINFER_MTP_DEBUG`).

### Vision pipeline (`src/frontend/models/vision_tower.cpp`, `src/zedinfer/multimodal_processor.cpp`)
Qwen3.5-VL only. `MultiModalProcessor` decodes image data URIs and computes patch grids; `VisionTower` runs the
ViT + spatial merger to produce image embeddings that are scattered into `<|image_pad|>` positions to form the
layer-0 input embeddings consumed by `hybrid_transformer_forward`.

### GPTQ-Int4 loading (`src/frontend/models/base.cpp`)
The loader detects GPTQ checkpoints, transposes/repacks `qweight`, and adjusts packed nibbles to the kernel's
zero-point convention (`detect_gptq_zero_point`, `transpose_gptq_qweight`). Quantized linears dispatch through
`ModelForwardConfig::dispatch_linear` / `dispatch_expert_linear` to `ops::linear_quantized`.

---

## Operator Backends

| Operator | CPU | GPU |
|----------|-----|-----|
| linear | oneDNN GEMM (BF16/FP32 native) | cuBLAS/cuBLASLt (auto-tuned, fused bias) |
| linear (GPTQ-Int4) | scalar/vectorized dequant matmul | packed-int4 kernels |
| attention (decode) | Paged GQA, OMP parallel | FlashInfer when enabled and supported; otherwise custom paged kernel |
| attention (prefill) | Paged GQA, causal mask | FlashInfer when enabled and supported; otherwise custom paged kernel |
| attention (batched decode) | Loop over single decode | FlashInfer when enabled and supported; otherwise custom batched paged kernel |
| mamba (ssu / gdn / causal_conv1d) | causal_conv1d only | GatedDeltaNet scan — **FlashInfer-only** (`ssu` throws without it) |
| moe (top-k softmax, expert FFN) | — | top-k softmax + per-expert GEMM via ExpertPool |
| rms_norm | AVX-512 vectorized | Vectorized 128-bit packed, warp+block reduction |
| rope | CPU scalar + OMP | Per-token CUDA kernel |
| mrope (3D, Qwen3.5) | CPU scalar | Per-token CUDA kernel |
| vision_attention | — | dense (non-paged) attention for the ViT |
| add, swiglu | AVX-512 vectorized | Vectorized 128-bit packed |
| embedding | OMP memcpy | Vectorized lookup |
| argmax | OMP reduction | Block-scope reduction, pre-allocated pinned host buffer |

> Some ops are NVIDIA-only by design (moe, vision_attention, mamba::ssu/gdn, kv_scatter); the hybrid and
> vision model families therefore require the NVIDIA backend (and FlashInfer for the linear-attention scan).

---

## Build System

- **Tool:** XMake (`xmake.lua`)
- **C++ Standard:** C++17
- **GPU:** Optional (`--nv-gpu=y`), links cuBLAS/cuBLASLt
- **FlashInfer:** Optional (`--flashinfer=y`), source tracked in `third_party/flashinfer`
- **CPU GEMM:** Optional oneDNN (`--onednn=y`)
- **Targets:** Static libraries (zedinfer, frontend, backend, ops, etc.) + example binaries (`bench`, `chat`, `ping`, `ppl`, `serve`, `batch_bench`)

---

## Test Suite

~24 test binaries (run via `xmake run <name>`). Tests that need real weights are gated by
`ZEDINFER_TEST_MODEL_PATH` and skip cleanly when unset:
- Core infra: `test-memorypool`, `test-storage`, `test-tensor`
- KV cache: `test-blockpool`, `test-prefixcache`
- Sampling / templates: `test-sampler`, `test-chattemplate`, `test-chat-template-jinja`
- Frontend I/O (model path): `test-tokenizer`, `test-loader`, `test-models`
- Qwen3.5 hybrid: `test-qwen3-5-config-parse`, `test-qwen3-5-weight-name-map`, `test-qwen3-5-load`,
  `test-hybrid-forward-config`, `test-ssm-state-pool`
- Ops (NVIDIA): `test-ops-mamba-ssu`, `test-ops-causal-conv1d`, `test-ops-mrope-3d`,
  `test-ops-attn-output-gate`, `test-gdn`, `test-flashinfer-ssu-link`
- Smoke: `test-minja-smoke`, `test-stb-smoke`

---

## Environment Variables

Runtime/debug toggles read via `getenv`. None are required for normal operation; they exist for tuning,
A/B testing, and debugging. Resolved once per process (immutable at runtime).

| Variable | Effect |
|----------|--------|
| `ZEDINFER_MOE_GPU_SLOTS` | Force MoE ExpertPool strategy: `>= num_experts` → ALL_GPU, else N PINNED_LRU slots/layer |
| `ZEDINFER_REPETITION_PENALTY` | Override the repetition penalty from `generation_config.json` |
| `ZEDINFER_GPTQ_ZEROPOINT` | Override auto-detected GPTQ zero-point (debug GPTQ checkpoints) |
| `ZEDINFER_MTP_SPEC` | Enable MTP speculative decoding (active spec mode), same as `--mtp` |
| `ZEDINFER_MTP_DEBUG` | Log MTP drafts without committing them (observer mode) |
| `ZEDINFER_FORCE_ARGMAX` | Force greedy/argmax sampling regardless of generation config |
| `ZEDINFER_DISABLE_FLASHINFER` | Fall back to the custom paged-attention kernels |
| `ZEDINFER_FLASHINFER_DISABLE_FASTPATH` | Disable the FlashInfer single-decode fast path |
| `ZEDINFER_DISABLE_SCRATCH` | Disable DecodeScratch pre-allocation (forces per-step Tensor::create) |
| `ZEDINFER_DISABLE_SMALL_NQ` | Disable the small-batch decode attention optimization |
| `ZEDINFER_DISABLE_WARMUP` | Skip the profiler warmup pass |
| `ZEDINFER_DISABLE_LOAD_PROGRESS` | Disable the model-load progress bars |
| `ZEDINFER_DUMP_TOKEN_IDS` | Dump every committed token id to stderr (debug) |
| `ZEDINFER_DUMP_TOP_LOGITS` | Dump top-5 logits per sampled token to stderr (debug) |
| `ZEDINFER_TEST_MODEL_PATH` | Model path used by the model-dependent tests |
