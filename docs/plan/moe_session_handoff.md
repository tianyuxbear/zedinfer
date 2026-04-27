# MoE Session Handoff (Phase 2 in progress)

> Carries state between sessions on `feat/hetero-moe-support`. Update at the end of
> every session that touches the MoE path.

## 1. Current state at a glance

- ✅ **Phase 1**: Qwen3-30B-A3B-GPTQ-Int4 runs end-to-end on NVIDIA GPU, all experts
  GPU-resident. Validated ("Who are you?" → coherent "Qwen..." output).
- ✅ **Cleanup rounds**: code organization, ops layout, dispatch unification,
  `ExpertWeights` indexed storage, per-layer MoE decision, HF-matching topk_softmax,
  `moe_forward.cpp` split into `compute_router_topk` / `moe_decode` / `moe_prefill` /
  `apply_shared_expert`.
- ✅ **Phase 2 runtime infrastructure**: transfer stream in `Runtime`,
  `register_pinned` / `unregister_pinned` API on CPU and NVIDIA backends
  (`nvidia_runtime_api.cu:113`), pre-allocated prefill buffers.
- ✅ **M1 — ExpertPool MVP (ALL_GPU)** — commit `36e73cc`.
- ✅ **M2 — CPU pinned storage + synchronous H2D + `ZEDINFER_MOE_GPU_SLOTS`**.
- ✅ **M3 — Async prefetch on transfer stream + sliding-window**. Async H2D on
  `Runtime::transfer_stream()` guarded by per-slot `cudaEvent_t`; `moe_decode` and
  `moe_prefill` issue prefetch batches before the compute loop (prefill uses a
  sliding window sized to `max_prefetch_depth()` to avoid self-eviction). Clean
  single-card A6000 bench at N=32 on Qwen3-30B-A3B-GPTQ-Int4: prefill 122 → 87.8
  tok/s (1.39× latency), decode 26.3 → 31.1 tok/s (0.84× latency — PINNED_LRU
  actually faster, probably L2 locality + warmup ordering). See
  `heterogeneous_moe.md §6 M3`.
- ⏳ **M4 (optional) — Predictive prefetch** — not needed for current workloads;
  revisit only if a model with extreme fan-out shows under-utilized compute.
- ✅ **D-series — BF16 large-model support**. Loader predicate routes expert tensors
  directly to CPU pinned memory via `Model::load_weights(..., to_cpu_pinned)`.
  `ZEDINFER_MOE_GPU_SLOTS` is now the single user-visible knob (it controls both the
  loader predicate and the pool strategy). `ExpertPool` ctor detects whether experts
  arrived on GPU (legacy) or CPU pinned (new) and picks the right path. BF16
  Qwen3-30B-A3B (~60 GB) validated end-to-end on 24 GB VRAM budget — see
  `heterogeneous_moe.md §10` for the full run log and numbers.

Phase 2 core is functionally done. Remaining open items are polish.

Key code entry points:
- `include/frontend/models/expert_pool.hpp` + `.cpp` — ExpertPool, slot arena, LRU,
  async H2D, sliding-window support via `compute_touched` flag + `pick_lru_slot`
  returning `-1` when no evictable slot.
- `include/backend/device/runtime_api.hpp` — event API (`create_event` /
  `destroy_event` / `record_event` / `stream_wait_event`).
- `include/frontend/models/forward_config.hpp:52` — `ExpertPool* expert_pool`.
- `forward_config.hpp:118-153` — `dispatch_expert_linear` calls `ensure_on_gpu`.
- `src/frontend/models/moe_forward.cpp` — `moe_decode` batch prefetch;
  `moe_prefill` active-experts ordering + sliding window.
- `src/frontend/models/qwen3_moe.cpp` — `choose_pool_config()` parses
  `ZEDINFER_MOE_GPU_SLOTS`; ctor constructs pool.

## 2. Validation strategy

Two models:

1. `Qwen/Qwen3-30B-A3B-GPTQ-Int4` (~15 GB, downloaded)
   - Bit-exact comparison vs. ALL_GPU for correctness.
   - Force offloading via `ZEDINFER_MOE_GPU_SLOTS=N` for `N < num_experts`.
   - Already validated through M2 at `N=32`.

2. `Qwen/Qwen3-30B-A3B` BF16 (~60 GB, user may not have downloaded)
   - Real "doesn't fit on 24 GB GPU" scenario — requires a streaming load path
     (experts read from safetensors mmap directly into CPU pinned, not via GPU).
   - Milestone validation only; ground truth via HuggingFace `transformers`.
   - Also the first test of the dense (non-quantized) `gate_weight` / `up_weight` /
     `down_weight` path through `allocate_slot_tensors` and `dispatch_expert_linear`.

## 3. Pre-existing caveats (still relevant)

1. **Qwen3-30B-A3B `head_dim = 128`** even though `hidden_size / num_heads = 64`.
   The forward loop reads `head_dim` from config.

2. **Qwen3-30B-A3B has no shared expert** despite HF naming. `apply_shared_expert`
   respects `has_shared_expert` (`src/frontend/models/moe_forward.cpp:151`).

3. **GPTQ format quirks**: `qweight` transposed in loader (`[K/8, N]` → `[N, K/8]`);
   zero-point convention adjusted in `process_gptq_group`.

4. **`moe_prefill` per-row memcpy gather**: known perf TODO
   (`moe_forward.cpp:118-125`). A CUDA gather kernel would batch launches. Not on
   critical path for M3.

5. **Router logits D2H sync**: per-MoE-layer `cudaDeviceSynchronize` forced by
   `compute_router_topk`'s copy to CPU. GPU top-k kernel would eliminate it.
   Separate optimization.

6. **M2 peak VRAM**: PINNED_LRU ctor assumes experts arrive on GPU then D2H-migrates.
   Peak briefly holds full expert set on GPU. Enough for INT4 30B on 24 GB; BF16 60B
   requires the streaming-load path (see §5 item 1).

7. *(Resolved)* Shared_ptr cycle in `InferenceEngine` — `ServingLoop` and `Profiler`
   now hold a non-owning `InferenceEngine*`, so the engine destructor runs cleanly
   and the ExpertPool stats log lands in `logs/ping.log` from `~ExpertPool`.

## 4. Build and run

```bash
# Configure + build (clean build after struct layout changes)
xmake f -m release --nv-gpu=y --onednn=y
xmake build

# ALL_GPU (unchanged behavior)
xmake run ping ~/data/models/Qwen3-30B-A3B-GPTQ-Int4 --nvidia

# PINNED_LRU: force offloading
ZEDINFER_MOE_GPU_SLOTS=32 xmake run ping ~/data/models/Qwen3-30B-A3B-GPTQ-Int4 --nvidia

# Edge of per-layer slots: top_k=8 is the minimum before correctness-preserving thrashing
ZEDINFER_MOE_GPU_SLOTS=8 xmake run ping ~/data/models/Qwen3-30B-A3B-GPTQ-Int4 --nvidia

# Iteration shortcut
ZEDINFER_DISABLE_WARMUP=1 xmake run ping ~/data/models/Qwen3-30B-A3B-GPTQ-Int4 --nvidia
```

## 5. Suggested next work (Phase 2 polish, any order)

Phase 2 M1–M3 + BF16 large-model support all landed. These polish items remain but
none is blocking the thesis's core claim (quantization + memory-tiered expert
execution on a single consumer GPU):

1. *(Done)* **Auto N sizing.** When `ZEDINFER_MOE_GPU_SLOTS` is unset on a MoE model,
   `compute_moe_pool_config` (`src/frontend/models/base.cpp`) queries
   `get_memory_info()` and picks ALL_GPU vs PINNED_LRU(N) with a 70%-of-free-VRAM
   expert budget heuristic. The decision happens once in `Model::parse` so the
   loader's CPU-pinned routing predicate and the pool strategy share a single source
   of truth. Verified on Qwen3-30B-A3B-GPTQ-Int4: free=31 GB → budget=22 GB ≥ total
   experts=14 GB → ALL_GPU. Caveat: the "after block_pool init" timing from the
   original note isn't satisfied — KV cache actual footprint isn't known up front
   when the loader needs to decide routing. The 70% heuristic is conservative
   (favors fewer slots over KV starvation); tunable via `kExpertVramFraction`.

2. **Fix the prefill per-row gather** (`moe_forward.cpp:140-147`). PERF-TODO predates
   M3 but is the dominant remaining prefill cost — M3 sliding-window prefetch exposed
   it. A single CUDA gather kernel batching the per-row copies is the obvious fix.
   Helps short-prompt workloads the most (ping 12-token prefill today still shows
   per-layer fixed overhead).

3. **Fiddler-style CPU execution** of cold experts. `docs/plan/heterogeneous_moe.md §9`
   kept this out of Phase 2 scope. Would be the step that actually makes the thesis
   claim "heterogeneous inference" literal (CPU compute + GPU compute) rather than
   "heterogeneous memory" (the current implementation). Revisit if time allows or if
   the thesis framing needs it.

4. **Expand BF16 validation**: cross-check against HuggingFace `transformers` on the
   same model (same prompt, greedy, compare first N token ids). Currently we've only
   verified output is coherent, not that it matches HF token-for-token.

5. **Load chat templates from `tokenizer_config.json` via real Jinja2 evaluator.**
   Currently `ChatTemplate::load` (`src/zedinfer/chat_template.cpp:124-135`) hardcodes
   one ChatML formatter for all Qwen variants and one DeepSeek-R1 formatter. The HF
   chat_template field in `tokenizer_config.json` is a full Jinja2 program with
   `namespace`, slicing, `is` tests, string methods, filters — modern Qwen3 uses it
   for tool calling and thinking-mode. Right approach: vendor minja
   (llama.cpp-style minimal Jinja2 evaluator, header-only ~3k LOC, MIT) and route
   `apply()` through it; delete the hardcoded `default_qwen_chatml` /
   `default_deepseek_r1` once the Jinja path is verified. ROI is low for the thesis
   (none of decode/prefill numbers change) but it's the architecturally correct
   answer and would be a clean post-defense PR. ~1 day work.

## 6. Pre-existing issues worth a future pass

1. *(Resolved)* Non-quantized MoE path — BF16 Qwen3-30B-A3B (D.5) was the first run
   that exercised `allocate_slot_tensors`'s dense branch (the
   `gate_weight`/`up_weight`/`down_weight` fields) and `dispatch_expert_linear`
   falling through to `ops::linear`. Both worked without code changes.

2. **`xmake run` drops env vars.** Observed during bench (see commit messages around
   the `CUDA_VISIBLE_DEVICES` investigation). Workaround: invoke binaries directly
   with `LD_LIBRARY_PATH=$PWD/python/zedinfer:$LD_LIBRARY_PATH`. No plan to change
   xmake's behavior.

## 7. Development rules

- Small, reversible steps. `xmake build && ping` per change, compare text output.
- Preserve the per-layer slot scope — it's the right fit for sequential-layer decode.
- `log_stats()` fires from the destructor now (shared_ptr cycle fixed). Writes to
  `logs/ping.log`. Useful for sanity-checking hit rates after a test.
- At session end, **update this file**.
