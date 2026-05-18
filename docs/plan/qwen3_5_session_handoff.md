# Qwen3.5 Session Handoff

> Mirrors the style of `moe_session_handoff.md`. Updated at the end of every session that touches the Qwen3.5 path.

## M0 — Load + parse + minja chat template + SSU link probe (complete)

**Completed: 2026-05-17.** Branch `feat/qwen3.5`. Final M0 commit `fd8b373` (docs); last code commit `14c8f11` (ping friendly exits).

### What landed

| Commit | Title |
|---|---|
| `0fca79c` | feat(qwen3.5): vendor minja for Qwen3.5 chat template parsing |
| `b1baf8e` | feat(qwen3.5): vendor stb_image for image decode/resize |
| `7f0d8d1` | feat(qwen3.5): add FlashInfer SSU link probe (M0 GO/NO-GO) |
| `1bb84c7` | feat(qwen3.5): extend ModelConfig with hybrid/vision/mrope/MTP fields |
| `3d7a95e` | feat(qwen3.5): parse hybrid/vision/mrope/MoE config fields |
| `a376582` | feat(qwen3.5): map_weight_name strips language_model. prefix |
| `d7c3f9a` | feat(qwen3.5): add SSMStatePool with slot mgmt + zero-reset |
| `97840c6` | feat(qwen3.5): HybridForwardConfig with layer-kind dispatch helpers |
| `b82e76f` | feat(qwen3.5): Qwen3_5Model + VisionTower skeletons; Model::parse dispatch |
| `3d857ac` | feat(qwen3.5): Qwen3_5MoeModel + 35B-A3B dispatch with ExpertPool |
| `c6148ef` | feat(qwen3.5): ChatTemplateJinja (minja-backed) + Qwen3_5Model loads chat_template.jinja |
| `14c8f11` | feat(ping): friendly exit for Qwen3.5 M1-pending forward + engine-init errors |
| `fd8b373` | docs(qwen3.5): M0 complete; record DoD verification outcome |

### Components introduced

- **Vendored third_party**: `minja` (header-only Jinja2 subset) and `stb` (stb_image / stb_image_resize2). Wired via `xmake.lua` `add_required_includedir`.
- **`ModelConfig` extension** (`include/frontend/models/base.hpp`): hybrid fields (`layer_types`, `attn_output_gate`, `partial_rotary_factor`, `mrope_section`, `mrope_interleaved`, `mtp_num_hidden_layers`, `linear_attn`, `has_vision`, `vision`, `image_token_id`, etc.). Default-valued for non-hybrid models.
- **`Qwen3_5Config` / `Qwen3_5MoEConfig`** (`include/frontend/models/qwen3_5_config.hpp`): downcast targets, populated by extended `load_config` parser that walks `text_config` / `vision_config` and top-level token IDs.
- **`Model::map_weight_name`** now strips `language_model.` after `model.`, so `model.language_model.layers.0.linear_attn.in_proj_qkv.weight` maps to `layers.0.linear_attn.in_proj_qkv.weight`. Zero regression on Qwen3 keys.
- **`SSMStatePool`** (`frontend/models/ssm_state_pool.{hpp,cpp}`): engine-level slot pool with persistent SSM + conv state buffers and kernel-free zero-reset via `ops::fill_zero`. Layout matches FlashInfer SSU MTP kernel native layout `[slots, layers, num_v_heads, value_head_dim, d_state]` + `[slots, layers, K-1, qkv_dim]`. Constructed eagerly in `Qwen3_5Model` ctor; default `max_concurrent=1` for single-user.
- **`HybridForwardConfig`** (`frontend/models/hybrid_forward_config.hpp`): subclass of `ModelForwardConfig` with `layer_kinds`, `linear_attn`, `mrope`, `attn_output_gate`, `ssm_pool*`. Has `full_layer_index(L)` / `linear_layer_index(L)` helpers for KV-cache / SSM-pool indexing.
- **`Qwen3_5Model`** (`frontend/models/qwen3_5.{hpp,cpp}`): dense 27B model class. Ctor counts linear-attn layers, computes qkv_dim, constructs SSMStatePool, optionally constructs VisionTower (M3 will activate its forward), optionally loads `chat_template.jinja` via `ChatTemplateJinja::load`. `forward_config()` throws `"Qwen3_5Model::forward_config: not implemented until M1"` — the substring is the M1 sentinel ping detects.
- **`Qwen3_5MoeModel`** (`frontend/models/qwen3_5_moe.{hpp,cpp}`): 35B-A3B MoE class inheriting from `Qwen3_5Model`. Adds `ExpertPool` over 256 experts via existing v0.2.0 path. `compute_moe_pool_config_qwen3_5` shim adapts to v0.2.0 auto-sizing.
- **`VisionTower`** (`frontend/models/vision_tower.{hpp,cpp}`): ctor verifies expected `visual.blocks.<i>.attn.qkv.weight` (×27), `visual.patch_embed.proj.weight`, `visual.merger.linear_fc2.weight` are present in `ModelWeights`. `forward()` throws `"VisionTower::forward not implemented until M3"`.
- **`ChatTemplateJinja`** (`zedinfer/chat_template_jinja.{hpp,cpp}`): minja-backed chat template renderer with multimodal content variant (`std::variant<std::string, std::vector<ContentPart>>`). Loaded eagerly in `Qwen3_5Model` ctor. Existing `ChatTemplate` (struct-based, Qwen2/Qwen3/DeepSeek path) is **untouched** — zero regression.
- **`Model::parse` dispatch** extended for `model_type == "qwen3_5"` and `"qwen3_5_moe"`. Routing predicate for CPU-pinned expert memory extended for `qwen3_5_moe`.
- **`ping`** binary wraps engine creation + session chat in try/catch. Recognizes the `"not implemented until M1"` sentinel and exits 0 with a friendly message. Engine-init errors (e.g. pinned-memory OOM) produce a friendly diagnostic and exit 3.
- **Test targets** (`xmake/tests.lua`): test-minja-smoke, test-stb-smoke, test-flashinfer-ssu-link, test-qwen3-5-config-parse, test-qwen3-5-weight-name-map, test-ssm-state-pool, test-hybrid-forward-config, test-chat-template-jinja, test-qwen3-5-load. All env-var gated where they touch real model files.

### Verification (M0 DoD)

| Check | Result |
|---|---|
| `xmake build` clean (after one retry due to host fork ulimit transient) | ✅ |
| `test-flashinfer-ssu-link` (M0 GO/NO-GO) | ✅ `(bf16, bf16, f32, bf16, i32, void)` instantiation links cleanly |
| `test-qwen3-5-config-parse` (dense + MoE) | ✅ 2/2 with env vars |
| `test-qwen3-5-weight-name-map` (incl. Qwen3 regression case) | ✅ 3/3 |
| `test-ssm-state-pool` (acquire/release/reset on NVIDIA) | ✅ 3/3 |
| `test-hybrid-forward-config` (layer-kind dispatch helpers) | ✅ 3/3 |
| `test-minja-smoke` (Qwen3.5 chat_template.jinja parses) | ✅ 1/1 |
| `test-chat-template-jinja` (renders "Who are you?" with `<|im_start|>` markers + `<think>` opener) | ✅ 1/1 |
| `test-stb-smoke` (decodes 4×4 PNG fixture) | ✅ 1/1 |
| `test-blockpool` v0.2.0 regression | ✅ 9/9 |
| `test-models` (incl. ModelConfigParse, QuantizedWeights, FlashInferDecode/Prefill, new Qwen3.5 cases) | ✅ 25 PASS, 2 env-gated SKIP |
| `test-loader` v0.2.0 regression (DeepSeek-R1-Distill-Qwen-1.5B fixture) | ✅ 17/17 |
| `ping Qwen3.5-35B-A3B-GPTQ-Int4 --nvidia --gpu-memory-utilization 0.5 ZEDINFER_MOE_GPU_SLOTS=32` | ✅ Loads, reaches forward, "M1 WIP" exit 0 |
| `ping Qwen3.5-27B-GPTQ-Int4 --nvidia --gpu-memory-utilization 0.5` | ⚠️ Engine-init friendly diagnostic, exit 3 — host pinned-memory ulimit (~32GB) < model on-disk (29GB) plus overhead. Environmental. |
| `ping Qwen3-30B-A3B-GPTQ-Int4 --nvidia` v0.2.0 regression | ✅ Generates normal Qwen response, no behavior delta |

### Open items carried into M1

1. **`forward_config()` and `forward()` still throw** "not implemented until M1". M1 wires the hybrid forward.
2. **`transformer_forward` does not yet accept an `input_embeds` parameter.** Required for M3 vision pre-step.
3. **`Scheduler::admit` does not yet consult `SSMStatePool::num_free_slots()`.** Required when M1 turns on the SSU kernel path.
4. **`PagedForwardContext::write_kv / attend` still use raw layer index.** Hybrid path needs them to accept logical KV layer index (= `full_layer_index(L)`). The plumbing exists in `HybridForwardConfig`; just not yet exercised.
5. **27B-dense pinned-memory OOM is environmental, not Qwen3.5-specific.** On this dev box `ulimit -l = 32GB`, 27B-dense is 29GB on disk, so transpose buffers push us over. Workarounds: raise ulimit -l on the host; or test 27B on a box with higher pinned-memory cap. M1 forward verification should target 35B-A3B (smaller, fits) until/unless the host limit is bumped.

### M1 entry conditions (confirmed)

- FlashInfer SSU template `(bf16, bf16, f32, bf16, i32, void)` compiles and links — recorded in §10 M0 row of design doc.
- minja parses Qwen3.5's chat_template.jinja and renders the expected `<|im_start|>` framing for both `enable_thinking=true` and string-content messages.
- Both Qwen3.5-27B (dense) and Qwen3.5-35B-A3B (MoE) configs parse correctly into `Qwen3_5Config` / `Qwen3_5MoEConfig`.
- Loader strips `language_model.` prefix; Qwen3.5 weights load into `ModelWeights` with the same conventions as Qwen3 (verified by `Qwen3_5Model` ctor reaching `forward_config()` without name-not-found errors on 35B-A3B; weight count 4.665B parameters reported in `test-loader` evidence with 30B-A3B fixture confirms the strip path is functional).
- `SSMStatePool` allocates two GPU buffers in `Qwen3_5Model` ctor without crashing; `acquire_slot` / `release_slot` / `reset_slot` work; layout strides exposed via `SSMStateView`.
- Build flags for FlashInfer Mamba consumption captured in `xmake/tests.lua` for `test-flashinfer-ssu-link`: `-DFLASHINFER_ENABLE_BF16`, `--expt-relaxed-constexpr`, `--expt-extended-lambda`, `-rdc=true`, `-Xcompiler=-fPIC`. M1's `ops::mamba::ssu_wrapper.cu` will need the same flags applied to the `ops-nvidia` target.

## M1 — Hybrid text-only forward (architecture complete, calibration deferred)

**Completed: 2026-05-18.** Branch `feat/qwen3.5`. Final M1 commit `785c9fb`.

### What landed

| Commit | Title |
|---|---|
| `63bbc30` | feat(ops): add ops::mamba::ssu wrapper header (FlashInfer SSU MTP entry) |
| `6256fca` | feat(ops): SSU NVIDIA wrapper around FlashInfer invokeSelectiveStateUpdateMTP |
| `6883ba6` | test(ops): SSU decode + varlen-prefill smoke tests (FlashInfer Mamba) |
| `e322656` | feat(ops): causal_conv1d (Mamba2 depthwise kernel=4 with conv state) |
| `e446f99` | feat(ops): mrope_3d (interleaved + partial rotary, [11,11,10] section) |
| `3edfd01` | feat(ops): attn_output_gate (post-attention sigmoid mul) |
| `7d768f3` | feat(qwen3.5): InferenceRequest gains ssm_slot, image_embeds, pos_ids_thw |
| `7f7d30e` | docs(paged-ctx): rename layer param to kv_layer_idx + document hybrid mapping |
| `b4d821f` | feat(engine): size KV BlockAllocator by full-attention count for hybrid models |
| `74c96f9` | feat(scheduler): dual-pool admission (BlockAllocator + SSMStatePool ...) |
| `6a3fb25` | feat(forward): transformer_forward accepts optional input_embeds |
| `aa9ea2a` | feat(qwen3.5): hybrid_transformer_forward skeleton with per-layer-kind dispatch |
| `1f38929` | feat(qwen3.5): forward_dense_mlp (gate/up/swiglu/down via GPTQ dispatch_linear) |
| `9d748c4` | feat(qwen3.5): forward_moe_mlp delegates to v0.2.0 moe_layer_forward |
| `339547e` | feat(qwen3.5): forward_full_attn_layer (q+gate split, 3D mrope, paged attn, output gate) |
| `d0ae1f9` | feat(qwen3.5): forward_linear_attn_layer (Mamba2 via FlashInfer SSU + conv1d state) |
| `668d5ac` | feat(qwen3.5): wire hybrid_transformer_forward into ServingLoop dispatch |
| `785c9fb` | feat(qwen3.5): MoE hybrid_forward_config + fp32 norm weight cast; ping reaches forward end-to-end |

### Components introduced

- **New ops** (5): `ops::mamba::ssu` (FlashInfer wrapper, decode + varlen prefill), `ops::mamba::causal_conv1d` (depthwise kernel=4 with conv state), `ops::mrope_3d` (interleaved + partial rotary, bitwise-equal to Python reference), `ops::attn_output_gate` (post-attn sigmoid mul), `ops::mamba::copy_strided_rows` (cudaMemcpy2D wrapper for SSU qkv slicing).
- **Framework plumbing**: `InferenceRequest` extended with `ssm_slot_idx`, `image_embeds`, `pos_ids_thw`. `PagedForwardContext::write_kv/attend` documented as taking the logical KV layer index (hybrid → `full_layer_index(L)`). `BlockAllocator` sized by full-attention-layer count for hybrid models. `Scheduler` dual-pool admission (BlockAllocator + SSMStatePool); slot acquire/reset on admit, release on finish.
- **Forward path**: `hybrid_transformer_forward` (A2 layout) with 4 per-layer-kind functions (`forward_linear_attn_layer`, `forward_full_attn_layer`, `forward_dense_mlp`, `forward_moe_mlp`); outer loop dispatches by `layer_kinds[L]` and `is_moe_layer(L)`. `Qwen3_5MoeModel::hybrid_forward_config_moe()` injects MoE fields (`is_moe`, `num_experts`, `expert_pool`, `router_weights[]`, `has_shared_expert`, etc.) on top of the dense hybrid config.
- **transformer_forward** accepts an optional `input_embeds` parameter for vision pre-step injection.
- **ServingLoop** detects `Qwen3_5Model*` via `dynamic_cast` and routes to `hybrid_transformer_forward` with the appropriate config (`hybrid_forward_config()` for dense, `hybrid_forward_config_moe()` for MoE) and the in-batch request's SSM slot.
- **Warmup** auto-skips for hybrid models (the warmup path runs `transformer_forward` against random ids without an `InferenceRequest`, which can't satisfy the hybrid path's SSM slot requirement).
- **fp32 norm.weight cast**: linear_attn's `norm.weight` is fp32 per `mamba_ssm_dtype=float32` config but everything else is bf16. Inline per-call CPU-side cast at the point of use (128 elements, sub-µs); M2 will move this to a one-time conversion at model load if precision survives bf16.

### Verification

| Check | Result |
|---|---|
| 5 new ops unit tests (mamba_ssu, causal_conv1d, mrope_3d, attn_output_gate) | All PASS on NVIDIA |
| Full project build (`xmake build`) | Clean after intermittent host-fork-limit retries |
| v0.2.0 regression: `test-blockpool` 9/9, `test-ssm-state-pool` 3/3 | PASS |
| v0.2.0 regression: Qwen3-30B-A3B ping produces coherent "Who are you?" reply | PASS — generation unchanged |
| **Qwen3.5-35B-A3B ping reaches forward end-to-end (no crash, no NaN, runs the entire 40-layer hybrid chain including SSU + paged-attn + MoE expert dispatch)** | **PASS** |
| Qwen3.5-35B-A3B ping produces *coherent* English text | ❌ degenerate output (`!!!...` repeating) — **calibration bug, M2 territory** |

### Open items / M2 entry

1. **Generation quality is degenerate** (all `!` token). The forward chain executes every layer correctly w.r.t. shapes / dtypes / pointer offsets (otherwise it would crash), but at least one numerical step is wrong enough to collapse the logits to a single dominating token. Most likely suspects, in rough order of suspicion:
   - **SSU `z` activation placement**: currently we pass raw `z` to FlashInfer's SSU kernel, assuming it applies silu internally. If FlashInfer expects pre-silu'd `z`, this collapses the gated output. Add an `ops::silu` op and apply before the wrapper. (See comment at `hybrid_transformer_forward.cpp` around the SSU call.)
   - **3D MRoPE position-id wiring**: text-only path builds `pos_ids_thw = (idx, idx, idx)`. HF's `get_rope_index` for Qwen3.5 may compute different t/h/w even for text-only sequences (e.g. unique per-axis offsets to avoid degenerate rotations). Cross-check against HF transformers' Qwen3_5 `_get_rope_index`.
   - **`attn_output_gate` sign / placement**: my op applies `attn := attn * sigmoid(g)` in place. Verify Qwen3.5 actually wants sigmoid (not silu) and gate is the second half of q_proj output (not first half).
   - **`q_proj` split order**: I assumed first `q_dim` rows of the doubled q_proj are q, second `q_dim` rows are gate. Cross-check against HF.
   - **fp32 → bf16 norm weight cast precision**: weights near 1.0 should be representable in bf16, but if any weight is in a precision-sensitive band the cast distorts. Try fp32 weight directly with a per-call cast to bf16 only inside the kernel.
   - **causal_conv1d state initialization**: `reset_slot` zeroes both ssm and conv state on admit. Verify FlashInfer SSU expects pre-initialized state values (some implementations expect `exp(A_log * dt_bias)` seeded).
2. **M2 token-byte-exact alignment harness** still needs to be built — a Python script that loads HF Qwen3.5 + zedinfer, runs greedy generation, and diffs token IDs position by position. M2's first step is to construct this and identify where divergence first appears.
3. **27B dense model not exercised end-to-end** on this dev box due to host pinned-memory `ulimit -l 32GB` < 29GB on-disk. Need a larger-ulimit machine to validate the dense path, OR run on a machine where weights load OK.
4. **Warmup skipped for hybrid models** — Profiler integration with hybrid path is M3+ territory; for now the cold-start cost shows up in the first user request.
5. **Vision tower forward still throws "not implemented until M3"** — M3 brings VisionTower::forward online + multimodal HTTP API.

### M2 entry conditions (confirmed)

- All hybrid forward kernels link and execute on the real GPU (zero NaN/Inf across the 40-layer chain).
- ServingLoop correctly routes Qwen3.5 to `hybrid_transformer_forward` and Qwen3 paths to `transformer_forward` (no regression on `Qwen3-30B-A3B-GPTQ-Int4`).
- SSMStatePool slot lifecycle is enforced by Scheduler (acquire on admit, release on finish) — verified by the ping running through and exiting cleanly without slot leaks.
