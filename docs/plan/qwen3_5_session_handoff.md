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
