# Qwen3.5 Multimodal Hybrid SSM+Attention Support — Design

> Status: Design phase. Targets Qwen3.5-27B-GPTQ-Int4 (dense) and Qwen3.5-35B-A3B-GPTQ-Int4 (MoE).
>
> Companion docs: `architecture.md` (current system), `roadmap.md` (priorities),
> `heterogeneous_moe.md` (Phase 2 baseline this design composes on top of).

## 1. Goals

- Run **Qwen3_5ForConditionalGeneration** (27B dense) and **Qwen3_5MoeForConditionalGeneration**
  (35B-A3B) end-to-end on a consumer GPU (24–48 GB), with token-level byte-exact alignment
  against HuggingFace `transformers` (greedy / argmax sampling).
- Cover the model families' novel surfaces:
  - **Hybrid layer types**: 75% Mamba2-style linear-attention SSM + 25% gated softmax attention.
  - **3D MRoPE interleaved + partial rotary** (`partial_rotary_factor=0.25`, sections `[11,11,10]`).
  - **Attention output gate** (`q_proj` output doubled — second half is a sigmoid gate on attention output).
  - **Vision tower** (27-layer ViT, Qwen2.5-VL family, dynamic resolution).
  - **MTP head** (DeepSeek-V3 style; weights loaded but inference path not activated in this design).
  - **Selective GPTQ Int4**: only `mlp.*` (dense) or `mlp.experts.*` (MoE) are quantized; attention,
    shared expert, MTP, and vision are BF16.
- Expose **OpenAI Vision API**–compatible HTTP endpoint (`content` array with `text` /
  `image_url` parts, data URI base64 images).
- Maintain framework health: **zero regression** on existing Qwen2 / Qwen3 / Qwen3-MoE.

Non-goals (in this design):

- Active MTP inference. Weights load; not used. Future work tied to phase3 `§A.2` (native spec decoding).
- Remote URL image download (only data URI in M4).
- Refactoring existing Qwen2/Qwen3/Qwen3MoE forward paths.

## 2. Architecture observations (reverse-engineered)

### 2.1 Two-model comparison (config + weight shape implementation-grade)

| Field | 27B (dense) | 35B-A3B (MoE) |
|---|---|---|
| `architectures` | `Qwen3_5ForConditionalGeneration` | `Qwen3_5MoeForConditionalGeneration` |
| `hidden_size` | 5120 | 2048 |
| `num_hidden_layers` | 64 (48 linear + 16 full) | 40 (30 linear + 10 full) |
| `layer_types` pattern | `[L,L,L,F] × 16` | `[L,L,L,F] × 10` |
| `full_attention_interval` | 4 | 4 |
| `head_dim` | 256 | 256 |
| `num_attention_heads` | 24 | 16 |
| `num_key_value_heads` | 4 (GQA 6:1) | 2 (GQA 8:1) |
| `intermediate_size` (dense MLP) | 17408 | — |
| `num_experts` | — | 256 |
| `num_experts_per_tok` | — | 8 |
| `moe_intermediate_size` | — | 512 |
| `shared_expert_intermediate_size` | — | 512 (with sigmoid gate) |
| `vocab_size` | 248320 | 248320 |
| `max_position_embeddings` | 262144 | 262144 |
| `rope_theta` | 1e7 | 1e7 |
| `partial_rotary_factor` | 0.25 | 0.25 |
| `mrope_section` | [11, 11, 10] | [11, 11, 10] |
| `attn_output_gate` | true | true |
| **Linear attention (Mamba2)** | | |
| `linear_num_value_heads` | 48 | 32 |
| `linear_value_head_dim` | 128 | 128 |
| `linear_num_key_heads` | 16 | 16 |
| `linear_key_head_dim` | 128 | 128 |
| `linear_conv_kernel_dim` | 4 | 4 |
| `mamba_ssm_dtype` | float32 | float32 |
| **MTP** | | |
| `mtp_num_hidden_layers` | 1 (dense MLP) | 1 (MoE MLP) |
| **Vision** | | |
| `depth` / `hidden_size` / `out_hidden_size` | 27 / 1152 / 5120 | 27 / 1152 / 2048 |
| `patch_size` / `temporal_patch_size` / `spatial_merge_size` | 16 / 2 / 2 | identical |
| `num_position_embeddings` | 2304 | 2304 |
| **Quantization** | | |
| GPTQ `bits / group_size / sym` | 4 / 128 / true | 4 / 128 / true |
| `dynamic` excludes | `attn`, `shared_expert`, `mtp`, `visual` | identical |

### 2.2 Linear-attention block (Mamba2-style)

Per-layer weight tensors (27B example):

```
in_proj_qkv   [10240, 5120]   # (Hk·Dk + Hk·Dk + Hv·Dv) = (16·128 + 16·128 + 48·128)
in_proj_z     [ 6144, 5120]   # gate path (Hv·Dv)
in_proj_a     [   48, 5120]   # per-V-head Δt projection
in_proj_b     [   48, 5120]   # per-V-head B modulation
A_log         [   48]         # per-V-head log(-A)
dt_bias       [   48]         # per-V-head Δt bias
conv1d.weight [10240, 1, 4]   # depthwise causal conv
norm.weight   [  128]         # RMSNorm on value_head_dim
out_proj      [ 5120, 6144]   # → hidden_size
```

Semantics (per layer, per token / chunk):

```
qkv = in_proj_qkv(x)
qkv = silu(causal_conv1d(qkv, kernel=4, state=conv_state))
q, k, v = split(qkv, [Hk·Dk, Hk·Dk, Hv·Dv])

a = in_proj_a(x); b = in_proj_b(x)
Δt = softplus(a + dt_bias)
A  = -exp(A_log)
state_t = exp(Δt·A) · state_{t-1} + (Δt·B) · outer(b_t, k_t · v_t)
y_t     = q_t · state_t                              (per V-head)

z = silu(in_proj_z(x))
y = rmsnorm(y, norm) · z                              (gated output)
out = out_proj(y)
```

State is per-(request, layer): `[num_v_heads, value_head_dim, d_state]`. **No KV cache**;
state size is fixed regardless of sequence length.

### 2.3 Gated softmax attention block (full-attention layers)

Per-layer weight tensors (27B example):

```
q_proj  [12288, 5120]   # 12288 = 2 · Hq · Dh = 2·24·256   ← doubled (q + gate)
k_proj  [ 1024, 5120]   # Hkv · Dh (GQA)
v_proj  [ 1024, 5120]
q_norm  [  256]         # per-head RMSNorm on Dh
k_norm  [  256]
o_proj  [ 5120, 6144]   # Hq · Dh → hidden               ← takes only q part (not gate)
```

Semantics:

```
qg = q_proj(x)
q, g = split(qg, [Hq·Dh, Hq·Dh])
k = k_proj(x); v = v_proj(x)

q = rmsnorm(q, q_norm)  # per-head
k = rmsnorm(k, k_norm)

q[:, :, :64] = mrope_3d_interleaved(q[:, :, :64], pos_ids_thw)   # partial rotary
k[:, :, :64] = mrope_3d_interleaved(k[:, :, :64], pos_ids_thw)
                                                                 # last 192 dims pass-through

write paged KV (only full-attention layers contribute to KV cache)
attn = paged_attention(q, k_cache, v_cache)                       # [N, Hq, Dh]

attn = attn · sigmoid(g)                                          # attn_output_gate
out  = o_proj(attn)
```

### 2.4 Vision tower (27-layer ViT, Qwen2.5-VL family)

Weight tensors:

```
patch_embed.proj   [1152, 3, 2, 16, 16]   # = linear(1152, 3·2·16·16=1536) via reshape
patch_embed.proj.bias [1152]
pos_embed.weight   [2304, 1152]
blocks[0..26]:
  norm1{.weight,.bias}  [1152]            # LayerNorm with bias
  norm2{.weight,.bias}  [1152]
  attn.qkv{.weight,.bias} [3456, 1152]    # combined QKV
  attn.proj{.weight,.bias} [1152, 1152]
  mlp.linear_fc1{.weight,.bias} [4304, 1152]
  mlp.linear_fc2{.weight,.bias} [1152, 4304]
merger.norm{.weight,.bias}      [1152]
merger.linear_fc1{.weight,.bias} [4608, 4608]   # 4608 = 1152 · 4 (2×2 spatial merge)
merger.linear_fc2{.weight,.bias} [out_hidden, 4608]
```

Notes:

- LayerNorm with bias (not RMSNorm), GELU with tanh approximation (not SwiGLU).
- Full attention, no KV cache, no paged scheduling — single forward per image batch.
- `merger` concatenates 2×2 neighboring patches and projects to LM hidden_size.

### 2.5 MTP head (DeepSeek-V3 style; weights loaded, not used in this design)

```
mtp.pre_fc_norm_embedding [hidden]
mtp.pre_fc_norm_hidden    [hidden]
mtp.fc                    [hidden, 2·hidden]
mtp.layers.0.*            # one transformer block (self_attn + dense MLP or MoE MLP)
mtp.norm                  [hidden]
# lm_head shared with the main path
```

Inference-time semantics (when activated; deferred):

```
h_t = main_lm_hidden_state(t)
e_{t+1} = embed(token_{t+1})
fc_in = concat([rmsnorm(e_{t+1}, pre_fc_norm_embedding),
                rmsnorm(h_t,     pre_fc_norm_hidden)])
h'_t = fc(fc_in)
h'_t = mtp_transformer_block(h'_t)
logits_{t+2} = lm_head(rmsnorm(h'_t, mtp.norm))
```

### 2.6 Quantization scope

GPTQ `dynamic` rules use negative patterns (`-:`) to exclude tensor name regexes:

```
-:.*attn.*           # self_attn + linear_attn — BF16
-:.*shared_expert.*  # MoE shared expert — BF16
-:.*mtp.*            # MTP head — BF16
-:.*visual.*         # Vision tower — BF16
```

Effectively quantized: 27B = 64-layer dense MLP `gate/up/down`; 35B-A3B = 40-layer × 256 expert `gate/up/down`.
`g_idx` is dummy under `desc_act=false` (consistent with Qwen3-30B-A3B-GPTQ).

## 3. Decisions (locked in brainstorming, 2026-05-16)

| # | Decision | Rationale |
|---|---|---|
| 1 | Scope: 27B + 35B-A3B designed together; 27B implemented first | Shared hybrid / vision / MTP / quant code; staged risk |
| 2 | Refactor: hybrid path is standalone (new `hybrid_transformer_forward.cpp`); existing Qwen2/Qwen3/Qwen3MoE untouched | Zero regression on v0.2.0 |
| 3 | Mamba2 kernel: FlashInfer `mamba` module (already vendored under `third_party/flashinfer`) | Eliminates the single largest implementation risk; numerically matches HF (HF transformers also calls `mamba_ssm`) |
| 4 | MTP: weights loaded, inference path not activated | YAGNI; future spec-decoding hook |
| 5 | Multimodal API: OpenAI Vision–compatible (content array + data URI base64) | Industry standard, reusable clients |
| 6 | Accuracy bar: token byte-exact vs HF (greedy / argmax, first 50 tokens) | Release-grade quality bar |
| 7 | Chat template: vendor `minja` (header-only Jinja2 subset), Qwen3.5 path only | Existing hardcoded templates can't cover Qwen3.5 features (namespace, macro, reverse-iter, tojson, type tests) |
| A2 | Forward dispatch: per-layer-kind independent functions (`forward_full_attn_layer`, `forward_linear_attn_layer`, `forward_dense_mlp`, `forward_moe_mlp`) | Matches `moe_forward.cpp` style; no virtual dispatch; easy unit tests |
| B2 | Mixed cache: `SSMStatePool` engine-level slot pool, paged KV pool unchanged | FlashInfer kernel native layout `{state_cache_size, nheads, dim, dstate}`; zero runtime alloc; mirrors `ExpertPool` design |
| C1 | Vision tower: pre-step in engine entry; `transformer_forward` gains an optional `input_embeds` parameter | Vision tower stateless, decoupled from main forward; zero regression for non-vision models |
| stb | Image decode: stb_image + stb_image_resize (header-only vendor) | Matches existing `third_party/*` header-only style; not a hot path; covers JPEG/PNG/BMP/TGA |
| flashinfer | flashinfer is hard-required (build error if `--flashinfer=y` not set, same as v0.2.0 default) | No SSM fallback path; aligns with v0.2.0 release default |
| scatter | `scatter_image_embeds` is a small new op (~50 LOC CUDA + 30 CPU) | Semantics differ from `gather_rows`/`scatter_add_rows` (condition on input_ids == target_value) |

## 4. File / directory layout

### 4.1 New files

```
include/
├── frontend/models/
│   ├── qwen3_5.hpp                    NEW dense
│   ├── qwen3_5_moe.hpp                NEW MoE
│   ├── hybrid_forward_config.hpp      NEW (extends ModelForwardConfig)
│   ├── ssm_state_pool.hpp             NEW (SSM + conv state pool)
│   ├── vision_tower.hpp               NEW
│   └── multimodal_input.hpp           NEW (image + text payload types)
├── backend/ops/
│   ├── mamba/ssu.hpp                  NEW (FlashInfer SSU wrapper entry)
│   ├── mamba/causal_conv1d.hpp        NEW
│   ├── mrope/mrope_3d.hpp             NEW
│   ├── attn_output_gate/
│   │     attn_output_gate.hpp         NEW
│   ├── scatter_image_embeds/
│   │     scatter_image_embeds.hpp     NEW
│   ├── layer_norm/layer_norm_bias.hpp NEW (ViT path)
│   ├── gelu_tanh/gelu_tanh.hpp        NEW (ViT path)
│   └── vision_attention/
│         vision_attention.hpp         NEW (FlashInfer ragged prefill wrapper)

src/
├── frontend/models/
│   ├── qwen3_5.cpp
│   ├── qwen3_5_moe.cpp
│   ├── hybrid_transformer_forward.cpp NEW (A2 layout)
│   ├── ssm_state_pool.cpp
│   ├── vision_tower.cpp
│   └── multimodal_input.cpp
├── backend/ops/
│   ├── mamba/nvidia/ssu_wrapper.cu
│   ├── mamba/{cpu,nvidia}/causal_conv1d.{cpp,cu}
│   ├── mrope/{cpu,nvidia}/mrope_3d.{cpp,cu}
│   ├── attn_output_gate/{cpu,nvidia}/attn_output_gate.{cpp,cu}
│   ├── scatter_image_embeds/{cpu,nvidia}/scatter_image_embeds.{cpp,cu}
│   ├── layer_norm/{cpu,nvidia}/layer_norm_bias.{cpp,cu}
│   ├── gelu_tanh/{cpu,nvidia}/gelu_tanh.{cpp,cu}
│   └── vision_attention/{cpu,nvidia}/vision_attention.{cpp,cu}
└── zedinfer/
    ├── chat_template_jinja.cpp        NEW (minja-backed)
    └── multimodal_processor.cpp       NEW (stb_image preprocessing)

third_party/
├── minja/                             NEW vendored (MIT, header-only ~3k LOC)
└── stb/                               NEW vendored (public domain header-only)

tests/
├── unit/
│   ├── test_ssm_state_pool.cpp
│   ├── test_ops_mamba_ssu.cpp
│   ├── test_ops_causal_conv1d.cpp
│   ├── test_ops_mrope_3d.cpp
│   ├── test_ops_attn_output_gate.cpp
│   ├── test_ops_scatter_image_embeds.cpp
│   ├── test_vision_tower.cpp
│   ├── test_minja_chat_template.cpp
│   └── test_multimodal_processor.cpp
└── e2e/
    └── test_qwen3_5_hf_alignment.py   NEW (HF token-level alignment harness)

docs/plan/
└── qwen3_5_support.md                 ← this document
```

### 4.2 Existing files extended (small, zero-regression)

| File | Change |
|---|---|
| `include/frontend/models/base.hpp` | `ModelConfig` gains hybrid-only fields: `layer_types`, `attn_output_gate`, `partial_rotary_factor`, `mrope_section`, `mrope_interleaved`, `mtp_num_hidden_layers`, `linear_attn`, `vision` |
| `include/frontend/models/forward_config.hpp` | No change. `HybridForwardConfig` subclasses externally |
| `src/frontend/models/transformer_forward.cpp` | Accept optional `input_embeds` parameter. Existing Qwen2/Qwen3/Qwen3MoE callers pass nullptr (zero behavior change) |
| `src/zedinfer/loader.cpp` | Strip `model.language_model.` / `model.` prefixes uniformly. Keep `mtp.*` and `visual.*` as-is. GPTQ rename logic unchanged |
| `include/zedinfer/request.hpp` | Add `ssm_slot_idx_`, `image_embeds_`, `pos_ids_thw_`, `has_images_` |
| `include/zedinfer/chat_template.hpp` | Add `ChatTemplate::load_jinja()` factory; `ChatMessage::content` becomes `std::variant<string, vector<ContentPart>>` |
| `src/zedinfer/scheduler.cpp` | `admit` checks both `block_allocator.available()` and `ssm_state_pool.num_free_slots()` |
| `src/zedinfer/engine.cpp` | `init_block_pool` accepts `num_kv_layers` (= full attention layer count for hybrid, total for non-hybrid) |
| `include/frontend/models/paged_forward_context.hpp` | `write_kv` / `attend` accept logical `kv_layer_idx` (= `full_layer_index(L)` for hybrid; = L for non-hybrid) |
| `src/zedinfer/http_server.cpp` | Parse OpenAI Vision content arrays |
| `xmake.lua` | Add `add_required_includedir` for `third_party/minja/include`, `third_party/stb`. **No new option flag**; flashinfer remains mandatory (error if not set, mirroring v0.2.0 release default) |

## 5. Core new modules

### 5.1 `SSMStatePool`

```cpp
// include/frontend/models/ssm_state_pool.hpp
struct SSMStatePoolConfig {
    int num_linear_layers   = 0;
    int num_v_heads         = 0;
    int value_head_dim      = 0;
    int d_state             = 0;
    int conv_kernel_dim     = 4;       // for conv state size
    int qkv_dim             = 0;       // for conv state size
    int max_concurrent      = 1;       // default single-user
    zedinferDataType_t state_dtype = ZEDINFER_DTYPE_BF16;
};

struct SSMStateView {
    void* ssm_base;                    // GPU ptr to ssm buffer
    void* conv_base;                   // GPU ptr to conv buffer
    int64_t ssm_stride_slot, ssm_stride_layer;
    int64_t conv_stride_slot, conv_stride_layer;
    int num_v_heads, value_head_dim, d_state;
    int conv_kernel_dim, qkv_dim;
    zedinferDataType_t dtype;
};

class SSMStatePool {
public:
    SSMStatePool(const SSMStatePoolConfig& cfg, const ExecutorConfig& exec);
    ~SSMStatePool();

    int acquire_slot();                // O(1); throws if none free
    void release_slot(int slot_idx);
    void reset_slot(int slot_idx);     // zeros SSM + conv state across all linear layers
    SSMStateView view() const;
    int num_free_slots() const;
    size_t bytes_per_slot() const;
private:
    SSMStatePoolConfig cfg_;
    tensor_t ssm_buffer_;              // [slots, L, Hv, Dv, D_state]
    tensor_t conv_buffer_;             // [slots, L, K-1, qkv_dim]
    std::vector<bool> slot_in_use_;
    int next_hint_;
};
```

Memory budget example (27B, single-user, BF16 state):

```
ssm_buffer  = 1 slot × 48 linear layers × 48 V-heads × 128 dim × 128 d_state × 2B = 75 MB
conv_buffer = 1 slot × 48 linear layers × 3 (kernel-1) × 10240 qkv_dim       × 2B =  2.8 MB
Total       ≈ 78 MB
```

For `max_concurrent=4`: ≈ 312 MB. Acceptable on 24 GB VRAM.

If M6 alignment fails on BF16 state, fall back to fp32 state — `state_dtype = ZEDINFER_DTYPE_F32`,
doubles memory to 156 MB / slot. Still acceptable.

### 5.2 `VisionTower`

```cpp
// include/frontend/models/vision_tower.hpp
struct VisionConfig {
    int depth, hidden_size, out_hidden_size;
    int num_heads, patch_size, temporal_patch_size, spatial_merge_size;
    int num_pos_embed, intermediate_size;
};

class VisionTower {
public:
    VisionTower(const VisionConfig& cfg, const ModelWeights& w, const ExecutorConfig& exec);
    // patches:     [N_patches, in_channels · T_patch · H_patch · W_patch]
    // pos_ids_thw: [N_patches, 3]
    // returns:     [N_image_tokens, out_hidden_size], where N_image_tokens = N_patches / merge_size²
    tensor_t forward(tensor_t patches, tensor_t pos_ids_thw, const ExecutorConfig& exec);
};
```

Key differences from LM path:

- LayerNorm with bias (`ops::layer_norm_bias`).
- GELU tanh-approximation (`ops::gelu_tanh`).
- Full attention without paged KV (`ops::vision_attention`, wraps FlashInfer ragged prefill).
- `patch_embed.proj` is a 3D conv implemented as reshape → `ops::linear` (no new conv op).
- `pos_embed` is a lookup table reused via `ops::embedding`.

### 5.3 `Qwen3_5Model` / `Qwen3_5MoeModel`

```cpp
// include/frontend/models/qwen3_5.hpp
class Qwen3_5Model : public Model {
public:
    Qwen3_5Model(const std::string& model_path, const ExecutorConfig& exec,
                 const SchedulerConfig& sched);
    HybridForwardConfig forward_config() const;
    SSMStatePool& ssm_state_pool();
    const VisionTower* vision_tower() const;
    tensor_t forward(InferenceRequest& req, PagedForwardContext& ctx,
                     const ExecutorConfig& exec, DecodeScratch* scratch) override;
private:
    std::unique_ptr<SSMStatePool> ssm_pool_;
    std::unique_ptr<VisionTower>  vision_;
};
```

`Qwen3_5MoeModel` additionally holds an `ExpertPool` (35B-A3B routes 256 experts through
Phase 2 infrastructure). Both share `hybrid_transformer_forward()`.

### 5.4 `HybridForwardConfig`

```cpp
// include/frontend/models/hybrid_forward_config.hpp
enum class LayerKind : uint8_t { Full, Linear };

struct LinearAttnConfig {
    int num_v_heads, value_head_dim;
    int num_k_heads, key_head_dim;
    int d_state, conv_kernel_dim;
    zedinferDataType_t state_dtype;
};

struct MRoPEConfig {
    bool   interleaved = true;
    std::array<int,3> section { 11, 11, 10 };
    float  partial_factor = 0.25f;
    float  theta = 1e7f;
};

struct HybridForwardConfig : public ModelForwardConfig {
    HybridForwardConfig(const ModelConfig& c, const ModelWeights& w)
        : ModelForwardConfig(c, w) {}

    std::vector<LayerKind> layer_kinds;            // per layer
    LinearAttnConfig       linear_attn;
    MRoPEConfig            mrope;
    bool                   attn_output_gate = true;
    SSMStatePool*          ssm_pool         = nullptr;

    // Index translations:
    //   full_layer_index(L)   → 0..num_full_attn_layers-1 (for paged KV cache indexing)
    //   linear_layer_index(L) → 0..num_linear_attn_layers-1 (for SSMStatePool indexing)
    int full_layer_index(size_t L) const;
    int linear_layer_index(size_t L) const;
    bool is_linear_attn_layer(size_t L) const { return layer_kinds[L] == LayerKind::Linear; }
};
```

Compile-time guarantee: only `hybrid_transformer_forward` sees `HybridForwardConfig`. Existing
`transformer_forward` keeps using `ModelForwardConfig`. Cross-pollination is impossible.

### 5.5 ChatTemplate (minja vendor)

```cpp
class ChatTemplate {
public:
    static ChatTemplate load(const std::string& model_path);         // existing
    static ChatTemplate load_jinja(const std::string& template_file); // NEW

    std::string apply(const std::vector<ChatMessage>& msgs,
                      bool add_generation_prompt = true,
                      bool enable_thinking      = true) const;
};

struct ChatMessage {
    std::string role;
    std::variant<std::string, std::vector<ContentPart>> content;     // string or array
    // existing tool_calls, reasoning_content, etc.
};

struct TextPart  { std::string text; };
struct ImagePart { std::string data_uri; };           // height/width unknown until decode
using ContentPart = std::variant<TextPart, ImagePart>;
```

minja exposes `Value` (variant-like). The chat template's `content is string` / `content is iterable`
tests work directly. If minja lacks a Qwen3.5 feature, patch header-only inline; trivial.

### 5.6 MultiModalProcessor

```cpp
struct ImagePayload { int height, width; std::vector<uint8_t> rgb_pixels; };
struct ProcessedImage {
    tensor_t patches;          // [N_patches, 3·2·16·16] on GPU
    tensor_t pos_ids_thw;      // [N_patches, 3]
    int      num_image_tokens; // = N_patches / merge_size²
};

class MultiModalProcessor {
public:
    MultiModalProcessor(const VisionConfig& cfg);
    ProcessedImage process(const ImagePayload& img, const ExecutorConfig& exec);
    static ImagePayload decode_base64(std::string_view data_uri);
    int num_image_tokens_for(int h, int w) const;  // used by template post-processor
};
```

Decode path: stb_image (JPEG/PNG/BMP/TGA) → resize via stb_image_resize → normalize by
[0.5, 0.5, 0.5] mean/std → patchify (`temporal_patch_size=2`, `patch_size=16`) → H2D.

## 6. Forward data flow

### 6.1 End-to-end call graph

```
[HTTP {messages, content:[text|image_url]}]
        │
        ▼
   OpenAIVisionParser
        │
   ┌────┴────┐
   ▼         ▼
Tokenizer   MultiModalProcessor   (stb_image decode → resize → patchify → H2D)
   │         │
   │         ▼  ProcessedImage{patches, pos_ids_thw, N_img_tokens}
   │         │
   │         ▼  VisionTower::forward
   │         │
   │   image_embeds [N_img_tokens, lm_hidden]
   ├─────────┘
   ▼  (<|image_pad|> placeholders already expanded to N_img_tokens copies)
ChatTemplate::apply (minja)  →  prompt text → tokenize → input_ids [N_total]
   │
   ▼
InferenceEngine::submit
   │  Scheduler::admit:
   │    ssm_state_pool.acquire_slot() → req.ssm_slot_idx
   │    block_allocator.allocate_blocks() (existing)
   │
   ▼
hybrid_transformer_forward(model, ctx, req, exec, scratch, input_embeds?)
   │
   │  Embed lookup, then (if req.has_images):
   │  ops::scatter_image_embeds(hidden, input_ids, req.image_embeds, 248056)
   │
   ▼
[loop L in 0..num_hidden_layers]
       ├── h_in = rmsnorm(hidden, input_layernorm[L])
       ├── attn_out =
       │     layer_kinds[L] == Linear
       │       ? forward_linear_attn_layer(m, h_in, L, req.ssm_slot_idx)
       │       : forward_full_attn_layer(m, ctx, h_in, L, exec)
       ├── h1 = hidden + attn_out
       ├── h_post = rmsnorm(h1, post_attention_layernorm[L])
       ├── mlp_out =
       │     m.is_moe ? forward_moe_mlp(m, h_post, L, exec, scratch)
       │              : forward_dense_mlp(m, h_post, L)
       └── hidden = h1 + mlp_out
[end loop]
   │
   ▼
logits = lm_head(rmsnorm(hidden, "norm.weight"))
   │
   ▼  Sample, stream; Scheduler::on_finish releases ssm slot + KV blocks
```

### 6.2 `forward_linear_attn_layer` (75% of layers)

```cpp
tensor_t forward_linear_attn_layer(const HybridForwardConfig& m,
                                    tensor_t h_in, size_t L, int slot_idx) {
    auto p = m.prefix(L) + "linear_attn.";
    const auto& cfg = m.linear_attn;

    auto qkv = ops::linear(m.W(p+"in_proj_qkv.weight"), h_in);
    auto z   = ops::linear(m.W(p+"in_proj_z.weight"),   h_in);
    auto a   = ops::linear(m.W(p+"in_proj_a.weight"),   h_in);
    auto b   = ops::linear(m.W(p+"in_proj_b.weight"),   h_in);

    qkv = ops::silu(ops::mamba::causal_conv1d(
        qkv, m.W(p+"conv1d.weight"),
        m.ssm_pool->view(), slot_idx, m.linear_layer_index(L)));

    auto [q_ssm, k_ssm, v_ssm] = split_qkv(qkv, cfg);

    auto y = ops::mamba::ssu({
        .state_view = m.ssm_pool->view(),
        .slot_idx   = slot_idx,
        .layer_idx  = m.linear_layer_index(L),
        .q = q_ssm, .k = k_ssm, .v = v_ssm,
        .a = a, .b = b,
        .A_log   = m.W(p+"A_log"),
        .dt_bias = m.W(p+"dt_bias"),
        .z = ops::silu(z),
        .num_tokens = N,
    });

    y = ops::rms_norm(y.view({N*cfg.num_v_heads, cfg.value_head_dim}),
                      m.W(p+"norm.weight"), eps);
    return ops::linear(m.W(p+"out_proj.weight"),
                       y.view({N, cfg.num_v_heads * cfg.value_head_dim}));
}
```

### 6.3 `forward_full_attn_layer` (25% of layers)

```cpp
tensor_t forward_full_attn_layer(const HybridForwardConfig& m,
                                  PagedForwardContext& ctx,
                                  tensor_t h_in, size_t L,
                                  const ExecutorConfig& exec) {
    auto p = m.prefix(L) + "self_attn.";
    const size_t Hq = m.config.num_attention_heads;
    const size_t Hkv = m.config.num_key_value_heads;
    const size_t Dh = m.config.head_dim;

    auto qg = ops::linear(m.W(p+"q_proj.weight"), h_in);
    auto k  = ops::linear(m.W(p+"k_proj.weight"), h_in);
    auto v  = ops::linear(m.W(p+"v_proj.weight"), h_in);

    auto q = qg.view({N, Hq, Dh}, /*offset=*/0);
    auto g = qg.view({N, Hq, Dh}, /*offset=*/Hq*Dh);

    q = ops::rms_norm(q.view({N*Hq, Dh}),  m.W(p+"q_norm.weight"), eps).view({N, Hq, Dh});
    k = ops::rms_norm(k.view({N*Hkv, Dh}), m.W(p+"k_norm.weight"), eps).view({N, Hkv, Dh});

    ops::mrope_3d(q, ctx.pos_ids_thw(), m.mrope);
    ops::mrope_3d(k, ctx.pos_ids_thw(), m.mrope);

    ctx.write_kv(m.full_layer_index(L), k.view({N, Hkv*Dh}), v);

    float scale = 1.0f / sqrtf((float)Dh);
    auto attn = ctx.attend(m.full_layer_index(L), q, scale, exec, Hq, Hkv, Dh, nullptr);

    ops::attn_output_gate(attn, g);

    return ops::linear(m.W(p+"o_proj.weight"), attn.view({N, Hq*Dh}));
}
```

### 6.4 `forward_dense_mlp` / `forward_moe_mlp`

Dense reuses `dispatch_linear` GPTQ path; MoE reuses `moe_layer_forward()` from v0.2.0 verbatim
(possible because `HybridForwardConfig` is a subclass of `ModelForwardConfig`).

### 6.5 Prefill vs Decode dispatch

| Stage | Full-attention layer | Linear-attention layer |
|---|---|---|
| **Prefill** (N tokens) | FlashInfer paged prefill (one-shot N tokens, ragged batch metadata) | FlashInfer `selective_state_update` **varlen** path (`cu_seqlens=[0,N]`, `num_accepted_tokens=N`); causal conv processes N tokens against prior conv state |
| **Decode** (1 token) | FlashInfer paged decode (batched if multi-request) | FlashInfer `selective_state_update` **STP** path (single token, in-place state update); causal conv single-step shift |

Both reach FlashInfer through the same `ops::mamba::ssu` wrapper; the wrapper dispatches on `num_tokens`.
No dependency on FlashInfer's SSDCombined / Triton path.

### 6.6 Image embeddings injection

ChatTemplate post-processes the `<|image_pad|>` placeholder by literal expansion to `N_img_tokens`
copies before tokenization. Tokenizer maps each to `image_token_id = 248056`. At forward entry:

```cpp
auto hidden = ops::embedding(input_ids, embed_tokens_weight);
if (req.has_images())
    ops::scatter_image_embeds(hidden, input_ids, req.image_embeds(), /*token_id=*/248056);
```

`pos_ids_thw` computed by `MultiModalProcessor::compute_pos_ids_thw()` before forward — pure CPU
work, single H2D transfer. Pure-text tokens get `(t, h, w) = (idx, idx, idx)`; image tokens occupy
a `(T_chunk, H_patches, W_patches)` rectangle in row-major order.

## 7. Ops layer (FlashInfer mamba wrapper)

### 7.1 `ops::mamba::ssu` wrapper internals

Pseudo-code for `src/backend/ops/mamba/nvidia/ssu_wrapper.cu`:

```cpp
#include <flashinfer/mamba/invoke_selective_state_update_mtp.cuh>
#include <flashinfer/mamba/selective_state_update.cuh>

void ssu(const SSUParams& p) {
    flashinfer::mamba::SelectiveStateMTPParams params{};
    params.state    = ptr_with_offset(p.state_view.ssm_base,
                                      p.slot_idx  * p.state_view.ssm_stride_slot
                                    + p.layer_idx * p.state_view.ssm_stride_layer);
    params.x        = p.v->data();
    params.dt       = p.a->data();
    params.A_log    = p.A_log->data();
    params.B        = p.b->data();
    params.C        = p.q->data();
    params.D        = nullptr;                  // Qwen3.5 has no D skip-connect
    params.z        = p.z->data();
    params.dt_bias  = p.dt_bias->data();
    params.output   = p.out->data();
    params.batch    = 1;
    params.dim      = num_v_heads * value_head_dim;
    params.dstate   = d_state;
    params.nheads   = num_v_heads;
    params.ngroups  = num_k_heads;
    params.ntokens_mtp = p.num_tokens;
    params.cu_seqlens  = p.num_tokens > 1 ? build_cu_seqlens({0, p.num_tokens}) : nullptr;
    params.num_accepted_tokens = nullptr;
    params.dt_softplus = true;

    flashinfer::mamba::mtp::invokeSelectiveStateUpdateMTP<
        __nv_bfloat16, __nv_bfloat16, float, __nv_bfloat16, int32_t, void>(
        params, flashinfer::mamba::SSUAlgorithm::kAuto,
        core::context().runtime().compute_stream());
}
```

Key properties:

- No PyTorch / TVM-ffi dependency. We call the C++ template directly with raw pointers.
- Dtype combo `(bf16 input, bf16 weight, f32 A, bf16 state, i32 stateIndex, void state_scale)`
  is instantiated at compile time. Verify FlashInfer JIT cache emits this combo at M0 (GO/NO-GO).
- SM dispatch (80 / 90 / 100) handled by FlashInfer internally via `cudaGetDeviceProperties`.
- Algorithm `kAuto` (FlashInfer picks `simple` / `horizontal` / `async_horizontal` by batch size).
- Runs on `core::context().runtime().compute_stream()` — FIFO with subsequent zedinfer ops.

### 7.2 Caller contract

1. `SSMStatePool::acquire_slot()` already returned `slot_idx`.
2. Before first SSU call of a request, `SSMStatePool::reset_slot(slot_idx)` (zero state).
3. Per-step calls with same `(slot_idx, layer_idx)` are allowed; FlashInfer updates state in place.
4. Cross-layer (same slot, varying `layer_idx`) and cross-request (different slot) calls are
   non-interfering by layout.
5. Caller never records a CUDA event; FlashInfer kernel runs on the compute stream — synchronizes
   with subsequent ops by stream order.

### 7.3 Other new ops

```cpp
// ops::mamba::causal_conv1d
//   x:     [N, qkv_dim]  (decode N=1, prefill N>1)
//   w:     [qkv_dim, 1, kernel_dim]  (depthwise weight)
//   state:  per-slot, per-layer conv state buffer (loaded from SSMStateView::conv_base)
//   returns: [N, qkv_dim]  (state updated in place)
tensor_t causal_conv1d(tensor_t x, tensor_t w, SSMStateView v, int slot, int layer);

// ops::mrope_3d
//   x: [N, num_heads, head_dim]  (rotates in place; first head_dim*partial_factor dims)
//   pos_ids_thw: [3, N]
void mrope_3d(tensor_t x, tensor_t pos_ids_thw, const MRoPE3DConfig& cfg);

// ops::attn_output_gate
//   attn: [N, H, D]; g: [N, H, D]
//   attn := attn * sigmoid(g)
void attn_output_gate(tensor_t attn, tensor_t g);

// ops::scatter_image_embeds
//   hidden: [N_total, hidden]  (in/out)
//   input_ids: [N_total]
//   image_embeds: [N_image_total, hidden]
//   For positions where input_ids[i] == image_token_id, write image_embeds[next] into hidden[i].
void scatter_image_embeds(tensor_t hidden, tensor_t input_ids,
                          tensor_t image_embeds, int image_token_id);

// ops::layer_norm_bias  (ViT path)
void layer_norm_bias(tensor_t y, tensor_t x, tensor_t w, tensor_t b, float eps);

// ops::gelu_tanh       (ViT path)
void gelu_tanh(tensor_t y, tensor_t x);

// ops::vision_attention  (ViT path; wraps FlashInfer ragged prefill)
struct VisionAttentionParams { tensor_t q, k, v, out; float scale; };
void vision_attention(const VisionAttentionParams& p);
```

### 7.4 Risk: FlashInfer template instantiation

The first concrete check at M0 is whether `invokeSelectiveStateUpdateMTP<bf16, bf16, f32, bf16, i32, void>`
compiles & links against our build setup. If the combination isn't pre-instantiated in
`csrc/selective_state_update_kernel_inst.cu`, we must either:

1. Add an instantiation entry (FlashInfer uses jinja-generated `*_kernel_inst.cu` files — extend
   the generator inputs in xmake).
2. Switch to a covered combo (e.g., fp16 state instead of bf16).

This is the **M0 GO/NO-GO checkpoint**.

## 8. Loader / Tokenizer / Chat / MultiModalProcessor changes

### 8.1 Loader

- Uniform prefix stripping: `model.language_model.` → ``, `model.` → ``. Vision (`visual.*`) and
  MTP (`mtp.*`) keep their prefixes.
- GPTQ rename: `*.qweight` → `*.weight_packed`, `*.scales` → `*.weight_scale`, `*.qzeros` dropped
  (sym=true). No change vs v0.2.0; `quant_config.dynamic` negative patterns are implicitly handled
  because non-MLP tensors simply lack `qweight`.
- MTP weights load into `ModelWeights` with `mtp.*` keys. Forward never queries them in this design.
- CPU pinned routing predicate same as v0.2.0 D-series path (driven by `ZEDINFER_MOE_GPU_SLOTS`
  for 35B-A3B).

### 8.2 Tokenizer

No change. Existing BPE (vocab.json + merges.txt) supports Qwen3.5 directly. Special-token IDs
(image_token_id=248056, vision_start=248053, vision_end=248054, im_start=248045, etc.) are exposed
as ModelConfig fields.

### 8.3 Chat template (minja-backed)

- Vendor minja into `third_party/minja/`.
- `ChatTemplate::load_jinja(path)` parses & caches the compiled AST.
- `apply()` evaluates against `ChatMessage`s, with `content` upgraded to the variant type.
- Post-process the rendered string: replace each `<|image_pad|>` with `<|image_pad|>` × `num_image_tokens`
  before tokenization.

### 8.4 MultiModalProcessor

Pipeline per image:

1. Decode data URI (strip `data:image/...;base64,`), base64 → bytes.
2. stb_image_load → `RGB pixels[H×W×3]`.
3. Resize (stb_image_resize, bilinear) to fit constraints from `preprocessor_config.json`
   (`longest_edge`, `shortest_edge`). Round H and W to multiples of `patch_size * spatial_merge_size = 32`.
4. Normalize: `(pixel/255 - 0.5) / 0.5` (per-channel).
5. Patchify: split into `(T=temporal_patch_size, H_patch=patch_size, W_patch=patch_size)` blocks.
   For static images, replicate to `T=2` (matches Qwen2-VL preprocessor convention).
6. H2D into `patches` tensor.
7. Compute `pos_ids_thw`: row-major spatial positions for this image; flattened.
8. Return `ProcessedImage{patches, pos_ids_thw, num_image_tokens}`.

For purely-text requests, `MultiModalProcessor` is not invoked; engine constructs `pos_ids_thw`
trivially from `0..N-1`.

## 9. Scheduler / Request / Allocator changes

### 9.1 `InferenceRequest`

```cpp
class InferenceRequest {
    // existing...
    int      ssm_slot_idx_   = -1;
    tensor_t image_embeds_;        // concatenated across all images in this request
    tensor_t pos_ids_thw_;         // [3, N_total]
    bool     has_images_     = false;
};
```

### 9.2 `Scheduler::admit`

```
existing admit (non-hybrid model):
  blocks_needed = ceil(prompt_len / block_size)
  return block_allocator.available() >= blocks_needed

new admit (hybrid model):
  blocks_needed = ceil(prompt_len_in_full_layers / block_size)
  return block_allocator.available() >= blocks_needed
      && ssm_state_pool.num_free_slots() >= 1

on admit (hybrid):
  req.ssm_slot_idx = ssm_state_pool.acquire_slot()
  ssm_state_pool.reset_slot(req.ssm_slot_idx)
  allocate_blocks(req)
```

### 9.3 `BlockAllocator` + `init_block_pool`

```
existing init_block_pool:
  total_blocks = vram_budget / (block_size · num_hidden_layers · kv_per_token_bytes)

new init_block_pool (hybrid):
  num_kv_layers = count(layer_types == "full_attention")
  total_blocks = vram_budget / (block_size · num_kv_layers · kv_per_token_bytes)
```

`Engine::init_block_pool` accepts `num_kv_layers` parameter (defaults to `num_hidden_layers` for
backward compatibility).

### 9.4 `PagedForwardContext`

`write_kv(L, k, v)` and `attend(L, ...)` accept logical KV layer index:

- Hybrid: caller passes `full_layer_index(L)` → 0..num_full_attn_layers-1.
- Non-hybrid: caller passes `L` → identity, no behavior change.

### 9.5 Request release

`InferenceRequest` does **not** hold an `Engine` reference (existing v0.2.0 design). KV blocks
are released by Scheduler-driven `BlockAllocator::deallocate`. SSM slot release follows the same
pattern in `Scheduler::on_request_finish`:

```cpp
// Inside Scheduler::on_request_finish(InferenceRequest& req)
if (model_is_hybrid && req.ssm_slot_idx_ >= 0) {
    ssm_state_pool_.release_slot(req.ssm_slot_idx_);
    req.ssm_slot_idx_ = -1;
}
// KV blocks released via SequenceBlockTable as before
```

## 10. Milestones

| M | Name | Body | DoD | Estimate |
|---|---|---|---|---|
| **M0** | Load + parse | Loader strips `model.language_model.` prefix. `ModelConfig` gains hybrid fields. `Qwen3_5Model::Qwen3_5Model()` succeeds. `SSMStatePool` + `VisionTower` construct (no forward). minja vendor + text-only chat_template renders. **GO/NO-GO**: `tests/integration/test_flashinfer_ssu_link.cu` (minimal `.cu` that includes `flashinfer/mamba/*.cuh` and instantiates `invokeSelectiveStateUpdateMTP<__nv_bfloat16, __nv_bfloat16, float, __nv_bfloat16, int32_t, void>`) compiles and links via xmake. If link fails (instantiation missing), pick fallback before proceeding: (a) switch state to `__half`; or (b) extend FlashInfer `*_kernel_inst.cu` generator inputs. Decision recorded inline in design doc before M1 starts. _M0 GO confirmed 2026-05-17: `xmake run test-flashinfer-ssu-link` prints `[ok] SSU MTP template (bf16 input/state, f32 A) linked successfully` — the `(bf16, bf16, f32, bf16, i32, void)` instantiation compiles and links cleanly via header-only consumption of `flashinfer/mamba/selective_state_update.cuh` (required xmake flags: `-DFLASHINFER_ENABLE_BF16`, `--expt-relaxed-constexpr`, `--expt-extended-lambda`). SSMStatePool will use BF16 state._ | `xmake run ping ~/data/models/Qwen3.5-27B-GPTQ-Int4 --nvidia` enters main loop, prints "unsupported forward" but does not crash. Loader log shows all 1775 weights loaded. SSU link test compiles | 1 week |
| **M1** | Hybrid text-only forward | `ops::mamba::ssu` wrapper. `ops::mamba::causal_conv1d` (self-written). `ops::mrope_3d` + `ops::attn_output_gate`. `hybrid_transformer_forward` (A2 layout). Scheduler/Allocator/PagedForwardContext `full_layer_index` plumbing | `ping --nvidia` on 27B with "Who are you?" emits coherent Qwen reply | 3–4 weeks |
| **M2** | 27B token byte-exact alignment | `tests/e2e/test_qwen3_5_hf_alignment.py`. Per-op tests (SSU, conv1d, mrope_3d, attn_gate) against fixed-input Python references. Iterate fix→test | 27B greedy first 50 tokens byte-exact vs HF | 1–1.5 weeks |
| **M3** | Vision tower integration | `ops::vision_attention` (FlashInfer ragged prefill wrapper). `ops::layer_norm_bias`, `ops::gelu_tanh`. `VisionTower::forward`. stb_image vendor + `MultiModalProcessor::process()`. `ops::scatter_image_embeds`. `transformer_forward` gains `input_embeds` entry | CLI `ping --image path.jpg --prompt "describe this"` emits sensible description. Image embed count matches HF | 1.5 weeks |
| **M4** | OpenAI Vision HTTP | Parse content arrays. `<|image_pad|>` placeholder expansion. SSE streaming with multimodal. Optional Web UI image upload | `curl POST /v1/chat/completions` with data URI image streams reply. Web UI can drag-image to chat | 1 week |
| **M5** | 35B-A3B (MoE path) | `Qwen3_5MoeModel` ctor + `ExpertPool` over 256 experts. `HybridForwardConfig::is_moe=true`. `forward_moe_mlp` invokes existing `moe_layer_forward`. `ZEDINFER_MOE_GPU_SLOTS` matches v0.2.0 behavior. Validate text + multimodal + byte-exact alignment | 35B-A3B text + image Q&A runs. Token byte-exact vs HF on both text and multimodal | 1.5–2 weeks |
| **M6** | Perf bench polish | `xmake run bench` covers hybrid path. SSMStatePool + ExpertPool VRAM auto-N decision integrated. Verify SSU varlen prefill doesn't regress | bench prints prefill / decode tok/s table for 27B and 35B-A3B | 1 week |
| **M7** | Docs + release | Update `architecture.md` hybrid section. Mark `roadmap.md`. v0.3.0 release notes. Optional: activate MTP as native spec draft (phase3 §A.2) | v0.3.0 tag | 1 week |

**Total: 10–12 weeks.** M1 is the single largest risk (FlashInfer SSU + numerical alignment).

## 11. Testing strategy

### 11.1 Per-op unit tests (tests/unit/test_<op>.cpp)

- `test_ssm_state_pool` — slot acquire/release; layout (ssm + conv); reset_slot.
- `test_ops_mamba_ssu` — fixed (q, k, v, A_log, dt_bias, b, z) vs Python ground-truth (HF mamba_ssm); max-abs-diff < 1e-3 BF16.
- `test_ops_causal_conv1d` — state correctly carried across decode-step calls.
- `test_ops_mrope_3d` — vs HF `apply_multimodal_rotary_pos_emb`. Critical for byte-exact.
- `test_ops_attn_output_gate` — trivial.
- `test_ops_scatter_image_embeds` — scatter correctness on a small input_ids vector.
- `test_vision_tower` — fixed (3×448×448) RGB input, compare merger output vs HF Qwen3VLModel.
- `test_minja_chat_template` — Qwen3.5 jinja vs `transformers.tokenizer.apply_chat_template`.
- `test_multimodal_processor` — patches shape, pos_ids_thw vs HF Qwen3VLProcessor.

### 11.2 End-to-end token alignment

```python
# tests/e2e/test_qwen3_5_hf_alignment.py
hf_model = AutoModelForCausalLM.from_pretrained(model_path)
hf_ids = hf_model.generate(prompt_ids, do_sample=False, max_new_tokens=50)

zi_ids = run_zedinfer_ping("--nvidia", "--no-sample", "--max-tokens=50", prompt)

assert hf_ids[:50].tolist() == zi_ids[:50]
```

Run at M2 (27B text), M3 (27B text + image), M5 (35B-A3B text), M5 (35B-A3B text + image).

### 11.3 Performance benchmark

Extend `xmake run bench` to accept Qwen3.5 paths. Output prefill / decode tok/s columns alongside
v0.2.0 reference (Qwen3-30B-A3B-GPTQ-Int4). Same `-p / -d / -r / --gpu-memory-utilization` knobs.

## 12. Risks and mitigations

| Risk | Severity | Mitigation |
|---|---|---|
| FlashInfer SSU template instantiation missing for `(bf16, bf16, f32, bf16, i32, void)` | **High** | M0 first task: `tests/integration/test_flashinfer_ssu_link.cu` — minimal .cu including `flashinfer/mamba/invoke_selective_state_update_mtp.cuh` + a 5-line `int main()` that invokes the template (zero-input is fine, only need link). xmake target `test-flashinfer-ssu-link`. If undefined-symbol at link: (a) probe `csrc/selective_state_update_kernel_inst.cu` jinja inputs and add our dtype combo; or (b) switch state_dtype to `__half` (also covered, smaller VRAM footprint). Decision documented in §10 before M1 |
| `causal_conv1d` bug causes layer-0 drift | Medium | Per-op unit test against HF reference. Compare zedinfer layer-0 output vs HF byte-exact at M2 |
| 3D MRoPE interleaved + partial_rotary alignment | Medium | Largest M2 attention point. Implement carefully in fp32 intermediate where needed; compare against HF token-by-token |
| SSMStatePool layout indexing across layers / slots | Medium | Dedicated unit tests cover layout. Cross-slot isolation tested explicitly |
| VisionTower patch order divergence from HF Qwen3VLProcessor | Medium | At M3, byte-exact compare `patches` and `pos_ids_thw` for a fixed image |
| minja missing a Qwen3.5 Jinja feature | Medium | Header-only; patch inline or contribute upstream. Worst case: small Qwen3.5-specific monkey-patch in `chat_template_jinja.cpp` |
| 35B-A3B + hybrid SSMStatePool VRAM negotiation | Low | Extend v0.2.0 auto-N logic at M5 to include SSMStatePool budget |
| OpenAI Vision URL fetch | Low | M4 supports only `data:` URIs; document remote URL as unsupported |
| Loader warns on unused `mtp.*` weights | Low | Add `is_known_unused_prefix({"mtp."})` quiet list |
| Multi-request ExpertPool + SSMStatePool race | Low | Single-user default (max_concurrent=1) sidesteps; revisit in M5 |

## 13. Out of scope (this design)

- Active MTP inference. Weights load; inference path deferred to a separate future design tied to
  phase3 `§A.2` (native speculative decoding draft).
- Remote URL image download in OpenAI Vision payloads. M4 supports only `data:` URIs.
- FlashInfer SSDCombined / Triton chunked-scan path. `selective_state_update` varlen suffices for
  prefill; SSD evaluation deferred to M6 perf polish only if prefill is slow.
- Refactoring existing Qwen2 / Qwen3 / Qwen3-MoE forward paths. Hybrid stays in its own files.
- Multi-image batched optimization. M3/M4 supports single image per request; multi-image works but
  is not perf-optimized.

## 14. File summary

| File | Status | Purpose |
|---|---|---|
| `include/frontend/models/qwen3_5.hpp` + `.cpp` | new | 27B model class |
| `include/frontend/models/qwen3_5_moe.hpp` + `.cpp` | new | 35B-A3B model class |
| `include/frontend/models/hybrid_forward_config.hpp` | new | hybrid config subclass |
| `include/frontend/models/ssm_state_pool.hpp` + `.cpp` | new | SSM + conv state slot pool |
| `include/frontend/models/vision_tower.hpp` + `.cpp` | new | ViT forward module |
| `include/frontend/models/multimodal_input.hpp` + `.cpp` | new | image + text payloads |
| `src/frontend/models/hybrid_transformer_forward.cpp` | new | A2 layout: 5 per-layer-kind functions + outer loop |
| `include/backend/ops/mamba/ssu.hpp` + `nvidia/ssu_wrapper.cu` | new | FlashInfer SSU wrapper |
| `include/backend/ops/mamba/causal_conv1d.hpp` + `{cpu,nvidia}/*.{cpp,cu}` | new | causal conv1d w/ state |
| `include/backend/ops/mrope/mrope_3d.hpp` + `{cpu,nvidia}/*.{cpp,cu}` | new | 3D MRoPE |
| `include/backend/ops/attn_output_gate/*.hpp` + `{cpu,nvidia}/*.{cpp,cu}` | new | attn output gate |
| `include/backend/ops/scatter_image_embeds/*` | new | scatter image embeds |
| `include/backend/ops/layer_norm/layer_norm_bias.hpp` + `{cpu,nvidia}/*.{cpp,cu}` | new | ViT LayerNorm w/ bias |
| `include/backend/ops/gelu_tanh/*.hpp` + `{cpu,nvidia}/*.{cpp,cu}` | new | GELU tanh |
| `include/backend/ops/vision_attention/*` | new | ViT attention (FlashInfer ragged prefill) |
| `src/zedinfer/chat_template_jinja.cpp` | new | minja-backed ChatTemplate |
| `src/zedinfer/multimodal_processor.cpp` | new | stb-based image preprocessing |
| `third_party/minja/` | new vendored | header-only Jinja2 subset |
| `third_party/stb/` | new vendored | header-only image decode/resize |
| `tests/unit/test_*.cpp` (8 new) | new | per-op + per-module tests |
| `tests/e2e/test_qwen3_5_hf_alignment.py` | new | HF token-level alignment harness |
| `include/frontend/models/base.hpp` | extend | ModelConfig hybrid fields |
| `src/frontend/models/transformer_forward.cpp` | extend | optional input_embeds entry |
| `src/zedinfer/loader.cpp` | extend | prefix-strip, unused-prefix quiet list |
| `include/zedinfer/request.hpp` | extend | ssm_slot_idx, image_embeds, pos_ids_thw |
| `include/zedinfer/chat_template.hpp` | extend | load_jinja factory + content variant |
| `src/zedinfer/scheduler.cpp` | extend | dual-pool admission check |
| `src/zedinfer/engine.cpp` | extend | init_block_pool accepts num_kv_layers |
| `include/frontend/models/paged_forward_context.hpp` | extend | logical kv_layer_idx |
| `src/zedinfer/http_server.cpp` | extend | OpenAI Vision parsing |
| `xmake.lua` | extend | vendor minja + stb include dirs |
