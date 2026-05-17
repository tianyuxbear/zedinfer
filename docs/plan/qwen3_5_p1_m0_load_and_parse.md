# Qwen3.5 P1 (M0) — Load + Parse + FlashInfer SSU Link GO/NO-GO

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `xmake run ping ~/data/models/Qwen3.5-27B-GPTQ-Int4 --nvidia` enter main loop, load all 1775 weights, construct SSMStatePool + VisionTower (no forward), render Qwen3.5 chat template via minja, and verify FlashInfer SSU template instantiates & links — without running any inference forward.

**Architecture:** Hybrid path stays standalone (decision C of design doc §3). All new files live in `frontend/models/qwen3_5*`, `backend/ops/mamba`, `third_party/{minja,stb}`. Existing Qwen2/Qwen3/Qwen3MoE untouched. Forward returns "unsupported" at end of M0.

**Tech Stack:** C++17, CUDA 12.x, FlashInfer (vendored), minja (header-only), stb_image (header-only), xmake.

**Reference:** Design doc `docs/plan/qwen3_5_support.md`. M0 row in §10.

---

### Task 1: Verify branch + tooling baseline

**Files:**
- Verify: `feat/qwen3.5` branch is checked out

- [ ] **Step 1: Confirm git state**

Run: `git status && git rev-parse --abbrev-ref HEAD`
Expected: branch `feat/qwen3.5`, working tree clean except for plan files.

- [ ] **Step 2: Verify model files exist**

Run: `ls -lh ~/data/models/Qwen3.5-27B-GPTQ-Int4/config.json ~/data/models/Qwen3.5-35B-A3B-GPTQ-Int4/config.json`
Expected: Both files exist.

- [ ] **Step 3: Verify build baseline (v0.2.0 still compiles + tests pass)**

Run: `xmake f -m release --nv-gpu=y --onednn=y --flashinfer=y && xmake build && xmake run test-blockpool`
Expected: Clean build; tests pass. (Establishes "before-state" for regression detection at M0 end.)

---

### Task 2: Vendor minja header-only Jinja2

**Files:**
- Create directory: `third_party/minja/`

- [ ] **Step 1: Add minja submodule (or copy headers)**

Run:
```bash
git submodule add https://github.com/google/minja third_party/minja
git submodule update --init third_party/minja
```
If upstream URL differs, copy `minja.hpp` + `chat-template.hpp` into `third_party/minja/include/minja/`.

Verify: `ls third_party/minja/include/minja/*.hpp`
Expected: shows `minja.hpp`, `chat-template.hpp` (header-only).

- [ ] **Step 2: Update xmake.lua**

Open `xmake.lua` and find the existing `add_required_includedir(...)` block around line 90. Add:
```lua
add_required_includedir("third_party/minja/include", third_party_hint)
```

- [ ] **Step 3: Verify build still works**

Run: `xmake build`
Expected: succeeds (minja is header-only, nothing yet uses it).

- [ ] **Step 4: Smoke-test parse of Qwen3.5 chat template**

Create `tests/integration/test_minja_smoke.cpp`:
```cpp
#include <fstream>
#include <iostream>
#include <minja/minja.hpp>
#include <minja/chat-template.hpp>

int main() {
    std::ifstream f("/home/tianyux/data/models/Qwen3.5-27B-GPTQ-Int4/chat_template.jinja");
    std::string tpl((std::istreambuf_iterator<char>(f)), {});
    try {
        auto compiled = minja::chat_template(tpl, /*bos=*/"", /*eos=*/"");
        std::cout << "[ok] minja parsed Qwen3.5 chat_template.jinja\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "[fail] minja parse: " << e.what() << "\n";
        return 1;
    }
}
```

Add xmake target in `tests/xmake.lua` (or root `xmake.lua` at test section):
```lua
target("test-minja-smoke")
    set_kind("binary")
    add_files("tests/integration/test_minja_smoke.cpp")
    set_default(false)
```

- [ ] **Step 5: Build and run smoke test**

Run: `xmake build test-minja-smoke && xmake run test-minja-smoke`
Expected output contains: `[ok] minja parsed Qwen3.5 chat_template.jinja`. If parse fails, capture the exception message and add a TODO comment in the test file with the exact error — defer to Task 16 (minja patching).

- [ ] **Step 6: Commit**

```bash
git add third_party/minja .gitmodules xmake.lua tests/integration/test_minja_smoke.cpp
git commit -m "feat(qwen3.5): vendor minja for Qwen3.5 chat template parsing"
```

---

### Task 3: Vendor stb_image + stb_image_resize

**Files:**
- Create directory: `third_party/stb/`
- Create: `third_party/stb/stb_image.h`, `third_party/stb/stb_image_resize2.h`

- [ ] **Step 1: Download stb headers**

Run:
```bash
mkdir -p third_party/stb
curl -L https://raw.githubusercontent.com/nothings/stb/master/stb_image.h -o third_party/stb/stb_image.h
curl -L https://raw.githubusercontent.com/nothings/stb/master/stb_image_resize2.h -o third_party/stb/stb_image_resize2.h
```

Verify file sizes are non-trivial (each > 100 KB).

- [ ] **Step 2: Update xmake.lua**

Add to the include-dir block in `xmake.lua`:
```lua
add_required_includedir("third_party/stb", third_party_hint)
```

- [ ] **Step 3: Smoke test stb_image decodes Qwen sample image**

Create a 4x4 RGB test PNG (use any small fixture) at `tests/fixtures/tiny_4x4.png` — if creating from scratch:
```bash
python3 -c "import struct
data = bytes([
    137,80,78,71,13,10,26,10,
    # ...minimal PNG signature... use Pillow if available:
])"
```
Or use Pillow:
```bash
python3 -c "from PIL import Image; img=Image.new('RGB',(4,4),(255,0,0)); img.save('tests/fixtures/tiny_4x4.png')"
```

Create `tests/integration/test_stb_smoke.cpp`:
```cpp
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#include <iostream>

int main() {
    int w, h, ch;
    unsigned char* pixels = stbi_load("tests/fixtures/tiny_4x4.png", &w, &h, &ch, 3);
    if (!pixels) {
        std::cerr << "[fail] " << stbi_failure_reason() << "\n";
        return 1;
    }
    std::cout << "[ok] decoded " << w << "x" << h << " ch=" << ch << "\n";
    stbi_image_free(pixels);
    return 0;
}
```

xmake target:
```lua
target("test-stb-smoke")
    set_kind("binary")
    add_files("tests/integration/test_stb_smoke.cpp")
    set_default(false)
```

- [ ] **Step 4: Build and run**

Run: `xmake build test-stb-smoke && xmake run test-stb-smoke`
Expected: `[ok] decoded 4x4 ch=3`.

- [ ] **Step 5: Commit**

```bash
git add third_party/stb xmake.lua tests/integration/test_stb_smoke.cpp tests/fixtures/tiny_4x4.png
git commit -m "feat(qwen3.5): vendor stb_image for image decode/resize"
```

---

### Task 4: FlashInfer SSU link test (GO/NO-GO checkpoint)

**Files:**
- Create: `tests/integration/test_flashinfer_ssu_link.cu`
- Modify: `xmake.lua` (add target)

- [ ] **Step 1: Write the link test source**

Create `tests/integration/test_flashinfer_ssu_link.cu`:
```cuda
// Minimal compile + link probe for FlashInfer Mamba selective_state_update
// MTP path. Goal: verify the template instantiation
// (bf16 input, bf16 weight, f32 A, bf16 state, i32 stateIndex, void state_scale)
// is available at link time. Runtime correctness is NOT tested here.

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <iostream>

// FlashInfer needs DIM / DSTATE / NTOKENS_MTP as constexpr; provide them.
#define DIM 128
#define DSTATE 128
#define NTOKENS_MTP 1
#define PHILOX_ROUNDS 0

#include <flashinfer/mamba/invoke_selective_state_update_mtp.cuh>

int main() {
    using namespace flashinfer::mamba;
    SelectiveStateMTPParams params{};
    // Zero-fill params; we are only checking link, not correctness.
    cudaStream_t stream = nullptr;
    cudaStreamCreate(&stream);

    try {
        mtp::invokeSelectiveStateUpdateMTP<
            __nv_bfloat16,  // input_t
            __nv_bfloat16,  // weight_t
            float,          // matrixA_t
            __nv_bfloat16,  // state_t
            int32_t,        // stateIndex_t
            void            // state_scale_t
        >(params, SSUAlgorithm::kSimple, stream);
    } catch (...) {
        // Runtime errors expected with zero params; we only care about link.
    }
    cudaStreamDestroy(stream);
    std::cout << "[ok] SSU MTP template (bf16 input/state, f32 A) linked successfully\n";
    return 0;
}
```

- [ ] **Step 2: Add xmake target**

In `xmake.lua` (test section), add:
```lua
target("test-flashinfer-ssu-link")
    set_kind("binary")
    if has_config("flashinfer") then
        add_defines("USE_FLASHINFER")
        add_files("tests/integration/test_flashinfer_ssu_link.cu")
        add_cugencodes("native")
    else
        on_load(function() raise("test-flashinfer-ssu-link requires --flashinfer=y") end)
    end
    set_default(false)
```

- [ ] **Step 3: Build the link test**

Run: `xmake build test-flashinfer-ssu-link`

**GO/NO-GO outcomes:**
- **GO**: Build succeeds. Note exact compilation time and any warnings. Proceed.
- **NO-GO (missing instantiation)**: Linker error like `undefined reference to invokeSelectiveStateUpdateMTP<bfloat16, bfloat16, float, bfloat16, int, void>`. **Fallback A**: change `state_t` from `__nv_bfloat16` to `__half`. Rebuild. If still fails, **Fallback B**: edit `third_party/flashinfer/csrc/selective_state_update_dtype_inst.jinja` to add our dtype combo, regenerate `selective_state_update_kernel_inst.cu`, rebuild FlashInfer headers. Document the actual chosen `state_t` in `docs/plan/qwen3_5_support.md` §5.1 and §10 M0 row.

- [ ] **Step 4: Run the link test**

Run: `xmake run test-flashinfer-ssu-link`
Expected: prints `[ok] SSU MTP template (...) linked successfully` and exits 0. Runtime errors from zero-filled params are OK; we suppress them.

- [ ] **Step 5: Record GO/NO-GO outcome in design doc**

Edit `docs/plan/qwen3_5_support.md`. In §10 M0 row, append a line like:
> M0 GO confirmed 2026-05-XX: state_dtype=bf16 instantiation linked cleanly. SSMStatePool will use BF16 state.

(Or fallback note if NO-GO.)

- [ ] **Step 6: Commit**

```bash
git add tests/integration/test_flashinfer_ssu_link.cu xmake.lua docs/plan/qwen3_5_support.md
git commit -m "feat(qwen3.5): add FlashInfer SSU link probe (M0 GO/NO-GO)"
```

---

### Task 5: Extend `ModelConfig` with hybrid fields

**Files:**
- Modify: `include/frontend/models/base.hpp`

- [ ] **Step 1: Add hybrid + vision sub-config types to `base.hpp`**

After the `QuantizationConfig` struct but before `struct ModelConfig`, add:
```cpp
struct LinearAttnConfig {
    int num_v_heads = 0;
    int value_head_dim = 0;
    int num_k_heads = 0;
    int key_head_dim = 0;
    int d_state = 0;          // derived from in_proj_b shape
    int conv_kernel_dim = 4;
    std::string state_dtype;  // "bfloat16" | "float32" | "float16"
};

struct VisionConfig {
    int depth = 0;
    int hidden_size = 0;
    int out_hidden_size = 0;
    int num_heads = 0;
    int patch_size = 16;
    int temporal_patch_size = 2;
    int spatial_merge_size = 2;
    int num_position_embeddings = 0;
    int intermediate_size = 0;
};
```

Inside `struct ModelConfig`, add the new fields at the bottom (just above the destructor):
```cpp
    // Hybrid SSM+attention support (zero-valued for non-hybrid models)
    std::vector<std::string> layer_types;       // "linear_attention" | "full_attention" per layer
    bool   attn_output_gate = false;
    float  partial_rotary_factor = 1.0f;
    std::array<int,3> mrope_section { 0, 0, 0 };
    bool   mrope_interleaved = false;
    int    mtp_num_hidden_layers = 0;
    LinearAttnConfig linear_attn;
    bool   has_vision = false;
    VisionConfig vision;

    // Special-token IDs surfaced for multimodal preprocessing
    int    image_token_id = -1;
    int    video_token_id = -1;
    int    vision_start_token_id = -1;
    int    vision_end_token_id = -1;
```

Add `#include <array>` at top of `base.hpp`.

- [ ] **Step 2: Compile-check**

Run: `xmake build`
Expected: success. (No code uses new fields yet.)

- [ ] **Step 3: Commit**

```bash
git add include/frontend/models/base.hpp
git commit -m "feat(qwen3.5): extend ModelConfig with hybrid/vision/mrope fields"
```

---

### Task 6: Create `Qwen3_5Config` and parse logic

**Files:**
- Create: `include/frontend/models/qwen3_5_config.hpp`
- Modify: `src/frontend/models/base.cpp` (config parser)

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_qwen3_5_config_parse.cpp`:
```cpp
#include "frontend/models/base.hpp"
#include "frontend/models/qwen3_5_config.hpp"
#include <cassert>
#include <iostream>

using namespace zedinfer::model;

int main() {
    auto config = Model::load_config("/home/tianyux/data/models/Qwen3.5-27B-GPTQ-Int4/config.json");
    auto* qcfg = dynamic_cast<Qwen3_5Config*>(config.get());
    if (!qcfg) { std::cerr << "config is not Qwen3_5Config\n"; return 1; }

    assert(qcfg->model_type == "qwen3_5");
    assert(qcfg->hidden_size == 5120);
    assert(qcfg->num_hidden_layers == 64);
    assert(qcfg->head_dim == 256);
    assert(qcfg->num_attention_heads == 24);
    assert(qcfg->num_key_value_heads == 4);
    assert(qcfg->layer_types.size() == 64);
    assert(qcfg->layer_types[0] == "linear_attention");
    assert(qcfg->layer_types[3] == "full_attention");
    assert(qcfg->attn_output_gate == true);
    assert(qcfg->partial_rotary_factor == 0.25f);
    assert(qcfg->mrope_section[0] == 11);
    assert(qcfg->mrope_section[1] == 11);
    assert(qcfg->mrope_section[2] == 10);
    assert(qcfg->mrope_interleaved == true);
    assert(qcfg->mtp_num_hidden_layers == 1);
    assert(qcfg->linear_attn.num_v_heads == 48);
    assert(qcfg->linear_attn.value_head_dim == 128);
    assert(qcfg->linear_attn.num_k_heads == 16);
    assert(qcfg->linear_attn.conv_kernel_dim == 4);
    assert(qcfg->has_vision == true);
    assert(qcfg->vision.depth == 27);
    assert(qcfg->vision.hidden_size == 1152);
    assert(qcfg->vision.out_hidden_size == 5120);
    assert(qcfg->image_token_id == 248056);
    std::cout << "[ok] Qwen3_5Config parsed correctly\n";
    return 0;
}
```

Add xmake target:
```lua
target("test-qwen3-5-config-parse")
    set_kind("binary")
    add_files("tests/unit/test_qwen3_5_config_parse.cpp")
    add_deps("zedinfer")
    set_default(false)
```

- [ ] **Step 2: Run; expect failure**

Run: `xmake build test-qwen3-5-config-parse 2>&1 | head -20`
Expected: fails because `Qwen3_5Config` doesn't exist.

- [ ] **Step 3: Create the config class**

Create `include/frontend/models/qwen3_5_config.hpp`:
```cpp
#pragma once
#include "frontend/models/base.hpp"

namespace zedinfer::model {

struct Qwen3_5Config : public ModelConfig {
    Qwen3_5Config() = default;
    Qwen3_5Config(ModelConfig base) : ModelConfig(std::move(base)) {}
    int full_attention_interval = 4;
};

// MoE variant
struct Qwen3_5MoEConfig : public Qwen3_5Config {
    Qwen3_5MoEConfig() = default;
    Qwen3_5MoEConfig(ModelConfig base) : Qwen3_5Config(std::move(base)) {}
    int num_experts = 0;
    int num_experts_per_tok = 0;
    int moe_intermediate_size = 0;
    int shared_expert_intermediate_size = 0;
    int decoder_sparse_step = 1;
    std::vector<int> mlp_only_layers;
};

} // namespace zedinfer::model
```

- [ ] **Step 4: Extend the parser**

Open `src/frontend/models/base.cpp`. Find `load_config` (around line 575). After the existing `if (config->model_type == "qwen3_moe") { ... }` branches, add:
```cpp
} else if (model_type == "qwen3_5" || model_type == "qwen3_5_moe") {
    // Qwen3.5 nests text/vision configs under "text_config" / "vision_config"
    json text_json = j.contains("text_config") ? j["text_config"] : j;
    json vision_json = j.value("vision_config", json::object());

    // Re-apply base loads against text_config so hidden_size/etc come from the right slot
    load_base_config(base_config, text_json);
    base_config.model_type = model_type;

    // Hybrid fields
    if (text_json.contains("layer_types")) {
        for (auto& el : text_json["layer_types"]) {
            base_config.layer_types.push_back(el.get<std::string>());
        }
    }
    base_config.attn_output_gate = text_json.value("attn_output_gate", false);
    base_config.mtp_num_hidden_layers = text_json.value("mtp_num_hidden_layers", 0);

    // Rope parameters nested
    if (text_json.contains("rope_parameters")) {
        auto& rp = text_json["rope_parameters"];
        base_config.partial_rotary_factor = rp.value("partial_rotary_factor", 1.0f);
        base_config.mrope_interleaved = rp.value("mrope_interleaved", false);
        if (rp.contains("mrope_section") && rp["mrope_section"].is_array() &&
            rp["mrope_section"].size() == 3) {
            for (int i = 0; i < 3; ++i) base_config.mrope_section[i] = rp["mrope_section"][i].get<int>();
        }
        base_config.rope_theta = rp.value("rope_theta", base_config.rope_theta);
    }

    // Linear attention sub-config
    base_config.linear_attn.num_v_heads      = text_json.value("linear_num_value_heads", 0);
    base_config.linear_attn.value_head_dim   = text_json.value("linear_value_head_dim", 0);
    base_config.linear_attn.num_k_heads      = text_json.value("linear_num_key_heads", 0);
    base_config.linear_attn.key_head_dim     = text_json.value("linear_key_head_dim", 0);
    base_config.linear_attn.conv_kernel_dim  = text_json.value("linear_conv_kernel_dim", 4);
    base_config.linear_attn.state_dtype      = text_json.value("mamba_ssm_dtype", "bfloat16");
    base_config.linear_attn.d_state          = base_config.linear_attn.value_head_dim;  // d_state == value_head_dim default

    // Vision sub-config
    if (!vision_json.empty()) {
        base_config.has_vision = true;
        base_config.vision.depth                  = vision_json.value("depth", 0);
        base_config.vision.hidden_size            = vision_json.value("hidden_size", 0);
        base_config.vision.out_hidden_size        = vision_json.value("out_hidden_size", 0);
        base_config.vision.num_heads              = vision_json.value("num_heads", 0);
        base_config.vision.patch_size             = vision_json.value("patch_size", 16);
        base_config.vision.temporal_patch_size    = vision_json.value("temporal_patch_size", 2);
        base_config.vision.spatial_merge_size     = vision_json.value("spatial_merge_size", 2);
        base_config.vision.num_position_embeddings = vision_json.value("num_position_embeddings", 0);
        base_config.vision.intermediate_size      = vision_json.value("intermediate_size", 0);
    }

    // Special-token IDs
    base_config.image_token_id        = j.value("image_token_id", -1);
    base_config.video_token_id        = j.value("video_token_id", -1);
    base_config.vision_start_token_id = j.value("vision_start_token_id", -1);
    base_config.vision_end_token_id   = j.value("vision_end_token_id", -1);

    if (model_type == "qwen3_5_moe") {
        auto cfg = std::make_unique<Qwen3_5MoEConfig>(base_config);
        cfg->num_experts                       = text_json.value("num_experts", 0);
        cfg->num_experts_per_tok               = text_json.value("num_experts_per_tok", 0);
        cfg->moe_intermediate_size             = text_json.value("moe_intermediate_size", 0);
        cfg->shared_expert_intermediate_size   = text_json.value("shared_expert_intermediate_size", 0);
        cfg->decoder_sparse_step               = text_json.value("decoder_sparse_step", 1);
        if (text_json.contains("mlp_only_layers")) {
            for (auto& el : text_json["mlp_only_layers"]) cfg->mlp_only_layers.push_back(el.get<int>());
        }
        return cfg;
    }
    return std::make_unique<Qwen3_5Config>(base_config);
}
```

Add at top of `base.cpp`: `#include "frontend/models/qwen3_5_config.hpp"`

- [ ] **Step 5: Run the test**

Run: `xmake build test-qwen3-5-config-parse && xmake run test-qwen3-5-config-parse`
Expected: `[ok] Qwen3_5Config parsed correctly`.

- [ ] **Step 6: Add 35B-A3B parse assertion**

Append to `tests/unit/test_qwen3_5_config_parse.cpp` before `return 0`:
```cpp
    auto moe_cfg = Model::load_config("/home/tianyux/data/models/Qwen3.5-35B-A3B-GPTQ-Int4/config.json");
    auto* m = dynamic_cast<Qwen3_5MoEConfig*>(moe_cfg.get());
    assert(m && "expected Qwen3_5MoEConfig");
    assert(m->num_experts == 256);
    assert(m->num_experts_per_tok == 8);
    assert(m->moe_intermediate_size == 512);
    assert(m->shared_expert_intermediate_size == 512);
    assert(m->num_hidden_layers == 40);
    assert(m->linear_attn.num_v_heads == 32);
    std::cout << "[ok] Qwen3_5MoEConfig parsed correctly\n";
```

Run: `xmake build test-qwen3-5-config-parse && xmake run test-qwen3-5-config-parse`
Expected: both `[ok]` lines.

- [ ] **Step 7: Commit**

```bash
git add include/frontend/models/qwen3_5_config.hpp src/frontend/models/base.cpp \
        tests/unit/test_qwen3_5_config_parse.cpp xmake.lua
git commit -m "feat(qwen3.5): parse hybrid/vision/mrope/MoE config fields"
```

---

### Task 7: Extend `Model::map_weight_name` to strip `language_model.`

**Files:**
- Modify: `src/frontend/models/base.cpp` (`map_weight_name`)

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_qwen3_5_weight_name_map.cpp`:
```cpp
#include "frontend/models/base.hpp"
#include <cassert>
#include <iostream>
using namespace zedinfer::model;

int main() {
    // Qwen3.5 keys
    assert(Model::map_weight_name("model.language_model.embed_tokens.weight") == "embed_tokens.weight");
    assert(Model::map_weight_name("model.language_model.layers.0.linear_attn.in_proj_qkv.weight")
           == "layers.0.linear_attn.in_proj_qkv.weight");
    assert(Model::map_weight_name("model.language_model.layers.0.mlp.gate_proj.qweight")
           == "layers.0.mlp.gate_proj.weight_packed");

    // Vision keys keep their visual. prefix
    assert(Model::map_weight_name("model.visual.blocks.0.attn.qkv.weight")
           == "visual.blocks.0.attn.qkv.weight");

    // MTP keys preserved
    assert(Model::map_weight_name("mtp.layers.0.self_attn.q_proj.weight")
           == "mtp.layers.0.self_attn.q_proj.weight");

    // lm_head untouched
    assert(Model::map_weight_name("lm_head.weight") == "lm_head.weight");

    // Existing Qwen3 keys still work (no language_model. prefix)
    assert(Model::map_weight_name("model.layers.0.self_attn.q_proj.weight")
           == "layers.0.self_attn.q_proj.weight");

    std::cout << "[ok] map_weight_name handles Qwen3.5 keys + zero-regression on Qwen3\n";
    return 0;
}
```

Add xmake target.

Run: `xmake build test-qwen3-5-weight-name-map`
Expected: build succeeds but test fails (assertions trip).

- [ ] **Step 2: Update `map_weight_name` in `base.cpp`**

Find `Model::map_weight_name` (around line 1120). Replace the body:
```cpp
std::string Model::map_weight_name(const std::string& raw_name) {
    std::string name = raw_name;

    // Strip "model." prefix
    if (name.size() > 6 && name.substr(0, 6) == "model.") {
        name = name.substr(6);
    }
    // Then strip "language_model." prefix (Qwen3.5)
    if (name.size() > 15 && name.substr(0, 15) == "language_model.") {
        name = name.substr(15);
    }

    // Map GPTQ suffixes
    if (name.size() > 8 && name.substr(name.size() - 8) == ".qweight") {
        name = name.substr(0, name.size() - 8) + ".weight_packed";
    } else if (name.size() > 7 && name.substr(name.size() - 7) == ".scales") {
        name = name.substr(0, name.size() - 7) + ".weight_scale";
    } else if (name.size() > 6 && name.substr(name.size() - 6) == ".g_idx") {
        name = name.substr(0, name.size() - 6) + ".weight_g_idx";
    } else if (name.size() > 7 && name.substr(name.size() - 7) == ".qzeros") {
        name = name.substr(0, name.size() - 7) + ".weight_zeros";
    }
    return name;
}
```

- [ ] **Step 3: Run the test**

Run: `xmake build test-qwen3-5-weight-name-map && xmake run test-qwen3-5-weight-name-map`
Expected: `[ok] map_weight_name ...`.

- [ ] **Step 4: Verify regression suite still passes**

Run: `xmake run test-loader ZEDINFER_TEST_MODEL_PATH=/home/tianyux/data/models/Qwen3-30B-A3B-GPTQ-Int4` (Qwen3 path) — ensure existing Qwen3 loader still works.
Expected: all existing weight names map identically (Qwen3 keys lack `language_model.` so the second strip is a no-op).

- [ ] **Step 5: Commit**

```bash
git add src/frontend/models/base.cpp tests/unit/test_qwen3_5_weight_name_map.cpp xmake.lua
git commit -m "feat(qwen3.5): map_weight_name strips language_model. prefix"
```

---

### Task 8: `SSMStatePool` header + impl (no kernel)

**Files:**
- Create: `include/frontend/models/ssm_state_pool.hpp`
- Create: `src/frontend/models/ssm_state_pool.cpp`
- Create: `tests/unit/test_ssm_state_pool.cpp`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_ssm_state_pool.cpp`:
```cpp
#include "frontend/models/ssm_state_pool.hpp"
#include "backend/core/runtime/runtime.hpp"
#include <cassert>
#include <iostream>

using namespace zedinfer;
using namespace zedinfer::model;

int main() {
    ExecutorConfig exec{ZEDINFER_DTYPE_BF16, ZEDINFER_DEVICE_NVIDIA, 0};
    core::context().setRuntime(zedinfer::createRuntime(ZEDINFER_DEVICE_NVIDIA, 0));

    SSMStatePoolConfig cfg;
    cfg.num_linear_layers = 4;
    cfg.num_v_heads = 8;
    cfg.value_head_dim = 16;
    cfg.d_state = 16;
    cfg.conv_kernel_dim = 4;
    cfg.qkv_dim = 256;
    cfg.max_concurrent = 2;
    cfg.state_dtype = ZEDINFER_DTYPE_BF16;

    SSMStatePool pool(cfg, exec);

    // 2 slots available
    assert(pool.num_free_slots() == 2);

    int s1 = pool.acquire_slot();
    int s2 = pool.acquire_slot();
    assert(s1 != s2);
    assert(pool.num_free_slots() == 0);

    // 3rd acquire throws
    bool threw = false;
    try { pool.acquire_slot(); } catch (const std::exception&) { threw = true; }
    assert(threw);

    pool.release_slot(s1);
    assert(pool.num_free_slots() == 1);

    // reset_slot doesn't throw (kernel-free zero fill)
    pool.reset_slot(s2);

    auto view = pool.view();
    assert(view.ssm_base != nullptr);
    assert(view.conv_base != nullptr);
    assert(view.num_v_heads == 8);

    std::cout << "[ok] SSMStatePool basic slot mgmt\n";
    return 0;
}
```

Add xmake target.

Run: `xmake build test-ssm-state-pool`
Expected: fail (SSMStatePool doesn't exist).

- [ ] **Step 2: Create the header**

Create `include/frontend/models/ssm_state_pool.hpp`:
```cpp
#pragma once
#include "backend/tensor/tensor.hpp"
#include "zedinfer.h"
#include <cstdint>
#include <memory>
#include <vector>

namespace zedinfer {
struct ExecutorConfig;
}

namespace zedinfer::model {

struct SSMStatePoolConfig {
    int num_linear_layers   = 0;
    int num_v_heads         = 0;
    int value_head_dim      = 0;
    int d_state             = 0;
    int conv_kernel_dim     = 4;
    int qkv_dim             = 0;
    int max_concurrent      = 1;
    zedinferDataType_t state_dtype = ZEDINFER_DTYPE_BF16;
};

struct SSMStateView {
    void*    ssm_base = nullptr;
    void*    conv_base = nullptr;
    int64_t  ssm_stride_slot = 0;
    int64_t  ssm_stride_layer = 0;
    int64_t  conv_stride_slot = 0;
    int64_t  conv_stride_layer = 0;
    int      num_v_heads = 0;
    int      value_head_dim = 0;
    int      d_state = 0;
    int      conv_kernel_dim = 0;
    int      qkv_dim = 0;
    zedinferDataType_t dtype = ZEDINFER_DTYPE_BF16;
};

class SSMStatePool {
public:
    SSMStatePool(const SSMStatePoolConfig& cfg, const ExecutorConfig& exec);
    ~SSMStatePool() = default;

    int  acquire_slot();
    void release_slot(int slot_idx);
    void reset_slot(int slot_idx);  // zeros SSM + conv state for this slot across all linear layers

    SSMStateView view() const { return view_; }
    int num_free_slots() const;
    size_t bytes_per_slot() const { return ssm_bytes_per_slot_ + conv_bytes_per_slot_; }
    const SSMStatePoolConfig& config() const { return cfg_; }

private:
    SSMStatePoolConfig cfg_;
    tensor_t ssm_buffer_;
    tensor_t conv_buffer_;
    std::vector<char> slot_in_use_;  // bool-vector replaced for thread-safety later
    int next_hint_ = 0;
    size_t ssm_bytes_per_slot_ = 0;
    size_t conv_bytes_per_slot_ = 0;
    SSMStateView view_{};
};

} // namespace zedinfer::model
```

- [ ] **Step 3: Create the impl**

Create `src/frontend/models/ssm_state_pool.cpp`:
```cpp
#include "frontend/models/ssm_state_pool.hpp"
#include "backend/core/context/context.hpp"
#include "backend/tensor/tensor.hpp"
#include "zedinfer.h"

#include <plog/Log.h>
#include <stdexcept>

namespace zedinfer::model {

static size_t dtype_bytes(zedinferDataType_t dt) {
    switch (dt) {
        case ZEDINFER_DTYPE_F32:  return 4;
        case ZEDINFER_DTYPE_BF16: return 2;
        case ZEDINFER_DTYPE_F16:  return 2;
        default: throw std::runtime_error("Unsupported SSM state dtype");
    }
}

SSMStatePool::SSMStatePool(const SSMStatePoolConfig& cfg, const ExecutorConfig& exec)
    : cfg_(cfg), slot_in_use_(cfg.max_concurrent, 0) {
    const size_t state_dt = dtype_bytes(cfg.state_dtype);

    // ssm buffer shape: [slots, layers, num_v_heads, value_head_dim, d_state]
    ssm_buffer_ = Tensor::create(
        {(size_t)cfg.max_concurrent, (size_t)cfg.num_linear_layers,
         (size_t)cfg.num_v_heads, (size_t)cfg.value_head_dim, (size_t)cfg.d_state},
        cfg.state_dtype, exec.device_type, exec.device_id);

    // conv buffer shape: [slots, layers, kernel-1, qkv_dim]
    conv_buffer_ = Tensor::create(
        {(size_t)cfg.max_concurrent, (size_t)cfg.num_linear_layers,
         (size_t)(cfg.conv_kernel_dim - 1), (size_t)cfg.qkv_dim},
        cfg.state_dtype, exec.device_type, exec.device_id);

    ssm_bytes_per_slot_  = (size_t)cfg.num_linear_layers * cfg.num_v_heads * cfg.value_head_dim * cfg.d_state * state_dt;
    conv_bytes_per_slot_ = (size_t)cfg.num_linear_layers * (cfg.conv_kernel_dim - 1) * cfg.qkv_dim * state_dt;

    view_.ssm_base          = ssm_buffer_->data();
    view_.conv_base         = conv_buffer_->data();
    view_.ssm_stride_slot   = (int64_t)ssm_bytes_per_slot_;
    view_.ssm_stride_layer  = (int64_t)cfg.num_v_heads * cfg.value_head_dim * cfg.d_state * state_dt;
    view_.conv_stride_slot  = (int64_t)conv_bytes_per_slot_;
    view_.conv_stride_layer = (int64_t)(cfg.conv_kernel_dim - 1) * cfg.qkv_dim * state_dt;
    view_.num_v_heads       = cfg.num_v_heads;
    view_.value_head_dim    = cfg.value_head_dim;
    view_.d_state           = cfg.d_state;
    view_.conv_kernel_dim   = cfg.conv_kernel_dim;
    view_.qkv_dim           = cfg.qkv_dim;
    view_.dtype             = cfg.state_dtype;

    LOGI << "[SSMStatePool] max_concurrent=" << cfg.max_concurrent
         << " layers=" << cfg.num_linear_layers
         << " ssm_bytes/slot=" << ssm_bytes_per_slot_
         << " conv_bytes/slot=" << conv_bytes_per_slot_
         << " total=" << (cfg.max_concurrent * bytes_per_slot()) << " bytes";
}

int SSMStatePool::acquire_slot() {
    for (int i = 0; i < cfg_.max_concurrent; ++i) {
        int idx = (next_hint_ + i) % cfg_.max_concurrent;
        if (!slot_in_use_[idx]) {
            slot_in_use_[idx] = 1;
            next_hint_ = (idx + 1) % cfg_.max_concurrent;
            return idx;
        }
    }
    throw std::runtime_error("SSMStatePool: no free slots (max_concurrent=" + std::to_string(cfg_.max_concurrent) + ")");
}

void SSMStatePool::release_slot(int slot_idx) {
    if (slot_idx < 0 || slot_idx >= cfg_.max_concurrent) return;
    slot_in_use_[slot_idx] = 0;
}

void SSMStatePool::reset_slot(int slot_idx) {
    // Zero SSM + conv state for this slot. Use fill_zero op (CPU + NVIDIA already exists).
    if (slot_idx < 0 || slot_idx >= cfg_.max_concurrent) return;
    auto* runtime = &core::context().runtime();
    auto stream = runtime->compute_stream();

    // ssm slice: ssm_buffer_[slot_idx, :, :, :, :]
    size_t bytes_ssm  = ssm_bytes_per_slot_;
    size_t bytes_conv = conv_bytes_per_slot_;
    runtime->memsetAsync((char*)ssm_buffer_->data()  + slot_idx * bytes_ssm,  0, bytes_ssm,  stream);
    runtime->memsetAsync((char*)conv_buffer_->data() + slot_idx * bytes_conv, 0, bytes_conv, stream);
}

int SSMStatePool::num_free_slots() const {
    int free_count = 0;
    for (char c : slot_in_use_) if (!c) free_count++;
    return free_count;
}

} // namespace zedinfer::model
```

- [ ] **Step 4: Add files to xmake**

In `xmake.lua` find the `frontend` target's `add_files` and ensure it includes `src/frontend/models/*.cpp` (it should already glob).

Verify Runtime has `memsetAsync`. If not, add it to `include/backend/device/runtime_api.hpp` + per-device impl. (Run `grep -rn memsetAsync include/backend src/backend` to check; if missing, add a stub that calls `cudaMemsetAsync` for NVIDIA / `std::memset` for CPU.)

- [ ] **Step 5: Run the test**

Run: `xmake build test-ssm-state-pool && xmake run test-ssm-state-pool`
Expected: `[ok] SSMStatePool basic slot mgmt`.

- [ ] **Step 6: Commit**

```bash
git add include/frontend/models/ssm_state_pool.hpp src/frontend/models/ssm_state_pool.cpp \
        tests/unit/test_ssm_state_pool.cpp xmake.lua
git commit -m "feat(qwen3.5): add SSMStatePool with slot mgmt + zero-reset"
```

---

### Task 9: `HybridForwardConfig` header

**Files:**
- Create: `include/frontend/models/hybrid_forward_config.hpp`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_hybrid_forward_config.cpp`:
```cpp
#include "frontend/models/hybrid_forward_config.hpp"
#include "frontend/models/qwen3_5_config.hpp"
#include <cassert>
#include <iostream>

using namespace zedinfer::model;

int main() {
    Qwen3_5Config qcfg;
    qcfg.layer_types = {
        "linear_attention","linear_attention","linear_attention","full_attention",
        "linear_attention","linear_attention","linear_attention","full_attention"
    };
    qcfg.num_hidden_layers = 8;
    ModelWeights weights;

    HybridForwardConfig hcfg(qcfg, weights);
    // populate layer kinds
    for (size_t L = 0; L < qcfg.num_hidden_layers; ++L) {
        hcfg.layer_kinds.push_back(qcfg.layer_types[L] == "linear_attention"
                                    ? LayerKind::Linear : LayerKind::Full);
    }

    assert(hcfg.is_linear_attn_layer(0));
    assert(hcfg.is_linear_attn_layer(1));
    assert(hcfg.is_linear_attn_layer(2));
    assert(!hcfg.is_linear_attn_layer(3));
    assert(hcfg.full_layer_index(3) == 0);
    assert(hcfg.full_layer_index(7) == 1);
    assert(hcfg.linear_layer_index(0) == 0);
    assert(hcfg.linear_layer_index(4) == 3);

    std::cout << "[ok] HybridForwardConfig indexes computed correctly\n";
    return 0;
}
```

Add xmake target. Run: `xmake build test-hybrid-forward-config` — expect fail.

- [ ] **Step 2: Create the header**

Create `include/frontend/models/hybrid_forward_config.hpp`:
```cpp
#pragma once
#include "frontend/models/forward_config.hpp"
#include "frontend/models/ssm_state_pool.hpp"
#include <array>
#include <vector>

namespace zedinfer::model {

enum class LayerKind : uint8_t { Full = 0, Linear = 1 };

struct MRoPEConfig {
    bool   interleaved = true;
    std::array<int,3> section { 11, 11, 10 };
    float  partial_factor = 0.25f;
    float  theta = 1e7f;
};

struct HybridForwardConfig : public ModelForwardConfig {
    HybridForwardConfig(const ModelConfig& c, const ModelWeights& w)
        : ModelForwardConfig(c, w) {}

    std::vector<LayerKind> layer_kinds;          // per-layer
    LinearAttnConfig       linear_attn;          // copy from ModelConfig
    MRoPEConfig            mrope;
    bool                   attn_output_gate = true;
    SSMStatePool*          ssm_pool = nullptr;

    bool is_linear_attn_layer(size_t L) const { return layer_kinds[L] == LayerKind::Linear; }

    int full_layer_index(size_t L) const {
        int idx = 0;
        for (size_t i = 0; i < L; ++i) if (layer_kinds[i] == LayerKind::Full) idx++;
        return idx;
    }
    int linear_layer_index(size_t L) const {
        int idx = 0;
        for (size_t i = 0; i < L; ++i) if (layer_kinds[i] == LayerKind::Linear) idx++;
        return idx;
    }
};

} // namespace zedinfer::model
```

- [ ] **Step 3: Run the test**

Run: `xmake build test-hybrid-forward-config && xmake run test-hybrid-forward-config`
Expected: `[ok] HybridForwardConfig indexes computed correctly`.

- [ ] **Step 4: Commit**

```bash
git add include/frontend/models/hybrid_forward_config.hpp tests/unit/test_hybrid_forward_config.cpp xmake.lua
git commit -m "feat(qwen3.5): HybridForwardConfig with layer-kind dispatch helpers"
```

---

### Task 10: `Qwen3_5Model` skeleton (ctor + forward stub)

**Files:**
- Create: `include/frontend/models/qwen3_5.hpp`
- Create: `src/frontend/models/qwen3_5.cpp`

- [ ] **Step 1: Write the header**

Create `include/frontend/models/qwen3_5.hpp`:
```cpp
#pragma once
#include "frontend/models/base.hpp"
#include "frontend/models/qwen3_5_config.hpp"
#include "frontend/models/hybrid_forward_config.hpp"
#include "frontend/models/ssm_state_pool.hpp"
#include <memory>

namespace zedinfer::model {

class VisionTower;  // fwd-decl; vision_tower.hpp included from cpp

class Qwen3_5Model : public Model {
public:
    Qwen3_5Model(Qwen3_5Config config, std::unique_ptr<ModelWeights> weights,
                 const ExecutorConfig& exec, int max_concurrent);
    ~Qwen3_5Model() override;

    const ModelConfig& config() const override { return config_; }
    const ModelWeights& weights() const override { return *weights_; }
    std::string model_type() const override { return "qwen3_5"; }
    size_t num_parameters() const override;

    ModelForwardConfig forward_config() const override;
    HybridForwardConfig hybrid_forward_config() const;

    SSMStatePool& ssm_state_pool() { return *ssm_pool_; }
    const VisionTower* vision_tower() const { return vision_.get(); }

protected:
    Qwen3_5Config config_;
    std::unique_ptr<ModelWeights> weights_;
    std::unique_ptr<SSMStatePool> ssm_pool_;
    std::unique_ptr<VisionTower> vision_;
};

} // namespace zedinfer::model
```

- [ ] **Step 2: Write the impl**

Create `src/frontend/models/qwen3_5.cpp`:
```cpp
#include "frontend/models/qwen3_5.hpp"
#include "frontend/models/vision_tower.hpp"
#include <plog/Log.h>
#include <stdexcept>

namespace zedinfer::model {

Qwen3_5Model::Qwen3_5Model(Qwen3_5Config config, std::unique_ptr<ModelWeights> weights,
                            const ExecutorConfig& exec, int max_concurrent)
    : config_(std::move(config)), weights_(std::move(weights)) {

    // Count linear-attention layers and qkv_dim for pool sizing
    int num_linear = 0;
    for (auto& t : config_.layer_types) {
        if (t == "linear_attention") num_linear++;
    }
    int qkv_dim = config_.linear_attn.num_k_heads * config_.linear_attn.key_head_dim * 2
                + config_.linear_attn.num_v_heads * config_.linear_attn.value_head_dim;

    SSMStatePoolConfig pool_cfg;
    pool_cfg.num_linear_layers = num_linear;
    pool_cfg.num_v_heads       = config_.linear_attn.num_v_heads;
    pool_cfg.value_head_dim    = config_.linear_attn.value_head_dim;
    pool_cfg.d_state           = config_.linear_attn.d_state;
    pool_cfg.conv_kernel_dim   = config_.linear_attn.conv_kernel_dim;
    pool_cfg.qkv_dim           = qkv_dim;
    pool_cfg.max_concurrent    = std::max(1, max_concurrent);
    pool_cfg.state_dtype       = config_.linear_attn.state_dtype == "float32"
                                  ? ZEDINFER_DTYPE_F32 : ZEDINFER_DTYPE_BF16;
    ssm_pool_ = std::make_unique<SSMStatePool>(pool_cfg, exec);

    // Vision tower (M3 will fill in forward; ctor loads weights now)
    if (config_.has_vision) {
        vision_ = std::make_unique<VisionTower>(config_.vision, *weights_, exec);
    }

    LOGI << "[Qwen3_5Model] constructed: " << num_linear << " linear layers, "
         << (config_.num_hidden_layers - num_linear) << " full-attn layers, "
         << "vision=" << (vision_ ? "yes" : "no");
}

Qwen3_5Model::~Qwen3_5Model() = default;

size_t Qwen3_5Model::num_parameters() const {
    size_t total = 0;
    for (const auto& [_, t] : weights_->get_all_weights()) total += t->numel();
    return total;
}

ModelForwardConfig Qwen3_5Model::forward_config() const {
    // Stub: forward not implemented in M0
    throw std::runtime_error("Qwen3_5Model::forward_config: not implemented until M1");
}

HybridForwardConfig Qwen3_5Model::hybrid_forward_config() const {
    HybridForwardConfig h(config_, *weights_);
    for (auto& t : config_.layer_types) {
        h.layer_kinds.push_back(t == "linear_attention" ? LayerKind::Linear : LayerKind::Full);
    }
    h.linear_attn = config_.linear_attn;
    h.mrope.interleaved    = config_.mrope_interleaved;
    h.mrope.section        = config_.mrope_section;
    h.mrope.partial_factor = config_.partial_rotary_factor;
    h.mrope.theta          = config_.rope_theta;
    h.attn_output_gate     = config_.attn_output_gate;
    h.ssm_pool             = ssm_pool_.get();
    return h;
}

} // namespace zedinfer::model
```

- [ ] **Step 3: Smoke test (load 27B, build model, do not forward)**

Create `tests/integration/test_qwen3_5_load.cpp`:
```cpp
#include "frontend/models/base.hpp"
#include "frontend/models/qwen3_5.hpp"
#include "backend/core/runtime/runtime.hpp"
#include <iostream>

int main() {
    using namespace zedinfer;
    using namespace zedinfer::model;
    core::context().setRuntime(createRuntime(ZEDINFER_DEVICE_NVIDIA, 0));

    auto model = Model::parse("/home/tianyux/data/models/Qwen3.5-27B-GPTQ-Int4",
                              ZEDINFER_DEVICE_NVIDIA);
    if (model->model_type() != "qwen3_5") {
        std::cerr << "expected qwen3_5, got " << model->model_type() << "\n";
        return 1;
    }
    auto* q = dynamic_cast<Qwen3_5Model*>(model.get());
    if (!q) { std::cerr << "downcast failed\n"; return 1; }

    std::cout << "[ok] loaded Qwen3.5-27B, params=" << q->num_parameters()
              << ", ssm_pool free_slots=" << q->ssm_state_pool().num_free_slots() << "\n";
    return 0;
}
```

- [ ] **Step 4: VisionTower stub (forward not implemented)**

Create minimal `include/frontend/models/vision_tower.hpp`:
```cpp
#pragma once
#include "frontend/models/base.hpp"
#include "backend/tensor/tensor.hpp"

namespace zedinfer {
struct ExecutorConfig;
}

namespace zedinfer::model {

class VisionTower {
public:
    VisionTower(const VisionConfig& cfg, const ModelWeights& w, const ExecutorConfig& exec);
    ~VisionTower();
    tensor_t forward(tensor_t patches, tensor_t pos_ids_thw, const ExecutorConfig& exec);
private:
    VisionConfig cfg_;
};

} // namespace zedinfer::model
```

And `src/frontend/models/vision_tower.cpp`:
```cpp
#include "frontend/models/vision_tower.hpp"
#include <plog/Log.h>
#include <stdexcept>

namespace zedinfer::model {

VisionTower::VisionTower(const VisionConfig& cfg, const ModelWeights& w, const ExecutorConfig& exec)
    : cfg_(cfg) {
    // M0 only verifies that vision weights are present; full forward arrives in M3.
    int needed_blocks = cfg.depth;
    for (int i = 0; i < needed_blocks; ++i) {
        std::string prefix = "visual.blocks." + std::to_string(i) + ".";
        if (!w.has_tensor(prefix + "attn.qkv.weight")) {
            throw std::runtime_error("[VisionTower] missing weight: " + prefix + "attn.qkv.weight");
        }
    }
    if (!w.has_tensor("visual.patch_embed.proj.weight")) {
        throw std::runtime_error("[VisionTower] missing patch_embed weights");
    }
    if (!w.has_tensor("visual.merger.linear_fc2.weight")) {
        throw std::runtime_error("[VisionTower] missing merger weights");
    }
    LOGI << "[VisionTower] ctor verified " << needed_blocks << " blocks + patch_embed + merger present";
}

VisionTower::~VisionTower() = default;

tensor_t VisionTower::forward(tensor_t, tensor_t, const ExecutorConfig&) {
    throw std::runtime_error("VisionTower::forward not implemented until M3");
}

} // namespace zedinfer::model
```

- [ ] **Step 5: Hook Qwen3.5 into `Model::parse`**

In `src/frontend/models/base.cpp::Model::parse`, after the `qwen3_moe` branch and before `throw "Unsupported model type"`, add:
```cpp
} else if (config->model_type == "qwen3_5") {
    auto* qcfg = dynamic_cast<Qwen3_5Config*>(config.get());
    if (!qcfg) throw std::logic_error("Config is not Qwen3_5Config");
    int max_concurrent = 1;  // M0 default; revisited in M5
    ExecutorConfig exec{ZEDINFER_DTYPE_BF16, target_device, 0};
    return std::make_shared<Qwen3_5Model>(*qcfg, std::move(weights), exec, max_concurrent);
```

(35B-A3B `qwen3_5_moe` branch comes in Task 11.)

Add `#include "frontend/models/qwen3_5.hpp"` at top of `base.cpp`.

- [ ] **Step 6: Build and run smoke test**

Run:
```bash
xmake build test-qwen3-5-load
xmake run test-qwen3-5-load 2>&1 | tee /tmp/qwen3_5_load.log
```
Expected: `[ok] loaded Qwen3.5-27B, params=...`. Parameter count should be in the 27 B range (~27.4 B).

If load fails with "Tensor not found": check the loader log for which weight name didn't get stripped. Capture in Task 14.

- [ ] **Step 7: Commit**

```bash
git add include/frontend/models/qwen3_5.hpp src/frontend/models/qwen3_5.cpp \
        include/frontend/models/vision_tower.hpp src/frontend/models/vision_tower.cpp \
        src/frontend/models/base.cpp tests/integration/test_qwen3_5_load.cpp xmake.lua
git commit -m "feat(qwen3.5): Qwen3_5Model + VisionTower skeletons; Model::parse dispatch"
```

---

### Task 11: `Qwen3_5MoeModel` skeleton

**Files:**
- Create: `include/frontend/models/qwen3_5_moe.hpp`
- Create: `src/frontend/models/qwen3_5_moe.cpp`

- [ ] **Step 1: Header**

```cpp
// include/frontend/models/qwen3_5_moe.hpp
#pragma once
#include "frontend/models/qwen3_5.hpp"
#include "frontend/models/expert_pool.hpp"
#include "frontend/models/expert_weights.hpp"

namespace zedinfer::model {

class Qwen3_5MoeModel : public Qwen3_5Model {
public:
    Qwen3_5MoeModel(Qwen3_5MoEConfig config, std::unique_ptr<ModelWeights> weights,
                     const ExecutorConfig& exec, int max_concurrent,
                     ExpertPoolConfig pool_cfg);
    ~Qwen3_5MoeModel() override;

    std::string model_type() const override { return "qwen3_5_moe"; }
    ExpertPool& expert_pool() { return *expert_pool_; }

private:
    Qwen3_5MoEConfig moe_config_;
    std::unique_ptr<ExpertPool> expert_pool_;
};

} // namespace zedinfer::model
```

- [ ] **Step 2: Impl**

```cpp
// src/frontend/models/qwen3_5_moe.cpp
#include "frontend/models/qwen3_5_moe.hpp"
#include <plog/Log.h>

namespace zedinfer::model {

Qwen3_5MoeModel::Qwen3_5MoeModel(Qwen3_5MoEConfig config, std::unique_ptr<ModelWeights> weights,
                                   const ExecutorConfig& exec, int max_concurrent,
                                   ExpertPoolConfig pool_cfg)
    : Qwen3_5Model(config, nullptr /*set below*/, exec, max_concurrent),
      moe_config_(std::move(config)) {

    // Re-take ownership; we passed nullptr to base ctor placeholder
    weights_ = std::move(weights);

    // Extract experts following v0.2.0 Qwen3MoEModel pattern (reuse expert_weights.hpp)
    auto experts = ExpertWeights::extract_from(*weights_, moe_config_);
    expert_pool_ = std::make_unique<ExpertPool>(std::move(experts), pool_cfg);

    LOGI << "[Qwen3_5MoeModel] constructed: experts=" << moe_config_.num_experts
         << " top_k=" << moe_config_.num_experts_per_tok
         << " shared_expert_size=" << moe_config_.shared_expert_intermediate_size;
}

Qwen3_5MoeModel::~Qwen3_5MoeModel() = default;

} // namespace zedinfer::model
```

Note: above `Qwen3_5Model` ctor must accept `nullptr` weights OR rework constructor signature. Simpler: change `Qwen3_5MoeModel` to take base config differently. Rewrite as composition: peek at existing `Qwen3MoEModel` for pattern.

If composition is cleaner, refactor: make `Qwen3_5Model` ctor accept `weights` by reference and don't transfer ownership inside MoE child. Update both as needed.

- [ ] **Step 3: Extend `ExpertWeights::extract_from` for `Qwen3_5MoEConfig`**

In `include/frontend/models/expert_weights.hpp`, find the existing `extract_from(weights, Qwen3MoEConfig)`. Add an overload for `Qwen3_5MoEConfig` that uses the same `mlp.experts.<eid>.{gate,up,down}_proj.weight_packed/weight_scale` naming (which after `language_model.` strip is identical to Qwen3-MoE post-strip).

If the existing extract_from is config-agnostic (just iterates layers × experts), no overload is needed.

- [ ] **Step 4: `Model::parse` MoE branch**

In `src/frontend/models/base.cpp::Model::parse`, after the new `qwen3_5` branch, add:
```cpp
} else if (config->model_type == "qwen3_5_moe") {
    auto* qcfg = dynamic_cast<Qwen3_5MoEConfig*>(config.get());
    if (!qcfg) throw std::logic_error("Config is not Qwen3_5MoEConfig");
    ExecutorConfig exec{ZEDINFER_DTYPE_BF16, target_device, 0};
    // ExpertPool config from env var, same as Qwen3-MoE
    ExpertPoolConfig pool_cfg = compute_moe_pool_config_qwen3_5(*qcfg, target_device, gpu_memory_utilization);
    return std::make_shared<Qwen3_5MoeModel>(*qcfg, std::move(weights), exec, 1, pool_cfg);
```

Provide `compute_moe_pool_config_qwen3_5` as a thin adapter — internally identical to v0.2.0's `compute_moe_pool_config` but typed for `Qwen3_5MoEConfig`. Add it in `base.cpp` near the existing helper.

Also extend the MoE-expert routing predicate above the load_weights call:
```cpp
if (config->model_type == "qwen3_5_moe" && target_device != ZEDINFER_DEVICE_CPU) {
    auto* moe_cfg = dynamic_cast<const Qwen3_5MoEConfig*>(config.get());
    if (moe_cfg) {
        moe_pool_config = compute_moe_pool_config_qwen3_5(*moe_cfg, target_device, gpu_memory_utilization);
        if (moe_pool_config.strategy == ExpertPoolStrategy::PINNED_LRU) {
            LOGI << "[Model] Routing Qwen3.5 MoE expert tensors to CPU pinned memory";
            route = [](const std::string& name) { return name.find(".mlp.experts.") != std::string::npos; };
        }
    }
}
```

- [ ] **Step 5: Smoke test 35B-A3B load**

Append to `tests/integration/test_qwen3_5_load.cpp`:
```cpp
{
    auto model = Model::parse("/home/tianyux/data/models/Qwen3.5-35B-A3B-GPTQ-Int4",
                              ZEDINFER_DEVICE_NVIDIA);
    if (model->model_type() != "qwen3_5_moe") { std::cerr << "expected qwen3_5_moe\n"; return 1; }
    auto* m = dynamic_cast<Qwen3_5MoeModel*>(model.get());
    std::cout << "[ok] loaded Qwen3.5-35B-A3B, params=" << m->num_parameters() << "\n";
}
```

Run: `xmake build test-qwen3-5-load && ZEDINFER_MOE_GPU_SLOTS=32 xmake run test-qwen3-5-load`
Expected: both 27B and 35B-A3B load with `[ok]`.

- [ ] **Step 6: Commit**

```bash
git add include/frontend/models/qwen3_5_moe.hpp src/frontend/models/qwen3_5_moe.cpp \
        src/frontend/models/base.cpp tests/integration/test_qwen3_5_load.cpp
git commit -m "feat(qwen3.5): Qwen3_5MoeModel + 35B-A3B dispatch with ExpertPool"
```

---

### Task 12: `ChatTemplate` minja-backed entry

**Files:**
- Modify: `include/zedinfer/chat_template.hpp`
- Create: `src/zedinfer/chat_template_jinja.cpp`

- [ ] **Step 1: Extend `chat_template.hpp` with content variant + factory**

Open `include/zedinfer/chat_template.hpp`. Add to top:
```cpp
#include <variant>
```
Add types before the existing `struct ChatMessage`:
```cpp
struct TextPart  { std::string text; };
struct ImagePart { std::string data_uri; };
using ContentPart = std::variant<TextPart, ImagePart>;
```
Update `ChatMessage`:
```cpp
struct ChatMessage {
    std::string role;
    // Either a plain string or an array of content parts (OpenAI Vision API).
    std::variant<std::string, std::vector<ContentPart>> content;
    // existing fields preserved (reasoning_content, tool_calls, etc.)
    ...
};
```

Inside `class ChatTemplate`, add a new factory:
```cpp
static ChatTemplate load_jinja(const std::string& template_file);
```

- [ ] **Step 2: Implement minja-backed apply**

Create `src/zedinfer/chat_template_jinja.cpp`:
```cpp
#include "zedinfer/chat_template.hpp"
#include <fstream>
#include <minja/chat-template.hpp>
#include <minja/minja.hpp>
#include <plog/Log.h>
#include <sstream>
#include <stdexcept>

namespace zedinfer {

namespace {

// Convert ChatMessage list to JSON understood by minja
nlohmann::json messages_to_json(const std::vector<ChatMessage>& msgs) {
    nlohmann::json arr = nlohmann::json::array();
    for (const auto& m : msgs) {
        nlohmann::json e;
        e["role"] = m.role;
        if (auto* s = std::get_if<std::string>(&m.content)) {
            e["content"] = *s;
        } else {
            const auto& parts = std::get<std::vector<ContentPart>>(m.content);
            nlohmann::json content_arr = nlohmann::json::array();
            for (const auto& p : parts) {
                nlohmann::json item;
                if (auto* tp = std::get_if<TextPart>(&p)) {
                    item["type"] = "text";
                    item["text"] = tp->text;
                } else if (auto* ip = std::get_if<ImagePart>(&p)) {
                    item["type"] = "image";
                    item["image"] = ip->data_uri;
                }
                content_arr.push_back(item);
            }
            e["content"] = content_arr;
        }
        arr.push_back(e);
    }
    return arr;
}

} // namespace

ChatTemplate ChatTemplate::load_jinja(const std::string& template_file) {
    std::ifstream f(template_file);
    if (!f.is_open()) throw std::runtime_error("Cannot open chat template: " + template_file);
    std::stringstream ss; ss << f.rdbuf();
    ChatTemplate self;
    self.template_source_ = ss.str();
    self.use_jinja_ = true;
    return self;
}

std::string ChatTemplate::apply(const std::vector<ChatMessage>& msgs,
                                  bool add_generation_prompt,
                                  bool enable_thinking) const {
    if (use_jinja_) {
        try {
            minja::chat_template tpl(template_source_, /*bos=*/"", /*eos=*/"");
            minja::chat_template_inputs in;
            in.messages = messages_to_json(msgs);
            in.add_generation_prompt = add_generation_prompt;
            in.extra_context["enable_thinking"] = enable_thinking;
            return tpl.apply(in);
        } catch (const std::exception& e) {
            throw std::runtime_error("[ChatTemplate jinja] apply failed: " + std::string(e.what()));
        }
    }
    // Fall through to existing hardcoded paths (Qwen2/Qwen3/DeepSeek-R1)
    return apply_hardcoded(msgs, add_generation_prompt, enable_thinking);
}

} // namespace zedinfer
```

(Adjust `apply_hardcoded` to whatever name the existing apply uses; rename existing `apply` to `apply_hardcoded` and route via the new `apply` wrapper.)

- [ ] **Step 3: Hook Qwen3.5 model into ChatTemplate**

In `src/frontend/models/qwen3_5.cpp` Qwen3_5Model ctor, add chat template load:
```cpp
chat_template_ = std::make_unique<ChatTemplate>(
    ChatTemplate::load_jinja(model_path + "/chat_template.jinja"));
```

(Add `chat_template_` member + `model_path` parameter to ctor.)

- [ ] **Step 4: Unit test**

Create `tests/unit/test_qwen3_5_chat_template.cpp`:
```cpp
#include "zedinfer/chat_template.hpp"
#include <cassert>
#include <iostream>

using namespace zedinfer;

int main() {
    auto tpl = ChatTemplate::load_jinja(
        "/home/tianyux/data/models/Qwen3.5-27B-GPTQ-Int4/chat_template.jinja");

    std::vector<ChatMessage> msgs;
    ChatMessage m;
    m.role = "user";
    m.content = std::string("Who are you?");
    msgs.push_back(m);

    std::string rendered = tpl.apply(msgs, /*add_generation_prompt=*/true);
    // Sanity check: rendered should contain <|im_start|>user and <|im_start|>assistant
    assert(rendered.find("<|im_start|>user") != std::string::npos);
    assert(rendered.find("Who are you?") != std::string::npos);
    assert(rendered.find("<|im_start|>assistant") != std::string::npos);
    std::cout << "[ok] Qwen3.5 chat template rendered:\n" << rendered << "\n";
    return 0;
}
```

Add xmake target. Run: `xmake build test-qwen3-5-chat-template && xmake run test-qwen3-5-chat-template`
Expected: prints rendered prompt containing `<|im_start|>user\nWho are you?<|im_end|>` etc.

If minja throws on a Qwen3.5-specific feature: drop into the minja source (`third_party/minja/include/minja/minja.hpp`) and locate the failing syntax. Patch minimally with a comment `// qwen3.5: <reason>`. Re-run.

- [ ] **Step 5: Commit**

```bash
git add include/zedinfer/chat_template.hpp src/zedinfer/chat_template_jinja.cpp \
        src/frontend/models/qwen3_5.cpp tests/unit/test_qwen3_5_chat_template.cpp xmake.lua
git commit -m "feat(qwen3.5): minja-backed chat template; Qwen3.5 loads chat_template.jinja"
```

---

### Task 13: `ping` end-to-end load + forward-stub

**Files:**
- Modify: `src/examples/ping.cpp` (or wherever the ping binary lives — `find . -name 'ping*.cpp'`)

- [ ] **Step 1: Locate ping source**

Run: `find . -name 'ping*' -not -path './build*' 2>&1`
Note the path (typically `src/examples/ping.cpp` or `examples/ping.cpp`).

- [ ] **Step 2: Verify ping can `Model::parse` Qwen3.5 path**

The existing ping should already call `Model::parse`. The only addition needed: in the unhandled-model-type branch (or wherever forward is invoked), catch the "not implemented until M1" exception and print a friendly message.

Open the ping source; find where it calls `model->forward(...)` (or scheduler→engine→model). Wrap with:
```cpp
try {
    // ... existing forward invocation ...
} catch (const std::exception& e) {
    std::string msg = e.what();
    if (msg.find("not implemented until M1") != std::string::npos) {
        std::cout << "[ping] Qwen3.5 forward path is M1 work-in-progress; exiting clean\n";
        return 0;
    }
    throw;
}
```

- [ ] **Step 3: Run ping on 27B**

Run:
```bash
xmake run ping ~/data/models/Qwen3.5-27B-GPTQ-Int4 --nvidia 2>&1 | tee /tmp/qwen3_5_ping.log
```
Expected: load succeeds; chat template renders; reaches forward; prints "M1 work-in-progress" and exits 0.

Verify log mentions:
- All 1775 weights loaded (or similar count line)
- SSMStatePool constructed line
- VisionTower constructed line (27 blocks verified)
- ChatTemplate jinja path used

- [ ] **Step 4: Run ping on 35B-A3B**

Run:
```bash
ZEDINFER_MOE_GPU_SLOTS=32 xmake run ping ~/data/models/Qwen3.5-35B-A3B-GPTQ-Int4 --nvidia 2>&1 | tee /tmp/qwen3_5_moe_ping.log
```
Expected: similar clean exit. ExpertPool constructed (256 experts).

- [ ] **Step 5: Regression check**

Run existing Qwen3 ping:
```bash
xmake run ping ~/data/models/Qwen3-30B-A3B-GPTQ-Int4 --nvidia
```
Expected: identical behavior to before. ChatML hardcoded template still used. No SSU / hybrid path taken.

- [ ] **Step 6: Commit**

```bash
git add src/examples/ping.cpp
git commit -m "feat(qwen3.5): ping accepts Qwen3.5 models with M1-pending forward exit"
```

---

### Task 14: M0 DoD verification + spec update

- [ ] **Step 1: Run full M0 checklist**

| Check | Command | Expected |
|---|---|---|
| Build clean | `xmake build` | success |
| SSU link probe | `xmake run test-flashinfer-ssu-link` | `[ok]` |
| Config parse | `xmake run test-qwen3-5-config-parse` | `[ok]` × 2 |
| Weight name map | `xmake run test-qwen3-5-weight-name-map` | `[ok]` |
| SSMStatePool | `xmake run test-ssm-state-pool` | `[ok]` |
| Hybrid config | `xmake run test-hybrid-forward-config` | `[ok]` |
| Chat template | `xmake run test-qwen3-5-chat-template` | `[ok]` |
| 27B load | `xmake run test-qwen3-5-load` | `[ok]` × 2 |
| 27B ping | `xmake run ping ~/data/models/Qwen3.5-27B-GPTQ-Int4 --nvidia` | "M1 WIP" exit 0 |
| 35B-A3B ping | `ZEDINFER_MOE_GPU_SLOTS=32 xmake run ping ...35B-A3B... --nvidia` | "M1 WIP" exit 0 |
| Qwen3 regression | `xmake run ping ~/data/models/Qwen3-30B-A3B-GPTQ-Int4 --nvidia` | unchanged behavior |
| Existing tests | `xmake run test-blockpool && xmake run test-prefixcache && xmake run test-sampler` | all pass |

Capture any failures in a `/tmp/m0_punchlist.md` and fix before moving on.

- [ ] **Step 2: Update spec doc with M0 done marker**

In `docs/plan/qwen3_5_support.md` §10 M0 row, append:
```
M0 complete: <commit-sha-of-last-M0-commit>, 27B + 35B-A3B both reach forward stub cleanly. SSU template instantiation: <state_dtype-chosen>.
```

- [ ] **Step 3: Commit doc update**

```bash
git add docs/plan/qwen3_5_support.md
git commit -m "docs(qwen3.5): M0 complete; record SSU instantiation outcome"
```

---

### Task 15: M0 retrospective + M1 handoff notes

- [ ] **Step 1: Write a brief M0 retro in `docs/plan/qwen3_5_session_handoff.md`**

Create `docs/plan/qwen3_5_session_handoff.md` (mirror style of `moe_session_handoff.md`):
```markdown
# Qwen3.5 Session Handoff

## M0 — Load + parse (complete)

### What landed
- Vendored minja (`third_party/minja`) + stb (`third_party/stb`)
- ModelConfig hybrid/vision/mrope/MoE fields
- `Qwen3_5Config` / `Qwen3_5MoEConfig`
- `Model::map_weight_name` strips `language_model.` prefix
- `SSMStatePool` (no kernel; ctor + slot mgmt + zero-reset)
- `HybridForwardConfig` with full/linear layer-index helpers
- `Qwen3_5Model` / `Qwen3_5MoeModel` ctor; forward stub
- VisionTower ctor only
- ChatTemplate jinja path (Qwen3.5 uses it; others unchanged)
- FlashInfer SSU link probe (state_dtype = <bf16 | half>)
- `ping` exits cleanly on Qwen3.5; existing Qwen3 untouched

### Caveats / open items for M1
- `forward_config()` and `forward()` still throw "not implemented"
- `transformer_forward` doesn't yet accept `input_embeds` parameter
- `Scheduler::admit` doesn't yet consult `SSMStatePool::num_free_slots()`
- `PagedForwardContext::write_kv/attend` still use raw layer index, not `full_layer_index`
- SSMStatePool memsetAsync may need to be added if Runtime API lacks it (see Task 8 step 4 note)

### M1 entry conditions confirmed
- FlashInfer SSU template (bf16 input, <chosen> state) compiles + links
- minja successfully parses chat_template.jinja for both Qwen3.5 variants
- Both Qwen3.5 model files parse and load all weights without name-not-found errors
```

- [ ] **Step 2: Commit handoff doc**

```bash
git add docs/plan/qwen3_5_session_handoff.md
git commit -m "docs(qwen3.5): M0 retrospective and M1 handoff notes"
```

---

## M0 Done. Total: ~15 tasks. Est. 1 week.

When done, the engineer should be able to run `xmake run ping ~/data/models/Qwen3.5-27B-GPTQ-Int4 --nvidia` and see all weights load, both pools construct, chat template render, and a clean "M1 work-in-progress" exit. Hand off to P2 plan.
