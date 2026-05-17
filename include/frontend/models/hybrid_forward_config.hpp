#pragma once

#include "frontend/models/forward_config.hpp"
#include "frontend/models/ssm_state_pool.hpp"

#include <array>
#include <cstdint>
#include <vector>

namespace zedinfer::model {

// Per-layer kind tag for hybrid SSM + softmax attention models (Qwen3.5 family).
// Stored compactly in HybridForwardConfig::layer_kinds (one entry per layer).
enum class LayerKind : uint8_t { Full = 0, Linear = 1 };

// Multi-axis RoPE config (Qwen3.5 / Qwen2-VL family). `section` partitions the
// rotary head dim across (T, H, W) axes; `partial_factor` selects the fraction
// of the head dim that gets rotated.
struct MRoPEConfig {
    bool                interleaved    = true;
    std::array<int, 3>  section{ 11, 11, 10 };
    float               partial_factor = 0.25f;
    float               theta          = 1e7f;
};

// Forward-pass config for the Qwen3.5 hybrid stack. Extends ModelForwardConfig
// with the layer-kind layout + SSM/MRoPE/linear-attention pointers that the
// shared transformer loop needs to dispatch per-layer to the correct path.
struct HybridForwardConfig : public ModelForwardConfig {
    HybridForwardConfig(const ModelConfig& c, const ModelWeights& w) : ModelForwardConfig(c, w) {}

    // One entry per decoder layer; size must equal num_hidden_layers after init.
    std::vector<LayerKind> layer_kinds;

    // Sub-config copies. Owned by the forward config so dispatch sites do not
    // chase pointers back through ModelConfig on every call.
    LinearAttnConfig       linear_attn;
    MRoPEConfig            mrope;

    // Whether the softmax-attention path applies an output gate (sigmoid * out).
    // Qwen3.5: true. Qwen2 / Qwen3: false.
    bool                   attn_output_gate = true;

    // Engine-owned SSM slot pool. nullptr for non-hybrid models or when the
    // pool has not yet been wired (early init).
    SSMStatePool*          ssm_pool = nullptr;

    // Precomputed lookup tables: full_layer_index_table[L] = index of layer L
    // among full-attention layers (and similarly for linear). Populated by
    // rebuild_layer_index_tables() after layer_kinds is filled. M1 decode hot
    // path indexes these in O(1) instead of scanning layer_kinds per call.
    std::vector<int> full_layer_index_table;
    std::vector<int> linear_layer_index_table;

    // Repopulate full_/linear_layer_index_table from the current layer_kinds.
    // Callers must invoke this once after assigning layer_kinds (and again if
    // the layout ever changes — which it does not for a constructed model).
    void rebuild_layer_index_tables() {
        full_layer_index_table.assign(layer_kinds.size(), 0);
        linear_layer_index_table.assign(layer_kinds.size(), 0);
        int full_n = 0;
        int linear_n = 0;
        for (size_t i = 0; i < layer_kinds.size(); ++i) {
            full_layer_index_table[i] = full_n;
            linear_layer_index_table[i] = linear_n;
            if (layer_kinds[i] == LayerKind::Full) {
                ++full_n;
            } else {
                ++linear_n;
            }
        }
    }

    bool is_linear_attn_layer(size_t L) const { return layer_kinds[L] == LayerKind::Linear; }

    // Index of layer L among the full-attention layers (0 for the first full
    // layer, 1 for the second, ...). Used to address per-full-layer scratch /
    // KV-cache slots without paying for the linear layers in between. O(1)
    // after rebuild_layer_index_tables; falls back to O(L) scan otherwise so
    // the helpers stay safe to call on partially-initialized configs.
    int full_layer_index(size_t L) const {
        if (L < full_layer_index_table.size()) {
            return full_layer_index_table[L];
        }
        int idx = 0;
        for (size_t i = 0; i < L; ++i) {
            if (layer_kinds[i] == LayerKind::Full) {
                ++idx;
            }
        }
        return idx;
    }

    // Index of layer L among the linear-attention layers. Mirrors
    // full_layer_index for the SSM slot/conv-state addressing path.
    int linear_layer_index(size_t L) const {
        if (L < linear_layer_index_table.size()) {
            return linear_layer_index_table[L];
        }
        int idx = 0;
        for (size_t i = 0; i < L; ++i) {
            if (layer_kinds[i] == LayerKind::Linear) {
                ++idx;
            }
        }
        return idx;
    }
};

} // namespace zedinfer::model
