#pragma once

#include "backend/tensor/tensor.hpp"
#include "frontend/models/base.hpp"

namespace zedinfer {
struct ExecutorConfig;
}

namespace zedinfer::model {

// Qwen3.5-VL vision tower. ctor verifies weights exist in the model's weight
// map; forward() runs the 27-block ViT and the 2x2 spatial merger to produce
// LLM-side image embeddings (shape: [num_image_tokens, out_hidden_size]).
class VisionTower {
public:
    VisionTower(const VisionConfig& cfg, const ModelWeights& w, const ExecutorConfig& exec);
    ~VisionTower();

    // Run the vision encoder on the patches produced by MultiModalProcessor.
    //   patches      [N_patches, in_channels * temporal_patch_size * patch_size * patch_size]
    //   pos_ids_thw  [N_patches, 3]  (t, h, w) — reserved for the future 3D
    //                MRoPE wiring; currently the simplified impl uses a flat
    //                pos_embed lookup keyed on (h * grid_w + w).
    //   grid_h, grid_w   spatial grid (in patches) of the input image. Needed
    //                for the merger's 2x2 spatial grouping.
    //   exec         executor config (device + dtype)
    // Returns: [num_image_tokens, out_hidden_size] tensor on `exec` device.
    tensor_t forward(tensor_t patches, tensor_t pos_ids_thw, int grid_h, int grid_w, const ExecutorConfig& exec);

    const VisionConfig& config() const { return cfg_; }

private:
    VisionConfig cfg_;
    const ModelWeights* weights_ = nullptr; // borrowed, valid for lifetime

    // Pre-computed static resources captured at ctor — invariant across all
    // images / requests. Caching them here avoids per-request D2H of the
    // pos_embed table (~5 MB for Qwen3.5-VL) and re-computation of the RoPE
    // inv_freq vector.
    std::vector<float> pos_embed_table_f32_; // [num_position_embeddings, hidden_size]
    int num_grid_per_side_ = 0;
    std::vector<float> rope_inv_freq_;       // [head_dim/4]
};

} // namespace zedinfer::model
