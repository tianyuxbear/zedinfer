#include "frontend/models/vision_tower.hpp"

#include <plog/Log.h>

#include <stdexcept>
#include <string>

namespace zedinfer::model {

VisionTower::VisionTower(const VisionConfig& cfg, const ModelWeights& w, const ExecutorConfig& /*exec*/)
    : cfg_(cfg) {
    // M0 only verifies that vision weights are present in the loaded weight map;
    // the full forward pass arrives in M3. Use a descriptive error message so
    // post-load failures are easy to trace back to a specific missing tensor.
    const int needed_blocks = cfg.depth;
    for (int i = 0; i < needed_blocks; ++i) {
        const std::string prefix = "visual.blocks." + std::to_string(i) + ".";
        if (!w.has_tensor(prefix + "attn.qkv.weight")) {
            throw std::runtime_error("[VisionTower] missing weight: " + prefix + "attn.qkv.weight");
        }
    }
    if (!w.has_tensor("visual.patch_embed.proj.weight")) {
        throw std::runtime_error("[VisionTower] missing patch_embed weight: visual.patch_embed.proj.weight");
    }
    if (!w.has_tensor("visual.merger.linear_fc2.weight")) {
        throw std::runtime_error("[VisionTower] missing merger weight: visual.merger.linear_fc2.weight");
    }
    LOGI.printf("[VisionTower] ctor verified %d blocks + patch_embed + merger present (hidden=%d, out=%d, heads=%d)",
                needed_blocks, cfg.hidden_size, cfg.out_hidden_size, cfg.num_heads);
}

VisionTower::~VisionTower() = default;

tensor_t VisionTower::forward(tensor_t /*patches*/, tensor_t /*pos_ids_thw*/, const ExecutorConfig& /*exec*/) {
    throw std::runtime_error("VisionTower::forward: not implemented until M3");
}

} // namespace zedinfer::model
