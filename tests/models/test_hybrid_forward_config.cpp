#include "frontend/models/base.hpp"
#include "frontend/models/hybrid_forward_config.hpp"
#include "frontend/models/qwen3_5_config.hpp"

#include <gtest/gtest.h>

using zedinfer::model::HybridForwardConfig;
using zedinfer::model::LayerKind;
using zedinfer::model::ModelWeights;
using zedinfer::model::Qwen3_5Config;

namespace {

// Build an 8-layer hybrid layout: 3 linear : 1 full (idx 3, idx 7 are Full).
// This mirrors the Qwen3.5 reference pattern (full_attention_interval = 4).
HybridForwardConfig make_8layer_config(const Qwen3_5Config& cfg, const ModelWeights& w) {
    HybridForwardConfig fcfg(cfg, w);
    fcfg.layer_kinds = {
        LayerKind::Linear, // 0
        LayerKind::Linear, // 1
        LayerKind::Linear, // 2
        LayerKind::Full,   // 3
        LayerKind::Linear, // 4
        LayerKind::Linear, // 5
        LayerKind::Linear, // 6
        LayerKind::Full,   // 7
    };
    return fcfg;
}

} // namespace

TEST(HybridForwardConfigTest, IsLinearAttnLayer) {
    Qwen3_5Config cfg;
    cfg.num_hidden_layers = 8;
    ModelWeights weights;
    const auto fcfg = make_8layer_config(cfg, weights);

    EXPECT_TRUE(fcfg.is_linear_attn_layer(0));
    EXPECT_TRUE(fcfg.is_linear_attn_layer(1));
    EXPECT_TRUE(fcfg.is_linear_attn_layer(2));
    EXPECT_FALSE(fcfg.is_linear_attn_layer(3));
    EXPECT_TRUE(fcfg.is_linear_attn_layer(4));
    EXPECT_FALSE(fcfg.is_linear_attn_layer(7));
}

TEST(HybridForwardConfigTest, FullLayerIndex) {
    Qwen3_5Config cfg;
    cfg.num_hidden_layers = 8;
    ModelWeights weights;
    const auto fcfg = make_8layer_config(cfg, weights);

    // Layer 3 is the first full layer -> index 0 among full layers.
    EXPECT_EQ(fcfg.full_layer_index(3), 0);
    // Layer 7 is the second full layer -> index 1 among full layers.
    EXPECT_EQ(fcfg.full_layer_index(7), 1);
}

TEST(HybridForwardConfigTest, LinearLayerIndex) {
    Qwen3_5Config cfg;
    cfg.num_hidden_layers = 8;
    ModelWeights weights;
    const auto fcfg = make_8layer_config(cfg, weights);

    // Layer 0 is the first linear layer -> index 0.
    EXPECT_EQ(fcfg.linear_layer_index(0), 0);
    // Layer 4 comes after 3 linear (0,1,2) + 1 full (3) -> linear index 3.
    EXPECT_EQ(fcfg.linear_layer_index(4), 3);
}
