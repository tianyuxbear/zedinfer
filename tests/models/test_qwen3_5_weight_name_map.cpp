#include "frontend/models/base.hpp"
#include <gtest/gtest.h>

using zedinfer::model::Model;

TEST(Qwen3_5WeightNameMap, StripsLanguageModelPrefix) {
    EXPECT_EQ(Model::map_weight_name("model.language_model.embed_tokens.weight"),
              "embed_tokens.weight");
    EXPECT_EQ(Model::map_weight_name("model.language_model.layers.0.linear_attn.in_proj_qkv.weight"),
              "layers.0.linear_attn.in_proj_qkv.weight");
    EXPECT_EQ(Model::map_weight_name("model.language_model.layers.0.mlp.gate_proj.qweight"),
              "layers.0.mlp.gate_proj.weight_packed");
}

TEST(Qwen3_5WeightNameMap, PreservesVisionAndMtpPrefixes) {
    // Vision keys: strip "model." but keep "visual." prefix.
    EXPECT_EQ(Model::map_weight_name("model.visual.blocks.0.attn.qkv.weight"),
              "visual.blocks.0.attn.qkv.weight");
    // MTP keys: have no "model." prefix at top, kept as-is.
    EXPECT_EQ(Model::map_weight_name("mtp.layers.0.self_attn.q_proj.weight"),
              "mtp.layers.0.self_attn.q_proj.weight");
    // lm_head untouched.
    EXPECT_EQ(Model::map_weight_name("lm_head.weight"), "lm_head.weight");
}

TEST(Qwen3_5WeightNameMap, Qwen3RegressionUnchanged) {
    // Existing Qwen3 keys (no "language_model." prefix) — second strip must be a no-op.
    EXPECT_EQ(Model::map_weight_name("model.layers.0.self_attn.q_proj.weight"),
              "layers.0.self_attn.q_proj.weight");
    EXPECT_EQ(Model::map_weight_name("model.embed_tokens.weight"),
              "embed_tokens.weight");
    // GPTQ-packed weight: strip "model." and remap ".qweight" -> ".weight_packed"
    // (the segment ".gate_proj" must be preserved, not collapsed).
    EXPECT_EQ(Model::map_weight_name("model.layers.5.mlp.experts.7.gate_proj.qweight"),
              "layers.5.mlp.experts.7.gate_proj.weight_packed");
}
