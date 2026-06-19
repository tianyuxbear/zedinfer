#include "frontend/models/base.hpp"
#include "frontend/models/qwen3_5_config.hpp"

#include <cstdlib>
#include <filesystem>
#include <gtest/gtest.h>
#include <string>

using namespace zedinfer::model;
namespace fs = std::filesystem;

namespace {

// Resolve a model directory either from an env var (preferred) or skip the test.
// Returns an empty string when the env var is unset OR the path does not contain
// a config.json — callers should GTEST_SKIP() in that case so the suite stays
// green on machines without the model files.
std::string resolve_model_dir(const char* env_name) {
    const char* env = std::getenv(env_name);
    if (env == nullptr || *env == '\0') {
        return {};
    }
    fs::path p(env);
    if (!fs::exists(p / "config.json")) {
        return {};
    }
    return p.string();
}

} // namespace

TEST(Qwen3_5ConfigParseTest, ParsesDenseQwen3_5Config) {
    const std::string model_dir = resolve_model_dir("ZEDINFER_TEST_QWEN3_5_DENSE_PATH");
    if (model_dir.empty()) {
        GTEST_SKIP() << "ZEDINFER_TEST_QWEN3_5_DENSE_PATH not set or config.json missing";
    }

    const std::string config_path = (fs::path(model_dir) / "config.json").string();
    auto config = Model::load_config(config_path);
    ASSERT_NE(config, nullptr);

    auto* qcfg = dynamic_cast<Qwen3_5Config*>(config.get());
    ASSERT_NE(qcfg, nullptr) << "expected Qwen3_5Config (got base ModelConfig)";

    // ---- Identity / base fields (sourced from text_config) ----
    EXPECT_EQ(qcfg->model_type, "qwen3_5");
    EXPECT_EQ(qcfg->hidden_size, 5120u);
    EXPECT_EQ(qcfg->intermediate_size, 17408u);
    EXPECT_EQ(qcfg->num_hidden_layers, 64u);
    EXPECT_EQ(qcfg->num_attention_heads, 24u);
    EXPECT_EQ(qcfg->num_key_value_heads, 4u);
    EXPECT_EQ(qcfg->head_dim, 256u);
    EXPECT_EQ(qcfg->vocab_size, 248320u);
    EXPECT_EQ(qcfg->max_position_embeddings, 262144u);
    EXPECT_EQ(qcfg->torch_dtype, "bfloat16");

    // ---- Hybrid attention layout ----
    ASSERT_EQ(qcfg->layer_types.size(), 64u);
    EXPECT_EQ(qcfg->layer_types[0], "linear_attention");
    EXPECT_EQ(qcfg->layer_types[3], "full_attention");
    EXPECT_TRUE(qcfg->attn_output_gate);
    EXPECT_EQ(qcfg->mtp_num_hidden_layers, 1);

    // ---- MRoPE / partial rotary ----
    EXPECT_FLOAT_EQ(qcfg->partial_rotary_factor, 0.25f);
    EXPECT_TRUE(qcfg->mrope_interleaved);
    EXPECT_EQ(qcfg->mrope_section[0], 11);
    EXPECT_EQ(qcfg->mrope_section[1], 11);
    EXPECT_EQ(qcfg->mrope_section[2], 10);
    EXPECT_FLOAT_EQ(qcfg->rope_theta, 10000000.0f);

    // ---- Linear attention sub-config ----
    EXPECT_EQ(qcfg->linear_attn.num_v_heads, 48);
    EXPECT_EQ(qcfg->linear_attn.value_head_dim, 128);
    EXPECT_EQ(qcfg->linear_attn.num_k_heads, 16);
    EXPECT_EQ(qcfg->linear_attn.key_head_dim, 128);
    EXPECT_EQ(qcfg->linear_attn.conv_kernel_dim, 4);
    EXPECT_EQ(qcfg->linear_attn.state_dtype, "float32");

    // ---- Vision sub-config ----
    EXPECT_TRUE(qcfg->has_vision);
    EXPECT_EQ(qcfg->vision.depth, 27);
    EXPECT_EQ(qcfg->vision.hidden_size, 1152);
    EXPECT_EQ(qcfg->vision.out_hidden_size, 5120);
    EXPECT_EQ(qcfg->vision.num_heads, 16);
    EXPECT_EQ(qcfg->vision.patch_size, 16);
    EXPECT_EQ(qcfg->vision.temporal_patch_size, 2);
    EXPECT_EQ(qcfg->vision.spatial_merge_size, 2);
    EXPECT_EQ(qcfg->vision.num_position_embeddings, 2304);
    EXPECT_EQ(qcfg->vision.intermediate_size, 4304);

    // ---- Top-level special-token IDs ----
    EXPECT_EQ(qcfg->image_token_id, 248056);
    EXPECT_EQ(qcfg->video_token_id, 248057);
    EXPECT_EQ(qcfg->vision_start_token_id, 248053);
    EXPECT_EQ(qcfg->vision_end_token_id, 248054);

    // ---- Quantization (GPTQ-Int4) ----
    EXPECT_TRUE(qcfg->quant_config.enabled);
    EXPECT_EQ(qcfg->quant_config.quant_method, "gptq");
    EXPECT_EQ(qcfg->quant_config.weights.num_bits, 4);
    EXPECT_EQ(qcfg->quant_config.weights.group_size, 128);
}

TEST(Qwen3_5ConfigParseTest, ParsesMoEQwen3_5Config) {
    const std::string model_dir = resolve_model_dir("ZEDINFER_TEST_QWEN3_5_MOE_PATH");
    if (model_dir.empty()) {
        GTEST_SKIP() << "ZEDINFER_TEST_QWEN3_5_MOE_PATH not set or config.json missing";
    }

    const std::string config_path = (fs::path(model_dir) / "config.json").string();
    auto config = Model::load_config(config_path);
    ASSERT_NE(config, nullptr);

    auto* mcfg = dynamic_cast<Qwen3_5MoEConfig*>(config.get());
    ASSERT_NE(mcfg, nullptr) << "expected Qwen3_5MoEConfig (got base/dense)";

    // ---- Identity / base fields (sourced from text_config) ----
    EXPECT_EQ(mcfg->model_type, "qwen3_5_moe");
    EXPECT_EQ(mcfg->hidden_size, 2048u);
    EXPECT_EQ(mcfg->num_hidden_layers, 40u);
    EXPECT_EQ(mcfg->num_attention_heads, 16u);
    EXPECT_EQ(mcfg->num_key_value_heads, 2u);
    EXPECT_EQ(mcfg->head_dim, 256u);
    EXPECT_EQ(mcfg->vocab_size, 248320u);
    EXPECT_EQ(mcfg->max_position_embeddings, 262144u);

    // ---- Hybrid attention layout ----
    ASSERT_EQ(mcfg->layer_types.size(), 40u);
    EXPECT_EQ(mcfg->layer_types[0], "linear_attention");
    EXPECT_EQ(mcfg->layer_types[3], "full_attention");
    EXPECT_TRUE(mcfg->attn_output_gate);
    EXPECT_EQ(mcfg->mtp_num_hidden_layers, 1);

    // ---- MoE-specific fields ----
    EXPECT_EQ(mcfg->num_experts, 256);
    EXPECT_EQ(mcfg->num_experts_per_tok, 8);
    EXPECT_EQ(mcfg->moe_intermediate_size, 512);
    EXPECT_EQ(mcfg->shared_expert_intermediate_size, 512);
    EXPECT_EQ(mcfg->decoder_sparse_step, 1);
    EXPECT_TRUE(mcfg->mlp_only_layers.empty());

    // ---- MRoPE / partial rotary ----
    EXPECT_FLOAT_EQ(mcfg->partial_rotary_factor, 0.25f);
    EXPECT_TRUE(mcfg->mrope_interleaved);
    EXPECT_EQ(mcfg->mrope_section[0], 11);
    EXPECT_EQ(mcfg->mrope_section[1], 11);
    EXPECT_EQ(mcfg->mrope_section[2], 10);
    EXPECT_FLOAT_EQ(mcfg->rope_theta, 10000000.0f);

    // ---- Linear attention sub-config (MoE variant has fewer V heads) ----
    EXPECT_EQ(mcfg->linear_attn.num_v_heads, 32);
    EXPECT_EQ(mcfg->linear_attn.value_head_dim, 128);
    EXPECT_EQ(mcfg->linear_attn.num_k_heads, 16);
    EXPECT_EQ(mcfg->linear_attn.key_head_dim, 128);
    EXPECT_EQ(mcfg->linear_attn.conv_kernel_dim, 4);
    EXPECT_EQ(mcfg->linear_attn.state_dtype, "float32");

    // ---- Vision sub-config (MoE out_hidden_size matches text hidden_size) ----
    EXPECT_TRUE(mcfg->has_vision);
    EXPECT_EQ(mcfg->vision.depth, 27);
    EXPECT_EQ(mcfg->vision.hidden_size, 1152);
    EXPECT_EQ(mcfg->vision.out_hidden_size, 2048);
    EXPECT_EQ(mcfg->vision.num_heads, 16);

    // ---- Top-level special-token IDs ----
    EXPECT_EQ(mcfg->image_token_id, 248056);
    EXPECT_EQ(mcfg->video_token_id, 248057);
    EXPECT_EQ(mcfg->vision_start_token_id, 248053);
    EXPECT_EQ(mcfg->vision_end_token_id, 248054);

    // ---- Quantization (GPTQ-Int4) ----
    EXPECT_TRUE(mcfg->quant_config.enabled);
    EXPECT_EQ(mcfg->quant_config.quant_method, "gptq");
    EXPECT_EQ(mcfg->quant_config.weights.num_bits, 4);
    EXPECT_EQ(mcfg->quant_config.weights.group_size, 128);
}
