#include "frontend/models/base.hpp"

#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <string>

using namespace zedinfer::model;
namespace fs = std::filesystem;

static std::string write_temp_config(const std::string& content) {
    const auto temp_path = fs::temp_directory_path() / "zedinfer_test_config.json";
    std::ofstream out(temp_path);
    out << content;
    out.close();
    return temp_path.string();
}

TEST(ModelConfigParseTest, ParsesExtendedQuantizationFields) {
    const std::string config_content = R"JSON({
  "architectures": ["Qwen3ForCausalLM"],
  "bos_token_id": 151643,
  "eos_token_id": 151645,
  "hidden_act": "silu",
  "hidden_size": 4096,
  "intermediate_size": 12288,
  "max_position_embeddings": 131072,
  "max_window_layers": 36,
  "model_type": "qwen3",
  "num_attention_heads": 32,
  "num_hidden_layers": 36,
  "num_key_value_heads": 8,
  "quantization_config": {
    "config_groups": {
      "group_0": {
        "input_activations": null,
        "output_activations": null,
        "targets": ["Linear"],
        "weights": {
          "actorder": null,
          "block_structure": null,
          "dynamic": false,
          "group_size": null,
          "num_bits": 8,
          "observer": "minmax",
          "observer_kwargs": {},
          "strategy": "channel",
          "symmetric": true,
          "type": "int"
        }
      }
    },
    "format": "pack-quantized",
    "global_compression_ratio": null,
    "ignore": ["lm_head"],
    "kv_cache_scheme": null,
    "quant_method": "compressed-tensors",
    "quantization_status": "compressed"
  },
  "rms_norm_eps": 1e-06,
  "rope_theta": 1000000,
  "tie_word_embeddings": false,
  "torch_dtype": "bfloat16",
  "use_sliding_window": false,
  "vocab_size": 151936
})JSON";

    const auto config_path = write_temp_config(config_content);
    auto config = Model::load_config(config_path);

    ASSERT_NE(config, nullptr);
    EXPECT_EQ(config->architectures.size(), 1);
    EXPECT_EQ(config->architectures[0], "Qwen3ForCausalLM");

    const auto& quant = config->quant_config;
    EXPECT_TRUE(quant.enabled);
    EXPECT_EQ(quant.quant_method, "compressed-tensors");
    EXPECT_EQ(quant.format, "pack-quantized");
    EXPECT_EQ(quant.quantization_status, "compressed");
    EXPECT_TRUE(quant.raw.is_object());
    EXPECT_TRUE(quant.config_groups_raw.is_object());
    EXPECT_TRUE(quant.global_compression_ratio.is_null());
    EXPECT_TRUE(quant.kv_cache_scheme.is_null());

    EXPECT_EQ(quant.ignored_layers.size(), 1);
    EXPECT_EQ(quant.ignored_layers[0], "lm_head");
    EXPECT_EQ(quant.target_modules.size(), 1);
    EXPECT_EQ(quant.target_modules[0], "Linear");

    EXPECT_EQ(quant.weights.num_bits, 8);
    EXPECT_EQ(quant.weights.group_size, -1);
    EXPECT_TRUE(quant.weights.symmetric);
    EXPECT_FALSE(quant.weights.dynamic);
    EXPECT_EQ(quant.weights.strategy, "channel");
    EXPECT_FALSE(quant.weights.act_order);
    EXPECT_EQ(quant.weights.observer, "minmax");
    EXPECT_EQ(quant.weights.value_type, "int");
    EXPECT_TRUE(quant.weights.block_structure.is_null());
    EXPECT_TRUE(quant.weights.observer_kwargs.is_object());
    EXPECT_TRUE(quant.weights.raw.is_object());

    EXPECT_EQ(quant.activations.num_bits, 0);
    EXPECT_EQ(quant.output_activations.num_bits, 0);
}

TEST(ModelConfigParseTest, KeepsBackwardCompatibilityForMinimalQuantConfig) {
    const std::string config_content = R"JSON({
  "bos_token_id": 1,
  "eos_token_id": 2,
  "hidden_act": "silu",
  "hidden_size": 128,
  "intermediate_size": 256,
  "max_position_embeddings": 2048,
  "max_window_layers": 8,
  "model_type": "qwen3",
  "num_attention_heads": 8,
  "num_hidden_layers": 8,
  "num_key_value_heads": 8,
  "quantization_config": {
    "config_groups": {
      "group_0": {
        "weights": {
          "num_bits": 4,
          "group_size": 128,
          "actorder": true,
          "strategy": "group"
        }
      }
    },
    "format": "packed",
    "quant_method": "gptq"
  },
  "rms_norm_eps": 1e-06,
  "rope_theta": 10000,
  "tie_word_embeddings": false,
  "torch_dtype": "bfloat16",
  "use_sliding_window": false,
  "vocab_size": 32000
})JSON";

    const auto config_path = write_temp_config(config_content);
    auto config = Model::load_config(config_path);

    ASSERT_NE(config, nullptr);
    EXPECT_TRUE(config->quant_config.enabled);
    EXPECT_EQ(config->quant_config.quant_method, "gptq");
    EXPECT_EQ(config->quant_config.format, "packed");
    EXPECT_EQ(config->quant_config.weights.num_bits, 4);
    EXPECT_EQ(config->quant_config.weights.group_size, 128);
    EXPECT_TRUE(config->quant_config.weights.act_order);
}

TEST(ModelConfigParseTest, ParsesCompressedTensorsWithArrayEosTokenIds) {
    const std::string config_content = R"JSON({
  "architectures": ["Qwen2ForCausalLM"],
  "bos_token_id": 151643,
  "eos_token_id": [151643, 151645],
  "hidden_act": "silu",
  "hidden_size": 1536,
  "intermediate_size": 8960,
  "max_position_embeddings": 131072,
  "max_window_layers": 21,
  "model_type": "qwen2",
  "num_attention_heads": 12,
  "num_hidden_layers": 28,
  "num_key_value_heads": 2,
  "quantization_config": {
    "config_groups": {
      "group_0": {
        "input_activations": {
          "dynamic": true,
          "group_size": null,
          "num_bits": 8,
          "observer": null,
          "observer_kwargs": {},
          "strategy": "token",
          "symmetric": true,
          "type": "int"
        },
        "output_activations": null,
        "targets": ["Linear"],
        "weights": {
          "dynamic": false,
          "group_size": null,
          "num_bits": 8,
          "observer": "mse",
          "observer_kwargs": {},
          "strategy": "channel",
          "symmetric": true,
          "type": "int"
        }
      }
    },
    "format": "int-quantized",
    "global_compression_ratio": 1.5267357912400792,
    "ignore": ["lm_head"],
    "kv_cache_scheme": null,
    "quant_method": "compressed-tensors",
    "quantization_status": "compressed"
  },
  "rms_norm_eps": 1e-06,
  "rope_theta": 10000,
  "tie_word_embeddings": false,
  "torch_dtype": "bfloat16",
  "use_sliding_window": false,
  "vocab_size": 151936
})JSON";

    const auto config_path = write_temp_config(config_content);
    auto config = Model::load_config(config_path);

    ASSERT_NE(config, nullptr);
    EXPECT_EQ(config->model_type, "qwen2");
    ASSERT_EQ(config->eos_token_ids.size(), 2);
    EXPECT_EQ(config->eos_token_ids[0], 151643);
    EXPECT_EQ(config->eos_token_ids[1], 151645);
    EXPECT_EQ(config->eos_token_id, 151643);

    const auto& quant = config->quant_config;
    EXPECT_TRUE(quant.enabled);
    EXPECT_EQ(quant.quant_method, "compressed-tensors");
    EXPECT_EQ(quant.format, "int-quantized");
    EXPECT_EQ(quant.weights.num_bits, 8);
    EXPECT_EQ(quant.weights.group_size, -1);
    EXPECT_EQ(quant.activations.strategy, "token");
    EXPECT_TRUE(quant.activations.dynamic);
}
