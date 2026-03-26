#include "backend/ops/ops.hpp"
#include "backend/tensor/tensor.hpp"
#include "frontend/models/forward_config.hpp"
#include "frontend/models/base.hpp"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <vector>

using namespace zedinfer;
using namespace zedinfer::model;
namespace fs = std::filesystem;
using json = nlohmann::json;

namespace {

struct TestTensorFile {
    std::string name;
    std::string dtype;
    std::vector<size_t> shape;
    std::vector<std::byte> data;
};

template <typename T>
std::vector<std::byte> to_bytes(const std::vector<T> &values) {
    std::vector<std::byte> bytes(values.size() * sizeof(T));
    std::memcpy(bytes.data(), values.data(), bytes.size());
    return bytes;
}

static fs::path make_temp_model_dir() {
    const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
    auto dir = fs::temp_directory_path() / ("zedinfer_quant_model_" + std::to_string(stamp));
    fs::create_directories(dir);
    return dir;
}

static void write_config(const fs::path &dir, const std::string &config_json) {
    std::ofstream out(dir / "config.json");
    out << config_json;
}

static void write_safetensors(
    const fs::path &dir,
    const std::vector<TestTensorFile> &tensors) {
    json header = json::object();
    size_t offset = 0;
    for (const auto &tensor : tensors) {
        header[tensor.name] = {
            {"dtype", tensor.dtype},
            {"shape", tensor.shape},
            {"data_offsets", {offset, offset + tensor.data.size()}}
        };
        offset += tensor.data.size();
    }

    const std::string header_str = header.dump();
    const uint64_t header_size = header_str.size();

    std::ofstream out(dir / "model.safetensors", std::ios::binary);
    out.write(reinterpret_cast<const char *>(&header_size), sizeof(header_size));
    out.write(header_str.data(), static_cast<std::streamsize>(header_str.size()));
    for (const auto &tensor : tensors) {
        out.write(reinterpret_cast<const char *>(tensor.data.data()),
                  static_cast<std::streamsize>(tensor.data.size()));
    }
}

static std::string int8_quant_config_json() {
    return R"JSON({
  "architectures": ["Qwen3ForCausalLM"],
  "bos_token_id": 151643,
  "eos_token_id": 151645,
  "hidden_act": "silu",
  "hidden_size": 8,
  "intermediate_size": 16,
  "max_position_embeddings": 64,
  "max_window_layers": 2,
  "model_type": "qwen3",
  "num_attention_heads": 2,
  "num_hidden_layers": 1,
  "num_key_value_heads": 2,
  "quantization_config": {
    "config_groups": {
      "group_0": {
        "targets": ["Linear"],
        "weights": {
          "dynamic": false,
          "group_size": null,
          "num_bits": 8,
          "strategy": "channel",
          "symmetric": true,
          "type": "int"
        }
      }
    },
    "format": "pack-quantized",
    "ignore": ["lm_head"],
    "quant_method": "compressed-tensors",
    "quantization_status": "compressed"
  },
  "rms_norm_eps": 1e-06,
  "rope_theta": 10000,
  "tie_word_embeddings": false,
  "torch_dtype": "float32",
  "use_sliding_window": false,
  "vocab_size": 32
})JSON";
}

static std::string int4_quant_config_json() {
    return R"JSON({
  "architectures": ["Qwen3ForCausalLM"],
  "bos_token_id": 151643,
  "eos_token_id": 151645,
  "hidden_act": "silu",
  "hidden_size": 8,
  "intermediate_size": 16,
  "max_position_embeddings": 64,
  "max_window_layers": 2,
  "model_type": "qwen3",
  "num_attention_heads": 2,
  "num_hidden_layers": 1,
  "num_key_value_heads": 2,
  "quantization_config": {
    "config_groups": {
      "group_0": {
        "targets": ["Linear"],
        "weights": {
          "actorder": true,
          "dynamic": false,
          "group_size": 8,
          "num_bits": 4,
          "strategy": "group",
          "symmetric": true,
          "type": "int"
        }
      }
    },
    "format": "packed",
    "quant_method": "gptq"
  },
  "rms_norm_eps": 1e-06,
  "rope_theta": 10000,
  "tie_word_embeddings": false,
  "torch_dtype": "float32",
  "use_sliding_window": false,
  "vocab_size": 32
})JSON";
}

static std::vector<int8_t> unpack_4bit_tensor_row(const tensor_t &packed) {
    const auto *src = reinterpret_cast<const int32_t *>(packed->data());
    const size_t k = packed->shape()[1] * 8;
    std::vector<int8_t> unpacked(k);
    for (size_t block = 0; block < k / 8; ++block) {
        int32_t value = src[block];
        for (int i = 0; i < 8; ++i) {
            unpacked[block * 8 + static_cast<size_t>(i)] =
                static_cast<int8_t>((value >> (i * 4)) & 0xF);
        }
    }
    return unpacked;
}

} // namespace

TEST(QuantizedWeightsTest, LoadWeightsPacksInt8WeightsAndExposesForwardMetadata) {
    const auto dir = make_temp_model_dir();
    write_config(dir, int8_quant_config_json());

    std::vector<int8_t> q_weight = {
        1, 2, 3, 4, 5, 6, 7, 8,
        -1, -2, -3, -4, -5, -6, -7, -8
    };
    std::vector<float> q_scale = {0.25f, 0.5f};
    std::vector<float> q_bias = {1.0f, -1.0f};
    std::vector<float> embed = {
        0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f
    };

    write_safetensors(dir, {
        {"model.layers.0.self_attn.q_proj.weight", "I8", {2, 8}, to_bytes(q_weight)},
        {"model.layers.0.self_attn.q_proj.weight_scale", "F32", {2}, to_bytes(q_scale)},
        {"model.layers.0.self_attn.q_proj.bias", "F32", {2}, to_bytes(q_bias)},
        {"model.embed_tokens.weight", "F32", {1, 8}, to_bytes(embed)}
    });

    auto config = Model::load_config((dir / "config.json").string());
    ASSERT_NE(config, nullptr);
    auto weights = Model::load_weights(dir.string(), ZEDINFER_DEVICE_CPU, *config);

    ASSERT_NE(weights, nullptr);
    EXPECT_FALSE(weights->has_tensor("layers.0.self_attn.q_proj.weight"));
    EXPECT_TRUE(weights->has_tensor("layers.0.self_attn.q_proj.weight_packed"));
    EXPECT_TRUE(weights->has_tensor("layers.0.self_attn.q_proj.weight_scale"));
    EXPECT_TRUE(weights->has_tensor("layers.0.self_attn.q_proj.bias"));

    auto packed = weights->get_tensor("layers.0.self_attn.q_proj.weight_packed");
    ASSERT_NE(packed, nullptr);
    EXPECT_EQ(packed->dtype(), ZEDINFER_DTYPE_I32);
    ASSERT_EQ(packed->shape().size(), 2u);
    EXPECT_EQ(packed->shape()[0], 2u);
    EXPECT_EQ(packed->shape()[1], 2u);

    ModelForwardConfig forward{*config, *weights, false, true};
    EXPECT_TRUE(forward.has_quantized_linear("layers.0.self_attn.q_proj"));

    auto quant = forward.quant_linear("layers.0.self_attn.q_proj");
    EXPECT_EQ(quant.num_bits, 8);
    EXPECT_EQ(quant.group_size, -1);
    ASSERT_NE(quant.weight, nullptr);
    ASSERT_NE(quant.scale, nullptr);
    ASSERT_NE(quant.bias, nullptr);
    EXPECT_EQ(quant.weight->shape()[1], 2u);
}

TEST(QuantizedWeightsTest, LoadWeightsReordersActOrderPackedWeights) {
    const auto dir = make_temp_model_dir();
    write_config(dir, int4_quant_config_json());

    const int32_t packed_row = 0x76543210;
    std::vector<int32_t> packed = {packed_row};
    std::vector<int32_t> g_idx = {2, 0, 1, 0, 2, 1, 0, 2};
    std::vector<float> scales = {1.0f};

    write_safetensors(dir, {
        {"model.layers.0.self_attn.q_proj.weight_packed", "I32", {1, 1}, to_bytes(packed)},
        {"model.layers.0.self_attn.q_proj.weight_g_idx", "I32", {8}, to_bytes(g_idx)},
        {"model.layers.0.self_attn.q_proj.weight_scale", "F32", {1, 1}, to_bytes(scales)}
    });

    auto config = Model::load_config((dir / "config.json").string());
    auto weights = Model::load_weights(dir.string(), ZEDINFER_DEVICE_CPU, *config);

    auto reordered_g_idx = weights->get_tensor("layers.0.self_attn.q_proj.weight_g_idx");
    ASSERT_NE(reordered_g_idx, nullptr);
    const auto *perm = reinterpret_cast<const int32_t *>(reordered_g_idx->data());
    const std::vector<int32_t> expected_perm = {1, 3, 6, 2, 5, 0, 4, 7};
    for (size_t i = 0; i < expected_perm.size(); ++i) {
        EXPECT_EQ(perm[i], expected_perm[i]);
    }

    auto reordered_weight = weights->get_tensor("layers.0.self_attn.q_proj.weight_packed");
    const auto unpacked = unpack_4bit_tensor_row(reordered_weight);
    const std::vector<int8_t> expected_unpacked = {1, 3, 6, 2, 5, 0, 4, 7};
    EXPECT_EQ(unpacked, expected_unpacked);
}

TEST(QuantizedWeightsTest, LinearQuantizedRunsCpuInt8Fallback) {
    auto out = Tensor::create({2, 2}, ZEDINFER_DTYPE_F32, ZEDINFER_DEVICE_CPU);
    auto in = Tensor::create({2, 4}, ZEDINFER_DTYPE_F32, ZEDINFER_DEVICE_CPU);
    auto weight = Tensor::create({2, 4}, ZEDINFER_DTYPE_I8, ZEDINFER_DEVICE_CPU);
    auto bias = Tensor::create({2}, ZEDINFER_DTYPE_F32, ZEDINFER_DEVICE_CPU);
    auto scale = Tensor::create({2, 2}, ZEDINFER_DTYPE_F32, ZEDINFER_DEVICE_CPU);

    const std::vector<float> input_values = {
        1.0f, 2.0f, 3.0f, 4.0f,
        2.0f, 0.0f, -1.0f, 1.0f
    };
    const std::vector<int8_t> weight_values = {
        1, 2, 3, 4,
        -1, 0, 1, 2
    };
    const std::vector<float> bias_values = {1.0f, -1.0f};
    const std::vector<float> scale_values = {
        0.5f, 1.0f,
        1.0f, 0.25f
    };

    std::memcpy(in->data(), input_values.data(), input_values.size() * sizeof(float));
    std::memcpy(weight->data(), weight_values.data(), weight_values.size() * sizeof(int8_t));
    std::memcpy(bias->data(), bias_values.data(), bias_values.size() * sizeof(float));
    std::memcpy(scale->data(), scale_values.data(), scale_values.size() * sizeof(float));

    ops::linear_quantized(out, in, weight, bias, scale, nullptr, 8, 2);

    const auto *out_values = reinterpret_cast<const float *>(out->data());
    EXPECT_FLOAT_EQ(out_values[0], 28.5f);
    EXPECT_FLOAT_EQ(out_values[1], 0.75f);
    EXPECT_FLOAT_EQ(out_values[2], 3.0f);
    EXPECT_FLOAT_EQ(out_values[3], -2.75f);
}

TEST(QuantizedWeightsTest, LinearQuantizedRunsCpuInt4Fallback) {
    auto out = Tensor::create({1, 1}, ZEDINFER_DTYPE_F32, ZEDINFER_DEVICE_CPU);
    auto in = Tensor::create({1, 8}, ZEDINFER_DTYPE_F32, ZEDINFER_DEVICE_CPU);
    auto weight = Tensor::create({1, 1}, ZEDINFER_DTYPE_I32, ZEDINFER_DEVICE_CPU);
    auto bias = Tensor::create({1}, ZEDINFER_DTYPE_F32, ZEDINFER_DEVICE_CPU);
    auto scale = Tensor::create({1}, ZEDINFER_DTYPE_F32, ZEDINFER_DEVICE_CPU);

    const std::vector<float> input_values = {
        1.0f, 2.0f, 3.0f, 4.0f,
        1.0f, 1.0f, 1.0f, 1.0f
    };
    const int32_t packed_weight = 0x99999999;
    const float bias_value = 1.0f;
    const float scale_value = 2.0f;

    std::memcpy(in->data(), input_values.data(), input_values.size() * sizeof(float));
    std::memcpy(weight->data(), &packed_weight, sizeof(packed_weight));
    std::memcpy(bias->data(), &bias_value, sizeof(bias_value));
    std::memcpy(scale->data(), &scale_value, sizeof(scale_value));

    ops::linear_quantized(out, in, weight, bias, scale, nullptr, 4, 8);

    const auto *out_values = reinterpret_cast<const float *>(out->data());
    EXPECT_FLOAT_EQ(out_values[0], 29.0f);
}

TEST(QuantizedWeightsTest, LinearQuantizedPermutesActOrderInputAtRuntime) {
    const auto dir = make_temp_model_dir();
    write_config(dir, int4_quant_config_json());

    const int32_t packed_row = 0x76543210;
    std::vector<int32_t> packed = {packed_row};
    std::vector<int32_t> g_idx = {2, 0, 1, 0, 2, 1, 0, 2};
    std::vector<float> scales = {1.0f};

    write_safetensors(dir, {
        {"model.layers.0.self_attn.q_proj.weight_packed", "I32", {1, 1}, to_bytes(packed)},
        {"model.layers.0.self_attn.q_proj.weight_g_idx", "I32", {8}, to_bytes(g_idx)},
        {"model.layers.0.self_attn.q_proj.weight_scale", "F32", {1}, to_bytes(scales)}
    });

    auto config = Model::load_config((dir / "config.json").string());
    auto weights = Model::load_weights(dir.string(), ZEDINFER_DEVICE_CPU, *config);

    auto input = Tensor::create({1, 8}, ZEDINFER_DTYPE_F32, ZEDINFER_DEVICE_CPU);
    auto permuted_input = Tensor::create({1, 8}, ZEDINFER_DTYPE_F32, ZEDINFER_DEVICE_CPU);
    auto output = Tensor::create({1, 1}, ZEDINFER_DTYPE_F32, ZEDINFER_DEVICE_CPU);
    auto expected = Tensor::create({1, 1}, ZEDINFER_DTYPE_F32, ZEDINFER_DEVICE_CPU);

    const std::vector<float> input_values = {
        1.0f, 2.0f, 3.0f, 4.0f,
        5.0f, 6.0f, 7.0f, 8.0f
    };
    const auto *perm = reinterpret_cast<const int32_t *>(
        weights->get_tensor("layers.0.self_attn.q_proj.weight_g_idx")->data());
    auto *permuted_values = reinterpret_cast<float *>(permuted_input->data());

    std::memcpy(input->data(), input_values.data(), input_values.size() * sizeof(float));
    for (size_t i = 0; i < input_values.size(); ++i) {
        permuted_values[i] = input_values[static_cast<size_t>(perm[i])];
    }

    auto weight = weights->get_tensor("layers.0.self_attn.q_proj.weight_packed");
    auto scale = weights->get_tensor("layers.0.self_attn.q_proj.weight_scale");
    auto reordered_g_idx = weights->get_tensor("layers.0.self_attn.q_proj.weight_g_idx");

    ops::linear_quantized(output, input, weight, nullptr, scale, reordered_g_idx, 4, 8);
    ops::linear_quantized(expected, permuted_input, weight, nullptr, scale, nullptr, 4, 8);

    const auto *output_value = reinterpret_cast<const float *>(output->data());
    const auto *expected_value = reinterpret_cast<const float *>(expected->data());
    EXPECT_FLOAT_EQ(output_value[0], expected_value[0]);
}
