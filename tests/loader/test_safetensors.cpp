#include "frontend/loader/interface.hpp"
#include "frontend/loader/safetensors.hpp"
#include "utils/types.hpp"
#include "zedinfer.h"

#include <cmath>
#include <cstddef>
#include <gtest/gtest.h>
#include <iomanip>
#include <iostream>

using namespace zedinfer::loader;

class ModelLoaderTest : public ::testing::Test {
protected:
    std::string model_path = "/home/xiongtianyu/data/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B";
    std::unique_ptr<IModelLoader> loader;

    void SetUp() override {
        loader = SafeTensorsLoader::create(model_path);
    }

    void TearDown() override {
        loader.reset();
    }
};

// ============================================
// 基本加载测试
// ============================================

TEST_F(ModelLoaderTest, ModelLoadsSuccessfully) {
    ASSERT_NE(loader, nullptr);
    EXPECT_GT(loader->get_num_files(), 0);
}

TEST_F(ModelLoaderTest, HasExpectedNumberOfTensors) {
    auto tensor_names = loader->get_all_tensor_names();
    EXPECT_GT(tensor_names.size(), 0);

    std::cout << "Total tensors: " << tensor_names.size() << std::endl;
}

TEST_F(ModelLoaderTest, HasMultipleSafetensorsFiles) {
    size_t num_files = loader->get_num_files();
    std::cout << "Number of safetensors files: " << num_files << std::endl;

    // DeepSeek-R1-Distill-Qwen-1.5B 通常有多个分片文件
    EXPECT_GE(num_files, 1);
}

// ============================================
// 模型结构测试
// ============================================

TEST_F(ModelLoaderTest, HasEmbeddingLayer) {
    auto names = loader->get_all_tensor_names();

    bool found_embeddings = false;
    for (const auto &name : names) {
        if (name.find("embed_tokens") != std::string::npos || name.find("embeddings") != std::string::npos) {
            found_embeddings = true;
            std::cout << "Found embedding layer: " << name << std::endl;
            break;
        }
    }

    EXPECT_TRUE(found_embeddings) << "Model should have embedding layer";
}

TEST_F(ModelLoaderTest, HasTransformerLayers) {
    auto names = loader->get_all_tensor_names();

    std::set<std::string> layer_numbers;
    for (const auto &name : names) {
        if (name.find("layers.") != std::string::npos) {
            size_t start = name.find("layers.") + 7;
            size_t end = name.find(".", start);
            if (end != std::string::npos) {
                std::string layer_num = name.substr(start, end - start);
                layer_numbers.insert(layer_num);
            }
        }
    }

    std::cout << "Number of transformer layers: " << layer_numbers.size() << std::endl;
    EXPECT_GT(layer_numbers.size(), 0) << "Model should have transformer layers";
}

TEST_F(ModelLoaderTest, HasAttentionWeights) {
    auto names = loader->get_all_tensor_names();

    bool has_q_proj = false;
    bool has_k_proj = false;
    bool has_v_proj = false;
    bool has_o_proj = false;

    for (const auto &name : names) {
        if (name.find("q_proj") != std::string::npos) {
            has_q_proj = true;
        }
        if (name.find("k_proj") != std::string::npos) {
            has_k_proj = true;
        }
        if (name.find("v_proj") != std::string::npos) {
            has_v_proj = true;
        }
        if (name.find("o_proj") != std::string::npos) {
            has_o_proj = true;
        }
    }

    EXPECT_TRUE(has_q_proj) << "Should have Q projection weights";
    EXPECT_TRUE(has_k_proj) << "Should have K projection weights";
    EXPECT_TRUE(has_v_proj) << "Should have V projection weights";
    EXPECT_TRUE(has_o_proj) << "Should have O projection weights";
}

TEST_F(ModelLoaderTest, HasMLPWeights) {
    auto names = loader->get_all_tensor_names();

    bool has_gate_proj = false;
    bool has_up_proj = false;
    bool has_down_proj = false;

    for (const auto &name : names) {
        if (name.find("gate_proj") != std::string::npos) {
            has_gate_proj = true;
        }
        if (name.find("up_proj") != std::string::npos) {
            has_up_proj = true;
        }
        if (name.find("down_proj") != std::string::npos) {
            has_down_proj = true;
        }
    }

    EXPECT_TRUE(has_gate_proj || has_up_proj || has_down_proj)
        << "Should have MLP weights";
}

TEST_F(ModelLoaderTest, HasNormalizationLayers) {
    auto names = loader->get_all_tensor_names();

    bool has_norm = false;
    for (const auto &name : names) {
        if (name.find("norm") != std::string::npos || name.find("ln") != std::string::npos) {
            has_norm = true;
            break;
        }
    }

    EXPECT_TRUE(has_norm) << "Should have normalization layers";
}

// ============================================
// Tensor信息测试
// ============================================

TEST_F(ModelLoaderTest, TensorInfoIsValid) {
    auto names = loader->get_all_tensor_names();
    ASSERT_FALSE(names.empty());

    // 测试第一个tensor
    const std::string &first_tensor = names[0];
    const TensorInfo *info = loader->get_tensor_info(first_tensor);

    ASSERT_NE(info, nullptr);
    EXPECT_EQ(info->name, first_tensor);
    EXPECT_FALSE(info->shape.empty());
    EXPECT_GT(info->numel(), 0);

    std::cout << "First tensor: " << first_tensor << std::endl;
    std::cout << "  Shape: [";
    for (size_t i = 0; i < info->shape.size(); ++i) {
        std::cout << info->shape[i];
        if (i < info->shape.size() - 1) {
            std::cout << ", ";
        }
    }
    std::cout << "]" << std::endl;
}

TEST_F(ModelLoaderTest, AllTensorsHaveValidInfo) {
    auto names = loader->get_all_tensor_names();

    for (const auto &name : names) {
        const TensorInfo *info = loader->get_tensor_info(name);
        ASSERT_NE(info, nullptr) << "Tensor " << name << " has no info";
        EXPECT_GT(info->numel(), 0) << "Tensor " << name << " has 0 elements";
    }
}

TEST_F(ModelLoaderTest, ListAllTensorNames) {
    auto names = loader->get_all_tensor_names();

    std::cout << "\n"
              << std::string(80, '=') << std::endl;
    std::cout << "ALL TENSOR NAMES (first 50)" << std::endl;
    std::cout << std::string(80, '=') << std::endl;

    size_t count = 0;
    for (const auto &name : names) {
        if (count >= 50) {
            std::cout << "... (" << (names.size() - 50) << " more)" << std::endl;
            break;
        }

        const TensorInfo *info = loader->get_tensor_info(name);
        if (info) {
            std::cout << std::left << std::setw(60) << name;
            std::cout << " [";
            for (size_t i = 0; i < info->shape.size(); ++i) {
                std::cout << info->shape[i];
                if (i < info->shape.size() - 1) {
                    std::cout << ", ";
                }
            }
            std::cout << "]";

            // 显示数据类型
            std::cout << " (";
            switch (info->dtype) {
            case ZEDINFER_DTYPE_F32:
                std::cout << "F32";
                break;
            case ZEDINFER_DTYPE_F16:
                std::cout << "F16";
                break;
            case ZEDINFER_DTYPE_BF16:
                std::cout << "BF16";
                break;
            default:
                std::cout << "?";
                break;
            }
            std::cout << ")" << std::endl;
        }
        count++;
    }
}

// ============================================
// Tensor数据测试
// ============================================

TEST_F(ModelLoaderTest, CanAccessTensorData) {
    auto names = loader->get_all_tensor_names();
    ASSERT_FALSE(names.empty());

    // 测试前5个tensor的数据访问
    size_t test_count = std::min(5UL, names.size());

    for (size_t i = 0; i < test_count; ++i) {
        const std::string &name = names[i];
        const void *data = loader->get_tensor_data(name);

        EXPECT_NE(data, nullptr) << "Cannot access data for tensor: " << name;
    }
}

TEST_F(ModelLoaderTest, TensorDataIsNotNull) {
    auto names = loader->get_all_tensor_names();

    // 随机测试10个tensor
    size_t test_count = std::min(10UL, names.size());

    for (size_t i = 0; i < test_count; ++i) {
        const void *data = loader->get_tensor_data(names[i]);
        ASSERT_NE(data, nullptr) << "Tensor data is null: " << names[i];
    }
}

TEST_F(ModelLoaderTest, CanReadFloatData) {
    auto names = loader->get_all_tensor_names();

    std::cout << "\n"
              << std::string(80, '=') << std::endl;
    std::cout << "READ TENSOR DATA (first 50)" << std::endl;
    std::cout << std::string(80, '=') << std::endl;

    size_t count = 0;
    for (const auto &name : names) {
        if (count >= 50) {
            std::cout << "... (" << (names.size() - 50) << " more)" << std::endl;
            break;
        }

        const TensorInfo *info = loader->get_tensor_info(name);

        if (info && info->dtype == ZEDINFER_DTYPE_BF16) {
            const void *data = loader->get_tensor_data(name);
            ASSERT_NE(data, nullptr);

            const zedinfer::bf16_t *bf16_data = reinterpret_cast<const zedinfer::bf16_t *>(data);

            // 简单验证：检查前几个值不是NaN或Inf
            for (size_t i = 0; i < std::min(10UL, info->numel()); ++i) {
                float item = zedinfer::utils::cast<float>(bf16_data[i]);
                EXPECT_FALSE(std::isnan(item))
                    << "Found NaN in tensor " << name << " at index " << i;
                EXPECT_FALSE(std::isinf(item))
                    << "Found Inf in tensor " << name << " at index " << i;
            }
            std::cout << "Validated tensor: " << name << " (BF16)" << std::endl;
        }
        count++;
    }
}

TEST_F(ModelLoaderTest, ReadNormData) {
    std::string tensor_name = "model.norm.weight";
    const void *data = loader->get_tensor_data(tensor_name);
    const zedinfer::bf16_t *bf16_data = reinterpret_cast<const zedinfer::bf16_t *>(data);

    std::cout << "First 10 elem of " << tensor_name << std::endl;

    std::cout << "[";
    for (size_t i = 0; i < 10UL; ++i) {
        float item = zedinfer::utils::cast<float>(bf16_data[i]);
        std::cout << item;
        if (i != 9) {
            std::cout << ", ";
        }
    }
    std::cout << "]" << std::endl;
}

// ============================================
// 内存测试
// ============================================

TEST_F(ModelLoaderTest, CalculateTotalParameters) {
    auto names = loader->get_all_tensor_names();

    size_t total_params = 0;
    for (const auto &name : names) {
        const TensorInfo *info = loader->get_tensor_info(name);
        if (info) {
            total_params += info->numel();
        }
    }

    std::cout << "Total parameters: " << total_params
              << " (~" << std::fixed << std::setprecision(2)
              << (total_params / 1e9) << "B)" << std::endl;

    // DeepSeek-R1-Distill-Qwen-1.5B 应该约有1.5B参数
    EXPECT_GT(total_params, 1e9) << "Parameter count seems too low";
    EXPECT_LT(total_params, 3e9) << "Parameter count seems too high";
}

TEST_F(ModelLoaderTest, MemoryMappingWorks) {
    auto names = loader->get_all_tensor_names();

    // 多次访问同一个tensor，应该返回相同的指针（mmap）
    if (!names.empty()) {
        const void *ptr1 = loader->get_tensor_data(names[0]);
        const void *ptr2 = loader->get_tensor_data(names[0]);

        EXPECT_EQ(ptr1, ptr2) << "Mmap should return same pointer";
    }
}
