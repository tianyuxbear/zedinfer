#include "backend/core/context/context.hpp"
#include "backend/tensor/tensor.hpp"
#include <cmath>
#include <gtest/gtest.h>
#include <numeric>

namespace neollm {
namespace test {

// ============================================================================
// 测试夹具 - 提供通用的测试环境
// ============================================================================
class TensorTest : public ::testing::Test {
protected:
    void SetUp() override {}

    void TearDown() override {}

    // 辅助函数：比较两个浮点数向量
    bool compareFloatVectors(const std::vector<float> &a,
                             const std::vector<float> &b,
                             float epsilon = 1e-5f) {
        if (a.size() != b.size()) {
            return false;
        }
        for (size_t i = 0; i < a.size(); ++i) {
            if (std::abs(a[i] - b[i]) > epsilon) {
                return false;
            }
        }
        return true;
    }

    // 辅助函数：填充张量数据
    template <typename T>
    void fillTensorData(tensor_t tensor, const std::vector<T> &data) {
        tensor->load(data.data());
    }

    // 辅助函数：读取张量数据
    template <typename T>
    std::vector<T> readTensorData(tensor_t tensor) {
        std::vector<T> result(tensor->numel());
        if (tensor->deviceType() == NEOLLM_DEVICE_CPU) {
            memcpy(result.data(), tensor->data(),
                   tensor->numel() * sizeof(T));
        } else {
            // GPU -> CPU
            core::context().runtime().api()->memcpy_sync(
                result.data(), tensor->data(),
                tensor->numel() * sizeof(T),
                NEOLLM_MEMCPY_D2H);
        }
        return result;
    }
};

// ============================================================================
// 基础功能测试
// ============================================================================
TEST_F(TensorTest, CreateTensorCPU) {
    auto tensor = Tensor::create({2, 3, 4}, NEOLLM_DTYPE_F32, NEOLLM_DEVICE_CPU);

    ASSERT_NE(tensor, nullptr);
    EXPECT_EQ(tensor->ndim(), 3);
    EXPECT_EQ(tensor->numel(), 24);
    EXPECT_EQ(tensor->deviceType(), NEOLLM_DEVICE_CPU);
    EXPECT_EQ(tensor->dtype(), NEOLLM_DTYPE_F32);
    EXPECT_EQ(tensor->elementSize(), 4); // sizeof(float)
}

TEST_F(TensorTest, ShapeAndStrides) {
    auto tensor = Tensor::create({2, 3, 4}, NEOLLM_DTYPE_F32);

    // 验证 shape
    EXPECT_EQ(tensor->shape(), std::vector<size_t>({2, 3, 4}));
    EXPECT_EQ(tensor->dim(0), 2);
    EXPECT_EQ(tensor->dim(1), 3);
    EXPECT_EQ(tensor->dim(2), 4);

    // 验证 strides (row-major: [12, 4, 1])
    EXPECT_EQ(tensor->strides(), std::vector<ptrdiff_t>({12, 4, 1}));
    EXPECT_EQ(tensor->stride(0), 12);
    EXPECT_EQ(tensor->stride(1), 4);
    EXPECT_EQ(tensor->stride(2), 1);
}

TEST_F(TensorTest, LoadAndReadData) {
    auto tensor = Tensor::create({2, 3}, NEOLLM_DTYPE_F32);

    // 准备测试数据
    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    fillTensorData(tensor, data);

    // 读取并验证
    auto result = readTensorData<float>(tensor);
    EXPECT_TRUE(compareFloatVectors(result, data));
}

// ============================================================================
// isContiguous 测试
// ============================================================================
TEST_F(TensorTest, IsContiguousAfterCreation) {
    auto tensor = Tensor::create({3, 4, 5}, NEOLLM_DTYPE_F32);
    EXPECT_TRUE(tensor->isContiguous());
}

TEST_F(TensorTest, IsContiguousAfterPermute) {
    auto tensor = Tensor::create({2, 3, 4}, NEOLLM_DTYPE_F32);

    // 原始张量是连续的
    EXPECT_TRUE(tensor->isContiguous());

    // 转置后变为非连续
    auto transposed = tensor->permute({2, 1, 0});
    EXPECT_FALSE(transposed->isContiguous());
}

TEST_F(TensorTest, IsContiguousAfterSlice) {
    auto tensor = Tensor::create({5, 6}, NEOLLM_DTYPE_F32);

    // 沿第二维切片仍然连续（最内层维度完整）
    auto sliced = tensor->slice(0, 1, 4);
    EXPECT_TRUE(sliced->isContiguous());

    // 沿第一维切片后，stride 不变但 shape 改变，仍然连续
    EXPECT_EQ(sliced->shape(), std::vector<size_t>({3, 6}));
}

// ============================================================================
// permute 测试
// ============================================================================
TEST_F(TensorTest, PermuteBasic) {
    auto tensor = Tensor::create({2, 3, 4}, NEOLLM_DTYPE_F32);
    std::vector<float> data(24);
    std::iota(data.begin(), data.end(), 0.0f); // 0, 1, 2, ..., 23
    fillTensorData(tensor, data);

    // 转置: (2,3,4) -> (4,3,2)
    auto permuted = tensor->permute({2, 1, 0});

    EXPECT_EQ(permuted->shape(), std::vector<size_t>({4, 3, 2}));
    EXPECT_EQ(permuted->strides(), std::vector<ptrdiff_t>({1, 4, 12}));
    EXPECT_EQ(permuted->numel(), 24);
}

TEST_F(TensorTest, PermuteInvalidOrder) {
    auto tensor = Tensor::create({2, 3, 4}, NEOLLM_DTYPE_F32);

    // 维度数量不匹配
    EXPECT_THROW(tensor->permute({0, 1}), std::invalid_argument);

    // 包含重复索引
    EXPECT_THROW(tensor->permute({0, 1, 1}), std::invalid_argument);

    // 包含越界索引
    EXPECT_THROW(tensor->permute({0, 1, 5}), std::invalid_argument);
}

TEST_F(TensorTest, PermutePreservesData) {
    auto tensor = Tensor::create({2, 3}, NEOLLM_DTYPE_F32);
    std::vector<float> data = {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f};
    fillTensorData(tensor, data);

    // 转置: (2,3) -> (3,2)
    auto transposed = tensor->permute({1, 0});

    EXPECT_EQ(transposed->shape(), std::vector<size_t>({3, 2}));

    // 验证数据通过 contiguous 操作
    auto cont = transposed->contiguous();
    auto result = readTensorData<float>(cont);

    std::vector<float> expected = {
        1.0f, 4.0f,
        2.0f, 5.0f,
        3.0f, 6.0f};
    EXPECT_TRUE(compareFloatVectors(result, expected));
}

// ============================================================================
// slice 测试
// ============================================================================
TEST_F(TensorTest, SliceBasic) {
    auto tensor = Tensor::create({5, 6}, NEOLLM_DTYPE_F32);
    std::vector<float> data(30);
    std::iota(data.begin(), data.end(), 0.0f);
    fillTensorData(tensor, data);

    // 切片第一维: [1:4] -> shape=(3,6)
    auto sliced = tensor->slice(0, 1, 4);

    EXPECT_EQ(sliced->shape(), std::vector<size_t>({3, 6}));
    EXPECT_EQ(sliced->numel(), 18);

    // 验证数据（第1,2,3行，共18个元素）
    auto result = readTensorData<float>(sliced);
    std::vector<float> expected(18);
    std::iota(expected.begin(), expected.end(), 6.0f); // 从第6个元素开始
    EXPECT_TRUE(compareFloatVectors(result, expected));
}

TEST_F(TensorTest, SliceInvalidRange) {
    auto tensor = Tensor::create({5, 6}, NEOLLM_DTYPE_F32);

    // start >= end
    EXPECT_THROW(tensor->slice(0, 3, 3), std::invalid_argument);
    EXPECT_THROW(tensor->slice(0, 4, 2), std::invalid_argument);

    // end 超出范围
    EXPECT_THROW(tensor->slice(0, 0, 10), std::invalid_argument);

    // dim 越界
    EXPECT_THROW(tensor->slice(5, 0, 2), std::invalid_argument);
}

TEST_F(TensorTest, SliceChaining) {
    auto tensor = Tensor::create({5, 6, 7}, NEOLLM_DTYPE_F32);

    // 连续切片
    auto sliced1 = tensor->slice(0, 1, 4);  // (3,6,7)
    auto sliced2 = sliced1->slice(1, 2, 5); // (3,3,7)
    auto sliced3 = sliced2->slice(2, 1, 6); // (3,3,5)

    EXPECT_EQ(sliced3->shape(), std::vector<size_t>({3, 3, 5}));
    EXPECT_EQ(sliced3->numel(), 45);
}

// ============================================================================
// view 测试
// ============================================================================
TEST_F(TensorTest, ViewBasic) {
    auto tensor = Tensor::create({2, 3, 4}, NEOLLM_DTYPE_F32);

    // Reshape: (2,3,4) -> (6,4)
    auto viewed = tensor->view({6, 4});

    EXPECT_EQ(viewed->shape(), std::vector<size_t>({6, 4}));
    EXPECT_EQ(viewed->numel(), 24);
    EXPECT_TRUE(viewed->isContiguous());
}

TEST_F(TensorTest, ViewFlatten) {
    auto tensor = Tensor::create({2, 3, 4}, NEOLLM_DTYPE_F32);

    // 展平为一维
    auto flattened = tensor->view({24});

    EXPECT_EQ(flattened->shape(), std::vector<size_t>({24}));
    EXPECT_EQ(flattened->ndim(), 1);
}

TEST_F(TensorTest, ViewIncompatibleShape) {
    auto tensor = Tensor::create({2, 3, 4}, NEOLLM_DTYPE_F32);

    // 元素数量不匹配
    EXPECT_THROW(tensor->view({2, 10}), std::invalid_argument);
    EXPECT_THROW(tensor->view({5, 5}), std::invalid_argument);
}

TEST_F(TensorTest, ViewRequiresContiguous) {
    auto tensor = Tensor::create({2, 3, 4}, NEOLLM_DTYPE_F32);
    auto transposed = tensor->permute({2, 1, 0}); // 非连续

    // view 要求张量连续
    EXPECT_THROW(transposed->view({4, 6}), std::runtime_error);
}

TEST_F(TensorTest, ViewSharesStorage) {
    auto tensor = Tensor::create({2, 3}, NEOLLM_DTYPE_F32);
    std::vector<float> data = {1, 2, 3, 4, 5, 6};
    fillTensorData(tensor, data);

    auto viewed = tensor->view({3, 2});

    // 修改原张量应该影响 view
    std::vector<float> new_data = {10, 20, 30, 40, 50, 60};
    fillTensorData(tensor, new_data);

    auto result = readTensorData<float>(viewed);
    EXPECT_TRUE(compareFloatVectors(result, new_data));
}

// ============================================================================
// contiguous 测试
// ============================================================================
TEST_F(TensorTest, ContiguousOnContiguousTensor) {
    auto tensor = Tensor::create({2, 3, 4}, NEOLLM_DTYPE_F32);
    std::vector<float> data(24);
    std::iota(data.begin(), data.end(), 0.0f);
    fillTensorData(tensor, data);

    auto cont = tensor->contiguous();

    // 应该返回共享存储的视图（零拷贝）
    EXPECT_TRUE(cont->isContiguous());
    auto result = readTensorData<float>(cont);
    EXPECT_TRUE(compareFloatVectors(result, data));
}

TEST_F(TensorTest, ContiguousOnNonContiguousTensor) {
    auto tensor = Tensor::create({2, 3}, NEOLLM_DTYPE_F32);
    std::vector<float> data = {1, 2, 3, 4, 5, 6};
    fillTensorData(tensor, data);

    // 转置后非连续
    auto transposed = tensor->permute({1, 0});
    EXPECT_FALSE(transposed->isContiguous());

    // contiguous 应该创建新的连续存储
    auto cont = transposed->contiguous();
    EXPECT_TRUE(cont->isContiguous());

    // 验证数据正确性
    auto result = readTensorData<float>(cont);
    std::vector<float> expected = {1, 4, 2, 5, 3, 6};
    EXPECT_TRUE(compareFloatVectors(result, expected));
}

TEST_F(TensorTest, ContiguousComplexCase) {
    // 创建一个复杂的非连续张量
    auto tensor = Tensor::create({2, 3, 4}, NEOLLM_DTYPE_F32);
    std::vector<float> data(24);
    std::iota(data.begin(), data.end(), 1.0f); // 1-24
    fillTensorData(tensor, data);

    // 多次变换
    auto perm1 = tensor->permute({2, 0, 1}); // (4,2,3)
    auto sliced = perm1->slice(0, 1, 3);     // (2,2,3)
    auto perm2 = sliced->permute({2, 1, 0}); // (3,2,2)

    EXPECT_FALSE(perm2->isContiguous());

    // 转为连续
    auto cont = perm2->contiguous();
    EXPECT_TRUE(cont->isContiguous());
    EXPECT_EQ(cont->shape(), std::vector<size_t>({3, 2, 2}));
    EXPECT_EQ(cont->numel(), 12);
}

// ============================================================================
// reshape 测试
// ============================================================================
TEST_F(TensorTest, ReshapeOnContiguousTensor) {
    auto tensor = Tensor::create({2, 3, 4}, NEOLLM_DTYPE_F32);

    // 连续张量的 reshape 应该是零拷贝
    auto reshaped = tensor->reshape({6, 4});

    EXPECT_EQ(reshaped->shape(), std::vector<size_t>({6, 4}));
    EXPECT_TRUE(reshaped->isContiguous());
}

TEST_F(TensorTest, ReshapeOnNonContiguousTensor) {
    auto tensor = Tensor::create({2, 3, 4}, NEOLLM_DTYPE_F32);
    std::vector<float> data(24);
    std::iota(data.begin(), data.end(), 0.0f);
    fillTensorData(tensor, data);

    auto transposed = tensor->permute({2, 1, 0});
    EXPECT_FALSE(transposed->isContiguous());

    // 非连续张量的 reshape 会先 contiguous
    auto reshaped = transposed->reshape({24});

    EXPECT_TRUE(reshaped->isContiguous());
    EXPECT_EQ(reshaped->shape(), std::vector<size_t>({24}));
}

TEST_F(TensorTest, ReshapeInvalidShape) {
    auto tensor = Tensor::create({2, 3, 4}, NEOLLM_DTYPE_F32);

    EXPECT_THROW(tensor->reshape({2, 10}), std::invalid_argument);
}

// ============================================================================
// to (设备迁移) 测试
// ============================================================================
TEST_F(TensorTest, ToSameDevice) {
    auto tensor = Tensor::create({2, 3}, NEOLLM_DTYPE_F32, NEOLLM_DEVICE_CPU);
    std::vector<float> data = {1, 2, 3, 4, 5, 6};
    fillTensorData(tensor, data);

    // 移到相同设备应该零拷贝
    auto same_device = tensor->to(NEOLLM_DEVICE_CPU, 0);

    EXPECT_EQ(same_device->deviceType(), NEOLLM_DEVICE_CPU);
    auto result = readTensorData<float>(same_device);
    EXPECT_TRUE(compareFloatVectors(result, data));
}

#ifdef ENABLE_NVIDIA_API
TEST_F(TensorTest, ToCPUToGPU) {
    auto cpu_tensor = Tensor::create({2, 3}, NEOLLM_DTYPE_F32, NEOLLM_DEVICE_CPU);
    std::vector<float> data = {1, 2, 3, 4, 5, 6};
    fillTensorData(cpu_tensor, data);

    // CPU -> GPU
    auto gpu_tensor = cpu_tensor->to(NEOLLM_DEVICE_CUDA, 0);

    EXPECT_EQ(gpu_tensor->deviceType(), NEOLLM_DEVICE_CUDA);
    EXPECT_EQ(gpu_tensor->shape(), std::vector<size_t>({2, 3}));

    // 验证数据
    auto result = readTensorData<float>(gpu_tensor);
    EXPECT_TRUE(compareFloatVectors(result, data));
}

TEST_F(TensorTest, ToGPUToCPU) {
    auto gpu_tensor = Tensor::create({2, 3}, NEOLLM_DTYPE_F32, NEOLLM_DEVICE_CUDA);
    std::vector<float> data = {1, 2, 3, 4, 5, 6};
    fillTensorData(gpu_tensor, data);

    // GPU -> CPU
    auto cpu_tensor = gpu_tensor->to(NEOLLM_DEVICE_CPU, 0);

    EXPECT_EQ(cpu_tensor->deviceType(), NEOLLM_DEVICE_CPU);
    auto result = readTensorData<float>(cpu_tensor);
    EXPECT_TRUE(compareFloatVectors(result, data));
}

TEST_F(TensorTest, ToNonContiguousTensor) {
    auto tensor = Tensor::create({2, 3, 4}, NEOLLM_DTYPE_F32, NEOLLM_DEVICE_CPU);
    std::vector<float> data(24);
    std::iota(data.begin(), data.end(), 0.0f);
    fillTensorData(tensor, data);

    auto transposed = tensor->permute({2, 1, 0});
    EXPECT_FALSE(transposed->isContiguous());

    // 非连续张量迁移到 GPU
    auto gpu_tensor = transposed->to(NEOLLM_DEVICE_CUDA, 0);

    EXPECT_EQ(gpu_tensor->deviceType(), NEOLLM_DEVICE_CUDA);
    EXPECT_TRUE(gpu_tensor->isContiguous()); // 优化后应该变连续
}
#endif

// ============================================================================
// 综合测试
// ============================================================================
TEST_F(TensorTest, ComplexOperationChain) {
    // 创建张量: (2,3,4)
    auto tensor = Tensor::create({2, 3, 4}, NEOLLM_DTYPE_F32);
    std::vector<float> data(24);
    std::iota(data.begin(), data.end(), 1.0f);
    fillTensorData(tensor, data);

    // 链式操作
    auto result = tensor
                      ->permute({2, 1, 0}) // (4,3,2)
                      ->slice(0, 1, 4)     // (3,3,2)
                      ->contiguous()       // 连续化
                      ->view({9, 2})       // 重塑
                      ->slice(1, 0, 1);    // 取第一列 (9,1)

    EXPECT_EQ(result->shape(), std::vector<size_t>({9, 1}));
    EXPECT_EQ(result->numel(), 9);
}

TEST_F(TensorTest, DataIntegrityAfterMultipleTransforms) {
    auto tensor = Tensor::create({3, 4}, NEOLLM_DTYPE_F32);
    std::vector<float> data = {
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12};
    fillTensorData(tensor, data);

    // 转置
    auto t1 = tensor->permute({1, 0}); // (4,3)

    // 切片
    auto t2 = t1->slice(0, 1, 3); // (2,3)

    // 连续化
    auto t3 = t2->contiguous();

    auto result = readTensorData<float>(t3);

    // 预期: 原矩阵的第2,3行（索引1,2）
    // 转置后是第2,3列
    std::vector<float> expected = {
        2, 6, 10,
        3, 7, 11};

    EXPECT_TRUE(compareFloatVectors(result, expected));
}

// ============================================================================
// 边界条件测试
// ============================================================================
TEST_F(TensorTest, EmptyShapeDimension) {
    // 包含 0 的 shape
    auto tensor = Tensor::create({2, 0, 3}, NEOLLM_DTYPE_F32);
    EXPECT_EQ(tensor->numel(), 0);
}

TEST_F(TensorTest, SingleElementTensor) {
    auto tensor = Tensor::create({1}, NEOLLM_DTYPE_F32);
    EXPECT_EQ(tensor->numel(), 1);
    EXPECT_TRUE(tensor->isContiguous());

    std::vector<float> data = {42.0f};
    fillTensorData(tensor, data);

    auto result = readTensorData<float>(tensor);
    EXPECT_FLOAT_EQ(result[0], 42.0f);
}

TEST_F(TensorTest, HighDimensionalTensor) {
    auto tensor = Tensor::create({2, 3, 4, 5, 6}, NEOLLM_DTYPE_F32);
    EXPECT_EQ(tensor->ndim(), 5);
    EXPECT_EQ(tensor->numel(), 720);
    EXPECT_TRUE(tensor->isContiguous());
}

} // namespace test
} // namespace neollm