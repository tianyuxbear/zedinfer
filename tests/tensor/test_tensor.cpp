#include "backend/core/context/context.hpp"
#include "backend/tensor/tensor.hpp"

#include <cmath>
#include <gtest/gtest.h>
#include <numeric>

namespace zedinfer {
namespace test {

// ============================================================================
// Test fixture - provides common test environment
// ============================================================================
class TensorTest : public ::testing::Test {
protected:
    void SetUp() override {}

    void TearDown() override {}

    // Helper function: compare two float vectors
    bool compareFloatVectors(const std::vector<float>& a, const std::vector<float>& b, float epsilon = 1e-5f) {
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

    // Helper function: fill tensor data
    template <typename T> void fillTensorData(tensor_t tensor, const std::vector<T>& data) {
        tensor->load(data.data());
    }

    // Helper function: read tensor data
    template <typename T> std::vector<T> readTensorData(tensor_t tensor) {
        std::vector<T> result(tensor->numel());
        if (tensor->deviceType() == ZEDINFER_DEVICE_CPU) {
            memcpy(result.data(), tensor->data(), tensor->numel() * sizeof(T));
        } else {
            // GPU -> CPU
            core::context().runtime().api()->memcpy_sync(result.data(), tensor->data(), tensor->numel() * sizeof(T),
                                                         ZEDINFER_MEMCPY_D2H);
        }
        return result;
    }
};

// ============================================================================
// Basic functionality tests
// ============================================================================
TEST_F(TensorTest, CreateTensorCPU) {
    auto tensor = Tensor::create({2, 3, 4}, ZEDINFER_DTYPE_F32, ZEDINFER_DEVICE_CPU);

    ASSERT_NE(tensor, nullptr);
    EXPECT_EQ(tensor->ndim(), 3);
    EXPECT_EQ(tensor->numel(), 24);
    EXPECT_EQ(tensor->deviceType(), ZEDINFER_DEVICE_CPU);
    EXPECT_EQ(tensor->dtype(), ZEDINFER_DTYPE_F32);
    EXPECT_EQ(tensor->elementSize(), 4); // sizeof(float)
}

TEST_F(TensorTest, ShapeAndStrides) {
    auto tensor = Tensor::create({2, 3, 4}, ZEDINFER_DTYPE_F32);

    // Verify shape
    EXPECT_EQ(tensor->shape(), std::vector<size_t>({2, 3, 4}));
    EXPECT_EQ(tensor->dim(0), 2);
    EXPECT_EQ(tensor->dim(1), 3);
    EXPECT_EQ(tensor->dim(2), 4);

    // Verify strides (row-major: [12, 4, 1])
    EXPECT_EQ(tensor->strides(), std::vector<ptrdiff_t>({12, 4, 1}));
    EXPECT_EQ(tensor->stride(0), 12);
    EXPECT_EQ(tensor->stride(1), 4);
    EXPECT_EQ(tensor->stride(2), 1);
}

TEST_F(TensorTest, LoadAndReadData) {
    auto tensor = Tensor::create({2, 3}, ZEDINFER_DTYPE_F32);

    // Prepare test data
    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    fillTensorData(tensor, data);

    // Read and verify
    auto result = readTensorData<float>(tensor);
    EXPECT_TRUE(compareFloatVectors(result, data));
}

// ============================================================================
// isContiguous tests
// ============================================================================
TEST_F(TensorTest, IsContiguousAfterCreation) {
    auto tensor = Tensor::create({3, 4, 5}, ZEDINFER_DTYPE_F32);
    EXPECT_TRUE(tensor->isContiguous());
}

TEST_F(TensorTest, IsContiguousAfterPermute) {
    auto tensor = Tensor::create({2, 3, 4}, ZEDINFER_DTYPE_F32);

    // Original tensor is contiguous
    EXPECT_TRUE(tensor->isContiguous());

    // Becomes non-contiguous after transpose
    auto transposed = tensor->permute({2, 1, 0});
    EXPECT_FALSE(transposed->isContiguous());
}

TEST_F(TensorTest, IsContiguousAfterSlice) {
    auto tensor = Tensor::create({5, 6}, ZEDINFER_DTYPE_F32);

    // Slicing along the first dim is still contiguous (innermost dim intact)
    auto sliced = tensor->slice(0, 1, 4);
    EXPECT_TRUE(sliced->isContiguous());

    // After slicing along the first dim, strides unchanged but shape changed, still contiguous
    EXPECT_EQ(sliced->shape(), std::vector<size_t>({3, 6}));
}

// ============================================================================
// permute tests
// ============================================================================
TEST_F(TensorTest, PermuteBasic) {
    auto tensor = Tensor::create({2, 3, 4}, ZEDINFER_DTYPE_F32);
    std::vector<float> data(24);
    std::iota(data.begin(), data.end(), 0.0f); // 0, 1, 2, ..., 23
    fillTensorData(tensor, data);

    // Transpose: (2,3,4) -> (4,3,2)
    auto permuted = tensor->permute({2, 1, 0});

    EXPECT_EQ(permuted->shape(), std::vector<size_t>({4, 3, 2}));
    EXPECT_EQ(permuted->strides(), std::vector<ptrdiff_t>({1, 4, 12}));
    EXPECT_EQ(permuted->numel(), 24);
}

TEST_F(TensorTest, PermuteInvalidOrder) {
    auto tensor = Tensor::create({2, 3, 4}, ZEDINFER_DTYPE_F32);

    // Dimension count mismatch
    EXPECT_THROW(tensor->permute({0, 1}), std::invalid_argument);

    // Contains duplicate indices
    EXPECT_THROW(tensor->permute({0, 1, 1}), std::invalid_argument);

    // Contains out-of-bounds index
    EXPECT_THROW(tensor->permute({0, 1, 5}), std::invalid_argument);
}

TEST_F(TensorTest, PermutePreservesData) {
    auto tensor = Tensor::create({2, 3}, ZEDINFER_DTYPE_F32);
    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    fillTensorData(tensor, data);

    // Transpose: (2,3) -> (3,2)
    auto transposed = tensor->permute({1, 0});

    EXPECT_EQ(transposed->shape(), std::vector<size_t>({3, 2}));

    // Verify data via contiguous operation
    auto cont = transposed->contiguous();
    auto result = readTensorData<float>(cont);

    std::vector<float> expected = {1.0f, 4.0f, 2.0f, 5.0f, 3.0f, 6.0f};
    EXPECT_TRUE(compareFloatVectors(result, expected));
}

// ============================================================================
// slice tests
// ============================================================================
TEST_F(TensorTest, SliceBasic) {
    auto tensor = Tensor::create({5, 6}, ZEDINFER_DTYPE_F32);
    std::vector<float> data(30);
    std::iota(data.begin(), data.end(), 0.0f);
    fillTensorData(tensor, data);

    // Slice first dim: [1:4] -> shape=(3,6)
    auto sliced = tensor->slice(0, 1, 4);

    EXPECT_EQ(sliced->shape(), std::vector<size_t>({3, 6}));
    EXPECT_EQ(sliced->numel(), 18);

    // Verify data (rows 1,2,3, total 18 elements)
    auto result = readTensorData<float>(sliced);
    std::vector<float> expected(18);
    std::iota(expected.begin(), expected.end(), 6.0f); // Starting from the 6th element
    EXPECT_TRUE(compareFloatVectors(result, expected));
}

TEST_F(TensorTest, SliceInvalidRange) {
    auto tensor = Tensor::create({5, 6}, ZEDINFER_DTYPE_F32);

    // start >= end
    EXPECT_THROW(tensor->slice(0, 3, 3), std::invalid_argument);
    EXPECT_THROW(tensor->slice(0, 4, 2), std::invalid_argument);

    // end out of range
    EXPECT_THROW(tensor->slice(0, 0, 10), std::invalid_argument);

    // dim out of bounds
    EXPECT_THROW(tensor->slice(5, 0, 2), std::invalid_argument);
}

TEST_F(TensorTest, SliceChaining) {
    auto tensor = Tensor::create({5, 6, 7}, ZEDINFER_DTYPE_F32);

    // Chained slicing
    auto sliced1 = tensor->slice(0, 1, 4);  // (3,6,7)
    auto sliced2 = sliced1->slice(1, 2, 5); // (3,3,7)
    auto sliced3 = sliced2->slice(2, 1, 6); // (3,3,5)

    EXPECT_EQ(sliced3->shape(), std::vector<size_t>({3, 3, 5}));
    EXPECT_EQ(sliced3->numel(), 45);
}

// ============================================================================
// view tests
// ============================================================================
TEST_F(TensorTest, ViewBasic) {
    auto tensor = Tensor::create({2, 3, 4}, ZEDINFER_DTYPE_F32);

    // Reshape: (2,3,4) -> (6,4)
    auto viewed = tensor->view({6, 4});

    EXPECT_EQ(viewed->shape(), std::vector<size_t>({6, 4}));
    EXPECT_EQ(viewed->numel(), 24);
    EXPECT_TRUE(viewed->isContiguous());
}

TEST_F(TensorTest, ViewFlatten) {
    auto tensor = Tensor::create({2, 3, 4}, ZEDINFER_DTYPE_F32);

    // Flatten to 1D
    auto flattened = tensor->view({24});

    EXPECT_EQ(flattened->shape(), std::vector<size_t>({24}));
    EXPECT_EQ(flattened->ndim(), 1);
}

TEST_F(TensorTest, ViewIncompatibleShape) {
    auto tensor = Tensor::create({2, 3, 4}, ZEDINFER_DTYPE_F32);

    // Element count mismatch
    EXPECT_THROW(tensor->view({2, 10}), std::invalid_argument);
    EXPECT_THROW(tensor->view({5, 5}), std::invalid_argument);
}

TEST_F(TensorTest, ViewRequiresContiguous) {
    auto tensor = Tensor::create({2, 3, 4}, ZEDINFER_DTYPE_F32);
    auto transposed = tensor->permute({2, 1, 0}); // Non-contiguous

    // view requires the tensor to be contiguous
    EXPECT_THROW(transposed->view({4, 6}), std::runtime_error);
}

TEST_F(TensorTest, ViewSharesStorage) {
    auto tensor = Tensor::create({2, 3}, ZEDINFER_DTYPE_F32);
    std::vector<float> data = {1, 2, 3, 4, 5, 6};
    fillTensorData(tensor, data);

    auto viewed = tensor->view({3, 2});

    // Modifying the original tensor should affect the view
    std::vector<float> new_data = {10, 20, 30, 40, 50, 60};
    fillTensorData(tensor, new_data);

    auto result = readTensorData<float>(viewed);
    EXPECT_TRUE(compareFloatVectors(result, new_data));
}

// ============================================================================
// contiguous tests
// ============================================================================
TEST_F(TensorTest, ContiguousOnContiguousTensor) {
    auto tensor = Tensor::create({2, 3, 4}, ZEDINFER_DTYPE_F32);
    std::vector<float> data(24);
    std::iota(data.begin(), data.end(), 0.0f);
    fillTensorData(tensor, data);

    auto cont = tensor->contiguous();

    // Should return a view sharing storage (zero-copy)
    EXPECT_TRUE(cont->isContiguous());
    auto result = readTensorData<float>(cont);
    EXPECT_TRUE(compareFloatVectors(result, data));
}

TEST_F(TensorTest, ContiguousOnNonContiguousTensor) {
    auto tensor = Tensor::create({2, 3}, ZEDINFER_DTYPE_F32);
    std::vector<float> data = {1, 2, 3, 4, 5, 6};
    fillTensorData(tensor, data);

    // Non-contiguous after transpose
    auto transposed = tensor->permute({1, 0});
    EXPECT_FALSE(transposed->isContiguous());

    // contiguous should create new contiguous storage
    auto cont = transposed->contiguous();
    EXPECT_TRUE(cont->isContiguous());

    // Verify data correctness
    auto result = readTensorData<float>(cont);
    std::vector<float> expected = {1, 4, 2, 5, 3, 6};
    EXPECT_TRUE(compareFloatVectors(result, expected));
}

TEST_F(TensorTest, ContiguousComplexCase) {
    // Create a complex non-contiguous tensor
    auto tensor = Tensor::create({2, 3, 4}, ZEDINFER_DTYPE_F32);
    std::vector<float> data(24);
    std::iota(data.begin(), data.end(), 1.0f); // 1-24
    fillTensorData(tensor, data);

    // Multiple transformations
    auto perm1 = tensor->permute({2, 0, 1}); // (4,2,3)
    auto sliced = perm1->slice(0, 1, 3);     // (2,2,3)
    auto perm2 = sliced->permute({2, 1, 0}); // (3,2,2)

    EXPECT_FALSE(perm2->isContiguous());

    // Convert to contiguous
    auto cont = perm2->contiguous();
    EXPECT_TRUE(cont->isContiguous());
    EXPECT_EQ(cont->shape(), std::vector<size_t>({3, 2, 2}));
    EXPECT_EQ(cont->numel(), 12);
}

// ============================================================================
// reshape tests
// ============================================================================
TEST_F(TensorTest, ReshapeOnContiguousTensor) {
    auto tensor = Tensor::create({2, 3, 4}, ZEDINFER_DTYPE_F32);

    // reshape on a contiguous tensor should be zero-copy
    auto reshaped = tensor->reshape({6, 4});

    EXPECT_EQ(reshaped->shape(), std::vector<size_t>({6, 4}));
    EXPECT_TRUE(reshaped->isContiguous());
}

TEST_F(TensorTest, ReshapeOnNonContiguousTensor) {
    auto tensor = Tensor::create({2, 3, 4}, ZEDINFER_DTYPE_F32);
    std::vector<float> data(24);
    std::iota(data.begin(), data.end(), 0.0f);
    fillTensorData(tensor, data);

    auto transposed = tensor->permute({2, 1, 0});
    EXPECT_FALSE(transposed->isContiguous());

    // reshape on a non-contiguous tensor will call contiguous first
    auto reshaped = transposed->reshape({24});

    EXPECT_TRUE(reshaped->isContiguous());
    EXPECT_EQ(reshaped->shape(), std::vector<size_t>({24}));
}

TEST_F(TensorTest, ReshapeInvalidShape) {
    auto tensor = Tensor::create({2, 3, 4}, ZEDINFER_DTYPE_F32);

    EXPECT_THROW(tensor->reshape({2, 10}), std::invalid_argument);
}

// ============================================================================
// to (device transfer) tests
// ============================================================================
TEST_F(TensorTest, ToSameDevice) {
    auto tensor = Tensor::create({2, 3}, ZEDINFER_DTYPE_F32, ZEDINFER_DEVICE_CPU);
    std::vector<float> data = {1, 2, 3, 4, 5, 6};
    fillTensorData(tensor, data);

    // Moving to the same device should be zero-copy
    auto same_device = tensor->to(ZEDINFER_DEVICE_CPU, 0);

    EXPECT_EQ(same_device->deviceType(), ZEDINFER_DEVICE_CPU);
    auto result = readTensorData<float>(same_device);
    EXPECT_TRUE(compareFloatVectors(result, data));
}

#ifdef ENABLE_NVIDIA_API
TEST_F(TensorTest, ToCPUToGPU) {
    auto cpu_tensor = Tensor::create({2, 3}, ZEDINFER_DTYPE_F32, ZEDINFER_DEVICE_CPU);
    std::vector<float> data = {1, 2, 3, 4, 5, 6};
    fillTensorData(cpu_tensor, data);

    // CPU -> GPU
    auto gpu_tensor = cpu_tensor->to(ZEDINFER_DEVICE_NVIDIA, 0);

    EXPECT_EQ(gpu_tensor->deviceType(), ZEDINFER_DEVICE_NVIDIA);
    EXPECT_EQ(gpu_tensor->shape(), std::vector<size_t>({2, 3}));

    // Verify data
    auto result = readTensorData<float>(gpu_tensor);
    EXPECT_TRUE(compareFloatVectors(result, data));
}

TEST_F(TensorTest, ToGPUToCPU) {
    auto gpu_tensor = Tensor::create({2, 3}, ZEDINFER_DTYPE_F32, ZEDINFER_DEVICE_NVIDIA);
    std::vector<float> data = {1, 2, 3, 4, 5, 6};
    fillTensorData(gpu_tensor, data);

    // GPU -> CPU
    auto cpu_tensor = gpu_tensor->to(ZEDINFER_DEVICE_CPU, 0);

    EXPECT_EQ(cpu_tensor->deviceType(), ZEDINFER_DEVICE_CPU);
    auto result = readTensorData<float>(cpu_tensor);
    EXPECT_TRUE(compareFloatVectors(result, data));
}

TEST_F(TensorTest, ToNonContiguousTensor) {
    auto tensor = Tensor::create({2, 3, 4}, ZEDINFER_DTYPE_F32, ZEDINFER_DEVICE_CPU);
    std::vector<float> data(24);
    std::iota(data.begin(), data.end(), 0.0f);
    fillTensorData(tensor, data);

    auto transposed = tensor->permute({2, 1, 0});
    EXPECT_FALSE(transposed->isContiguous());

    // Transfer non-contiguous tensor to GPU
    auto gpu_tensor = transposed->to(ZEDINFER_DEVICE_NVIDIA, 0);

    EXPECT_EQ(gpu_tensor->deviceType(), ZEDINFER_DEVICE_NVIDIA);
    // Note: to() does not guarantee contiguous — it preserves the layout.
    // Call contiguous() explicitly if needed.
}
#endif

// ============================================================================
// Integration tests
// ============================================================================
TEST_F(TensorTest, ComplexOperationChain) {
    // Create tensor: (2,3,4)
    auto tensor = Tensor::create({2, 3, 4}, ZEDINFER_DTYPE_F32);
    std::vector<float> data(24);
    std::iota(data.begin(), data.end(), 1.0f);
    fillTensorData(tensor, data);

    // Chained operations
    auto result = tensor
                      ->permute({2, 1, 0}) // (4,3,2)
                      ->slice(0, 1, 4)     // (3,3,2)
                      ->contiguous()       // Make contiguous
                      ->view({9, 2})       // Reshape
                      ->slice(1, 0, 1);    // Take first column (9,1)

    EXPECT_EQ(result->shape(), std::vector<size_t>({9, 1}));
    EXPECT_EQ(result->numel(), 9);
}

TEST_F(TensorTest, DataIntegrityAfterMultipleTransforms) {
    auto tensor = Tensor::create({3, 4}, ZEDINFER_DTYPE_F32);
    std::vector<float> data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    fillTensorData(tensor, data);

    // Transpose
    auto t1 = tensor->permute({1, 0}); // (4,3)

    // Slice
    auto t2 = t1->slice(0, 1, 3); // (2,3)

    // Make contiguous
    auto t3 = t2->contiguous();

    auto result = readTensorData<float>(t3);

    // Expected: rows 2,3 of original matrix (indices 1,2)
    // After transpose, these are columns 2,3
    std::vector<float> expected = {2, 6, 10, 3, 7, 11};

    EXPECT_TRUE(compareFloatVectors(result, expected));
}

// ============================================================================
// Edge case tests
// ============================================================================
TEST_F(TensorTest, EmptyShapeDimension) {
    // Shape containing 0
    auto tensor = Tensor::create({2, 0, 3}, ZEDINFER_DTYPE_F32);
    EXPECT_EQ(tensor->numel(), 0);
}

TEST_F(TensorTest, SingleElementTensor) {
    auto tensor = Tensor::create({1}, ZEDINFER_DTYPE_F32);
    EXPECT_EQ(tensor->numel(), 1);
    EXPECT_TRUE(tensor->isContiguous());

    std::vector<float> data = {42.0f};
    fillTensorData(tensor, data);

    auto result = readTensorData<float>(tensor);
    EXPECT_FLOAT_EQ(result[0], 42.0f);
}

TEST_F(TensorTest, HighDimensionalTensor) {
    auto tensor = Tensor::create({2, 3, 4, 5, 6}, ZEDINFER_DTYPE_F32);
    EXPECT_EQ(tensor->ndim(), 5);
    EXPECT_EQ(tensor->numel(), 720);
    EXPECT_TRUE(tensor->isContiguous());
}

} // namespace test
} // namespace zedinfer