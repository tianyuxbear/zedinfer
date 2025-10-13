#include "backend/core/context/context.hpp"
#include "backend/core/runtime/runtime.hpp"
#include "backend/core/storage/storage.hpp"
#include "neollm.h"
#include "utils/check.hpp"
#include <gtest/gtest.h>
#include <unordered_map>
#include <vector>


namespace neollm::test {
// 全局 Mock 状态
struct MockState {
    std::unordered_map<void *, size_t> allocated_memory;
    int current_device = 0;
    size_t alloc_count = 0;
    size_t free_count = 0;

    void reset() {
        for (auto &[ptr, size] : allocated_memory) {
            ::operator delete(ptr);
        }
        allocated_memory.clear();
        current_device = 0;
        alloc_count = 0;
        free_count = 0;
    }

    size_t getAllocatedCount() const {
        return allocated_memory.size();
    }
};

static MockState g_mock_state;

// Mock 函数实现
static void *mock_malloc_device(size_t size) {
    void *ptr = ::operator new(size);
    g_mock_state.allocated_memory[ptr] = size;
    g_mock_state.alloc_count++;
    return ptr;
}

static void mock_free_device(void *ptr) {
    if (g_mock_state.allocated_memory.find(ptr) != g_mock_state.allocated_memory.end()) {
        g_mock_state.allocated_memory.erase(ptr);
        g_mock_state.free_count++;
        ::operator delete(ptr);
    }
}

static void *mock_malloc_host(size_t size) {
    void *ptr = ::operator new(size);
    g_mock_state.allocated_memory[ptr] = size;
    g_mock_state.alloc_count++;
    return ptr;
}

static void mock_free_host(void *ptr) {
    if (g_mock_state.allocated_memory.find(ptr) != g_mock_state.allocated_memory.end()) {
        g_mock_state.allocated_memory.erase(ptr);
        g_mock_state.free_count++;
        ::operator delete(ptr);
    }
}

static int mock_get_device_count() {
    return 4;
}

static void mock_set_device(int device_id) {
    g_mock_state.current_device = device_id;
}

static void *mock_create_stream() {
    return reinterpret_cast<void *>(0x1234);
}

static void mock_destroy_stream(void *stream) {
    ASSERT(stream == (void *)0x1234, "CPU does not support explicit streams");
}

static void mock_stream_synchronize(void *stream) {
    ASSERT(stream == (void *)0x1234, "CPU does not support explicit streams");
}

// 创建 Mock API
static NeollmRuntimeAPI createMockAPI() {
    NeollmRuntimeAPI api;
    api.malloc_device = mock_malloc_device;
    api.free_device = mock_free_device;
    api.malloc_host = mock_malloc_host;
    api.free_host = mock_free_host;
    api.get_device_count = mock_get_device_count;
    api.set_device = mock_set_device;
    api.create_stream = mock_create_stream;
    api.destroy_stream = mock_destroy_stream;
    api.stream_synchronize = mock_stream_synchronize;
    return api;
}

} // namespace neollm::test

namespace neollm::device {
static NeollmRuntimeAPI g_mock_api = neollm::test::createMockAPI();

const NeollmRuntimeAPI *getRuntimeAPI(NeollmDeviceType_t device) {
    ASSERT(device == NEOLLM_DEVICE_CPU, "Only support CPU in test");
    return &g_mock_api;
}
} // namespace neollm::device

// ============================================================================
// Storage 测试
// ============================================================================
using namespace neollm::core;
using namespace neollm::test;

class StorageTest : public ::testing::Test {
protected:
    void SetUp() override {
        g_mock_state.reset();
    }

    void TearDown() override {
        // 验证没有内存泄漏
        EXPECT_EQ(g_mock_state.getAllocatedCount(), 0)
            << "Memory leak detected: "
            << g_mock_state.getAllocatedCount() << " blocks not freed";
    }
};

TEST_F(StorageTest, DeviceStorageCreation) {
    Context &ctx = context();
    ctx.setDevice(NEOLLM_DEVICE_CPU, 0);

    size_t size = 1024 * 1024; // 1MB
    auto storage = ctx.runtime().allocateDeviceStorage(size);

    ASSERT_NE(storage, nullptr);
    EXPECT_NE(storage->memory(), nullptr);
    EXPECT_GE(storage->size(), size);
    EXPECT_FALSE(storage->isHost());
    EXPECT_EQ(storage->deviceType(), NEOLLM_DEVICE_CPU);
    EXPECT_EQ(storage->deviceId(), 0);

    storage.reset();
    ctx.reset();
}

TEST_F(StorageTest, HostStorageCreation) {
    Context &ctx = context();
    ctx.setDevice(NEOLLM_DEVICE_CPU, 0);

    size_t size = 2 * 1024 * 1024; // 2MB
    auto storage = ctx.runtime().allocateHostStorage(size);

    ASSERT_NE(storage, nullptr);
    EXPECT_NE(storage->memory(), nullptr);
    EXPECT_EQ(storage->size(), size);
    EXPECT_TRUE(storage->isHost());
    EXPECT_EQ(storage->deviceType(), NEOLLM_DEVICE_CPU);
    EXPECT_EQ(storage->deviceId(), 0);

    storage.reset();
    ctx.reset();
}

TEST_F(StorageTest, StorageAutoRelease) {
    Context &ctx = context();
    ctx.setDevice(NEOLLM_DEVICE_CPU, 0);

    size_t count_before = g_mock_state.getAllocatedCount();

    {
        auto storage = ctx.runtime().allocateDeviceStorage(1024);
        EXPECT_GE(g_mock_state.getAllocatedCount(), count_before);
    } // storage 离开作用域，自动释放

    // 内存应该被释放（可能还有内存池中的块）
    // 所以这里只验证 storage 本身的内存被管理了
    ctx.reset();
}

TEST_F(StorageTest, MultipleStorages) {
    Context &ctx = context();
    ctx.setDevice(NEOLLM_DEVICE_CPU, 0);

    std::vector<storage_t> storages;

    for (int i = 0; i < 10; ++i) {
        storages.push_back(
            ctx.runtime().allocateDeviceStorage((i + 1) * 1024));
    }

    EXPECT_EQ(storages.size(), 10);

    for (size_t i = 0; i < storages.size(); ++i) {
        EXPECT_NE(storages[i], nullptr);
        EXPECT_NE(storages[i]->memory(), nullptr);
    }

    storages.clear();
    ctx.reset();
}

TEST_F(StorageTest, StorageMemoryAccess) {
    Context &ctx = context();
    ctx.setDevice(NEOLLM_DEVICE_CPU, 0);

    auto storage = ctx.runtime().allocateDeviceStorage(1024);

    // 写入数据
    std::byte *mem = storage->memory();
    for (size_t i = 0; i < 1024; ++i) {
        mem[i] = static_cast<std::byte>(i % 256);
    }

    // 验证数据
    for (size_t i = 0; i < 1024; ++i) {
        EXPECT_EQ(mem[i], static_cast<std::byte>(i % 256));
    }

    storage.reset();
    ctx.reset();
}

TEST_F(StorageTest, StorageOnDifferentDevices) {
    Context &ctx = context();

    // 在设备 0 上分配
    ctx.setDevice(NEOLLM_DEVICE_CPU, 0);
    auto storage0 = ctx.runtime().allocateDeviceStorage(1024);
    EXPECT_EQ(storage0->deviceId(), 0);

    // 在设备 1 上分配
    ctx.setDevice(NEOLLM_DEVICE_CPU, 1);
    auto storage1 = ctx.runtime().allocateDeviceStorage(1024);
    EXPECT_EQ(storage1->deviceId(), 1);

    // 两个 storage 应该在不同设备上
    EXPECT_NE(storage0->deviceId(), storage1->deviceId());

    storage0.reset();
    storage1.reset();
    ctx.reset();
}
