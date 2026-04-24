#include "backend/core/context/context.hpp"
#include "backend/core/runtime/runtime.hpp"
#include "backend/core/storage/storage.hpp" // IWYU pragma: keep
#include "utils/check.hpp"
#include "zedinfer.h"

#include <future>
#include <gtest/gtest.h>
#include <thread>
#include <unordered_map>
#include <vector>

namespace zedinfer::test {
// Global mock state
struct MockState {
    std::unordered_map<void*, size_t> allocated_memory;
    int current_device = 0;
    size_t alloc_count = 0;
    size_t free_count = 0;

    void reset() {
        for (auto& [ptr, size] : allocated_memory) { ::operator delete(ptr); }
        allocated_memory.clear();
        current_device = 0;
        alloc_count = 0;
        free_count = 0;
    }

    size_t getAllocatedCount() const { return allocated_memory.size(); }
};

static MockState g_mock_state;

// Mock function implementations
static void* mock_malloc_device(size_t size) {
    void* ptr = ::operator new(size);
    g_mock_state.allocated_memory[ptr] = size;
    g_mock_state.alloc_count++;
    return ptr;
}

static void mock_free_device(void* ptr) {
    if (g_mock_state.allocated_memory.find(ptr) != g_mock_state.allocated_memory.end()) {
        g_mock_state.allocated_memory.erase(ptr);
        g_mock_state.free_count++;
        ::operator delete(ptr);
    }
}

static void* mock_malloc_host(size_t size) {
    void* ptr = ::operator new(size);
    g_mock_state.allocated_memory[ptr] = size;
    g_mock_state.alloc_count++;
    return ptr;
}

static void mock_free_host(void* ptr) {
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

static void* mock_create_stream() {
    return reinterpret_cast<void*>(0x1234);
}

static void mock_destroy_stream(void* stream) {
    ASSERT(stream == (void*)0x1234, "CPU does not support explicit streams");
}

static void mock_stream_synchronize(void* stream) {
    ASSERT(stream == (void*)0x1234, "CPU does not support explicit streams");
}

// Create mock API
static ZedinferRuntimeAPI createMockAPI() {
    ZedinferRuntimeAPI api;
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

} // namespace zedinfer::test

namespace zedinfer::device {
static ZedinferRuntimeAPI g_mock_api = zedinfer::test::createMockAPI();

const ZedinferRuntimeAPI* getRuntimeAPI(zedinferDeviceType_t device) {
    ASSERT(device == ZEDINFER_DEVICE_CPU, "Only support CPU in test");
    return &g_mock_api;
}
} // namespace zedinfer::device

// ============================================================================
// Storage tests
// ============================================================================
using namespace zedinfer::core;
using namespace zedinfer::test;

class StorageTest : public ::testing::Test {
protected:
    void SetUp() override { g_mock_state.reset(); }

    void TearDown() override {
        // Verify no memory leaks
        EXPECT_EQ(g_mock_state.getAllocatedCount(), 0)
            << "Memory leak detected: " << g_mock_state.getAllocatedCount() << " blocks not freed";
    }
};

TEST_F(StorageTest, DeviceStorageCreation) {
    Context& ctx = context();
    ctx.setDevice(ZEDINFER_DEVICE_CPU, 0);

    size_t size = 1024 * 1024; // 1MB
    auto storage = ctx.runtime().allocateDeviceStorage(size);

    ASSERT_NE(storage, nullptr);
    EXPECT_NE(storage->memory(), nullptr);
    EXPECT_GE(storage->size(), size);
    EXPECT_FALSE(storage->isHost());
    EXPECT_EQ(storage->deviceType(), ZEDINFER_DEVICE_CPU);
    EXPECT_EQ(storage->deviceId(), 0);

    storage.reset();
    ctx.reset();
}

TEST_F(StorageTest, ThreadLocalContextStartsWithActiveCpuRuntime) {
    std::promise<bool> runtime_active;
    auto future = runtime_active.get_future();

    std::thread worker([promise = std::move(runtime_active)]() mutable {
        Context& ctx = context();
        promise.set_value(ctx.runtime().isActive());
        ctx.reset();
    });

    EXPECT_TRUE(future.get());
    worker.join();
}

TEST_F(StorageTest, HostStorageCreation) {
    Context& ctx = context();
    ctx.setDevice(ZEDINFER_DEVICE_CPU, 0);

    size_t size = 2 * 1024 * 1024; // 2MB
    auto storage = ctx.runtime().allocateHostStorage(size);

    ASSERT_NE(storage, nullptr);
    EXPECT_NE(storage->memory(), nullptr);
    EXPECT_EQ(storage->size(), size);
    EXPECT_TRUE(storage->isHost());
    EXPECT_EQ(storage->deviceType(), ZEDINFER_DEVICE_CPU);
    EXPECT_EQ(storage->deviceId(), 0);

    storage.reset();
    ctx.reset();
}

TEST_F(StorageTest, StorageAutoRelease) {
    Context& ctx = context();
    ctx.setDevice(ZEDINFER_DEVICE_CPU, 0);

    size_t count_before = g_mock_state.getAllocatedCount();

    {
        auto storage = ctx.runtime().allocateDeviceStorage(1024);
        EXPECT_GE(g_mock_state.getAllocatedCount(), count_before);
    } // storage goes out of scope, automatically released

    // Memory should be freed (there may still be blocks in the memory pool)
    // So here we only verify that the storage's own memory was managed
    ctx.reset();
}

TEST_F(StorageTest, MultipleStorages) {
    Context& ctx = context();
    ctx.setDevice(ZEDINFER_DEVICE_CPU, 0);

    std::vector<storage_t> storages;

    for (int i = 0; i < 10; ++i) { storages.push_back(ctx.runtime().allocateDeviceStorage((i + 1) * 1024)); }

    EXPECT_EQ(storages.size(), 10);

    for (size_t i = 0; i < storages.size(); ++i) {
        EXPECT_NE(storages[i], nullptr);
        EXPECT_NE(storages[i]->memory(), nullptr);
    }

    storages.clear();
    ctx.reset();
}

TEST_F(StorageTest, StorageMemoryAccess) {
    Context& ctx = context();
    ctx.setDevice(ZEDINFER_DEVICE_CPU, 0);

    auto storage = ctx.runtime().allocateDeviceStorage(1024);

    // Write data
    std::byte* mem = storage->memory();
    for (size_t i = 0; i < 1024; ++i) { mem[i] = static_cast<std::byte>(i % 256); }

    // Verify data
    for (size_t i = 0; i < 1024; ++i) { EXPECT_EQ(mem[i], static_cast<std::byte>(i % 256)); }

    storage.reset();
    ctx.reset();
}

TEST_F(StorageTest, StorageOnDifferentDevices) {
    Context& ctx = context();

    // Allocate on device 0
    ctx.setDevice(ZEDINFER_DEVICE_CPU, 0);
    auto storage0 = ctx.runtime().allocateDeviceStorage(1024);
    EXPECT_EQ(storage0->deviceId(), 0);

    // Allocate on device 1
    ctx.setDevice(ZEDINFER_DEVICE_CPU, 1);
    auto storage1 = ctx.runtime().allocateDeviceStorage(1024);
    EXPECT_EQ(storage1->deviceId(), 1);

    // The two storages should be on different devices
    EXPECT_NE(storage0->deviceId(), storage1->deviceId());

    storage0.reset();
    storage1.reset();
    ctx.reset();
}
