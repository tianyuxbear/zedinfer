#pragma once

#include "backend/core/context/context.hpp"
#include "backend/core/memory/allocator.hpp"
#include "backend/device/runtime_api.hpp"
#include "neollm.h"
#include <cstddef>

namespace neollm::core::memory {

// A memory allocator that uses a memory pool for efficient device allocations.
class PooledAllocator : public MemoryAllocator {
public:
    PooledAllocator(const NeollmRuntimeAPI *api,
                    NeollmDeviceType_t device_type,
                    int device_id,
                    const MemoryPoolConfig &config = MemoryPoolConfig())
        : MemoryAllocator(api, device_type, device_id) {

        // Initialize the memory pool with device-specific allocation/deallocation functions.
        auto alloc_func = [this](size_t size) -> void * {
            return api_->malloc_device(size);
        };

        auto free_func = [this](void *ptr) {
            api_->free_device(ptr);
        };

        memory_pool_ = std::make_unique<BestFitMemoryPool>(alloc_func, free_func, config);
    }

    ~PooledAllocator() override {
        // Ensure proper device context before releasing the memory pool.
        if (memory_pool_) {
            /*
             * During context destruction, the runtime is automatically released,
             * which in turn triggers the destruction of the allocator. At this point,
             * the context is already in the process of being torn down, so it is no
             * longer safe to use context.setDevice() to switch the device context.
             * To ensure the correct device is active when releasing the memory pool,
             * we directly invoke the underlying API to set the device.
             */
            api_->set_device(device_id_);
            memory_pool_.reset();
        }
    }

    // Allocates memory from the pool, switching to the correct device context.
    std::byte *allocate(size_t size) override {
        core::context().setDevice(device_type_, device_id_);
        return memory_pool_->allocate(size);
    }

    // Returns memory to the pool, switching to the correct device context.
    void release(std::byte *memory) override {
        if (!memory) {
            return;
        }
        core::context().setDevice(device_type_, device_id_);
        memory_pool_->deallocate(memory);
    }
};

} // namespace neollm::core::memory