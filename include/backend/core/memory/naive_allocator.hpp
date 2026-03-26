#pragma once

#include "backend/core/context/context.hpp"
#include "backend/core/memory/allocator.hpp"
#include "backend/device/runtime_api.hpp"
#include "zedinfer.h"

#include <cstddef>

namespace zedinfer::core::memory {

// A simple allocator that directly uses the device runtime API.
class NaiveAllocator : public MemoryAllocator {
public:
    NaiveAllocator(const ZedinferRuntimeAPI* api, zedinferDeviceType_t device_type, int device_id)
        : MemoryAllocator(api, device_type, device_id) {}

    // Allocates device memory via the runtime API.
    std::byte* allocate(size_t size) override {
        core::context().setDevice(device_type_, device_id_);
        return static_cast<std::byte*>(api_->malloc_device(size));
    }

    // Frees device memory via the runtime API.
    void release(std::byte* memory) override {
        if (!memory) {
            return;
        }
        core::context().setDevice(device_type_, device_id_);
        api_->free_device(memory);
    }
};

} // namespace zedinfer::core::memory