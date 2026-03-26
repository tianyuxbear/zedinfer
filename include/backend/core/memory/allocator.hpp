#pragma once

#include "backend/core/memory/memory_pool.hpp"
#include "backend/device/runtime_api.hpp"
#include "zedinfer.h"

#include <cstddef>
#include <memory>

namespace zedinfer::core::memory {

class MemoryAllocator {
protected:
    const ZedinferRuntimeAPI* api_;
    zedinferDeviceType_t device_type_;
    int device_id_;
    std::unique_ptr<MemoryPool> memory_pool_;

    MemoryAllocator(const ZedinferRuntimeAPI* api, zedinferDeviceType_t device_type, int device_id)
        : api_(api), device_type_(device_type), device_id_(device_id) {}

public:
    virtual ~MemoryAllocator() = default;

    // Non-copyable and non-movable.
    MemoryAllocator(const MemoryAllocator&) = delete;
    MemoryAllocator& operator=(const MemoryAllocator&) = delete;
    MemoryAllocator(MemoryAllocator&&) = delete;
    MemoryAllocator& operator=(MemoryAllocator&&) = delete;

    // Core allocation interface.
    virtual std::byte* allocate(size_t size) = 0;
    virtual void release(std::byte* memory) = 0;

    // Device info.
    int deviceId() const { return device_id_; }
    const ZedinferRuntimeAPI* api() const { return api_; }
};

} // namespace zedinfer::core::memory