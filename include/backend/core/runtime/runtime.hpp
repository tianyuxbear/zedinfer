#pragma once

#include "backend/core/core.hpp"
#include "backend/device/runtime_api.hpp"
#include <memory>

namespace neollm::core {

class Runtime {
private:
    NeollmDeviceType_t device_type_;
    int device_id_;
    bool is_active_;
    const NeollmRuntimeAPI *api_;
    allocator_t allocator_;
    NeollmStream_t stream_;

    Runtime(NeollmDeviceType_t device_type, int device_id);

    void activate();
    void deactivate();

public:
    // Creates a new Runtime instance for the specified device.
    static std::unique_ptr<Runtime> create(NeollmDeviceType_t device_type, int device_id);

    ~Runtime();

    // Non-copyable and non-movable.
    Runtime(const Runtime &) = delete;
    Runtime &operator=(const Runtime &) = delete;
    Runtime(Runtime &&) = delete;
    Runtime &operator=(Runtime &&) = delete;

    // Device properties.
    NeollmDeviceType_t deviceType() const { return device_type_; }
    int deviceId() const { return device_id_; }
    bool isActive() const { return is_active_; }

    // Returns the associated runtime API.
    const NeollmRuntimeAPI *api() const { return api_; }

    // Device and host memory allocation.
    storage_t allocateDeviceStorage(size_t size);
    storage_t allocateHostStorage(size_t size);
    storage_t allocateMmapStorage(std::byte *data_ptr, size_t size);
    void freeStorage(Storage *storage);

    // Stream management.
    NeollmStream_t stream() const { return stream_; }
    void synchronize() const;

    friend class Context;
};

} // namespace neollm::core