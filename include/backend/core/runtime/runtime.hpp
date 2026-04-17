#pragma once

#include "backend/core/core.hpp"
#include "backend/device/runtime_api.hpp"

#include <memory>

namespace zedinfer::core {

class Runtime {
private:
    zedinferDeviceType_t device_type_;
    int device_id_;
    bool is_active_;
    const ZedinferRuntimeAPI* api_;
    allocator_t allocator_;
    zedinferStream_t stream_;          // compute stream: kernel launches, activation memcpys
    zedinferStream_t stream_transfer_; // transfer stream: H2D/D2H weight movement (Phase 2 expert offloading)

    Runtime(zedinferDeviceType_t device_type, int device_id);

    void activate();
    void deactivate();

public:
    // Creates a new Runtime instance for the specified device.
    static std::unique_ptr<Runtime> create(zedinferDeviceType_t device_type, int device_id);

    ~Runtime();

    // Non-copyable and non-movable.
    Runtime(const Runtime&) = delete;
    Runtime& operator=(const Runtime&) = delete;
    Runtime(Runtime&&) = delete;
    Runtime& operator=(Runtime&&) = delete;

    // Device properties.
    zedinferDeviceType_t deviceType() const { return device_type_; }
    int deviceId() const { return device_id_; }
    bool isActive() const { return is_active_; }

    // Returns the associated runtime API.
    const ZedinferRuntimeAPI* api() const { return api_; }

    // Device and host memory allocation.
    storage_t allocateDeviceStorage(size_t size);
    storage_t allocateHostStorage(size_t size);
    storage_t allocateMmapStorage(std::byte* data_ptr, size_t size, bool is_host = true);
    void freeStorage(Storage* storage);

    // Stream management.
    zedinferStream_t stream() const { return stream_; }
    zedinferStream_t transfer_stream() const { return stream_transfer_; }
    void synchronize() const;
    // Synchronize a specific stream (does nothing on CPU).
    void synchronize_stream(zedinferStream_t s) const;

    friend class Context;
};

} // namespace zedinfer::core