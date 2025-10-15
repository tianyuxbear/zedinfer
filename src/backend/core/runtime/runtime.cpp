#include "backend/core/runtime/runtime.hpp"
#include "backend/core/memory/pooled_allocator.hpp"
#include "backend/core/storage/storage.hpp"
#include "backend/device/runtime_api.hpp"

#include <iostream>

namespace neollm::core {
Runtime::Runtime(NeollmDeviceType_t device_type, int device_id)
    : device_type_(device_type), device_id_(device_id), is_active_(false) {

    // Retrieve device-specific runtime API.
    api_ = neollm::device::getRuntimeAPI(device_type_);
    if (api_ == nullptr) {
        throw std::runtime_error(
            "No runtime API available for device type: " + std::to_string(static_cast<int>(device_type_)));
    }

    if (device_id_ < 0 || device_id_ >= api_->get_device_count()) {
        throw std::invalid_argument(
            "Invalid device ID: " + std::to_string(device_id_));
    }

    stream_ = api_->create_stream();
    if (device_type_ != NEOLLM_DEVICE_CPU && stream_ == nullptr) {
        throw std::runtime_error("Failed to create stream");
    }

    allocator_ = std::make_unique<memory::PooledAllocator>(api_, device_type, device_id);
}

std::unique_ptr<Runtime> Runtime::create(
    NeollmDeviceType_t device_type,
    int device_id) {

    std::unique_ptr<Runtime> runtime(new Runtime(device_type, device_id));

    return runtime;
}

Runtime::~Runtime() {
    if (!is_active_) {
        std::cerr << "Mallicious destruction of inactive runtime." << std::endl;
    }
    api_->destroy_stream(stream_);
    api_ = nullptr;
    // api_ is a borrowed pointer; do not delete.
}

void Runtime::activate() {
    api_->set_device(device_id_);
    is_active_ = true;
}

void Runtime::deactivate() {
    is_active_ = false;
}

storage_t Runtime::allocateDeviceStorage(size_t size) {
    return std::shared_ptr<Storage>(new Storage(allocator_->allocate(size), size, *this, false));
}

storage_t Runtime::allocateHostStorage(size_t size) {
    return std::shared_ptr<Storage>(new Storage((std::byte *)api_->malloc_host(size), size, *this, true));
}

storage_t Runtime::allocateMmapStorage(std::byte *data_ptr, size_t size) {
    return std::shared_ptr<Storage>(new Storage(data_ptr, size, *this, true, true));
}

void Runtime::freeStorage(Storage *storage) {
    if (storage->isMmap()) {
        return;
    }
    if (storage->isHost()) {
        api_->free_host(storage->memory());
    } else {
        allocator_->release(storage->memory());
    }
}

void Runtime::synchronize() const {
    api_->stream_synchronize(stream_);
}

} // namespace neollm::core
