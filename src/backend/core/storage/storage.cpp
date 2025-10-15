#include "backend/core/storage/storage.hpp"
#include "backend/core/runtime/runtime.hpp"

namespace neollm::core {
Storage::Storage(std::byte *memory, size_t size, Runtime &runtime, bool is_host, bool is_mmap)
    : memory_(memory), size_(size), runtime_(runtime), is_host_(is_host), is_mmap_(is_mmap) {}

Storage::~Storage() {
    runtime_.freeStorage(this);
}

std::byte *Storage::memory() const {
    return memory_;
}

size_t Storage::size() const {
    return size_;
}

NeollmDeviceType_t Storage::deviceType() const {
    if (isHost()) {
        return NEOLLM_DEVICE_CPU;
    } else {
        return runtime_.deviceType();
    }
}

int Storage::deviceId() const {
    if (isHost()) {
        return 0;
    } else {
        return runtime_.deviceId();
    }
}

bool Storage::isHost() const {
    return is_host_;
}

bool Storage::isMmap() const {
    return is_mmap_;
}
} // namespace neollm::core