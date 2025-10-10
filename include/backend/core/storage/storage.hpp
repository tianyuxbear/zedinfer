#pragma once
#include "backend/core/core.hpp"
#include "neollm.h"

namespace neollm::core {

// Memory block managed by a Runtime.
class Storage {
private:
    std::byte *memory_;
    size_t size_;
    Runtime &runtime_;
    bool is_host_;

    Storage(std::byte *memory, size_t size, Runtime &runtime, bool is_host);

public:
    friend class Runtime;
    ~Storage();

    std::byte *memory() const;
    size_t size() const;
    NeollmDeviceType_t deviceType() const;
    int deviceId() const;
    bool isHost() const;
};

} // namespace neollm::core