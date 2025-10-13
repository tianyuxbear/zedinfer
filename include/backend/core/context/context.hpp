#pragma once

#include "backend/core/core.hpp"
#include "backend/core/runtime/runtime.hpp"
#include "backend/device/device.hpp"
#include "neollm.h"

#include <unordered_map>

namespace neollm::core {

class Context {
private:
    std::unordered_map<device::Device, runtime_t, device::Device::Hash> runtime_map_;
    Runtime *current_runtime_;
    Context(); // Private constructor for singleton-like access

public:
    ~Context() = default;

    // Non-copyable
    Context(const Context &) = delete;
    Context &operator=(const Context &) = delete;

    // Non-movable
    Context(Context &&) = delete;
    Context &operator=(Context &&) = delete;

    // Sets the active device by type and ID.
    void setDevice(NeollmDeviceType_t device_type, int device_id);

    // Returns the currently active Runtime.
    Runtime &runtime();

    // Clears all runtimes and resets internal state (e.g., for test isolation).
    void reset();

    friend Context &context();
};

} // namespace neollm::core