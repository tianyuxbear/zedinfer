#include "backend/core/context/context.hpp"
#include "backend/core/runtime/runtime.hpp"
#include "backend/device/device.hpp"
#include "utils/check.hpp"
#include "zedinfer.h"

#include <memory>

namespace zedinfer::core {

Context::Context() : current_runtime_(nullptr) {
    // Initialize with a CPU runtime.
    auto runtime = Runtime::create(ZEDINFER_DEVICE_CPU, 0);
    current_runtime_ = runtime.get();
    runtime_map_[device::Device::cpu()] = std::move(runtime);
}

void Context::setDevice(zedinferDeviceType_t device_type, int device_id) {
    device::Device device(device_type, device_id);
    auto it = runtime_map_.find(device);

    if (it == runtime_map_.end()) {
        try {
            auto runtime = Runtime::create(device_type, device_id);
            current_runtime_ = runtime.get();
            runtime_map_[device] = std::move(runtime);
        } catch (const std::exception &e) {
            throw std::runtime_error(
                "Failed to create runtime for " + device.toString() + ": " + e.what());
        }
    } else {
        current_runtime_ = it->second.get();
    }

    if (current_runtime_) {
        current_runtime_->activate();
    }
}

Runtime &Context::runtime() {
    ASSERT(current_runtime_ != nullptr,
           "No runtime is activated; call setDevice() first.");
    return *current_runtime_;
}

void Context::reset() {
    runtime_map_.clear();
    current_runtime_ = nullptr;
}

// Returns the thread-local Context instance.
Context &context() {
    thread_local Context thread_context;
    return thread_context;
}

} // namespace zedinfer::core