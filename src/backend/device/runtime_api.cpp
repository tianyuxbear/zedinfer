#include "backend/device/runtime_api.hpp"
#include "utils/check.hpp"

namespace neollm::device {

// Returns the runtime API for the given device type
const NeollmRuntimeAPI *getRuntimeAPI(NeollmDeviceType_t device_type) {
    switch (device_type) {
    case NEOLLM_DEVICE_CPU:
        return neollm::device::cpu::getRuntimeAPI();
#ifdef ENABLE_NVIDIA_API
    case NEOLLM_DEVICE_NVIDIA:
        return neollm::device::nvidia::getRuntimeAPI();
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
        return nullptr;
    }
}

} // namespace neollm::device