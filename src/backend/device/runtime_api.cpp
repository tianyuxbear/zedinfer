#include "backend/device/runtime_api.hpp"
#include "utils/check.hpp"

namespace zedinfer::device {

// Returns the runtime API for the given device type
const ZedinferRuntimeAPI* getRuntimeAPI(zedinferDeviceType_t device_type) {
    switch (device_type) {
        case ZEDINFER_DEVICE_CPU:
            return zedinfer::device::cpu::getRuntimeAPI();
#ifdef ENABLE_NVIDIA_API
        case ZEDINFER_DEVICE_NVIDIA:
            return zedinfer::device::nvidia::getRuntimeAPI();
#endif
        default:
            EXCEPTION_UNSUPPORTED_DEVICE;
            return nullptr;
    }
}

} // namespace zedinfer::device