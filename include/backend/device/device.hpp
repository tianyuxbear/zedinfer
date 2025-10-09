#pragma once

#include "neollm.h"
#include <functional>
#include <stdexcept>
#include <string>

namespace neollm::device {

class Device {
public:
    static Device cpu() { return Device(NEOLLM_DEVICE_CPU, 0); }
    static Device cuda(int id = 0) { return Device(NEOLLM_DEVICE_NVIDIA, id); }

    Device() : type_(NEOLLM_DEVICE_CPU), id_(0) {}

    Device(NeollmDeviceType_t type, int id)
        : type_(type), id_(id) {
        validate();
    }

    Device(const Device &) = default;

    NeollmDeviceType_t type() const { return type_; }
    int id() const { return id_; }

    bool operator==(const Device &other) const {
        return type_ == other.type_ && id_ == other.id_;
    }

    bool operator!=(const Device &other) const {
        return !(*this == other);
    }

    struct Hash {
        size_t operator()(const Device &dev) const {
            return std::hash<int>()(static_cast<int>(dev.type_)) ^ (std::hash<int>()(dev.id_) << 1);
        }
    };

    std::string toString() const {
        const char *type_str = nullptr;
        switch (type_) {
        case NEOLLM_DEVICE_CPU:
            type_str = "cpu";
            break;
        case NEOLLM_DEVICE_NVIDIA:
            type_str = "nvidia";
            break;
        default:
            type_str = "unknown";
            break;
        }
        return std::string(type_str) + ":" + std::to_string(id_);
    }

    bool isHost() const { return type_ == NEOLLM_DEVICE_CPU; }
    bool isAccelerator() const { return !isHost(); }

private:
    NeollmDeviceType_t type_;
    int id_;

    void validate() const {
        if (id_ < 0) {
            throw std::invalid_argument("Device ID must be non-negative");
        }
    }
};

} // namespace neollm::device