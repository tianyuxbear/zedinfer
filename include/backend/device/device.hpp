#pragma once

#include "zedinfer.h"

#include <functional>
#include <stdexcept>
#include <string>

namespace zedinfer::device {

class Device {
public:
    static Device cpu() { return Device(ZEDINFER_DEVICE_CPU, 0); }
    static Device cuda(int id = 0) { return Device(ZEDINFER_DEVICE_NVIDIA, id); }

    Device() : type_(ZEDINFER_DEVICE_CPU), id_(0) {}

    Device(zedinferDeviceType_t type, int id)
        : type_(type), id_(id) {
        validate();
    }

    Device(const Device &) = default;

    zedinferDeviceType_t type() const { return type_; }
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
        case ZEDINFER_DEVICE_CPU:
            type_str = "cpu";
            break;
        case ZEDINFER_DEVICE_NVIDIA:
            type_str = "nvidia";
            break;
        default:
            type_str = "unknown";
            break;
        }
        return std::string(type_str) + ":" + std::to_string(id_);
    }

    bool isHost() const { return type_ == ZEDINFER_DEVICE_CPU; }
    bool isAccelerator() const { return !isHost(); }

private:
    zedinferDeviceType_t type_;
    int id_;

    void validate() const {
        if (id_ < 0) {
            throw std::invalid_argument("Device ID must be non-negative");
        }
    }
};

} // namespace zedinfer::device