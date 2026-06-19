#pragma once

#include "types.hpp" // IWYU pragma: keep
#include "utils/logging.hpp"
#include <iostream>  // IWYU pragma: keep
#include <stdexcept> // IWYU pragma: keep

#define EXCEPTION_LOCATION_MSG " from " << __func__ << " at " << __FILE__ << ":" << __LINE__ << "."

#define EXCEPTION_UNSUPPORTED_DEVICE                                                                                   \
    do {                                                                                                               \
        LOGE << "[ERROR] Unsupported device" << EXCEPTION_LOCATION_MSG;                                                \
        throw std::runtime_error("Unsupported device");                                                                \
    } while (0)

#define EXCEPTION_UNSUPPORTED_DATATYPE(DT__)                                                                           \
    do {                                                                                                               \
        LOGE << "[ERROR] Unsupported data type: " << zedinfer::utils::dtype_to_str(DT__) << EXCEPTION_LOCATION_MSG;    \
        throw std::runtime_error("Unsupported device");                                                                \
    } while (0)

#define CHECK_ARGUMENT(condition, message)                                                                             \
    do {                                                                                                               \
        if (!(condition)) {                                                                                            \
            LOGE << "[ERROR] Invalid argument: " << message << EXCEPTION_LOCATION_MSG;                                 \
            throw std::invalid_argument(message);                                                                      \
        }                                                                                                              \
    } while (0)

#define ASSERT(condition, message)                                                                                     \
    do {                                                                                                               \
        if (!(condition)) {                                                                                            \
            LOGE << "[ERROR] " << message << "\nAssertion failed: " << #condition << EXCEPTION_LOCATION_MSG;           \
            throw std::runtime_error("Assertion failed");                                                              \
        }                                                                                                              \
    } while (0)

#define TO_BE_IMPLEMENTED()                                                                                            \
    do {                                                                                                               \
        LOGE << "[ERROR] Unimplemented function" << EXCEPTION_LOCATION_MSG;                                            \
        throw std::runtime_error("Unimplemented function");                                                            \
    } while (0)

#define CHECK_SAME(ERR, FIRST, ...)                                                                                    \
    do {                                                                                                               \
        for (const auto& arg___ : {__VA_ARGS__}) {                                                                     \
            if (FIRST != arg___) {                                                                                     \
                { ERR; }                                                                                               \
            }                                                                                                          \
        }                                                                                                              \
    } while (0)

#define EXCEPTION_SHAPE_MISMATCH                                                                                       \
    do {                                                                                                               \
        LOGE << "[ERROR] Shapes mismatch" << EXCEPTION_LOCATION_MSG;                                                   \
        throw std::invalid_argument("Shapes mismatch");                                                                \
    } while (0)

#define CHECK_SAME_SHAPE(FIRST, ...) CHECK_SAME(EXCEPTION_SHAPE_MISMATCH, FIRST, __VA_ARGS__)

#define EXCEPTION_DATATYPE_MISMATCH                                                                                    \
    do {                                                                                                               \
        LOGE << "[ERROR] Datatypes mismatch" << EXCEPTION_LOCATION_MSG;                                                \
        throw std::invalid_argument("Datatypes mismatch");                                                             \
    } while (0)

#define CHECK_SAME_DTYPE(FIRST, ...) CHECK_SAME(EXCEPTION_DATATYPE_MISMATCH, FIRST, __VA_ARGS__)

#define EXCEPTION_DEVICE_MISMATCH                                                                                      \
    do {                                                                                                               \
        LOGE << "[ERROR] Input tensors must be on the same device!\nDevice mismatch" << EXCEPTION_LOCATION_MSG;        \
        throw std::runtime_error("device mismatch");                                                                   \
    } while (0)

#define CHECK_SAME_DEVICE(FIRST, ...)                                                                                  \
    do {                                                                                                               \
        for (const auto& tensor___ : {__VA_ARGS__}) {                                                                  \
            if (FIRST->deviceType() != tensor___->deviceType() || FIRST->deviceId() != tensor___->deviceId()) {        \
                { EXCEPTION_DEVICE_MISMATCH; }                                                                         \
            }                                                                                                          \
        }                                                                                                              \
    } while (0)
