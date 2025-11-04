#include "backend/device/runtime_api.hpp"
#include "utils/check.hpp"
#include "zedinfer.h"

#include <cstdlib>
#include <cstring>

namespace zedinfer::device::cpu {

namespace runtime_api {

int getDeviceCount() {
    return 1; // Only one logical CPU device
}

void setDevice(int) {
    // No-op: CPU has no device switching
}

void deviceSynchronize() {
    // No-op: CPU execution is synchronous
}

zedinferStream_t createStream() {
    return nullptr; // CPU uses default (null) stream
}

void destroyStream(zedinferStream_t stream) {
    ASSERT(stream == nullptr, "CPU does not support explicit streams");
}

void streamSynchronize(zedinferStream_t stream) {
    ASSERT(stream == nullptr, "CPU does not support explicit streams");
}

void *mallocDevice(size_t size) {
    return std::malloc(size);
}

void freeDevice(void *ptr) {
    std::free(ptr);
}

void *mallocHost(size_t size) {
    return mallocDevice(size); // Host and device memory are unified on CPU
}

void freeHost(void *ptr) {
    freeDevice(ptr);
}

void memcpySync(void *dst, const void *src, size_t size, zedinferMemcpyKind_t kind) {
    ASSERT(kind == ZEDINFER_MEMCPY_H2H, "CPU only supports host-to-host memory copy");
    std::memcpy(dst, src, size);
}

void memcpyAsync(void *dst, const void *src, size_t size, zedinferMemcpyKind_t kind, zedinferStream_t stream) {
    ASSERT(stream == nullptr, "CPU does not support explicit streams");
    memcpySync(dst, src, size, kind); // Async falls back to sync on CPU
}

static const ZedinferRuntimeAPI RUNTIME_API = {
    &getDeviceCount,
    &setDevice,
    &deviceSynchronize,
    &createStream,
    &destroyStream,
    &streamSynchronize,
    &mallocDevice,
    &freeDevice,
    &mallocHost,
    &freeHost,
    &memcpySync,
    &memcpyAsync};

} // namespace runtime_api

const ZedinferRuntimeAPI *getRuntimeAPI() {
    return &runtime_api::RUNTIME_API;
}

} // namespace zedinfer::device::cpu