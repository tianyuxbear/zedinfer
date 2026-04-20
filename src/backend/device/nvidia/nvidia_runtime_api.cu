#include "backend/device/runtime_api.hpp"
#include "utils/check.hpp"
#include "zedinfer.h"

#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>

namespace zedinfer::device::nvidia {

namespace runtime_api {

int getDeviceCount() {
    int count = 0;
    cudaGetDeviceCount(&count);
    return count;
}

void setDevice(int device_id) {
    cudaSetDevice(device_id);
}

void deviceSynchronize() {
    cudaDeviceSynchronize();
}

zedinferStream_t createStream() {
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    return reinterpret_cast<zedinferStream_t>(stream);
}

void destroyStream(zedinferStream_t stream) {
    if (stream != nullptr) {
        cudaStreamDestroy(reinterpret_cast<cudaStream_t>(stream));
    }
}

void streamSynchronize(zedinferStream_t stream) {
    if (stream != nullptr) {
        cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream));
    }
}

zedinferEvent_t createEvent() {
    cudaEvent_t event;
    // cudaEventDisableTiming: we only use events for stream-to-stream ordering,
    // not for elapsed-time measurement. Skipping timing cuts per-event overhead.
    cudaEventCreateWithFlags(&event, cudaEventDisableTiming);
    return reinterpret_cast<zedinferEvent_t>(event);
}

void destroyEvent(zedinferEvent_t event) {
    if (event != nullptr) {
        cudaEventDestroy(reinterpret_cast<cudaEvent_t>(event));
    }
}

void recordEvent(zedinferEvent_t event, zedinferStream_t stream) {
    cudaEventRecord(reinterpret_cast<cudaEvent_t>(event), reinterpret_cast<cudaStream_t>(stream));
}

void streamWaitEvent(zedinferStream_t stream, zedinferEvent_t event) {
    cudaStreamWaitEvent(reinterpret_cast<cudaStream_t>(stream), reinterpret_cast<cudaEvent_t>(event), 0);
}

void* mallocDevice(size_t size) {
    void* ptr = nullptr;
    cudaMalloc(&ptr, size);
    return ptr;
}

void freeDevice(void* ptr) {
    if (ptr != nullptr) {
        cudaFree(ptr);
    }
}

void* mallocHost(size_t size) {
    void* ptr = nullptr;
    cudaMallocHost(&ptr, size); // Pinned memory for faster transfers
    return ptr;
}

void freeHost(void* ptr) {
    if (ptr != nullptr) {
        cudaFreeHost(ptr);
    }
}

void memcpySync(void* dst, const void* src, size_t size, zedinferMemcpyKind_t kind) {
    cudaMemcpyKind cuda_kind;
    switch (kind) {
        case ZEDINFER_MEMCPY_H2D:
            cuda_kind = cudaMemcpyHostToDevice;
            break;
        case ZEDINFER_MEMCPY_D2H:
            cuda_kind = cudaMemcpyDeviceToHost;
            break;
        case ZEDINFER_MEMCPY_D2D:
            cuda_kind = cudaMemcpyDeviceToDevice;
            break;
        case ZEDINFER_MEMCPY_H2H:
            cuda_kind = cudaMemcpyHostToHost;
            break;
        default:
            ASSERT(false, "Unknown memory copy kind");
            return;
    }
    cudaMemcpy(dst, src, size, cuda_kind);
}

void memcpyAsync(void* dst, const void* src, size_t size, zedinferMemcpyKind_t kind, zedinferStream_t stream) {
    cudaMemcpyKind cuda_kind;
    switch (kind) {
        case ZEDINFER_MEMCPY_H2D:
            cuda_kind = cudaMemcpyHostToDevice;
            break;
        case ZEDINFER_MEMCPY_D2H:
            cuda_kind = cudaMemcpyDeviceToHost;
            break;
        case ZEDINFER_MEMCPY_D2D:
            cuda_kind = cudaMemcpyDeviceToDevice;
            break;
        case ZEDINFER_MEMCPY_H2H:
            cuda_kind = cudaMemcpyHostToHost;
            break;
        default:
            ASSERT(false, "Unknown memory copy kind");
            return;
    }
    cudaMemcpyAsync(dst, src, size, cuda_kind, reinterpret_cast<cudaStream_t>(stream));
}

void registerPinned(void* ptr, size_t size) {
    // cudaHostRegisterDefault: pageable→pinned. Subsequent cudaMemcpyAsync on this region
    // uses DMA for full PCIe bandwidth (~20+ GB/s vs ~6 GB/s pageable).
    cudaHostRegister(ptr, size, cudaHostRegisterDefault);
}

void unregisterPinned(void* ptr) {
    cudaHostUnregister(ptr);
}

void getMemoryInfo(size_t* free, size_t* total) {
    cudaMemGetInfo(free, total);
}

static const ZedinferRuntimeAPI RUNTIME_API = {&getDeviceCount,
                                               &setDevice,
                                               &deviceSynchronize,
                                               &createStream,
                                               &destroyStream,
                                               &streamSynchronize,
                                               &createEvent,
                                               &destroyEvent,
                                               &recordEvent,
                                               &streamWaitEvent,
                                               &mallocDevice,
                                               &freeDevice,
                                               &mallocHost,
                                               &freeHost,
                                               &memcpySync,
                                               &memcpyAsync,
                                               &registerPinned,
                                               &unregisterPinned,
                                               &getMemoryInfo};

} // namespace runtime_api

const ZedinferRuntimeAPI* getRuntimeAPI() {
    return &runtime_api::RUNTIME_API;
}

} // namespace zedinfer::device::nvidia