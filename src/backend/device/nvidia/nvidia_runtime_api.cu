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

void getMemoryInfo(size_t* free, size_t* total) {
    cudaMemGetInfo(free, total);
}

static const ZedinferRuntimeAPI RUNTIME_API = {
    &getDeviceCount, &setDevice,  &deviceSynchronize, &createStream, &destroyStream, &streamSynchronize, &mallocDevice,
    &freeDevice,     &mallocHost, &freeHost,          &memcpySync,   &memcpyAsync,   &getMemoryInfo};

} // namespace runtime_api

const ZedinferRuntimeAPI* getRuntimeAPI() {
    return &runtime_api::RUNTIME_API;
}

} // namespace zedinfer::device::nvidia