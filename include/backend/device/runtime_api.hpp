#pragma once

#include "zedinfer.h"

#include <stddef.h>

// Device management
typedef int (*get_device_count_api)();
typedef void (*set_device_api)(int);
typedef void (*device_synchronize_api)();

// Stream management
typedef zedinferStream_t (*create_stream_api)();
typedef void (*destroy_stream_api)(zedinferStream_t);
typedef void (*stream_synchronize_api)(zedinferStream_t);

// Device/host memory allocation
typedef void* (*malloc_device_api)(size_t);
typedef void (*free_device_api)(void*);
typedef void* (*malloc_host_api)(size_t);
typedef void (*free_host_api)(void*);

// Memory copy (sync/async)
typedef void (*memcpy_sync_api)(void*, const void*, size_t, zedinferMemcpyKind_t);
typedef void (*memcpy_async_api)(void*, const void*, size_t, zedinferMemcpyKind_t, zedinferStream_t);

// Pin/unpin existing host memory region for fast PCIe transfers.
// Used by Phase 2 expert offloading to pin CPU-resident expert weights.
// No-op on CPU runtime; cudaHostRegister/cudaHostUnregister on NVIDIA.
typedef void (*register_pinned_api)(void*, size_t);
typedef void (*unregister_pinned_api)(void*);

// Memory info query
typedef void (*get_memory_info_api)(size_t* free, size_t* total);

// Runtime API table for a backend (e.g., CUDA, CPU)
typedef struct ZedinferRuntimeAPI {
    // Device
    get_device_count_api get_device_count;
    set_device_api set_device;
    device_synchronize_api device_synchronize;

    // Stream
    create_stream_api create_stream;
    destroy_stream_api destroy_stream;
    stream_synchronize_api stream_synchronize;

    // Memory
    malloc_device_api malloc_device;
    free_device_api free_device;
    malloc_host_api malloc_host;
    free_host_api free_host;

    // Memory copy
    memcpy_sync_api memcpy_sync;
    memcpy_async_api memcpy_async;

    // Pinned memory (may be nullptr on backends that don't support pinning, e.g. CPU)
    register_pinned_api register_pinned;
    unregister_pinned_api unregister_pinned;

    // Memory info
    get_memory_info_api get_memory_info;
} ZedinferRuntimeAPI;

namespace zedinfer::device {

const ZedinferRuntimeAPI* getRuntimeAPI(zedinferDeviceType_t device_type);

namespace cpu {
const ZedinferRuntimeAPI* getRuntimeAPI();
}

#ifdef ENABLE_NVIDIA_API
namespace nvidia {
const ZedinferRuntimeAPI* getRuntimeAPI();
}
#endif

} // namespace zedinfer::device