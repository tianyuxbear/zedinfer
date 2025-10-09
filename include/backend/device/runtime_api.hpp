#pragma once

#include "neollm.h"
#include <stddef.h>

// Device management
typedef int (*get_device_count_api)();
typedef void (*set_device_api)(int);
typedef void (*device_synchronize_api)();

// Stream management
typedef NeollmStream_t (*create_stream_api)();
typedef void (*destroy_stream_api)(NeollmStream_t);
typedef void (*stream_synchronize_api)(NeollmStream_t);

// Device/host memory allocation
typedef void *(*malloc_device_api)(size_t);
typedef void (*free_device_api)(void *);
typedef void *(*malloc_host_api)(size_t);
typedef void (*free_host_api)(void *);

// Memory copy (sync/async)
typedef void (*memcpy_sync_api)(void *, const void *, size_t, NeollmMemcpyKind_t);
typedef void (*memcpy_async_api)(void *, const void *, size_t, NeollmMemcpyKind_t, NeollmStream_t);

// Runtime API table for a backend (e.g., CUDA, CPU)
typedef struct NeollmRuntimeAPI {
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
} NeollmRuntimeAPI;

namespace neollm::device {

const NeollmRuntimeAPI *getRuntimeAPI(NeollmDeviceType_t device_type);

namespace cpu {
const NeollmRuntimeAPI *getRuntimeAPI();
}

#ifdef ENABLE_NVIDIA_API
namespace nvidia {
const NeollmRuntimeAPI *getRuntimeAPI();
}
#endif

} // namespace neollm::device