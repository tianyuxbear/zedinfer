#ifndef __NEOLLM_H__
#define __NEOLLM_H__

#if defined(_WIN32)
#define __export __declspec(dllexport)
#elif defined(__GNUC__) && ((__GNUC__ >= 4) || (__GNUC__ == 3 && __GNUC_MINOR__ >= 3))
#define __export __attribute__((visibility("default")))
#else
#define __export
#endif

#ifdef __cplusplus
#define __NEOLLM__C extern "C"
#else
#define __NEOLLM__C
#endif

// Device Types
typedef enum {
    NEOLLM_DEVICE_CPU = 0,
    NEOLLM_DEVICE_NVIDIA = 1,
    NEOLLM_DEVICE_TYPE_COUNT
} neollmDeviceType_t;

// Data Types
typedef enum {
    NEOLLM_DTYPE_INVALID = 0,
    NEOLLM_DTYPE_BYTE = 1,
    NEOLLM_DTYPE_BOOL = 2,
    NEOLLM_DTYPE_I8 = 3,
    NEOLLM_DTYPE_I16 = 4,
    NEOLLM_DTYPE_I32 = 5,
    NEOLLM_DTYPE_I64 = 6,
    NEOLLM_DTYPE_U8 = 7,
    NEOLLM_DTYPE_U16 = 8,
    NEOLLM_DTYPE_U32 = 9,
    NEOLLM_DTYPE_U64 = 10,
    NEOLLM_DTYPE_F8 = 11,
    NEOLLM_DTYPE_F16 = 12,
    NEOLLM_DTYPE_BF16 = 13,
    NEOLLM_DTYPE_F32 = 14,
    NEOLLM_DTYPE_F64 = 15,
} neollmDataType_t;

// Runtime Types: Stream
typedef void *neollmStream_t;

// Memory Copy Directions
typedef enum {
    NEOLLM_MEMCPY_H2H = 0,
    NEOLLM_MEMCPY_H2D = 1,
    NEOLLM_MEMCPY_D2H = 2,
    NEOLLM_MEMCPY_D2D = 3,
} neollmMemcpyKind_t;

#endif // __NEOLLM_H__
