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

/**
 * @brief Stream handle for device-specific asynchronous execution.
 */
typedef void *NeollmStream_t;

/**
 * @brief Device types.
 */
typedef enum {
    NEOLLM_DEVICE_CPU = 0,    ///< CPU
    NEOLLM_DEVICE_NVIDIA = 1, ///< NVIDIA CUDA
    NEOLLM_DEVICE_TYPE_COUNT  ///< Number of device types
} NeollmDeviceType_t;

/**
 * @brief Supported data types.
 */
typedef enum {
    NEOLLM_DTYPE_BYTE = 0,  ///< 8-bit byte
    NEOLLM_DTYPE_BOOL = 1,  ///< Boolean
    NEOLLM_DTYPE_I8 = 2,    ///< int8
    NEOLLM_DTYPE_I16 = 3,   ///< int16
    NEOLLM_DTYPE_I32 = 4,   ///< int32
    NEOLLM_DTYPE_I64 = 5,   ///< int64
    NEOLLM_DTYPE_U8 = 6,    ///< uint8
    NEOLLM_DTYPE_U16 = 7,   ///< uint16
    NEOLLM_DTYPE_U32 = 8,   ///< uint32
    NEOLLM_DTYPE_U64 = 9,   ///< uint64
    NEOLLM_DTYPE_F16 = 10,  ///< float16
    NEOLLM_DTYPE_F32 = 11,  ///< float32
    NEOLLM_DTYPE_F64 = 12,  ///< float64
    NEOLLM_DTYPE_BF16 = 13, ///< bfloat16
    NEOLLM_DTYPE_COUNT      ///< Number of data types
} NeollmDataType_t;

/**
 * @brief Memory copy directions.
 */
typedef enum {
    NEOLLM_MEMCPY_H2H = 0,    ///< Host to Host
    NEOLLM_MEMCPY_H2D = 1,    ///< Host to Device
    NEOLLM_MEMCPY_D2H = 2,    ///< Device to Host
    NEOLLM_MEMCPY_D2D = 3,    ///< Device to Device
    NEOLLM_MEMCPY_DEFAULT = 4 ///< Auto-detect
} NeollmMemcpyKind_t;

#endif // __NEOLLM_H__
