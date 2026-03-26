#ifndef __ZEDINFER_H__
#define __ZEDINFER_H__

#if defined(_WIN32)
#define __export __declspec(dllexport)
#elif defined(__GNUC__) && ((__GNUC__ >= 4) || (__GNUC__ == 3 && __GNUC_MINOR__ >= 3))
#define __export __attribute__((visibility("default")))
#else
#define __export
#endif

#ifdef __cplusplus
#define __ZEDINFER__C extern "C"
#else
#define __ZEDINFER__C
#endif

/**
 * @brief Stream handle for device-specific asynchronous execution.
 */
typedef void* zedinferStream_t;

/**
 * @brief Device types.
 */
typedef enum {
    ZEDINFER_DEVICE_CPU = 0,    ///< CPU
    ZEDINFER_DEVICE_NVIDIA = 1, ///< NVIDIA CUDA
    ZEDINFER_DEVICE_TYPE_COUNT  ///< Number of device types
} zedinferDeviceType_t;

/**
 * @brief Supported data types.
 */
typedef enum {
    ZEDINFER_DTYPE_BYTE = 0,  ///< 8-bit byte
    ZEDINFER_DTYPE_BOOL = 1,  ///< Boolean
    ZEDINFER_DTYPE_I8 = 2,    ///< int8
    ZEDINFER_DTYPE_I16 = 3,   ///< int16
    ZEDINFER_DTYPE_I32 = 4,   ///< int32
    ZEDINFER_DTYPE_I64 = 5,   ///< int64
    ZEDINFER_DTYPE_U8 = 6,    ///< uint8
    ZEDINFER_DTYPE_U16 = 7,   ///< uint16
    ZEDINFER_DTYPE_U32 = 8,   ///< uint32
    ZEDINFER_DTYPE_U64 = 9,   ///< uint64
    ZEDINFER_DTYPE_F16 = 10,  ///< float16
    ZEDINFER_DTYPE_F32 = 11,  ///< float32
    ZEDINFER_DTYPE_F64 = 12,  ///< float64
    ZEDINFER_DTYPE_BF16 = 13, ///< bfloat16
    ZEDINFER_DTYPE_COUNT      ///< Number of data types
} zedinferDataType_t;

/**
 * @brief Memory copy directions.
 */
typedef enum {
    ZEDINFER_MEMCPY_H2H = 0,  ///< Host to Host
    ZEDINFER_MEMCPY_H2D = 1,  ///< Host to Device
    ZEDINFER_MEMCPY_D2H = 2,  ///< Device to Host
    ZEDINFER_MEMCPY_D2D = 3,  ///< Device to Device
    ZEDINFER_MEMCPY_COUNT = 4 ///< Number of memory copy directions
} zedinferMemcpyKind_t;

#endif // __ZEDINFER_H__
