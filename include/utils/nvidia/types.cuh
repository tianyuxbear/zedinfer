#pragma once

#include "common.cuh"
#include <cuda_bf16.h>
#include <cuda_fp16.h>

// ============================================================================
// Type Aliases
// ============================================================================

using cuda_bfloat16 = nv_bfloat16;
using cuda_bfloat162 = nv_bfloat162;

// ============================================================================
// Type Conversion Utilities
// ============================================================================

// Convert to float
template <typename T> DEVICE_INLINE float to_float(T x);

template <> DEVICE_INLINE float to_float(float x) {
    return x;
}
template <> DEVICE_INLINE float to_float(__half x) {
    return __half2float(x);
}
template <> DEVICE_INLINE float to_float(cuda_bfloat16 x) {
    return __bfloat162float(x);
}

// Convert from float
template <typename T> DEVICE_INLINE T from_float(float x);

template <> DEVICE_INLINE float from_float(float x) {
    return x;
}
template <> DEVICE_INLINE __half from_float(float x) {
    return __float2half(x);
}
template <> DEVICE_INLINE cuda_bfloat16 from_float(float x) {
    return __float2bfloat16(x);
}