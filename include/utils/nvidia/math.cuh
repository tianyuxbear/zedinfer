#pragma once

#include "memory.cuh"

// ============================================================================
// Dot Product Primitives
// ============================================================================

/**
 * @brief Computes dot product of two 128-bit packed vectors.
 * * Optimization Note:
 * Uses SIMD vector intrinsics (half2/bfloat162) with 4 iterations.
 * This effectively processes 2 elements per iteration.
 */
template <typename T>
DEVICE_INLINE float dot_packed_128b(const T *a, const T *b);

// FP32 Specialization: 4 elements
// Already native vector size, straightforward.
template <>
DEVICE_INLINE float dot_packed_128b<float>(const float *a, const float *b) {
    float4 va = load_128b(a);
    float4 vb = load_128b(b);
    return va.x * vb.x + va.y * vb.y + va.z * vb.z + va.w * vb.w;
}

// FP16 Specialization: 8 elements -> 4 iterations of half2
template <>
DEVICE_INLINE float dot_packed_128b<half>(const half *a, const half *b) {
    float4 va = load_128b(a);
    float4 vb = load_128b(b);

    // Reinterpret float4 (128-bit) as array of 4 x half2 (32-bit each)
    const __half2 *va_2 = reinterpret_cast<const __half2 *>(&va);
    const __half2 *vb_2 = reinterpret_cast<const __half2 *>(&vb);

    float sum = 0.0f;

#pragma unroll
    for (int i = 0; i < 4; ++i) {
        // Convert half2 -> float2 for precision accumulation
        float2 fa = __half22float2(va_2[i]);
        float2 fb = __half22float2(vb_2[i]);
        sum += fa.x * fb.x + fa.y * fb.y;
    }
    return sum;
}

// BF16 Specialization: 8 elements -> 4 iterations of cuda_bfloat162
template <>
DEVICE_INLINE float dot_packed_128b<cuda_bfloat16>(const cuda_bfloat16 *a, const cuda_bfloat16 *b) {
    float4 va = load_128b(a);
    float4 vb = load_128b(b);

    // Reinterpret float4 (128-bit) as array of 4 x bfloat162 (32-bit each)
    const cuda_bfloat162 *va_2 = reinterpret_cast<const cuda_bfloat162 *>(&va);
    const cuda_bfloat162 *vb_2 = reinterpret_cast<const cuda_bfloat162 *>(&vb);

    float sum = 0.0f;

#pragma unroll
    for (int i = 0; i < 4; ++i) {
#if __CUDA_ARCH__ >= 800
        // Use native hardware intrinsic if on Ampere+
        float2 fa = __bfloat1622float2(va_2[i]);
        float2 fb = __bfloat1622float2(vb_2[i]);
        sum += fa.x * fb.x + fa.y * fb.y;
#else
        // Fallback for older architectures or IDE warnings:
        // Manually unpack the opaque bfloat162 storage
        float2 fa, fb;
        fa.x = __bfloat162float(va_2[i].x);
        fa.y = __bfloat162float(va_2[i].y);
        fb.x = __bfloat162float(vb_2[i].x);
        fb.y = __bfloat162float(vb_2[i].y);
        sum += fa.x * fb.x + fa.y * fb.y;
#endif
    }
    return sum;
}