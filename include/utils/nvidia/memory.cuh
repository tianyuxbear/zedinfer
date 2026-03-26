#pragma once

#include "types.cuh"

// ============================================================================
// Packed Type Traits
// ============================================================================

template <typename T> struct PackedTraits;

// FP32: 1 float4 = 4 floats
template <> struct PackedTraits<float> {
    using Type = float4;
    static constexpr int size = 4;
};

// FP16: 1 float4 = 8 halfs
template <> struct PackedTraits<half> {
    using Type = float4;
    static constexpr int size = 8;
};

// BF16: 1 float4 = 8 bf16s
template <> struct PackedTraits<cuda_bfloat16> {
    using Type = float4;
    static constexpr int size = 8;
};

// ============================================================================
// Packed Data Access (128-bit)
// ============================================================================

/**
 * @brief Load 128-bit (16 bytes) from global/shared memory.
 * Compiles to: ld.global.v4.f32 / ld.shared.v4.f32
 * @note ptr must be 16-byte aligned.
 */
template <typename T> DEVICE_INLINE float4 load_128b(const T* ptr) {
    return *reinterpret_cast<const float4*>(ptr);
}

/**
 * @brief Store 128-bit (16 bytes) to global/shared memory.
 * Compiles to: st.global.v4.f32 / st.shared.v4.f32
 * @note ptr must be 16-byte aligned.
 */
template <typename T> DEVICE_INLINE void store_128b(T* ptr, float4 value) {
    *reinterpret_cast<float4*>(ptr) = value;
}