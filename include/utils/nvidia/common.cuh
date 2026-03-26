#pragma once

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

// ============================================================================
// Macros & Constants
// ============================================================================

// Optimization: Branch prediction expects true
#define likely(x) __builtin_expect(!!(x), 1)

// Optimization: Branch prediction expects false
#define unlikely(x) __builtin_expect(!!(x), 0)

// Standard Warp Size
#define WARP_SIZE 32

// Hardware Limits (Architectural Maximums)
#define MAX_BLOCK_SIZE 1024
#define MAX_NUM_WARPS (MAX_BLOCK_SIZE / WARP_SIZE) // = 32

// Default/Configured Block Size (Used for kernel launches)
// You can change this based on occupancy tuning, but it must be <= MAX_BLOCK_SIZE
#define BLOCK_SIZE 256
#define NUM_WARPS (BLOCK_SIZE / WARP_SIZE)

// Function modifiers
#define HOST_DEVICE __forceinline__ __host__ __device__
#define DEVICE_INLINE __forceinline__ __device__

// Error handling wrapper
#define CUDA_CHECK(call)                                                                                               \
    do {                                                                                                               \
        cudaError_t error = call;                                                                                      \
        if (error != cudaSuccess) {                                                                                    \
            fprintf(stderr, "CUDA Error: %s:%d, code: %d, reason: %s\n", __FILE__, __LINE__, error,                    \
                    cudaGetErrorString(error));                                                                        \
            exit(1);                                                                                                   \
        }                                                                                                              \
    } while (0)

// Index calculation helper
#define OFFSET(row, col, stride) ((row) * (stride) + (col))

// ============================================================================
// Basic Math Utils
// ============================================================================

// Ceiling division
HOST_DEVICE constexpr size_t div_ceil(size_t a, size_t b) {
    return (a + b - 1) / b;
}

/**
 * SiLU: x / (1 + exp(-x))
 */
DEVICE_INLINE float silu(float x) {
    return x / (1.0f + __expf(-x));
}
