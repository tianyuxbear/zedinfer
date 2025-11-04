#include "backend/ops/embedding/cpu/embedding_cpu.hpp"
#include "utils/check.hpp"
#include "utils/types.hpp"

#include <cmath>
#include <cstdint>
#include <cstring>

template <typename T>
void embedding_(T *out, const int *index, const T *weight, size_t numel, size_t len) {
    const size_t nlen = numel / len;

    // Avoid OpenMP for memory-bound operations;
    // single-threaded memcpy is faster due to CPU optimizations (SIMD/prefetch) and avoids thread scheduling overhead.
    for (size_t i = 0; i < nlen; ++i) {
        const int64_t idx = index[i];
        std::memcpy(out + (i * len), weight + (idx * len), len * sizeof(T));
    }
}

namespace zedinfer::ops::cpu {
void embedding(std::byte *out, const std::byte *index, const std::byte *weight, zedinferDataType_t type, size_t numel, size_t len) {
    switch (type) {
    case ZEDINFER_DTYPE_F32:
        return embedding_(reinterpret_cast<float *>(out), reinterpret_cast<const int *>(index), reinterpret_cast<const float *>(weight), numel, len);
    case ZEDINFER_DTYPE_BF16:
        return embedding_(reinterpret_cast<zedinfer::bf16_t *>(out), reinterpret_cast<const int *>(index),
                          reinterpret_cast<const zedinfer::bf16_t *>(weight), numel, len);
    case ZEDINFER_DTYPE_F16:
        return embedding_(reinterpret_cast<zedinfer::fp16_t *>(out), reinterpret_cast<const int *>(index),
                          reinterpret_cast<const zedinfer::fp16_t *>(weight), numel, len);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace zedinfer::ops::cpu
