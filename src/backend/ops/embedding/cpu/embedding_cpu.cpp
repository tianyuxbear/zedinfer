#include "backend/ops/embedding/cpu/embedding_cpu.hpp"
#include "utils/check.hpp"
#include "utils/types.hpp"

#include <cmath>
#include <cstdint>
#include <cstring>
namespace zedinfer::ops::cpu {

template <typename T>
void embedding_(T *output, const int *indices, const T *weight, size_t numel, size_t hidden_size) {
    const size_t seqlen = numel / hidden_size;

    // Avoid OpenMP for memory-bound operations;
    // single-threaded memcpy is faster due to CPU optimizations (SIMD/prefetch) and avoids thread scheduling overhead.
    for (size_t i = 0; i < seqlen; ++i) {
        const int64_t idx = indices[i];
        std::memcpy(output + (i * hidden_size), weight + (idx * hidden_size), hidden_size * sizeof(T));
    }
}

void embedding(std::byte *output, const std::byte *indices, const std::byte *weight, zedinferDataType_t type, size_t numel, size_t hidden_size) {
    switch (type) {
    case ZEDINFER_DTYPE_F32:
        return embedding_(reinterpret_cast<float *>(output),
                          reinterpret_cast<const int *>(indices),
                          reinterpret_cast<const float *>(weight),
                          numel, hidden_size);
    case ZEDINFER_DTYPE_F16:
        return embedding_(reinterpret_cast<zedinfer::fp16_t *>(output),
                          reinterpret_cast<const int *>(indices),
                          reinterpret_cast<const zedinfer::fp16_t *>(weight),
                          numel, hidden_size);
    case ZEDINFER_DTYPE_BF16:
        return embedding_(reinterpret_cast<zedinfer::bf16_t *>(output),
                          reinterpret_cast<const int *>(indices),
                          reinterpret_cast<const zedinfer::bf16_t *>(weight),
                          numel, hidden_size);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace zedinfer::ops::cpu
