#include "backend/ops/embedding/cpu/embedding_cpu.hpp"
#include "utils/check.hpp"
#include "utils/types.hpp"

#include <cmath>
#include <cstdint>
#include <cstring>
namespace zedinfer::ops::cpu {

template <typename T>
void embedding_(T *out, const int *index, const T *weight, size_t numel, size_t len) {
    const size_t nlen = numel / len;

    // Threshold to avoid threading overhead on small workloads
    const bool use_omp = nlen > 64;

// Parallelize to hide memory latency from random access (gather pattern).
// 'static' schedule is optimal due to uniform copy sizes.
#pragma omp parallel for schedule(static) if (use_omp)
    for (size_t i = 0; i < nlen; ++i) {
        const int64_t idx = index[i];
        std::memcpy(out + (i * len), weight + (idx * len), len * sizeof(T));
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
