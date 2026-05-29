#include "backend/ops/layer_norm/cpu/layer_norm_bias_cpu.hpp"
#include "utils/check.hpp"
#include "utils/types.hpp"

#include <cmath>
#include <cstddef>
#include <type_traits>
#include <vector>

namespace zedinfer::ops::cpu {

namespace {

template <typename T> static inline float to_f32(T v);
template <> inline float to_f32<float>(float v) {
    return v;
}
template <> inline float to_f32<zedinfer::bf16_t>(zedinfer::bf16_t v) {
    return zedinfer::utils::cast<float>(v);
}
template <> inline float to_f32<zedinfer::fp16_t>(zedinfer::fp16_t v) {
    return zedinfer::utils::cast<float>(v);
}

template <typename T> static inline T from_f32(float v);
template <> inline float from_f32<float>(float v) {
    return v;
}
template <> inline zedinfer::bf16_t from_f32<zedinfer::bf16_t>(float v) {
    return zedinfer::utils::cast<zedinfer::bf16_t>(v);
}
template <> inline zedinfer::fp16_t from_f32<zedinfer::fp16_t>(float v) {
    return zedinfer::utils::cast<zedinfer::fp16_t>(v);
}

template <typename T>
void layer_norm_bias_(T* output, const T* input, const T* weight, const T* bias, float eps, size_t seq_len,
                      size_t hidden_size) {
    // Pre-convert weight + bias to FP32 once (shared across all rows).
    std::vector<float> w_f32(hidden_size);
    std::vector<float> b_f32(hidden_size);
    for (size_t j = 0; j < hidden_size; ++j) {
        w_f32[j] = to_f32<T>(weight[j]);
        b_f32[j] = to_f32<T>(bias[j]);
    }

    std::vector<float> row(hidden_size);
    for (size_t i = 0; i < seq_len; ++i) {
        const T* in_row = input + i * hidden_size;
        T* out_row = output + i * hidden_size;

        float sum = 0.0f;
        for (size_t j = 0; j < hidden_size; ++j) {
            row[j] = to_f32<T>(in_row[j]);
            sum += row[j];
        }
        const float mean = sum / static_cast<float>(hidden_size);

        float sq_sum = 0.0f;
        for (size_t j = 0; j < hidden_size; ++j) {
            const float d = row[j] - mean;
            sq_sum += d * d;
        }
        const float inv_std = 1.0f / std::sqrt(sq_sum / static_cast<float>(hidden_size) + eps);

        for (size_t j = 0; j < hidden_size; ++j) {
            const float normed = (row[j] - mean) * inv_std;
            out_row[j] = from_f32<T>(normed * w_f32[j] + b_f32[j]);
        }
    }
}

} // namespace

void layer_norm_bias(std::byte* output, const std::byte* input, const std::byte* weight, const std::byte* bias,
                     float eps, zedinferDataType_t type, size_t seq_len, size_t hidden_size) {
    switch (type) {
        case ZEDINFER_DTYPE_F32:
            return layer_norm_bias_(reinterpret_cast<float*>(output), reinterpret_cast<const float*>(input),
                                    reinterpret_cast<const float*>(weight), reinterpret_cast<const float*>(bias), eps,
                                    seq_len, hidden_size);
        case ZEDINFER_DTYPE_F16:
            return layer_norm_bias_(reinterpret_cast<zedinfer::fp16_t*>(output),
                                    reinterpret_cast<const zedinfer::fp16_t*>(input),
                                    reinterpret_cast<const zedinfer::fp16_t*>(weight),
                                    reinterpret_cast<const zedinfer::fp16_t*>(bias), eps, seq_len, hidden_size);
        case ZEDINFER_DTYPE_BF16:
            return layer_norm_bias_(reinterpret_cast<zedinfer::bf16_t*>(output),
                                    reinterpret_cast<const zedinfer::bf16_t*>(input),
                                    reinterpret_cast<const zedinfer::bf16_t*>(weight),
                                    reinterpret_cast<const zedinfer::bf16_t*>(bias), eps, seq_len, hidden_size);
        default:
            EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}

} // namespace zedinfer::ops::cpu
