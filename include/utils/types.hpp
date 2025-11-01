#pragma once

#include "neollm.h"

#include <cstddef>
#include <cstdint>
#include <immintrin.h>
#include <stdexcept>

namespace neollm {
struct CustomFloat16 {
    uint16_t _v;
};
typedef struct CustomFloat16 fp16_t;

struct CustomBFloat16 {
    uint16_t _v;
};
typedef struct CustomBFloat16 bf16_t;

namespace utils {
inline size_t dsize(NeollmDataType_t dtype) {
    switch (dtype) {
    case NEOLLM_DTYPE_BYTE:
        return sizeof(unsigned char);
    case NEOLLM_DTYPE_BOOL:
        return sizeof(bool);
    case NEOLLM_DTYPE_I8:
        return sizeof(int8_t);
    case NEOLLM_DTYPE_I16:
        return sizeof(int16_t);
    case NEOLLM_DTYPE_I32:
        return sizeof(int32_t);
    case NEOLLM_DTYPE_I64:
        return sizeof(int64_t);
    case NEOLLM_DTYPE_U8:
        return sizeof(uint8_t);
    case NEOLLM_DTYPE_U16:
        return sizeof(uint16_t);
    case NEOLLM_DTYPE_U32:
        return sizeof(uint32_t);
    case NEOLLM_DTYPE_U64:
        return sizeof(uint64_t);
    case NEOLLM_DTYPE_F16:
        return 2;
    case NEOLLM_DTYPE_F32:
        return sizeof(float);
    case NEOLLM_DTYPE_F64:
        return sizeof(double);
    case NEOLLM_DTYPE_BF16:
        return 2;
    default:
        throw std::invalid_argument("Unsupported or invalid data type.");
    }
}

inline const char *dtype_to_str(NeollmDataType_t dtype) {
    switch (dtype) {
    case NEOLLM_DTYPE_BYTE:
        return "byte";
    case NEOLLM_DTYPE_BOOL:
        return "bool";
    case NEOLLM_DTYPE_I8:
        return "int8";
    case NEOLLM_DTYPE_I16:
        return "int16";
    case NEOLLM_DTYPE_I32:
        return "int32";
    case NEOLLM_DTYPE_I64:
        return "int64";
    case NEOLLM_DTYPE_U8:
        return "uint8";
    case NEOLLM_DTYPE_U16:
        return "uint16";
    case NEOLLM_DTYPE_U32:
        return "uint32";
    case NEOLLM_DTYPE_U64:
        return "uint64";
    case NEOLLM_DTYPE_F16:
        return "float16";
    case NEOLLM_DTYPE_F32:
        return "float32";
    case NEOLLM_DTYPE_F64:
        return "float64";
    case NEOLLM_DTYPE_BF16:
        return "bfloat16";
    default:
        throw std::invalid_argument("Unsupported or invalid data type.");
    }
}

inline NeollmDataType_t str_to_dtype(std::string_view str) {
    if (str == "byte") {
        return NEOLLM_DTYPE_BYTE;
    }
    if (str == "bool") {
        return NEOLLM_DTYPE_BOOL;
    }
    if (str == "int8") {
        return NEOLLM_DTYPE_I8;
    }
    if (str == "int16") {
        return NEOLLM_DTYPE_I16;
    }
    if (str == "int32") {
        return NEOLLM_DTYPE_I32;
    }
    if (str == "int64") {
        return NEOLLM_DTYPE_I64;
    }
    if (str == "uint8") {
        return NEOLLM_DTYPE_U8;
    }
    if (str == "uint16") {
        return NEOLLM_DTYPE_U16;
    }
    if (str == "uint32") {
        return NEOLLM_DTYPE_U32;
    }
    if (str == "uint64") {
        return NEOLLM_DTYPE_U64;
    }
    if (str == "float16") {
        return NEOLLM_DTYPE_F16;
    }
    if (str == "float32") {
        return NEOLLM_DTYPE_F32;
    }
    if (str == "float64") {
        return NEOLLM_DTYPE_F64;
    }
    if (str == "bfloat16") {
        return NEOLLM_DTYPE_BF16;
    }

    throw std::invalid_argument("Unsupported or invalid data type string: " + std::string(str));
}

float _f16_to_f32(fp16_t val);
fp16_t _f32_to_f16(float val);

float _bf16_to_f32(bf16_t val);
bf16_t _f32_to_bf16(float val);

float fp16_to_fp32_f16c(fp16_t x);
fp16_t fp32_to_fp16_f16c(float x);

void fp16_to_fp32_batch_f16c(float *dst, const fp16_t *src, size_t count);
void fp32_to_fp16_batch_f16c(fp16_t *dst, const float *src, size_t count);

void bf16_to_fp32_batch(float *dst, const bf16_t *src, size_t count);
void fp32_to_bf16_batch(bf16_t *dst, const float *src, size_t count);

template <typename TypeTo, typename TypeFrom>
TypeTo cast(TypeFrom val) {
    if constexpr (std::is_same<TypeTo, TypeFrom>::value) {
        return val;
    } else if constexpr (std::is_same<TypeTo, fp16_t>::value && std::is_same<TypeFrom, float>::value) {
        return fp32_to_fp16_f16c(val);
    } else if constexpr (std::is_same<TypeTo, fp16_t>::value && !std::is_same<TypeFrom, float>::value) {
        return fp32_to_fp16_f16c(static_cast<float>(val));
    } else if constexpr (std::is_same<TypeFrom, fp16_t>::value && std::is_same<TypeTo, float>::value) {
        return fp16_to_fp32_f16c(val);
    } else if constexpr (std::is_same<TypeFrom, fp16_t>::value && !std::is_same<TypeTo, float>::value) {
        return static_cast<TypeTo>(fp16_to_fp32_f16c(val));
    } else if constexpr (std::is_same<TypeTo, bf16_t>::value && std::is_same<TypeFrom, float>::value) {
        return _f32_to_bf16(val);
    } else if constexpr (std::is_same<TypeTo, bf16_t>::value && !std::is_same<TypeFrom, float>::value) {
        return _f32_to_bf16(static_cast<float>(val));
    } else if constexpr (std::is_same<TypeFrom, bf16_t>::value && std::is_same<TypeTo, float>::value) {
        return _bf16_to_f32(val);
    } else if constexpr (std::is_same<TypeFrom, bf16_t>::value && !std::is_same<TypeTo, float>::value) {
        return static_cast<TypeTo>(_bf16_to_f32(val));
    } else {
        return static_cast<TypeTo>(val);
    }
}

} // namespace utils
} // namespace neollm
