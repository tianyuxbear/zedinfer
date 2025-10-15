#include "backend/ops/add/cpu/add_cpu.hpp"
#include "backend/ops/add/cpu/add_cpu_opt.hpp"
#include "utils/check.hpp"

namespace neollm::ops::cpu {
void add(std::byte *c, const std::byte *a, const std::byte *b, NeollmDataType_t type, size_t numel) {
    switch (type) {
    case NEOLLM_DTYPE_F32:
        return add_f32(reinterpret_cast<float *>(c), reinterpret_cast<const float *>(a), reinterpret_cast<const float *>(b), numel);
    case NEOLLM_DTYPE_BF16:
        return add_bf16(reinterpret_cast<neollm::bf16_t *>(c), reinterpret_cast<const neollm::bf16_t *>(a),
                        reinterpret_cast<const neollm::bf16_t *>(b), numel);
    case NEOLLM_DTYPE_F16:
        return add_f16(reinterpret_cast<neollm::fp16_t *>(c), reinterpret_cast<const neollm::fp16_t *>(a),
                       reinterpret_cast<const neollm::fp16_t *>(b), numel);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace neollm::ops::cpu
