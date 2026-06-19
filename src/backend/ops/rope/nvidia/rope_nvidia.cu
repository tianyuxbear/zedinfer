#include "backend/core/context/context.hpp"
#include "backend/ops/rope/nvidia/rope_nvidia.cuh"
#include "utils/check.hpp"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <flashinfer/pos_enc.cuh>

#include <stdexcept>
#include <string>

namespace zedinfer::ops::nvidia {

namespace {

// FlashInfer's BatchQKApplyRotaryPosIds supports head_dim ∈ {64, 128, 256, 512}
// and dtype ∈ {fp16, bf16}. All Qwen2/Qwen3 supported models use head_dim=128
// in bf16, so the dispatch always lands in a covered case.
bool supports_flashinfer_rope(zedinferDataType_t type, size_t head_dim) {
    if (type != ZEDINFER_DTYPE_F16 && type != ZEDINFER_DTYPE_BF16) {
        return false;
    }
    switch (head_dim) {
        case 64:
        case 128:
        case 256:
        case 512:
            return true;
        default:
            return false;
    }
}

template <typename DType>
void rope_qk_via_flashinfer(std::byte* q_output, std::byte* k_output, const std::byte* q_input,
                            const std::byte* k_input, const std::byte* pos_ids, float theta, size_t seq_len,
                            size_t num_q_heads, size_t num_kv_heads, size_t head_dim) {
    auto stream = reinterpret_cast<cudaStream_t>(core::context().runtime().stream());
    auto* q_in = reinterpret_cast<DType*>(const_cast<std::byte*>(q_input));
    auto* k_in = reinterpret_cast<DType*>(const_cast<std::byte*>(k_input));
    auto* q_out = reinterpret_cast<DType*>(q_output);
    auto* k_out = reinterpret_cast<DType*>(k_output);
    auto* pos = reinterpret_cast<int64_t*>(const_cast<std::byte*>(pos_ids));
    auto status = flashinfer::BatchQKApplyRotaryPosIds<DType, int64_t>(
        q_in, k_in, q_out, k_out, pos, static_cast<uint32_t>(seq_len), static_cast<uint32_t>(num_q_heads),
        static_cast<uint32_t>(num_kv_heads), static_cast<uint32_t>(head_dim), static_cast<uint32_t>(head_dim),
        num_q_heads * head_dim, head_dim, num_kv_heads * head_dim, head_dim, num_q_heads * head_dim, head_dim,
        num_kv_heads * head_dim, head_dim,
        /*interleave=*/false, /*rope_scale=*/1.0f, /*rope_theta=*/theta, stream);
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string("FlashInfer BatchQKApplyRotaryPosIds failed: ")
                                 + cudaGetErrorString(status));
    }
}

} // namespace

// Single-tensor RoPE on NVIDIA path is intentionally unimplemented — no caller
// goes through this code (forward paths use rope_qk to fuse Q and K). If a
// future caller needs it, wire up FlashInfer with q=k pointing at the same
// buffer or add the single-tensor variant; throwing here makes the gap loud
// rather than silently producing the wrong shape via the qk path.
void rope(std::byte* /*output*/, const std::byte* /*input*/, const std::byte* /*pos_ids*/, float /*theta*/,
          zedinferDataType_t /*type*/, size_t /*seq_len*/, size_t /*num_heads*/, size_t /*head_dim*/) {
    throw std::runtime_error("ops::nvidia::rope (single-tensor) not implemented on NVIDIA. Use ops::rope_qk instead.");
}

void rope_qk(std::byte* q_output, std::byte* k_output, const std::byte* q_input, const std::byte* k_input,
             const std::byte* pos_ids, float theta, zedinferDataType_t type, size_t seq_len, size_t num_q_heads,
             size_t num_kv_heads, size_t head_dim) {
    if (!supports_flashinfer_rope(type, head_dim)) {
        throw std::runtime_error("ops::nvidia::rope_qk: FlashInfer RoPE requires bf16/f16 dtype and "
                                 "head_dim ∈ {64, 128, 256, 512} (got head_dim="
                                 + std::to_string(head_dim) + ")");
    }
    switch (type) {
        case ZEDINFER_DTYPE_F16:
            return rope_qk_via_flashinfer<half>(q_output, k_output, q_input, k_input, pos_ids, theta, seq_len,
                                                num_q_heads, num_kv_heads, head_dim);
        case ZEDINFER_DTYPE_BF16:
            return rope_qk_via_flashinfer<__nv_bfloat16>(q_output, k_output, q_input, k_input, pos_ids, theta, seq_len,
                                                         num_q_heads, num_kv_heads, head_dim);
        default:
            EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}

} // namespace zedinfer::ops::nvidia
