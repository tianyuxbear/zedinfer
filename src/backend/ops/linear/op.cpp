#include "backend/core/context/context.hpp"
#include "backend/core/core.hpp"
#include "backend/ops/linear/cpu/linear_cpu.hpp"
#include "backend/ops/linear/cpu/permute.hpp"
#include "backend/ops/linear/nvidia/linear_nvidia.cuh"
#include "backend/ops/linear/nvidia/permute.cuh"
#include "backend/ops/ops.hpp"
#include "utils/check.hpp"

#include <cstddef>

namespace zedinfer::ops {

static tensor_t permute_quantized_linear_input(tensor_t input, tensor_t g_idx) {
    ASSERT(input && g_idx, "permute_quantized_linear_input requires input and g_idx.");

    if (input->deviceType() == ZEDINFER_DEVICE_CPU) {
        return cpu::permute_cols(input, g_idx);
    }

    zedinfer::core::context().setDevice(input->deviceType(), input->deviceId());

    switch (input->deviceType()) {
    case ZEDINFER_DEVICE_CPU:
        return cpu::permute_cols(input, g_idx);
#ifdef ENABLE_NVIDIA_API
    case ZEDINFER_DEVICE_NVIDIA:
        return nvidia::permute_cols(input, g_idx);
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}

void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) {
    if (bias) {
        CHECK_SAME_DEVICE(out, in, weight, bias);
        // Only support contiguous inputs with same shape for now.
        CHECK_SAME_DTYPE(out->dtype(), in->dtype(), weight->dtype(), bias->dtype());
        ASSERT(out->isContiguous() && in->isContiguous() && weight->isContiguous() && bias->isContiguous(),
               "Linear: all tensors must be contiguous.");
    } else {
        CHECK_SAME_DEVICE(out, in, weight);
        // Only support contiguous inputs with same shape for now.
        CHECK_SAME_DTYPE(out->dtype(), in->dtype(), weight->dtype());
        ASSERT(out->isContiguous() && in->isContiguous() && weight->isContiguous(),
               "Linear: all tensors must be contiguous.");
    }

    std::byte* bias_data = nullptr;
    if (bias) {
        bias_data = bias->data();
    }

    // always support cpu calculation
    if (out->deviceType() == ZEDINFER_DEVICE_CPU) {
        return cpu::linear(out->data(), in->data(), weight->data(), bias_data, out->dtype(), out->dim(0), out->dim(1),
                           in->dim(1));
    }

    zedinfer::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
        case ZEDINFER_DEVICE_CPU:
            return cpu::linear(out->data(), in->data(), weight->data(), bias_data, out->dtype(), out->dim(0),
                               out->dim(1), in->dim(1));
#ifdef ENABLE_NVIDIA_API
        case ZEDINFER_DEVICE_NVIDIA:
            return nvidia::linear(out->data(), in->data(), weight->data(), bias_data, out->dtype(), out->dim(0),
                                  out->dim(1), in->dim(1));
#endif
        default:
            EXCEPTION_UNSUPPORTED_DEVICE;
    }
}

void linear_quantized(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias, tensor_t scale, tensor_t g_idx, int num_bits, int group_size) {
    if (bias) {
        if (g_idx) {
            CHECK_SAME_DEVICE(out, in, weight, scale, bias, g_idx);
        } else {
            CHECK_SAME_DEVICE(out, in, weight, scale, bias);
        }
    } else {
        if (g_idx) {
            CHECK_SAME_DEVICE(out, in, weight, scale, g_idx);
        } else {
            CHECK_SAME_DEVICE(out, in, weight, scale);
        }
    }

    ASSERT(weight->dtype() == ZEDINFER_DTYPE_I32 ||
           weight->dtype() == ZEDINFER_DTYPE_I8 ||
           weight->dtype() == ZEDINFER_DTYPE_U8,
           "Quantized linear weight must be INT32, INT8, or UINT8.");
    CHECK_SAME_DTYPE(out->dtype(), in->dtype(), scale->dtype());
    if (bias) {
        CHECK_SAME_DTYPE(out->dtype(), bias->dtype());
    }
    if (g_idx) {
        ASSERT(g_idx->dtype() == ZEDINFER_DTYPE_I32, "Linear quantized g_idx must be INT32.");
    }

    ASSERT(out->isContiguous() && in->isContiguous() &&
           weight->isContiguous() && scale->isContiguous(),
           "Linear quantized: tensors must be contiguous.");
    if (bias) {
        ASSERT(bias->isContiguous(), "Linear quantized bias must be contiguous.");
    }
    if (g_idx) {
        ASSERT(g_idx->isContiguous(), "Linear quantized g_idx must be contiguous.");
    }

    ASSERT(num_bits == 4 || num_bits == 8,
           "Linear quantized supports only INT4 and INT8 weights.");
    ASSERT(group_size == -1 || group_size > 0,
           "Linear quantized group_size must be -1 or positive.");

    tensor_t current_input = in;
    if (g_idx) {
        current_input = permute_quantized_linear_input(in, g_idx);
        g_idx = nullptr;
    }

    std::byte *bias_data = bias ? bias->data() : nullptr;
    std::byte *g_idx_data = nullptr;

    if (out->deviceType() == ZEDINFER_DEVICE_CPU) {
        return cpu::linear_quantized(
            out->data(), current_input->data(), weight->data(), bias_data,
            scale->data(), g_idx_data, out->dtype(), num_bits, group_size,
            out->dim(0), out->dim(1), current_input->dim(1));
    }

    zedinfer::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case ZEDINFER_DEVICE_CPU:
        return cpu::linear_quantized(
            out->data(), current_input->data(), weight->data(), bias_data,
            scale->data(), g_idx_data, out->dtype(), num_bits, group_size,
            out->dim(0), out->dim(1), current_input->dim(1));
#ifdef ENABLE_NVIDIA_API
    case ZEDINFER_DEVICE_NVIDIA:
        if (num_bits == 8 && group_size <= 0 && scale->numel() > out->dim(1)) {
            const size_t scale_groups = scale->numel() / out->dim(1);
            if (scale_groups > 1 && (current_input->dim(1) % scale_groups) == 0) {
                group_size = static_cast<int>(current_input->dim(1) / scale_groups);
            }
        }
        return nvidia::linear_quantized(
            out->data(), current_input->data(), weight->data(), bias_data,
            scale->data(), g_idx_data, out->dtype(), num_bits, group_size,
            out->dim(0), out->dim(1), current_input->dim(1));
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace zedinfer::ops
