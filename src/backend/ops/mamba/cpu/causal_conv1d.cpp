#include "backend/ops/mamba/causal_conv1d.hpp"

// CPU reference implementation of the Mamba2 depthwise causal conv1d.
//
// Compiled only when no NVIDIA backend is available. In NVIDIA builds the
// canonical `ops::mamba::causal_conv1d` is defined in the .cu translation
// unit (which executes on the device); shipping both as separate TUs would
// produce a multiple-definition link error on `ops::mamba::causal_conv1d`.
//
// This impl exists so:
//   - CPU-only builds still link (M1 forward path may instantiate the op
//     even when no GPU is present).
//   - We have a self-contained reference for cross-checking the GPU kernel
//     in future correctness work (M2 byte-exact alignment).
#ifndef ENABLE_NVIDIA_API

#include "utils/check.hpp"
#include "utils/types.hpp"

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace zedinfer::ops::mamba {

namespace {

// Float reference path (correctness baseline). All loads/stores are routed
// through utils::cast<> so the same code services bf16 and f32 inputs.
template <typename T>
void causal_conv1d_typed(T* out_ptr, const T* x_ptr, const T* w_ptr, T* state_ptr,
                         int N, int D, int K) {
    // Per-token: window[0..K-1] = (state | x)[n-K+1 .. n];
    //            y[n,d] = silu(sum_i w[d,0,i] * window[i,d]).
    for (int n = 0; n < N; ++n) {
        for (int d = 0; d < D; ++d) {
            float acc = 0.0f;
            for (int i = 0; i < K; ++i) {
                // window index i corresponds to absolute time (n - (K - 1 - i)).
                const int rel = n - (K - 1 - i);
                float v;
                if (rel < 0) {
                    // Read from state: state is the (K-1) tokens prior to x[0].
                    const int state_row = K - 1 + rel;
                    v = utils::cast<float>(state_ptr[state_row * D + d]);
                } else {
                    v = utils::cast<float>(x_ptr[rel * D + d]);
                }
                acc += v * utils::cast<float>(w_ptr[d * K + i]);
            }
            // SiLU: x * sigmoid(x).
            const float s      = 1.0f / (1.0f + std::exp(-acc));
            out_ptr[n * D + d] = utils::cast<T>(acc * s);
        }
    }

    // Update state to the last (K-1) tokens of (state | x).
    if (N >= K - 1) {
        for (int i = 0; i < K - 1; ++i) {
            for (int d = 0; d < D; ++d) {
                state_ptr[i * D + d] = x_ptr[(N - (K - 1) + i) * D + d];
            }
        }
    } else {
        // Save x tail first so the in-place state shift cannot clobber inputs
        // even if state and x happened to alias in a contrived test setup.
        std::vector<T> tail_x(static_cast<size_t>(N) * static_cast<size_t>(D));
        for (int i = 0; i < N; ++i) {
            for (int d = 0; d < D; ++d) {
                tail_x[static_cast<size_t>(i) * static_cast<size_t>(D) + static_cast<size_t>(d)] =
                    x_ptr[i * D + d];
            }
        }
        for (int i = 0; i < K - 1 - N; ++i) {
            for (int d = 0; d < D; ++d) {
                state_ptr[i * D + d] = state_ptr[(i + N) * D + d];
            }
        }
        for (int i = 0; i < N; ++i) {
            for (int d = 0; d < D; ++d) {
                state_ptr[(K - 1 - N + i) * D + d] =
                    tail_x[static_cast<size_t>(i) * static_cast<size_t>(D) + static_cast<size_t>(d)];
            }
        }
    }
}

} // namespace

void causal_conv1d(tensor_t out, tensor_t x, tensor_t weight,
                   model::SSMStateView v, int slot_idx, int layer_idx) {
    if (!out || !x || !weight) {
        throw std::runtime_error("ops::mamba::causal_conv1d(cpu): null tensor input");
    }
    if (v.conv_base == nullptr) {
        throw std::runtime_error("ops::mamba::causal_conv1d(cpu): conv state base is null");
    }
    const int N = static_cast<int>(x->shape()[0]);
    const int D = static_cast<int>(x->shape()[1]);
    const int K = v.conv_kernel_dim;
    if (K != 4) {
        throw std::runtime_error("ops::mamba::causal_conv1d(cpu): only K=4 is supported (Qwen3.5)");
    }
    if (D != v.qkv_dim) {
        throw std::runtime_error("ops::mamba::causal_conv1d(cpu): x channel dim must match qkv_dim");
    }

    // state_ptr is offset to the (slot_idx, layer_idx) sub-buffer;
    // v.conv_stride_* are already in bytes per SSMStatePool's view contract.
    auto* state_byte_layer = reinterpret_cast<std::byte*>(v.conv_base)
                            + static_cast<int64_t>(slot_idx)  * v.conv_stride_slot
                            + static_cast<int64_t>(layer_idx) * v.conv_stride_layer;

    switch (x->dtype()) {
        case ZEDINFER_DTYPE_BF16: {
            auto* xp = reinterpret_cast<const bf16_t*>(x->data());
            auto* wp = reinterpret_cast<const bf16_t*>(weight->data());
            auto* op = reinterpret_cast<bf16_t*>(out->data());
            auto* sp = reinterpret_cast<bf16_t*>(state_byte_layer);
            causal_conv1d_typed<bf16_t>(op, xp, wp, sp, N, D, K);
            break;
        }
        case ZEDINFER_DTYPE_F32: {
            auto* xp = reinterpret_cast<const float*>(x->data());
            auto* wp = reinterpret_cast<const float*>(weight->data());
            auto* op = reinterpret_cast<float*>(out->data());
            auto* sp = reinterpret_cast<float*>(state_byte_layer);
            causal_conv1d_typed<float>(op, xp, wp, sp, N, D, K);
            break;
        }
        default:
            EXCEPTION_UNSUPPORTED_DATATYPE(x->dtype());
    }
}

} // namespace zedinfer::ops::mamba

#endif // !ENABLE_NVIDIA_API
