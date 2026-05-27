#include "backend/ops/mrope/mrope_3d.hpp"

// CPU reference implementation of the Qwen3.5 3D mrope op.
//
// Compiled only when no NVIDIA backend is available. In NVIDIA builds the
// canonical `ops::mrope_3d` is defined in the .cu translation unit (which
// executes on the device); shipping both as separate TUs would produce a
// multiple-definition link error on `ops::mrope_3d`.
//
// This impl exists so:
//   - CPU-only builds still link (model loaders may instantiate the op even
//     when no GPU is present).
//   - We have a self-contained reference for cross-checking the GPU kernel
//     in future correctness work (M2 byte-exact alignment).
#ifndef ENABLE_NVIDIA_API

#include "utils/check.hpp"
#include "utils/types.hpp"

#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace zedinfer::ops {

namespace {

// Typed kernel; T is the element type of x (bf16_t or float).
// Uses HF's rotate_half pair layout (slot pi with slot pi+half) — see the
// matching .cu kernel docstring for the rationale.
template <typename T>
void mrope_3d_typed(T* xp, const int32_t* pos_t, const int32_t* pos_h, const int32_t* pos_w,
                    int N, int H, int Dh, int Dh_rot, int s0, int s1, int s2,
                    bool interleaved, float theta) {
    const int half = Dh_rot / 2;
    auto axis_for_pair = [s0, s1, s2, interleaved](int pi) -> int {
        if (interleaved) {
            // HF apply_interleaved_mrope: H owns pi where (pi%3==1 && pi<s1*3);
            // W owns pi where (pi%3==2 && pi<s2*3); else T.
            if ((pi % 3) == 1 && pi < s1 * 3) return 1;
            if ((pi % 3) == 2 && pi < s2 * 3) return 2;
            return 0;
        }
        // Chunked: [T..., H..., W...].
        if (pi < s0) return 0;
        if (pi < s0 + s1) return 1;
        return 2;
    };

    for (int n = 0; n < N; ++n) {
        const int axis_pos[3] = {pos_t[n], pos_h[n], pos_w[n]};
        for (int hd = 0; hd < H; ++hd) {
            T* row = xp + (static_cast<size_t>(n) * H + hd) * Dh;
            for (int pi = 0; pi < half; ++pi) {
                const int axis    = axis_for_pair(pi);
                const float pos_v = static_cast<float>(axis_pos[axis]);
                // freq = theta^(-(2*pi) / (2*half))
                const float freq  = 1.0f / std::pow(theta,
                                                    static_cast<float>(2 * pi) /
                                                        static_cast<float>(2 * half));
                const float angle = pos_v * freq;
                const float c     = std::cos(angle);
                const float s     = std::sin(angle);

                // HF rotate_half: pair (pi, pi+half), same angle on both.
                const float a = utils::cast<float>(row[pi]);
                const float b = utils::cast<float>(row[pi + half]);
                row[pi]        = utils::cast<T>(a * c - b * s);
                row[pi + half] = utils::cast<T>(a * s + b * c);
            }
            // dims [Dh_rot, Dh) untouched (pass-through).
        }
    }
}

} // namespace

void mrope_3d(tensor_t x, tensor_t pos_ids_thw, const model::MRoPEConfig& cfg) {
    if (!x || !pos_ids_thw) {
        throw std::runtime_error("ops::mrope_3d(cpu): null tensor input");
    }
    if (x->ndim() != 3) {
        throw std::runtime_error("ops::mrope_3d(cpu): x must be 3-D [N, H, Dh]");
    }
    if (pos_ids_thw->ndim() != 2 || pos_ids_thw->shape()[0] != 3) {
        throw std::runtime_error("ops::mrope_3d(cpu): pos_ids_thw must be [3, N]");
    }

    const int N  = static_cast<int>(x->shape()[0]);
    const int H  = static_cast<int>(x->shape()[1]);
    const int Dh = static_cast<int>(x->shape()[2]);
    int Dh_rot   = static_cast<int>(static_cast<float>(Dh) * cfg.partial_factor);
    if (Dh_rot % 2 != 0) --Dh_rot; // must be even for cos/sin pairing.
    if (Dh_rot <= 0) return;

    const auto* pos = reinterpret_cast<const int32_t*>(pos_ids_thw->data());
    const int32_t* pos_t = pos + 0 * N;
    const int32_t* pos_h = pos + 1 * N;
    const int32_t* pos_w = pos + 2 * N;

    switch (x->dtype()) {
        case ZEDINFER_DTYPE_BF16: {
            auto* xp = reinterpret_cast<bf16_t*>(x->data());
            mrope_3d_typed<bf16_t>(xp, pos_t, pos_h, pos_w, N, H, Dh, Dh_rot,
                                   cfg.section[0], cfg.section[1], cfg.section[2],
                                   cfg.interleaved, cfg.theta);
            break;
        }
        case ZEDINFER_DTYPE_F32: {
            auto* xp = reinterpret_cast<float*>(x->data());
            mrope_3d_typed<float>(xp, pos_t, pos_h, pos_w, N, H, Dh, Dh_rot,
                                  cfg.section[0], cfg.section[1], cfg.section[2],
                                  cfg.interleaved, cfg.theta);
            break;
        }
        default:
            EXCEPTION_UNSUPPORTED_DATATYPE(x->dtype());
    }
}

} // namespace zedinfer::ops

#endif // !ENABLE_NVIDIA_API
