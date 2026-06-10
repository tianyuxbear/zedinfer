#include "backend/ops/mamba/gdn.hpp"

#include <stdexcept>

namespace zedinfer::ops::mamba {

void gdn_decode_launch(const GDNParams& p);
void gdn_prefill_launch(const GDNParams& p);

void gdn(const GDNParams& p) {
    if (p.num_tokens <= 0) {
        throw std::runtime_error("[ops::mamba::gdn] num_tokens must be >= 1");
    }
    if (!p.state_view.ssm_base) {
        throw std::runtime_error("[ops::mamba::gdn] state_view.ssm_base is null");
    }
    if (!p.A_log || p.A_log->dtype() != ZEDINFER_DTYPE_F32) {
        throw std::runtime_error("[ops::mamba::gdn] A_log must be fp32");
    }
    if (!p.dt_bias || p.dt_bias->dtype() != ZEDINFER_DTYPE_BF16) {
        throw std::runtime_error("[ops::mamba::gdn] dt_bias must be bf16");
    }
    if (p.num_tokens == 1) {
        gdn_decode_launch(p);
    } else {
        gdn_prefill_launch(p);
    }
}

} // namespace zedinfer::ops::mamba
