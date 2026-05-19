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
    if (p.num_tokens == 1) {
        gdn_decode_launch(p);
    } else {
        gdn_prefill_launch(p);
    }
}

} // namespace zedinfer::ops::mamba
