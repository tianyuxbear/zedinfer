// Unit test for ops::attn_output_gate. Verifies attn := attn * sigmoid(g)
// against four (a, g) pairs spanning negative / zero / positive / saturating
// gate inputs. BF16 round-trip tolerance is 0.02f (one ULP at the magnitudes
// involved here).

#include "backend/core/context/context.hpp"
#include "backend/device/device.hpp"
#include "backend/ops/attn_output_gate/attn_output_gate.hpp"
#include "backend/tensor/tensor.hpp"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <gtest/gtest.h>
#include <vector>

namespace {

#if defined(ENABLE_NVIDIA_API)

uint16_t to_bf16_bits(float f) {
    uint32_t u = 0;
    std::memcpy(&u, &f, sizeof(u));
    return static_cast<uint16_t>(u >> 16);
}

float from_bf16_bits(uint16_t bits) {
    uint32_t u = static_cast<uint32_t>(bits) << 16;
    float f;
    std::memcpy(&f, &u, sizeof(f));
    return f;
}

TEST(OpsAttnOutputGate, AppliesSigmoidGateAtKnownPoints) {
    try {
        zedinfer::core::context().setDevice(ZEDINFER_DEVICE_NVIDIA, 0);
    } catch (const std::exception& e) {
        GTEST_SKIP() << "NVIDIA runtime init failed: " << e.what();
    }

    // Four scalars exercising negative / zero / negative-saturating / positive-saturating gate.
    const std::vector<float> attn_init = { 1.0f, 2.0f, -1.0f,  0.5f };
    const std::vector<float> gate_init = { 0.0f, 1.0f, -1.0f, 10.0f };
    const size_t n = attn_init.size();

    auto attn = zedinfer::Tensor::create({n}, ZEDINFER_DTYPE_BF16, ZEDINFER_DEVICE_NVIDIA, 0);
    auto g    = zedinfer::Tensor::create({n}, ZEDINFER_DTYPE_BF16, ZEDINFER_DEVICE_NVIDIA, 0);

    std::vector<uint16_t> attn_h(n), gate_h(n);
    for (size_t i = 0; i < n; ++i) {
        attn_h[i] = to_bf16_bits(attn_init[i]);
        gate_h[i] = to_bf16_bits(gate_init[i]);
    }
    auto* api = zedinfer::device::getRuntimeAPI(ZEDINFER_DEVICE_NVIDIA);
    api->memcpy_sync(attn->data(), attn_h.data(), n * sizeof(uint16_t), ZEDINFER_MEMCPY_H2D);
    api->memcpy_sync(g->data(),    gate_h.data(), n * sizeof(uint16_t), ZEDINFER_MEMCPY_H2D);

    zedinfer::ops::attn_output_gate(attn, g);
    zedinfer::core::context().runtime().synchronize();

    std::vector<uint16_t> result_h(n);
    api->memcpy_sync(result_h.data(), attn->data(), n * sizeof(uint16_t), ZEDINFER_MEMCPY_D2H);

    for (size_t i = 0; i < n; ++i) {
        const float sigmoid_g = 1.0f / (1.0f + std::exp(-gate_init[i]));
        const float expected = attn_init[i] * sigmoid_g;
        const float actual = from_bf16_bits(result_h[i]);
        EXPECT_NEAR(actual, expected, 0.02f) << "index=" << i << " attn=" << attn_init[i] << " g=" << gate_init[i];
    }
}

#else  // !ENABLE_NVIDIA_API

TEST(OpsAttnOutputGate, RequiresNvidiaBackend) {
    GTEST_SKIP() << "attn_output_gate test exercises NVIDIA path; CPU build skips";
}

#endif

} // namespace
