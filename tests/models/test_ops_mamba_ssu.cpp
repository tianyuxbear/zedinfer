// Smoke test for ops::mamba::ssu — verifies the FlashInfer wrapper actually
// runs and writes a finite, non-zero output on the GPU. Numerical correctness
// vs HuggingFace transformers is reserved for M2's byte-exact alignment work;
// this test only confirms the kernel launches, state-pool addressing is sane,
// and both single-token (decode) and varlen (prefill) modes complete.

#include "backend/core/context/context.hpp"
#include "backend/ops/mamba/ssu.hpp"
#include "backend/tensor/tensor.hpp"
#include "frontend/models/ssm_state_pool.hpp"
#include "zedinfer/activation.hpp"

#include "backend/device/device.hpp"

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <gtest/gtest.h>
#include <vector>

namespace {

#if defined(ENABLE_NVIDIA_API)

using zedinfer::Tensor;
using zedinfer::core::context;
using zedinfer::model::SSMStatePool;
using zedinfer::model::SSMStatePoolConfig;
using zedinfer::ops::mamba::ssu;
using zedinfer::ops::mamba::SSUParams;

// Skip whole suite if no NVIDIA runtime can be initialized (e.g. headless CI).
class OpsMambaSSU : public ::testing::Test {
protected:
    void SetUp() override {
        try {
            context().setDevice(ZEDINFER_DEVICE_NVIDIA, 0);
        } catch (const std::exception& e) {
            GTEST_SKIP() << "NVIDIA runtime init failed: " << e.what();
        }
    }
};

// Helper: bf16 host buffer from float seed.
std::vector<uint16_t> bf16_from_float(const std::vector<float>& src) {
    std::vector<uint16_t> out(src.size());
    for (size_t i = 0; i < src.size(); ++i) {
        uint32_t u = 0;
        std::memcpy(&u, &src[i], sizeof(u));
        out[i] = static_cast<uint16_t>(u >> 16); // truncate-to-bf16
    }
    return out;
}

// Allocate + fill bf16 tensor from host float values.
zedinfer::tensor_t make_bf16(const std::vector<size_t>& shape, const std::vector<float>& values) {
    auto t = Tensor::create(shape, ZEDINFER_DTYPE_BF16, ZEDINFER_DEVICE_NVIDIA, 0);
    auto bf = bf16_from_float(values);
    EXPECT_EQ(values.size(), t->numel());
    auto* api = zedinfer::device::getRuntimeAPI(ZEDINFER_DEVICE_NVIDIA);
    api->memcpy_sync(t->data(), bf.data(), bf.size() * sizeof(uint16_t), ZEDINFER_MEMCPY_H2D);
    return t;
}

// Same but for fp32 tensors (A_log lives in fp32 per Qwen3.5 config).
zedinfer::tensor_t make_f32(const std::vector<size_t>& shape, const std::vector<float>& values) {
    auto t = Tensor::create(shape, ZEDINFER_DTYPE_F32, ZEDINFER_DEVICE_NVIDIA, 0);
    auto* api = zedinfer::device::getRuntimeAPI(ZEDINFER_DEVICE_NVIDIA);
    api->memcpy_sync(t->data(), values.data(), values.size() * sizeof(float), ZEDINFER_MEMCPY_H2D);
    return t;
}

// Check every bf16 element in dev tensor is finite and at least one is non-zero.
void expect_finite_and_nonzero(zedinfer::tensor_t t) {
    std::vector<uint16_t> host(t->numel());
    auto* api = zedinfer::device::getRuntimeAPI(ZEDINFER_DEVICE_NVIDIA);
    api->memcpy_sync(host.data(), t->data(), host.size() * sizeof(uint16_t), ZEDINFER_MEMCPY_D2H);
    bool any_nonzero = false;
    for (auto v : host) {
        uint32_t u = static_cast<uint32_t>(v) << 16;
        float f;
        std::memcpy(&f, &u, sizeof(f));
        ASSERT_FALSE(std::isnan(f)) << "NaN in SSU output";
        ASSERT_FALSE(std::isinf(f)) << "Inf in SSU output";
        if (f != 0.0f) any_nonzero = true;
    }
    EXPECT_TRUE(any_nonzero) << "SSU output is all-zero; kernel likely no-op";
}

// Build inputs sized to match Qwen3.5's value/key head dims (Dv=128, Dk=128)
// but small Hv/Hk and dstate so the test stays sub-second.
constexpr int kHv = 8;     // value heads (small subset of Qwen3.5's 32-48)
constexpr int kDv = 128;
constexpr int kHk = 4;     // key heads
constexpr int kDk = 128;
constexpr int kDstate = 128;

SSMStatePoolConfig make_test_pool_cfg() {
    SSMStatePoolConfig cfg;
    cfg.num_linear_layers = 1;
    cfg.num_v_heads       = kHv;
    cfg.value_head_dim    = kDv;
    cfg.d_state           = kDstate;
    cfg.conv_kernel_dim   = 4;
    cfg.qkv_dim           = 2 * kHk * kDk + kHv * kDv;
    cfg.max_concurrent    = 1;
    cfg.state_dtype       = ZEDINFER_DTYPE_BF16;
    return cfg;
}

TEST_F(OpsMambaSSU, DecodeSingleTokenRuns) {
    zedinfer::ExecutorConfig exec(ZEDINFER_DEVICE_NVIDIA, 0, ZEDINFER_DTYPE_BF16);
    SSMStatePool pool(make_test_pool_cfg(), exec);

    const int slot = pool.acquire_slot();
    pool.reset_slot(slot);

    // N=1 decode: small randomized-looking inputs (deterministic seed).
    auto fill_seq = [](size_t n, float base, float step) {
        std::vector<float> v(n);
        for (size_t i = 0; i < n; ++i) v[i] = base + step * static_cast<float>(i);
        return v;
    };

    auto q = make_bf16({1, static_cast<size_t>(kHk * kDk)}, fill_seq(kHk * kDk, 0.01f, 0.001f));
    auto k = make_bf16({1, static_cast<size_t>(kHk * kDk)}, fill_seq(kHk * kDk, 0.02f, 0.001f));
    auto v = make_bf16({1, static_cast<size_t>(kHv * kDv)}, fill_seq(kHv * kDv, 0.03f, 0.001f));
    auto a = make_bf16({1, static_cast<size_t>(kHv)},       fill_seq(kHv, -0.5f, 0.05f));
    auto b = make_bf16({1, static_cast<size_t>(kHv)},       fill_seq(kHv, 0.4f, 0.05f));
    auto A_log   = make_f32({static_cast<size_t>(kHv)},     fill_seq(kHv, -0.1f, 0.01f));
    auto dt_bias = make_bf16({static_cast<size_t>(kHv)},    fill_seq(kHv, -0.2f, 0.02f));
    auto z       = make_bf16({1, static_cast<size_t>(kHv * kDv)}, fill_seq(kHv * kDv, 0.05f, 0.001f));
    auto out     = Tensor::create({1, static_cast<size_t>(kHv * kDv)},
                                   ZEDINFER_DTYPE_BF16, ZEDINFER_DEVICE_NVIDIA, 0);

    SSUParams p;
    p.state_view = pool.view();
    p.slot_idx   = slot;
    p.layer_idx  = 0;
    p.q = q; p.k = k; p.v = v; p.a = a; p.b = b;
    p.A_log = A_log; p.dt_bias = dt_bias; p.z = z;
    p.out = out;
    p.num_tokens = 1;

    ASSERT_NO_THROW(ssu(p));
    context().runtime().synchronize();
    expect_finite_and_nonzero(out);

    pool.release_slot(slot);
}

TEST_F(OpsMambaSSU, PrefillVarlenFourTokensRuns) {
    zedinfer::ExecutorConfig exec(ZEDINFER_DEVICE_NVIDIA, 0, ZEDINFER_DTYPE_BF16);
    SSMStatePool pool(make_test_pool_cfg(), exec);

    const int slot = pool.acquire_slot();
    pool.reset_slot(slot);
    const int N = 4;

    auto fill_seq = [](size_t n, float base, float step) {
        std::vector<float> v(n);
        for (size_t i = 0; i < n; ++i) v[i] = base + step * static_cast<float>(i);
        return v;
    };

    auto q = make_bf16({static_cast<size_t>(N), static_cast<size_t>(kHk * kDk)}, fill_seq(N * kHk * kDk, 0.01f, 0.0005f));
    auto k = make_bf16({static_cast<size_t>(N), static_cast<size_t>(kHk * kDk)}, fill_seq(N * kHk * kDk, 0.02f, 0.0005f));
    auto v = make_bf16({static_cast<size_t>(N), static_cast<size_t>(kHv * kDv)}, fill_seq(N * kHv * kDv, 0.03f, 0.0005f));
    auto a = make_bf16({static_cast<size_t>(N), static_cast<size_t>(kHv)}, fill_seq(N * kHv, -0.5f, 0.02f));
    auto b = make_bf16({static_cast<size_t>(N), static_cast<size_t>(kHv)}, fill_seq(N * kHv, 0.4f, 0.02f));
    auto A_log   = make_f32({static_cast<size_t>(kHv)},                  fill_seq(kHv, -0.1f, 0.01f));
    auto dt_bias = make_bf16({static_cast<size_t>(kHv)},                  fill_seq(kHv, -0.2f, 0.02f));
    auto z       = make_bf16({static_cast<size_t>(N), static_cast<size_t>(kHv * kDv)}, fill_seq(N * kHv * kDv, 0.05f, 0.0005f));
    auto out     = Tensor::create({static_cast<size_t>(N), static_cast<size_t>(kHv * kDv)},
                                   ZEDINFER_DTYPE_BF16, ZEDINFER_DEVICE_NVIDIA, 0);

    SSUParams p;
    p.state_view = pool.view();
    p.slot_idx   = slot;
    p.layer_idx  = 0;
    p.q = q; p.k = k; p.v = v; p.a = a; p.b = b;
    p.A_log = A_log; p.dt_bias = dt_bias; p.z = z;
    p.out = out;
    p.num_tokens = N;

    ASSERT_NO_THROW(ssu(p));
    context().runtime().synchronize();
    expect_finite_and_nonzero(out);

    pool.release_slot(slot);
}

#else  // !ENABLE_NVIDIA_API

TEST(OpsMambaSSU, RequiresNvidiaBackend) {
    GTEST_SKIP() << "ops::mamba::ssu has only an NVIDIA impl; CPU build skips this suite";
}

#endif

} // namespace
