// Unit test for ops::mamba::causal_conv1d — verifies the depthwise kernel-4
// causal conv runs end-to-end with the SSMStatePool conv state, that state
// rolls forward across tokens, and that the SiLU activation is applied.
//
// Numerical reference (all-ones x and w, K=4):
//   - n=0: state is all zeros after reset_slot, window=[0,0,0,1] -> sum=1 ->
//          silu(1) = 1 * sigmoid(1) ~= 0.7311.
//   - n=1: window=[0,0,1,1] -> sum=2 -> silu(2) ~= 1.7616.
//   - n=2: window=[0,1,1,1] -> sum=3 -> silu(3) ~= 2.857.
//   - n=3: window=[1,1,1,1] -> sum=4 -> silu(4) ~= 3.928.
//   - n=4: state has been updated to [x_1, x_2, x_3] = [1,1,1], window same
//          as n=3 -> silu(4) ~= 3.928.
// BF16 round-trip on x/state plus __expf precision loss is within ~0.05.

#include "backend/core/context/context.hpp"
#include "backend/device/device.hpp"
#include "backend/ops/mamba/causal_conv1d.hpp"
#include "backend/tensor/tensor.hpp"
#include "frontend/models/ssm_state_pool.hpp"
#include "zedinfer/activation.hpp"

#include <cmath>
#include <cstring>
#include <gtest/gtest.h>
#include <vector>

namespace {

#if defined(ENABLE_NVIDIA_API)

using zedinfer::Tensor;
using zedinfer::core::context;
using zedinfer::model::SSMStatePool;
using zedinfer::model::SSMStatePoolConfig;
using zedinfer::ops::mamba::causal_conv1d;

// Skip whole suite if no NVIDIA runtime can be initialized (e.g. headless CI).
class OpsCausalConv1d : public ::testing::Test {
protected:
    void SetUp() override {
        try {
            context().setDevice(ZEDINFER_DEVICE_NVIDIA, 0);
        } catch (const std::exception& e) {
            GTEST_SKIP() << "NVIDIA runtime init failed: " << e.what();
        }
    }
};

// Bf16 host buffer of value `v` repeated `n` times.
std::vector<uint16_t> bf16_ones(size_t n) {
    const float one = 1.0f;
    uint32_t u      = 0;
    std::memcpy(&u, &one, sizeof(u));
    const uint16_t bf = static_cast<uint16_t>(u >> 16); // truncate-to-bf16.
    return std::vector<uint16_t>(n, bf);
}

// Reconstruct a float from a bf16 host element.
float bf16_to_float(uint16_t v) {
    uint32_t u = static_cast<uint32_t>(v) << 16;
    float f;
    std::memcpy(&f, &u, sizeof(f));
    return f;
}

// All-ones x and w driving the canonical Mamba2 conv1d boundary case from the
// plan: K=4, D=32, N=5. Expects first-token silu(1) ~ 0.7311 and saturated
// silu(4) ~ 3.928 once the state is full of 1s.
TEST_F(OpsCausalConv1d, AllOnesFiveTokensRollsState) {
    constexpr int K = 4;
    constexpr int D = 32;
    constexpr int N = 5;

    SSMStatePoolConfig cfg;
    cfg.num_linear_layers = 1;
    cfg.num_v_heads       = 1;
    cfg.value_head_dim    = 1;
    cfg.d_state           = 1;
    cfg.conv_kernel_dim   = K;
    cfg.qkv_dim           = D;
    cfg.max_concurrent    = 1;
    cfg.state_dtype       = ZEDINFER_DTYPE_BF16;

    zedinfer::ExecutorConfig exec(ZEDINFER_DEVICE_NVIDIA, 0, ZEDINFER_DTYPE_BF16);
    SSMStatePool pool(cfg, exec);
    const int slot = pool.acquire_slot();
    pool.reset_slot(slot);

    auto x = Tensor::create({static_cast<size_t>(N), static_cast<size_t>(D)},
                             ZEDINFER_DTYPE_BF16, ZEDINFER_DEVICE_NVIDIA, 0);
    auto w = Tensor::create({static_cast<size_t>(D), 1, static_cast<size_t>(K)},
                             ZEDINFER_DTYPE_BF16, ZEDINFER_DEVICE_NVIDIA, 0);
    auto out = Tensor::create({static_cast<size_t>(N), static_cast<size_t>(D)},
                               ZEDINFER_DTYPE_BF16, ZEDINFER_DEVICE_NVIDIA, 0);

    auto x_ones = bf16_ones(static_cast<size_t>(N) * D);
    auto w_ones = bf16_ones(static_cast<size_t>(D) * K);

    auto* api = zedinfer::device::getRuntimeAPI(ZEDINFER_DEVICE_NVIDIA);
    api->memcpy_sync(x->data(), x_ones.data(), x_ones.size() * sizeof(uint16_t), ZEDINFER_MEMCPY_H2D);
    api->memcpy_sync(w->data(), w_ones.data(), w_ones.size() * sizeof(uint16_t), ZEDINFER_MEMCPY_H2D);

    ASSERT_NO_THROW(causal_conv1d(out, x, w, pool.view(), slot, 0));
    context().runtime().synchronize();

    std::vector<uint16_t> host(static_cast<size_t>(N) * D);
    api->memcpy_sync(host.data(), out->data(), host.size() * sizeof(uint16_t), ZEDINFER_MEMCPY_D2H);

    // Channel 0 at token 0: window=[0,0,0,1] -> sum=1 -> silu(1) ~ 0.7311.
    const float first = bf16_to_float(host[0 * D + 0]);
    // Channel 0 at last token: state full of 1s -> window=[1,1,1,1] -> silu(4) ~ 3.928.
    const float last = bf16_to_float(host[(N - 1) * D + 0]);

    EXPECT_NEAR(first, 0.7311f, 0.05f) << "first-token silu(1) deviates more than bf16 tolerance";
    EXPECT_NEAR(last, 3.928f, 0.05f)
        << "saturated-state silu(4) deviates more than bf16 tolerance";

    // Sanity: every channel should produce the same value at the same n since
    // x and w are constant across channels. Check token 0 across all D.
    for (int d = 0; d < D; ++d) {
        const float v = bf16_to_float(host[0 * D + d]);
        EXPECT_NEAR(v, first, 1e-3f) << "channel " << d << " mismatch at n=0";
    }

    // Sanity: monotonically increasing toward saturation across n.
    const float n0 = bf16_to_float(host[0 * D + 0]);
    const float n1 = bf16_to_float(host[1 * D + 0]);
    const float n2 = bf16_to_float(host[2 * D + 0]);
    const float n3 = bf16_to_float(host[3 * D + 0]);
    EXPECT_LT(n0, n1);
    EXPECT_LT(n1, n2);
    EXPECT_LT(n2, n3);

    pool.release_slot(slot);
}

// Decode mode: feed three single-token calls in sequence against the same
// slot, verifying that conv state persists across calls (so token #2 sees
// token #1's input as its history). After three decode steps the state is
// exactly two-of-K-1 deep, and the third step should yield silu(3).
TEST_F(OpsCausalConv1d, DecodeStatePersistsAcrossCalls) {
    constexpr int K = 4;
    constexpr int D = 16;

    SSMStatePoolConfig cfg;
    cfg.num_linear_layers = 1;
    cfg.num_v_heads       = 1;
    cfg.value_head_dim    = 1;
    cfg.d_state           = 1;
    cfg.conv_kernel_dim   = K;
    cfg.qkv_dim           = D;
    cfg.max_concurrent    = 1;
    cfg.state_dtype       = ZEDINFER_DTYPE_BF16;

    zedinfer::ExecutorConfig exec(ZEDINFER_DEVICE_NVIDIA, 0, ZEDINFER_DTYPE_BF16);
    SSMStatePool pool(cfg, exec);
    const int slot = pool.acquire_slot();
    pool.reset_slot(slot);

    auto x = Tensor::create({1, static_cast<size_t>(D)}, ZEDINFER_DTYPE_BF16, ZEDINFER_DEVICE_NVIDIA, 0);
    auto w = Tensor::create({static_cast<size_t>(D), 1, static_cast<size_t>(K)}, ZEDINFER_DTYPE_BF16,
                             ZEDINFER_DEVICE_NVIDIA, 0);
    auto out = Tensor::create({1, static_cast<size_t>(D)}, ZEDINFER_DTYPE_BF16, ZEDINFER_DEVICE_NVIDIA, 0);

    auto x_ones = bf16_ones(static_cast<size_t>(D));
    auto w_ones = bf16_ones(static_cast<size_t>(D) * K);

    auto* api = zedinfer::device::getRuntimeAPI(ZEDINFER_DEVICE_NVIDIA);
    api->memcpy_sync(x->data(), x_ones.data(), x_ones.size() * sizeof(uint16_t), ZEDINFER_MEMCPY_H2D);
    api->memcpy_sync(w->data(), w_ones.data(), w_ones.size() * sizeof(uint16_t), ZEDINFER_MEMCPY_H2D);

    // Expected per-step values: silu(1), silu(2), silu(3).
    const float expected[3] = {0.7311f, 1.7616f, 2.857f};
    for (int step = 0; step < 3; ++step) {
        ASSERT_NO_THROW(causal_conv1d(out, x, w, pool.view(), slot, 0))
            << "decode step " << step << " threw";
        context().runtime().synchronize();

        std::vector<uint16_t> host(static_cast<size_t>(D));
        api->memcpy_sync(host.data(), out->data(), host.size() * sizeof(uint16_t), ZEDINFER_MEMCPY_D2H);
        const float got = bf16_to_float(host[0]);
        EXPECT_NEAR(got, expected[step], 0.05f) << "decode step " << step << " value off";
    }

    pool.release_slot(slot);
}

#else // !ENABLE_NVIDIA_API

TEST(OpsCausalConv1d, RequiresNvidiaBackend) {
    GTEST_SKIP() << "ops::mamba::causal_conv1d has only an NVIDIA impl; CPU build skips this suite";
}

#endif

} // namespace
