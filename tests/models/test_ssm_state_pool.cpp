#include "backend/core/context/context.hpp"
#include "frontend/models/ssm_state_pool.hpp"
#include "zedinfer.h"
#include "zedinfer/activation.hpp"

#include <exception>
#include <gtest/gtest.h>

using zedinfer::ExecutorConfig;
using zedinfer::model::SSMStatePool;
using zedinfer::model::SSMStatePoolConfig;
using zedinfer::model::SSMStateView;

namespace {

// Initialize the active device runtime for the pool's target. Returns false if
// the runtime fails to come up (e.g. headless CI without a CUDA-capable GPU);
// callers should GTEST_SKIP() in that case so the suite stays green.
bool try_activate_device(zedinferDeviceType_t device_type, int device_id) {
    try {
        zedinfer::core::context().setDevice(device_type, device_id);
        return true;
    } catch (const std::exception&) { return false; }
}

SSMStatePoolConfig make_test_config() {
    SSMStatePoolConfig cfg;
    cfg.num_linear_layers = 4;
    cfg.num_v_heads = 8;
    cfg.value_head_dim = 16;
    cfg.d_state = 16;
    cfg.conv_kernel_dim = 4;
    cfg.qkv_dim = 256;
    cfg.max_concurrent = 2;
    cfg.state_dtype = ZEDINFER_DTYPE_BF16;
    return cfg;
}

} // namespace

TEST(SSMStatePoolTest, AllocatesBuffersAndExposesView) {
#if !defined(ENABLE_NVIDIA_API)
    GTEST_SKIP() << "NVIDIA support not enabled in this build";
#else
    if (!try_activate_device(ZEDINFER_DEVICE_NVIDIA, 0)) {
        GTEST_SKIP() << "NVIDIA runtime unavailable; skipping SSMStatePool GPU test";
    }

    const auto cfg = make_test_config();
    ExecutorConfig exec(ZEDINFER_DEVICE_NVIDIA, 0, ZEDINFER_DTYPE_BF16);
    SSMStatePool pool(cfg, exec);

    EXPECT_EQ(pool.num_free_slots(), 2);

    const SSMStateView view = pool.view();
    EXPECT_NE(view.ssm_base, nullptr);
    EXPECT_NE(view.conv_base, nullptr);
    EXPECT_EQ(view.num_v_heads, cfg.num_v_heads);
    EXPECT_EQ(view.value_head_dim, cfg.value_head_dim);
    EXPECT_EQ(view.d_state, cfg.d_state);
    EXPECT_EQ(view.conv_kernel_dim, cfg.conv_kernel_dim);
    EXPECT_EQ(view.qkv_dim, cfg.qkv_dim);
    EXPECT_EQ(view.dtype, cfg.state_dtype);
    EXPECT_GT(pool.bytes_per_slot(), 0u);
#endif
}

TEST(SSMStatePoolTest, AcquireReleaseRoundTrips) {
#if !defined(ENABLE_NVIDIA_API)
    GTEST_SKIP() << "NVIDIA support not enabled in this build";
#else
    if (!try_activate_device(ZEDINFER_DEVICE_NVIDIA, 0)) {
        GTEST_SKIP() << "NVIDIA runtime unavailable; skipping SSMStatePool GPU test";
    }

    SSMStatePool pool(make_test_config(), ExecutorConfig(ZEDINFER_DEVICE_NVIDIA, 0, ZEDINFER_DTYPE_BF16));

    const int s1 = pool.acquire_slot();
    const int s2 = pool.acquire_slot();
    EXPECT_NE(s1, s2);
    EXPECT_EQ(pool.num_free_slots(), 0);

    // 3rd acquire must throw — capacity is max_concurrent=2.
    EXPECT_THROW({ (void)pool.acquire_slot(); }, std::runtime_error);

    pool.release_slot(s1);
    EXPECT_EQ(pool.num_free_slots(), 1);

    // After release, the next acquire should succeed (likely returning s1 via the
    // round-robin hint, but the test only requires capacity is restored).
    const int s3 = pool.acquire_slot();
    EXPECT_NE(s3, s2);
    EXPECT_EQ(pool.num_free_slots(), 0);
    pool.release_slot(s3);
    pool.release_slot(s2);
    EXPECT_EQ(pool.num_free_slots(), 2);
#endif
}

TEST(SSMStatePoolTest, ResetSlotIsKernelFreeAndDoesNotThrow) {
#if !defined(ENABLE_NVIDIA_API)
    GTEST_SKIP() << "NVIDIA support not enabled in this build";
#else
    if (!try_activate_device(ZEDINFER_DEVICE_NVIDIA, 0)) {
        GTEST_SKIP() << "NVIDIA runtime unavailable; skipping SSMStatePool GPU test";
    }

    SSMStatePool pool(make_test_config(), ExecutorConfig(ZEDINFER_DEVICE_NVIDIA, 0, ZEDINFER_DTYPE_BF16));
    const int s1 = pool.acquire_slot();
    EXPECT_NO_THROW(pool.reset_slot(s1));
    // Out-of-range slot is a no-op (silently ignored), matching release_slot.
    EXPECT_NO_THROW(pool.reset_slot(-1));
    EXPECT_NO_THROW(pool.reset_slot(999));
    pool.release_slot(s1);
#endif
}
