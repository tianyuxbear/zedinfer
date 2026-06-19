// Fixture-driven correctness test for ops::mamba::gdn.
//
// Loads three pre-generated binary fixtures (decode_n1, prefill_n4, prefill_n12),
// copies inputs to device, runs the GDN kernel, and compares output y and final
// state S_T against the PyTorch reference values stored in each fixture.
//
// Fixture format (little-endian):
//   Header: 6 x int32  {N, Hv, Hk, Dv, Dk, _reserved}
//   Tensors (bf16):     q[N,Hk,Dk], k[N,Hk,Dk], v[N,Hv,Dv], b[N,Hv], a[N,Hv]
//   Tensors (f32):      A_log[Hv]
//   Tensors (bf16):     dt_bias[Hv]
//   Tensors (f32):      S0[Hv,Dv,Dk]
//   Tensors (bf16):     y_expected[N,Hv,Dv]
//   Tensors (f32):      S_T_expected[Hv,Dv,Dk]

#include "backend/core/context/context.hpp"
#include "backend/device/device.hpp"
#include "backend/ops/mamba/gdn.hpp"
#include "backend/tensor/tensor.hpp"
#include "frontend/models/ssm_state_pool.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <gtest/gtest.h>
#include <stdexcept>
#include <string>
#include <vector>

using namespace zedinfer;
using zedinfer::core::context;

namespace {

struct Fixture {
    int N, Hv, Hk, Dv, Dk;
    std::vector<uint16_t> q, k, v, b, a;
    std::vector<float> A_log;
    std::vector<uint16_t> dt_bias;
    std::vector<float> S0;
    std::vector<uint16_t> y_expected;
    std::vector<float> S_T_expected;
};

template <typename T> static void read_block(std::ifstream& f, std::vector<T>& vec, size_t n) {
    vec.resize(n);
    f.read(reinterpret_cast<char*>(vec.data()), n * sizeof(T));
    if (!f) {
        throw std::runtime_error("truncated fixture");
    }
}

static Fixture load(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        throw std::runtime_error("fixture not found: " + path);
    }
    Fixture fx;
    int32_t hdr[6]{};
    f.read(reinterpret_cast<char*>(hdr), sizeof(hdr));
    fx.N = hdr[0];
    fx.Hv = hdr[1];
    fx.Hk = hdr[2];
    fx.Dv = hdr[3];
    fx.Dk = hdr[4];
    read_block(f, fx.q, (size_t)fx.N * fx.Hk * fx.Dk);
    read_block(f, fx.k, (size_t)fx.N * fx.Hk * fx.Dk);
    read_block(f, fx.v, (size_t)fx.N * fx.Hv * fx.Dv);
    read_block(f, fx.b, (size_t)fx.N * fx.Hv);
    read_block(f, fx.a, (size_t)fx.N * fx.Hv);
    read_block(f, fx.A_log, (size_t)fx.Hv);
    read_block(f, fx.dt_bias, (size_t)fx.Hv);
    read_block(f, fx.S0, (size_t)fx.Hv * fx.Dv * fx.Dk);
    read_block(f, fx.y_expected, (size_t)fx.N * fx.Hv * fx.Dv);
    read_block(f, fx.S_T_expected, (size_t)fx.Hv * fx.Dv * fx.Dk);
    return fx;
}

static float bf16_to_f32(uint16_t bits) {
    uint32_t u = static_cast<uint32_t>(bits) << 16;
    float f;
    std::memcpy(&f, &u, sizeof(f));
    return f;
}

class GDNTest : public ::testing::Test {
protected:
    void SetUp() override {
        try {
            context().setDevice(ZEDINFER_DEVICE_NVIDIA, 0);
        } catch (const std::exception& e) { GTEST_SKIP() << "NVIDIA runtime init failed: " << e.what(); }
    }
};

static void run(const std::string& fixture_path) {
    auto fx = load(fixture_path);
    context().setDevice(ZEDINFER_DEVICE_NVIDIA, 0);
    auto* api = zedinfer::device::getRuntimeAPI(ZEDINFER_DEVICE_NVIDIA);
    auto dev = ZEDINFER_DEVICE_NVIDIA;

    // Upload inputs to device.
    auto to_dev_bf16 = [&](const std::vector<uint16_t>& host, std::vector<size_t> shape) -> tensor_t {
        auto t = Tensor::create(shape, ZEDINFER_DTYPE_BF16, dev, 0);
        api->memcpy_sync(t->data(), host.data(), host.size() * sizeof(uint16_t), ZEDINFER_MEMCPY_H2D);
        return t;
    };
    auto to_dev_f32 = [&](const std::vector<float>& host, std::vector<size_t> shape) -> tensor_t {
        auto t = Tensor::create(shape, ZEDINFER_DTYPE_F32, dev, 0);
        api->memcpy_sync(t->data(), host.data(), host.size() * sizeof(float), ZEDINFER_MEMCPY_H2D);
        return t;
    };

    auto q_t = to_dev_bf16(fx.q, {(size_t)fx.N, (size_t)fx.Hk * fx.Dk});
    auto k_t = to_dev_bf16(fx.k, {(size_t)fx.N, (size_t)fx.Hk * fx.Dk});
    auto v_t = to_dev_bf16(fx.v, {(size_t)fx.N, (size_t)fx.Hv * fx.Dv});
    auto b_t = to_dev_bf16(fx.b, {(size_t)fx.N, (size_t)fx.Hv});
    auto a_t = to_dev_bf16(fx.a, {(size_t)fx.N, (size_t)fx.Hv});
    auto Al_t = to_dev_f32(fx.A_log, {(size_t)fx.Hv});
    auto dtb_t = to_dev_bf16(fx.dt_bias, {(size_t)fx.Hv});

    // State tensor: S0 uploaded as fp32; kernel will update it in-place.
    auto state_t = to_dev_f32(fx.S0, {(size_t)fx.Hv, (size_t)fx.Dv, (size_t)fx.Dk});
    auto out_t = Tensor::create({(size_t)fx.N, (size_t)fx.Hv * fx.Dv}, ZEDINFER_DTYPE_BF16, dev, 0);

    // Construct a minimal SSMStateView that points directly at our state tensor
    // (single slot, single layer — strides and slot/layer offsets are zero).
    ops::mamba::GDNParams p;
    p.state_view.ssm_base = state_t->data();
    p.state_view.ssm_stride_slot = 0;
    p.state_view.ssm_stride_layer = 0;
    p.state_view.num_v_heads = fx.Hv;
    p.state_view.value_head_dim = fx.Dv;
    p.state_view.d_state = fx.Dk;
    p.slot_idx = 0;
    p.layer_idx = 0;
    p.q = q_t;
    p.k = k_t;
    p.v = v_t;
    p.b = b_t;
    p.a = a_t;
    p.A_log = Al_t;
    p.dt_bias = dtb_t;
    p.out = out_t;
    p.num_tokens = fx.N;

    ASSERT_NO_THROW(ops::mamba::gdn(p));
    context().runtime().synchronize();

    // --- Compare output y against y_expected ---
    std::vector<uint16_t> out_host((size_t)fx.N * fx.Hv * fx.Dv);
    api->memcpy_sync(out_host.data(), out_t->data(), out_host.size() * sizeof(uint16_t), ZEDINFER_MEMCPY_D2H);

    int bad_y = 0;
    int first_bad = -1;
    float first_got = 0.f, first_exp_y = 0.f;
    for (size_t i = 0; i < out_host.size(); ++i) {
        float got = bf16_to_f32(out_host[i]);
        float exp = bf16_to_f32(fx.y_expected[i]);
        float tol = std::max(1e-2f, 1e-2f * std::abs(exp));
        if (std::abs(got - exp) > tol) {
            if (first_bad < 0) {
                first_bad = (int)i;
                first_got = got;
                first_exp_y = exp;
            }
            ++bad_y;
        }
    }
    // Allow < 0.5 % outliers (rounding variance of bf16 ops on different hardware).
    const int y_limit = (int)(out_host.size() / 200);
    EXPECT_LT(bad_y, y_limit) << "fixture=" << fixture_path << " y mismatched=" << bad_y << "/" << out_host.size()
                              << " (limit=" << y_limit << ")"
                              << (first_bad >= 0 ? " first_bad_idx=" + std::to_string(first_bad)
                                                       + " got=" + std::to_string(first_got)
                                                       + " exp=" + std::to_string(first_exp_y)
                                                 : "");

    // --- Compare final state S_T against S_T_expected ---
    std::vector<float> S_host((size_t)fx.Hv * fx.Dv * fx.Dk);
    api->memcpy_sync(S_host.data(), state_t->data(), S_host.size() * sizeof(float), ZEDINFER_MEMCPY_D2H);

    int bad_s = 0;
    int first_bad_s = -1;
    float first_got_s = 0.f, first_exp_s = 0.f;
    for (size_t i = 0; i < S_host.size(); ++i) {
        float tol = std::max(1e-4f, 1e-4f * std::abs(fx.S_T_expected[i]));
        if (std::abs(S_host[i] - fx.S_T_expected[i]) > tol) {
            if (first_bad_s < 0) {
                first_bad_s = (int)i;
                first_got_s = S_host[i];
                first_exp_s = fx.S_T_expected[i];
            }
            ++bad_s;
        }
    }
    const int s_limit = (int)(S_host.size() / 1000);
    EXPECT_LT(bad_s, s_limit) << "fixture=" << fixture_path << " state mismatched=" << bad_s << "/" << S_host.size()
                              << " (limit=" << s_limit << ")"
                              << (first_bad_s >= 0 ? " first_bad_idx=" + std::to_string(first_bad_s)
                                                         + " got=" + std::to_string(first_got_s)
                                                         + " exp=" + std::to_string(first_exp_s)
                                                   : "");
}

static void run_prefill_decode_consistency(const std::string& fixture_path) {
    auto fx = load(fixture_path);
    context().setDevice(ZEDINFER_DEVICE_NVIDIA, 0);
    auto* api = zedinfer::device::getRuntimeAPI(ZEDINFER_DEVICE_NVIDIA);
    auto dev = ZEDINFER_DEVICE_NVIDIA;

    auto to_dev_bf16 = [&](const std::vector<uint16_t>& host, std::vector<size_t> shape) -> tensor_t {
        auto t = Tensor::create(shape, ZEDINFER_DTYPE_BF16, dev, 0);
        api->memcpy_sync(t->data(), host.data(), host.size() * sizeof(uint16_t), ZEDINFER_MEMCPY_H2D);
        return t;
    };
    auto to_dev_f32 = [&](const std::vector<float>& host, std::vector<size_t> shape) -> tensor_t {
        auto t = Tensor::create(shape, ZEDINFER_DTYPE_F32, dev, 0);
        api->memcpy_sync(t->data(), host.data(), host.size() * sizeof(float), ZEDINFER_MEMCPY_H2D);
        return t;
    };

    auto q_t = to_dev_bf16(fx.q, {(size_t)fx.N, (size_t)fx.Hk * fx.Dk});
    auto k_t = to_dev_bf16(fx.k, {(size_t)fx.N, (size_t)fx.Hk * fx.Dk});
    auto v_t = to_dev_bf16(fx.v, {(size_t)fx.N, (size_t)fx.Hv * fx.Dv});
    auto b_t = to_dev_bf16(fx.b, {(size_t)fx.N, (size_t)fx.Hv});
    auto a_t = to_dev_bf16(fx.a, {(size_t)fx.N, (size_t)fx.Hv});
    auto Al_t = to_dev_f32(fx.A_log, {(size_t)fx.Hv});
    auto dtb_t = to_dev_bf16(fx.dt_bias, {(size_t)fx.Hv});

    auto state_prefill = to_dev_f32(fx.S0, {(size_t)fx.Hv, (size_t)fx.Dv, (size_t)fx.Dk});
    auto state_decode = to_dev_f32(fx.S0, {(size_t)fx.Hv, (size_t)fx.Dv, (size_t)fx.Dk});
    auto out_prefill = Tensor::create({(size_t)fx.N, (size_t)fx.Hv * fx.Dv}, ZEDINFER_DTYPE_BF16, dev, 0);
    auto out_decode = Tensor::create({(size_t)fx.N, (size_t)fx.Hv * fx.Dv}, ZEDINFER_DTYPE_BF16, dev, 0);

    auto fill_params = [&](ops::mamba::GDNParams& p, tensor_t state, tensor_t out, int num_tokens) {
        p.state_view.ssm_base = state->data();
        p.state_view.ssm_stride_slot = 0;
        p.state_view.ssm_stride_layer = 0;
        p.state_view.num_v_heads = fx.Hv;
        p.state_view.value_head_dim = fx.Dv;
        p.state_view.d_state = fx.Dk;
        p.slot_idx = 0;
        p.layer_idx = 0;
        p.A_log = Al_t;
        p.dt_bias = dtb_t;
        p.out = out;
        p.num_tokens = num_tokens;
    };

    ops::mamba::GDNParams p_prefill;
    fill_params(p_prefill, state_prefill, out_prefill, fx.N);
    p_prefill.q = q_t;
    p_prefill.k = k_t;
    p_prefill.v = v_t;
    p_prefill.b = b_t;
    p_prefill.a = a_t;
    ASSERT_NO_THROW(ops::mamba::gdn(p_prefill));

    for (int t = 0; t < fx.N; ++t) {
        ops::mamba::GDNParams p_decode;
        fill_params(p_decode, state_decode, out_decode->slice(0, (size_t)t, (size_t)t + 1), 1);
        p_decode.q = q_t->slice(0, (size_t)t, (size_t)t + 1);
        p_decode.k = k_t->slice(0, (size_t)t, (size_t)t + 1);
        p_decode.v = v_t->slice(0, (size_t)t, (size_t)t + 1);
        p_decode.b = b_t->slice(0, (size_t)t, (size_t)t + 1);
        p_decode.a = a_t->slice(0, (size_t)t, (size_t)t + 1);
        ASSERT_NO_THROW(ops::mamba::gdn(p_decode)) << "decode step=" << t;
    }
    context().runtime().synchronize();

    std::vector<uint16_t> y_prefill((size_t)fx.N * fx.Hv * fx.Dv);
    std::vector<uint16_t> y_decode(y_prefill.size());
    api->memcpy_sync(y_prefill.data(), out_prefill->data(), y_prefill.size() * sizeof(uint16_t), ZEDINFER_MEMCPY_D2H);
    api->memcpy_sync(y_decode.data(), out_decode->data(), y_decode.size() * sizeof(uint16_t), ZEDINFER_MEMCPY_D2H);

    int bad_y = 0;
    int first_bad_y = -1;
    float first_prefill_y = 0.f, first_decode_y = 0.f;
    for (size_t i = 0; i < y_prefill.size(); ++i) {
        float a = bf16_to_f32(y_prefill[i]);
        float b = bf16_to_f32(y_decode[i]);
        float tol = std::max(1e-2f, 1e-2f * std::max(std::abs(a), std::abs(b)));
        if (std::abs(a - b) > tol) {
            if (first_bad_y < 0) {
                first_bad_y = (int)i;
                first_prefill_y = a;
                first_decode_y = b;
            }
            ++bad_y;
        }
    }
    EXPECT_EQ(bad_y, 0) << "fixture=" << fixture_path << " prefill/decode y mismatched=" << bad_y << "/"
                        << y_prefill.size() << " first_bad_idx=" << first_bad_y << " prefill=" << first_prefill_y
                        << " decode=" << first_decode_y;

    std::vector<float> S_prefill((size_t)fx.Hv * fx.Dv * fx.Dk);
    std::vector<float> S_decode(S_prefill.size());
    api->memcpy_sync(S_prefill.data(), state_prefill->data(), S_prefill.size() * sizeof(float), ZEDINFER_MEMCPY_D2H);
    api->memcpy_sync(S_decode.data(), state_decode->data(), S_decode.size() * sizeof(float), ZEDINFER_MEMCPY_D2H);

    int bad_s = 0;
    int first_bad_s = -1;
    float first_prefill_s = 0.f, first_decode_s = 0.f;
    for (size_t i = 0; i < S_prefill.size(); ++i) {
        float tol = std::max(1e-4f, 1e-4f * std::max(std::abs(S_prefill[i]), std::abs(S_decode[i])));
        if (std::abs(S_prefill[i] - S_decode[i]) > tol) {
            if (first_bad_s < 0) {
                first_bad_s = (int)i;
                first_prefill_s = S_prefill[i];
                first_decode_s = S_decode[i];
            }
            ++bad_s;
        }
    }
    EXPECT_EQ(bad_s, 0) << "fixture=" << fixture_path << " prefill/decode state mismatched=" << bad_s << "/"
                        << S_prefill.size() << " first_bad_idx=" << first_bad_s << " prefill=" << first_prefill_s
                        << " decode=" << first_decode_s;
}

TEST_F(GDNTest, DecodeN1) {
    run("tests/data/gdn_fixtures/decode_n1.bin");
}
TEST_F(GDNTest, PrefillN4) {
    run("tests/data/gdn_fixtures/prefill_n4.bin");
}
TEST_F(GDNTest, PrefillN12) {
    run("tests/data/gdn_fixtures/prefill_n12.bin");
}
TEST_F(GDNTest, PrefillMatchesSequentialDecode) {
    run_prefill_decode_consistency("tests/data/gdn_fixtures/prefill_n12.bin");
}

} // namespace

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
