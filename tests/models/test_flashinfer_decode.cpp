#include "backend/core/context/context.hpp"
#include "backend/ops/ops.hpp"
#include "backend/tensor/tensor.hpp"
#include "utils/types.hpp"

#include <cmath>
#include <gtest/gtest.h>
#include <numeric>
#include <random>
#include <vector>

namespace zedinfer::test {

namespace {

constexpr int kNHead = 32;
constexpr int kNKVHead = 8;
constexpr int kHeadDim = 128;
constexpr int kBlockSize = 16;

template <typename T> std::vector<T> make_random_low_precision(size_t count, uint32_t seed);

template <> std::vector<fp16_t> make_random_low_precision<fp16_t>(size_t count, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    std::vector<fp16_t> out(count);
    for (auto& v : out) { v = utils::cast<fp16_t>(dist(rng)); }
    return out;
}

template <> std::vector<bf16_t> make_random_low_precision<bf16_t>(size_t count, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    std::vector<bf16_t> out(count);
    for (auto& v : out) { v = utils::cast<bf16_t>(dist(rng)); }
    return out;
}

std::vector<int> make_i32_tensor_data(std::initializer_list<int> values) {
    return std::vector<int>(values);
}

std::vector<float> to_cpu_f32(tensor_t tensor) {
    auto cpu = tensor->to(ZEDINFER_DEVICE_CPU, 0);
    auto f32 = cpu->to(ZEDINFER_DTYPE_F32);
    const auto* ptr = reinterpret_cast<const float*>(f32->data());
    return std::vector<float>(ptr, ptr + f32->numel());
}

void load_random_tensor(tensor_t tensor, zedinferDataType_t dtype, uint32_t seed) {
    if (dtype == ZEDINFER_DTYPE_F16) {
        auto host = make_random_low_precision<fp16_t>(tensor->numel(), seed);
        tensor->load(host.data());
        return;
    }
    auto host = make_random_low_precision<bf16_t>(tensor->numel(), seed);
    tensor->load(host.data());
}

ops::AttentionConfig make_test_config(int nhead, int nkvhead, int head_dim, zedinferDataType_t dtype) {
    return ops::AttentionConfig{
        nhead,
        nkvhead,
        head_dim,
        1.0f / std::sqrt(static_cast<float>(head_dim)),
        kBlockSize,
        dtype,
        ZEDINFER_DEVICE_NVIDIA,
        0,
    };
}

ops::AttentionConfig make_test_config(zedinferDataType_t dtype) {
    return make_test_config(kNHead, kNKVHead, kHeadDim, dtype);
}

void expect_close(tensor_t legacy, tensor_t flash, float max_abs_limit, float mean_abs_limit) {
    const auto legacy_f32 = to_cpu_f32(legacy);
    const auto flash_f32 = to_cpu_f32(flash);

    ASSERT_EQ(legacy_f32.size(), flash_f32.size());

    float max_abs_diff = 0.0f;
    float mean_abs_diff = 0.0f;
    for (size_t i = 0; i < legacy_f32.size(); ++i) {
        const float diff = std::abs(legacy_f32[i] - flash_f32[i]);
        max_abs_diff = std::max(max_abs_diff, diff);
        mean_abs_diff += diff;
    }
    mean_abs_diff /= static_cast<float>(legacy_f32.size());

    EXPECT_LT(max_abs_diff, max_abs_limit) << "max_abs_diff=" << max_abs_diff;
    EXPECT_LT(mean_abs_diff, mean_abs_limit) << "mean_abs_diff=" << mean_abs_diff;
}

void run_single_decode_compare(zedinferDataType_t dtype) {
    constexpr int total_pages = 5;
    constexpr int active_pages = 4;
    constexpr int seq_len = 49;

    core::context().setDevice(ZEDINFER_DEVICE_NVIDIA, 0);

    const auto cfg = make_test_config(dtype);

    auto q = Tensor::create({1, kNHead, kHeadDim}, dtype, ZEDINFER_DEVICE_NVIDIA, 0);
    auto k_pool = Tensor::create({total_pages, kBlockSize, kNKVHead, kHeadDim}, dtype, ZEDINFER_DEVICE_NVIDIA, 0);
    auto v_pool = Tensor::create({total_pages, kBlockSize, kNKVHead, kHeadDim}, dtype, ZEDINFER_DEVICE_NVIDIA, 0);
    load_random_tensor(q, dtype, 7);
    load_random_tensor(k_pool, dtype, 11);
    load_random_tensor(v_pool, dtype, 19);

    const std::vector<int> page_table_host = {2, 0, 4, 1};
    auto page_table = Tensor::create({page_table_host.size()}, ZEDINFER_DTYPE_I32, ZEDINFER_DEVICE_NVIDIA, 0);
    page_table->load(page_table_host.data());

    auto legacy_out = Tensor::create({kNHead, kHeadDim}, dtype, ZEDINFER_DEVICE_NVIDIA, 0);
    ops::AttentionParams legacy{cfg};
    legacy.out = legacy_out;
    legacy.q = q->view({kNHead, kHeadDim});
    legacy.k_pool_base = k_pool->data();
    legacy.v_pool_base = v_pool->data();
    legacy.page_table = reinterpret_cast<const int*>(page_table->data());
    legacy.seq_len = seq_len;
    legacy.seqlen_q = 1;

    auto kv_indptr_host = make_i32_tensor_data({0, active_pages});
    auto kv_last_page_len_host = make_i32_tensor_data({1});
    auto kv_indptr = Tensor::create({kv_indptr_host.size()}, ZEDINFER_DTYPE_I32, ZEDINFER_DEVICE_NVIDIA, 0);
    auto kv_page_indices = Tensor::create({page_table_host.size()}, ZEDINFER_DTYPE_I32, ZEDINFER_DEVICE_NVIDIA, 0);
    auto kv_last_page_len
        = Tensor::create({kv_last_page_len_host.size()}, ZEDINFER_DTYPE_I32, ZEDINFER_DEVICE_NVIDIA, 0);
    kv_indptr->load(kv_indptr_host.data());
    kv_page_indices->load(page_table_host.data());
    kv_last_page_len->load(kv_last_page_len_host.data());

    auto flash_out = Tensor::create({1, kNHead, kHeadDim}, dtype, ZEDINFER_DEVICE_NVIDIA, 0);
    ops::AttentionParams flash{cfg};
    flash.use_flashinfer = true;
    flash.out = flash_out;
    flash.q = q;
    flash.k_pool_base = k_pool->data();
    flash.v_pool_base = v_pool->data();
    flash.kv_indptr = reinterpret_cast<const int*>(kv_indptr->data());
    flash.kv_page_indices = reinterpret_cast<const int*>(kv_page_indices->data());
    flash.kv_last_page_len = reinterpret_cast<const int*>(kv_last_page_len->data());
    flash.kv_indptr_host = kv_indptr_host.data();
    flash.kv_batch_size = 1;

    ops::attention(legacy);
    core::context().runtime().synchronize();
    ops::attention(flash);
    core::context().runtime().synchronize();

    expect_close(legacy_out, flash_out, 5e-2f, 5e-3f);
}

void run_single_prefill_compare(zedinferDataType_t dtype) {
    constexpr int total_pages = 6;
    constexpr int past_len = 17;
    constexpr int seqlen_q = 5;
    constexpr int active_pages = 2;

    core::context().setDevice(ZEDINFER_DEVICE_NVIDIA, 0);

    const auto cfg = make_test_config(dtype);

    auto q = Tensor::create({seqlen_q, kNHead, kHeadDim}, dtype, ZEDINFER_DEVICE_NVIDIA, 0);
    auto k_pool = Tensor::create({total_pages, kBlockSize, kNKVHead, kHeadDim}, dtype, ZEDINFER_DEVICE_NVIDIA, 0);
    auto v_pool = Tensor::create({total_pages, kBlockSize, kNKVHead, kHeadDim}, dtype, ZEDINFER_DEVICE_NVIDIA, 0);
    load_random_tensor(q, dtype, 23);
    load_random_tensor(k_pool, dtype, 29);
    load_random_tensor(v_pool, dtype, 31);

    const std::vector<int> page_table_host = {4, 1, 5};
    auto page_table = Tensor::create({page_table_host.size()}, ZEDINFER_DTYPE_I32, ZEDINFER_DEVICE_NVIDIA, 0);
    page_table->load(page_table_host.data());

    auto legacy_out = Tensor::create({seqlen_q, kNHead, kHeadDim}, dtype, ZEDINFER_DEVICE_NVIDIA, 0);
    ops::AttentionParams legacy{cfg};
    legacy.out = legacy_out;
    legacy.q = q;
    legacy.k_pool_base = k_pool->data();
    legacy.v_pool_base = v_pool->data();
    legacy.page_table = reinterpret_cast<const int*>(page_table->data());
    legacy.seqlen_q = seqlen_q;
    legacy.past_len = past_len;

    auto kv_indptr_host = make_i32_tensor_data({0, active_pages});
    auto kv_page_indices_host = make_i32_tensor_data({4, 1});
    auto kv_last_page_len_host = make_i32_tensor_data({6});
    auto qo_indptr_host = make_i32_tensor_data({0, seqlen_q});

    auto kv_indptr = Tensor::create({kv_indptr_host.size()}, ZEDINFER_DTYPE_I32, ZEDINFER_DEVICE_NVIDIA, 0);
    auto kv_page_indices = Tensor::create({kv_page_indices_host.size()}, ZEDINFER_DTYPE_I32, ZEDINFER_DEVICE_NVIDIA, 0);
    auto kv_last_page_len
        = Tensor::create({kv_last_page_len_host.size()}, ZEDINFER_DTYPE_I32, ZEDINFER_DEVICE_NVIDIA, 0);
    auto qo_indptr = Tensor::create({qo_indptr_host.size()}, ZEDINFER_DTYPE_I32, ZEDINFER_DEVICE_NVIDIA, 0);
    kv_indptr->load(kv_indptr_host.data());
    kv_page_indices->load(kv_page_indices_host.data());
    kv_last_page_len->load(kv_last_page_len_host.data());
    qo_indptr->load(qo_indptr_host.data());

    auto flash_out = Tensor::create({seqlen_q, kNHead, kHeadDim}, dtype, ZEDINFER_DEVICE_NVIDIA, 0);
    ops::AttentionParams flash{cfg};
    flash.use_flashinfer = true;
    flash.out = flash_out;
    flash.q = q;
    flash.k_pool_base = k_pool->data();
    flash.v_pool_base = v_pool->data();
    flash.kv_indptr = reinterpret_cast<const int*>(kv_indptr->data());
    flash.kv_page_indices = reinterpret_cast<const int*>(kv_page_indices->data());
    flash.kv_last_page_len = reinterpret_cast<const int*>(kv_last_page_len->data());
    flash.qo_indptr = reinterpret_cast<const int*>(qo_indptr->data());
    flash.kv_indptr_host = kv_indptr_host.data();
    flash.qo_indptr_host = qo_indptr_host.data();
    flash.kv_batch_size = 1;

    ops::attention(legacy);
    core::context().runtime().synchronize();
    ops::attention(flash);
    core::context().runtime().synchronize();

    expect_close(legacy_out, flash_out, 5e-2f, 5e-3f);
}

void run_single_decode_group_size_six_compare(zedinferDataType_t dtype) {
    constexpr int nhead = 12;
    constexpr int nkvhead = 2;
    constexpr int head_dim = 128;
    constexpr int total_pages = 5;
    constexpr int active_pages = 4;
    constexpr int seq_len = 49;

    core::context().setDevice(ZEDINFER_DEVICE_NVIDIA, 0);

    const auto cfg = make_test_config(nhead, nkvhead, head_dim, dtype);

    auto q = Tensor::create({1, nhead, head_dim}, dtype, ZEDINFER_DEVICE_NVIDIA, 0);
    auto k_pool = Tensor::create({total_pages, kBlockSize, nkvhead, head_dim}, dtype, ZEDINFER_DEVICE_NVIDIA, 0);
    auto v_pool = Tensor::create({total_pages, kBlockSize, nkvhead, head_dim}, dtype, ZEDINFER_DEVICE_NVIDIA, 0);
    load_random_tensor(q, dtype, 47);
    load_random_tensor(k_pool, dtype, 53);
    load_random_tensor(v_pool, dtype, 59);

    const std::vector<int> page_table_host = {2, 0, 4, 1};
    auto page_table = Tensor::create({page_table_host.size()}, ZEDINFER_DTYPE_I32, ZEDINFER_DEVICE_NVIDIA, 0);
    page_table->load(page_table_host.data());

    auto legacy_out = Tensor::create({nhead, head_dim}, dtype, ZEDINFER_DEVICE_NVIDIA, 0);
    ops::AttentionParams legacy{cfg};
    legacy.out = legacy_out;
    legacy.q = q->view({nhead, head_dim});
    legacy.k_pool_base = k_pool->data();
    legacy.v_pool_base = v_pool->data();
    legacy.page_table = reinterpret_cast<const int*>(page_table->data());
    legacy.seq_len = seq_len;
    legacy.seqlen_q = 1;

    auto kv_indptr_host = make_i32_tensor_data({0, active_pages});
    auto kv_last_page_len_host = make_i32_tensor_data({1});
    auto qo_indptr_host = make_i32_tensor_data({0, 1});
    auto kv_indptr = Tensor::create({kv_indptr_host.size()}, ZEDINFER_DTYPE_I32, ZEDINFER_DEVICE_NVIDIA, 0);
    auto kv_page_indices = Tensor::create({page_table_host.size()}, ZEDINFER_DTYPE_I32, ZEDINFER_DEVICE_NVIDIA, 0);
    auto kv_last_page_len
        = Tensor::create({kv_last_page_len_host.size()}, ZEDINFER_DTYPE_I32, ZEDINFER_DEVICE_NVIDIA, 0);
    auto qo_indptr = Tensor::create({qo_indptr_host.size()}, ZEDINFER_DTYPE_I32, ZEDINFER_DEVICE_NVIDIA, 0);
    kv_indptr->load(kv_indptr_host.data());
    kv_page_indices->load(page_table_host.data());
    kv_last_page_len->load(kv_last_page_len_host.data());
    qo_indptr->load(qo_indptr_host.data());

    auto flash_out = Tensor::create({1, nhead, head_dim}, dtype, ZEDINFER_DEVICE_NVIDIA, 0);
    ops::AttentionParams flash{cfg};
    flash.use_flashinfer = true;
    flash.out = flash_out;
    flash.q = q;
    flash.k_pool_base = k_pool->data();
    flash.v_pool_base = v_pool->data();
    flash.kv_indptr = reinterpret_cast<const int*>(kv_indptr->data());
    flash.kv_page_indices = reinterpret_cast<const int*>(kv_page_indices->data());
    flash.kv_last_page_len = reinterpret_cast<const int*>(kv_last_page_len->data());
    flash.qo_indptr = reinterpret_cast<const int*>(qo_indptr->data());
    flash.kv_indptr_host = kv_indptr_host.data();
    flash.qo_indptr_host = qo_indptr_host.data();
    flash.kv_batch_size = 1;

    ops::attention(legacy);
    core::context().runtime().synchronize();
    ops::attention(flash);
    core::context().runtime().synchronize();

    expect_close(legacy_out, flash_out, 5e-2f, 5e-3f);
}

void run_batched_decode_group_size_six_compare(zedinferDataType_t dtype) {
    constexpr int nhead = 12;
    constexpr int nkvhead = 2;
    constexpr int head_dim = 128;
    constexpr int total_pages = 8;
    constexpr int num_requests = 2;
    constexpr int max_pages_per_seq = 3;

    core::context().setDevice(ZEDINFER_DEVICE_NVIDIA, 0);

    const auto cfg = make_test_config(nhead, nkvhead, head_dim, dtype);

    auto q = Tensor::create({num_requests, nhead, head_dim}, dtype, ZEDINFER_DEVICE_NVIDIA, 0);
    auto k_pool = Tensor::create({total_pages, kBlockSize, nkvhead, head_dim}, dtype, ZEDINFER_DEVICE_NVIDIA, 0);
    auto v_pool = Tensor::create({total_pages, kBlockSize, nkvhead, head_dim}, dtype, ZEDINFER_DEVICE_NVIDIA, 0);
    load_random_tensor(q, dtype, 61);
    load_random_tensor(k_pool, dtype, 67);
    load_random_tensor(v_pool, dtype, 71);

    const std::vector<int> batched_page_tables_host = {
        5, 0, 0, 1, 7, 4,
    };
    const std::vector<int> seq_lens_host = {12, 34};
    auto batched_page_tables
        = Tensor::create({batched_page_tables_host.size()}, ZEDINFER_DTYPE_I32, ZEDINFER_DEVICE_NVIDIA, 0);
    auto seq_lens = Tensor::create({seq_lens_host.size()}, ZEDINFER_DTYPE_I32, ZEDINFER_DEVICE_NVIDIA, 0);
    batched_page_tables->load(batched_page_tables_host.data());
    seq_lens->load(seq_lens_host.data());

    auto legacy_out = Tensor::create({num_requests, nhead, head_dim}, dtype, ZEDINFER_DEVICE_NVIDIA, 0);
    ops::AttentionParams legacy{cfg};
    legacy.out = legacy_out;
    legacy.q = q;
    legacy.k_pool_base = k_pool->data();
    legacy.v_pool_base = v_pool->data();
    legacy.batched_page_tables = reinterpret_cast<const int*>(batched_page_tables->data());
    legacy.batched_seq_lens = reinterpret_cast<const int*>(seq_lens->data());
    legacy.num_requests = num_requests;
    legacy.max_blocks_per_seq = max_pages_per_seq;

    auto kv_indptr_host = make_i32_tensor_data({0, 1, 4});
    auto kv_page_indices_host = make_i32_tensor_data({5, 1, 7, 4});
    auto kv_last_page_len_host = make_i32_tensor_data({12, 2});
    auto qo_indptr_host = make_i32_tensor_data({0, 1, 2});
    auto kv_indptr = Tensor::create({kv_indptr_host.size()}, ZEDINFER_DTYPE_I32, ZEDINFER_DEVICE_NVIDIA, 0);
    auto kv_page_indices = Tensor::create({kv_page_indices_host.size()}, ZEDINFER_DTYPE_I32, ZEDINFER_DEVICE_NVIDIA, 0);
    auto kv_last_page_len
        = Tensor::create({kv_last_page_len_host.size()}, ZEDINFER_DTYPE_I32, ZEDINFER_DEVICE_NVIDIA, 0);
    auto qo_indptr = Tensor::create({qo_indptr_host.size()}, ZEDINFER_DTYPE_I32, ZEDINFER_DEVICE_NVIDIA, 0);
    kv_indptr->load(kv_indptr_host.data());
    kv_page_indices->load(kv_page_indices_host.data());
    kv_last_page_len->load(kv_last_page_len_host.data());
    qo_indptr->load(qo_indptr_host.data());

    auto flash_out = Tensor::create({num_requests, nhead, head_dim}, dtype, ZEDINFER_DEVICE_NVIDIA, 0);
    ops::AttentionParams flash{cfg};
    flash.use_flashinfer = true;
    flash.out = flash_out;
    flash.q = q;
    flash.k_pool_base = k_pool->data();
    flash.v_pool_base = v_pool->data();
    flash.kv_indptr = reinterpret_cast<const int*>(kv_indptr->data());
    flash.kv_page_indices = reinterpret_cast<const int*>(kv_page_indices->data());
    flash.kv_last_page_len = reinterpret_cast<const int*>(kv_last_page_len->data());
    flash.qo_indptr = reinterpret_cast<const int*>(qo_indptr->data());
    flash.kv_indptr_host = kv_indptr_host.data();
    flash.qo_indptr_host = qo_indptr_host.data();
    flash.kv_batch_size = num_requests;

    ops::attention(legacy);
    core::context().runtime().synchronize();
    ops::attention(flash);
    core::context().runtime().synchronize();

    expect_close(legacy_out, flash_out, 5e-2f, 5e-3f);
}

void run_batched_prefill_compare(zedinferDataType_t dtype) {
    constexpr int total_pages = 8;
    constexpr int past_len_a = 11;
    constexpr int seqlen_q_a = 4;
    constexpr int past_len_b = 19;
    constexpr int seqlen_q_b = 3;
    constexpr int total_q = seqlen_q_a + seqlen_q_b;

    core::context().setDevice(ZEDINFER_DEVICE_NVIDIA, 0);

    const auto cfg = make_test_config(dtype);

    auto q = Tensor::create({total_q, kNHead, kHeadDim}, dtype, ZEDINFER_DEVICE_NVIDIA, 0);
    auto k_pool = Tensor::create({total_pages, kBlockSize, kNKVHead, kHeadDim}, dtype, ZEDINFER_DEVICE_NVIDIA, 0);
    auto v_pool = Tensor::create({total_pages, kBlockSize, kNKVHead, kHeadDim}, dtype, ZEDINFER_DEVICE_NVIDIA, 0);
    load_random_tensor(q, dtype, 37);
    load_random_tensor(k_pool, dtype, 41);
    load_random_tensor(v_pool, dtype, 43);

    const std::vector<int> page_table_a_host = {5, 2};
    const std::vector<int> page_table_b_host = {1, 7, 4};
    auto page_table_a = Tensor::create({page_table_a_host.size()}, ZEDINFER_DTYPE_I32, ZEDINFER_DEVICE_NVIDIA, 0);
    auto page_table_b = Tensor::create({page_table_b_host.size()}, ZEDINFER_DTYPE_I32, ZEDINFER_DEVICE_NVIDIA, 0);
    page_table_a->load(page_table_a_host.data());
    page_table_b->load(page_table_b_host.data());

    auto legacy_out = Tensor::create({total_q, kNHead, kHeadDim}, dtype, ZEDINFER_DEVICE_NVIDIA, 0);

    ops::AttentionParams legacy_a{cfg};
    legacy_a.out = legacy_out->slice(0, 0, seqlen_q_a);
    legacy_a.q = q->slice(0, 0, seqlen_q_a);
    legacy_a.k_pool_base = k_pool->data();
    legacy_a.v_pool_base = v_pool->data();
    legacy_a.page_table = reinterpret_cast<const int*>(page_table_a->data());
    legacy_a.seqlen_q = seqlen_q_a;
    legacy_a.past_len = past_len_a;

    ops::AttentionParams legacy_b{cfg};
    legacy_b.out = legacy_out->slice(0, seqlen_q_a, total_q);
    legacy_b.q = q->slice(0, seqlen_q_a, total_q);
    legacy_b.k_pool_base = k_pool->data();
    legacy_b.v_pool_base = v_pool->data();
    legacy_b.page_table = reinterpret_cast<const int*>(page_table_b->data());
    legacy_b.seqlen_q = seqlen_q_b;
    legacy_b.past_len = past_len_b;

    auto kv_indptr_host = make_i32_tensor_data({0, 1, 3});
    auto kv_page_indices_host = make_i32_tensor_data({5, 1, 7});
    auto kv_last_page_len_host = make_i32_tensor_data({15, 6});
    auto qo_indptr_host = make_i32_tensor_data({0, seqlen_q_a, total_q});

    auto kv_indptr = Tensor::create({kv_indptr_host.size()}, ZEDINFER_DTYPE_I32, ZEDINFER_DEVICE_NVIDIA, 0);
    auto kv_page_indices = Tensor::create({kv_page_indices_host.size()}, ZEDINFER_DTYPE_I32, ZEDINFER_DEVICE_NVIDIA, 0);
    auto kv_last_page_len
        = Tensor::create({kv_last_page_len_host.size()}, ZEDINFER_DTYPE_I32, ZEDINFER_DEVICE_NVIDIA, 0);
    auto qo_indptr = Tensor::create({qo_indptr_host.size()}, ZEDINFER_DTYPE_I32, ZEDINFER_DEVICE_NVIDIA, 0);
    kv_indptr->load(kv_indptr_host.data());
    kv_page_indices->load(kv_page_indices_host.data());
    kv_last_page_len->load(kv_last_page_len_host.data());
    qo_indptr->load(qo_indptr_host.data());

    auto flash_out = Tensor::create({total_q, kNHead, kHeadDim}, dtype, ZEDINFER_DEVICE_NVIDIA, 0);
    ops::AttentionParams flash{cfg};
    flash.use_flashinfer = true;
    flash.out = flash_out;
    flash.q = q;
    flash.k_pool_base = k_pool->data();
    flash.v_pool_base = v_pool->data();
    flash.kv_indptr = reinterpret_cast<const int*>(kv_indptr->data());
    flash.kv_page_indices = reinterpret_cast<const int*>(kv_page_indices->data());
    flash.kv_last_page_len = reinterpret_cast<const int*>(kv_last_page_len->data());
    flash.qo_indptr = reinterpret_cast<const int*>(qo_indptr->data());
    flash.kv_indptr_host = kv_indptr_host.data();
    flash.qo_indptr_host = qo_indptr_host.data();
    flash.kv_batch_size = 2;

    ops::attention(legacy_a);
    ops::attention(legacy_b);
    core::context().runtime().synchronize();
    ops::attention(flash);
    core::context().runtime().synchronize();

    expect_close(legacy_out, flash_out, 5e-2f, 5e-3f);
}

} // namespace

TEST(FlashInferDecodeTest, SingleDecodeFp16MatchesLegacyPagedAttention) {
#if !defined(ENABLE_NVIDIA_API) || !defined(USE_FLASHINFER)
    GTEST_SKIP() << "FlashInfer/NVIDIA support not enabled in this build";
#else
    run_single_decode_compare(ZEDINFER_DTYPE_F16);
#endif
}

TEST(FlashInferDecodeTest, SingleDecodeBf16MatchesLegacyPagedAttention) {
#if !defined(ENABLE_NVIDIA_API) || !defined(USE_FLASHINFER)
    GTEST_SKIP() << "FlashInfer/NVIDIA support not enabled in this build";
#else
    run_single_decode_compare(ZEDINFER_DTYPE_BF16);
#endif
}

TEST(FlashInferDecodeTest, SingleDecodeGroupSizeSixFallsBackToPrefillKernelFp16) {
#if !defined(ENABLE_NVIDIA_API) || !defined(USE_FLASHINFER)
    GTEST_SKIP() << "FlashInfer/NVIDIA support not enabled in this build";
#else
    run_single_decode_group_size_six_compare(ZEDINFER_DTYPE_F16);
#endif
}

TEST(FlashInferDecodeTest, BatchedDecodeGroupSizeSixFallsBackToPrefillKernelFp16) {
#if !defined(ENABLE_NVIDIA_API) || !defined(USE_FLASHINFER)
    GTEST_SKIP() << "FlashInfer/NVIDIA support not enabled in this build";
#else
    run_batched_decode_group_size_six_compare(ZEDINFER_DTYPE_F16);
#endif
}

TEST(FlashInferPrefillTest, SinglePrefillFp16MatchesLegacyPagedAttention) {
#if !defined(ENABLE_NVIDIA_API) || !defined(USE_FLASHINFER)
    GTEST_SKIP() << "FlashInfer/NVIDIA support not enabled in this build";
#else
    run_single_prefill_compare(ZEDINFER_DTYPE_F16);
#endif
}

TEST(FlashInferPrefillTest, SinglePrefillBf16MatchesLegacyPagedAttention) {
#if !defined(ENABLE_NVIDIA_API) || !defined(USE_FLASHINFER)
    GTEST_SKIP() << "FlashInfer/NVIDIA support not enabled in this build";
#else
    run_single_prefill_compare(ZEDINFER_DTYPE_BF16);
#endif
}

TEST(FlashInferPrefillTest, BatchedPrefillFp16MatchesLegacyPagedAttention) {
#if !defined(ENABLE_NVIDIA_API) || !defined(USE_FLASHINFER)
    GTEST_SKIP() << "FlashInfer/NVIDIA support not enabled in this build";
#else
    run_batched_prefill_compare(ZEDINFER_DTYPE_F16);
#endif
}

TEST(FlashInferPrefillTest, BatchedPrefillBf16MatchesLegacyPagedAttention) {
#if !defined(ENABLE_NVIDIA_API) || !defined(USE_FLASHINFER)
    GTEST_SKIP() << "FlashInfer/NVIDIA support not enabled in this build";
#else
    run_batched_prefill_compare(ZEDINFER_DTYPE_BF16);
#endif
}

} // namespace zedinfer::test
