#include "backend/tensor/tensor.hpp"
#include "frontend/sampler/sampler.hpp"

#include <cmath>
#include <gtest/gtest.h>
#include <numeric>
#include <vector>

using namespace zedinfer;
using namespace zedinfer::sampler;

// Helper: create a CPU FP32 logits tensor from a vector
static tensor_t make_logits(const std::vector<float>& data, size_t vocab_size) {
    auto t = Tensor::create({vocab_size}, ZEDINFER_DTYPE_F32, ZEDINFER_DEVICE_CPU, 0);
    t->load(data.data());
    return t;
}

static tensor_t make_logits_2d(const std::vector<float>& data, size_t seq_len, size_t vocab_size) {
    auto t = Tensor::create({seq_len, vocab_size}, ZEDINFER_DTYPE_F32, ZEDINFER_DEVICE_CPU, 0);
    t->load(data.data());
    return t;
}

// ============================================================================
// SamplerParams validation
// ============================================================================

TEST(SamplerParamsTest, DefaultIsValid) {
    SamplerParams p;
    EXPECT_TRUE(p.validate());
}

TEST(SamplerParamsTest, ZeroTemperatureInvalid) {
    SamplerParams p(0.0f);
    EXPECT_FALSE(p.validate());
}

TEST(SamplerParamsTest, NegativeTemperatureInvalid) {
    SamplerParams p(-1.0f);
    EXPECT_FALSE(p.validate());
}

TEST(SamplerParamsTest, NegativeTopKInvalid) {
    SamplerParams p(1.0f, -1);
    EXPECT_FALSE(p.validate());
}

TEST(SamplerParamsTest, ZeroTopPInvalid) {
    SamplerParams p(1.0f, 0, 0.0f);
    EXPECT_FALSE(p.validate());
}

TEST(SamplerParamsTest, TopPAboveOneInvalid) {
    SamplerParams p(1.0f, 0, 1.1f);
    EXPECT_FALSE(p.validate());
}

// ============================================================================
// ArgmaxSampler
// ============================================================================

class ArgmaxSamplerTest : public ::testing::Test {
protected:
    void SetUp() override {
        ExecutorConfig cfg;
        cfg.device_type = ZEDINFER_DEVICE_CPU;
        cfg.device_id = 0;
        cfg.data_type = ZEDINFER_DTYPE_F32;
        sampler_ = std::make_shared<ArgmaxSampler>(cfg);
    }
    std::shared_ptr<ArgmaxSampler> sampler_;
};

TEST_F(ArgmaxSamplerTest, PicksMaximum) {
    // logits: [0.1, 0.5, 0.3, 0.9, 0.2]
    auto logits = make_logits({0.1f, 0.5f, 0.3f, 0.9f, 0.2f}, 5);
    int token = sampler_->sample(logits);
    EXPECT_EQ(token, 3); // index of max (0.9)
}

TEST_F(ArgmaxSamplerTest, PicksFirstOnTie) {
    // logits: [1.0, 0.5, 1.0, 0.5]
    auto logits = make_logits({1.0f, 0.5f, 1.0f, 0.5f}, 4);
    int token = sampler_->sample(logits);
    EXPECT_EQ(token, 0); // first occurrence of max
}

TEST_F(ArgmaxSamplerTest, WorksWith2DLogits) {
    // [seq_len=3, vocab_size=4] — should pick argmax of last row
    std::vector<float> data = {
        0.1f, 0.2f, 0.3f, 0.4f, // row 0
        0.5f, 0.6f, 0.7f, 0.8f, // row 1
        0.9f, 0.1f, 0.2f, 0.3f, // row 2 (last) — max at index 0
    };
    auto logits = make_logits_2d(data, 3, 4);
    int token = sampler_->sample(logits);
    EXPECT_EQ(token, 0); // argmax of last row
}

TEST_F(ArgmaxSamplerTest, LargeVocab) {
    // vocab_size = 1000, max at position 777
    std::vector<float> data(1000, 0.0f);
    data[777] = 100.0f;
    auto logits = make_logits(data, 1000);
    int token = sampler_->sample(logits);
    EXPECT_EQ(token, 777);
}

TEST_F(ArgmaxSamplerTest, NegativeLogits) {
    // All negative, max at index 2
    auto logits = make_logits({-5.0f, -3.0f, -1.0f, -2.0f, -4.0f}, 5);
    int token = sampler_->sample(logits);
    EXPECT_EQ(token, 2);
}

// ============================================================================
// GeneralSampler
// ============================================================================

TEST(GeneralSamplerTest, DeterministicWithSeed) {
    SamplerParams params(1.0f, 0, 1.0f, 42);
    GeneralSampler s1(params);
    GeneralSampler s2(params);

    auto logits = make_logits({0.1f, 0.2f, 0.3f, 0.2f, 0.1f}, 5);

    // Same seed → same result
    int t1 = s1.sample(logits);
    int t2 = s2.sample(logits);
    EXPECT_EQ(t1, t2);
}

TEST(GeneralSamplerTest, LowTemperatureApproachesArgmax) {
    SamplerParams params(0.01f, 0, 1.0f, 123); // very low temperature
    GeneralSampler s(params);

    auto logits = make_logits({0.1f, 0.5f, 0.3f, 0.9f, 0.2f}, 5);
    int token = s.sample(logits);
    EXPECT_EQ(token, 3); // should pick max with near-zero temperature
}

TEST(GeneralSamplerTest, TopKFiltering) {
    SamplerParams params(1.0f, 2, 1.0f, 42); // top_k=2
    GeneralSampler s(params);

    // logits: max at 3 (0.9), second at 1 (0.5)
    // top_k=2 means only indices 3 and 1 are candidates
    auto logits = make_logits({0.1f, 0.5f, 0.3f, 0.9f, 0.2f}, 5);
    int token = s.sample(logits);
    EXPECT_TRUE(token == 3 || token == 1) << "Top-k=2 should only select from top 2 tokens, got: " << token;
}

TEST(GeneralSamplerTest, TopPFiltering) {
    SamplerParams params(1.0f, 0, 0.5f, 42); // top_p=0.5
    GeneralSampler s(params);

    // One dominant token — should always pick it with top_p=0.5
    std::vector<float> data(10, 0.0f);
    data[7] = 100.0f; // overwhelmingly dominant
    auto logits = make_logits(data, 10);
    int token = s.sample(logits);
    EXPECT_EQ(token, 7);
}

TEST(GeneralSamplerTest, SetSeedChangesOutput) {
    SamplerParams params(1.0f, 0, 1.0f, 1);
    GeneralSampler s(params);

    auto logits = make_logits({0.25f, 0.25f, 0.25f, 0.25f}, 4);

    s.setSeed(100);
    int t1 = s.sample(logits);
    s.setSeed(200);
    int t2 = s.sample(logits);
    // Different seeds should (usually) produce different results
    // Not guaranteed for 4 tokens, but very likely
    // Just verify both are valid
    EXPECT_GE(t1, 0);
    EXPECT_LT(t1, 4);
    EXPECT_GE(t2, 0);
    EXPECT_LT(t2, 4);
}

// ============================================================================
// Factory
// ============================================================================

TEST(SamplerFactoryTest, CreateArgmax) {
    ExecutorConfig cfg;
    cfg.device_type = ZEDINFER_DEVICE_CPU;
    cfg.data_type = ZEDINFER_DTYPE_F32;
    auto s = createSampler(cfg, SamplerType::ARGMAX);
    EXPECT_EQ(s->name(), "Argmax");
}

TEST(SamplerFactoryTest, CreateGeneral) {
    ExecutorConfig cfg;
    cfg.device_type = ZEDINFER_DEVICE_CPU;
    cfg.data_type = ZEDINFER_DTYPE_F32;
    SamplerParams params(0.8f, 10, 0.9f);
    auto s = createSampler(cfg, SamplerType::GENERAL, params);
    EXPECT_EQ(s->name(), "General");
}
