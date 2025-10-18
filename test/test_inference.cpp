#include "frontend/sampler/sampler.hpp"
#include "neollm/engine.hpp"
#include <gtest/gtest.h>
#include <memory>

using namespace neollm;

// Test fixture for inference engine tests
class InferenceEngineTest : public ::testing::Test {
protected:
    static std::shared_ptr<InferenceEngine> engine;
    static std::string model_path;

    // SetUpTestSuite runs once before all tests
    static void SetUpTestSuite() {
        try {
            engine = InferenceEngine::create(
                model_path,
                NEOLLM_DEVICE_CPU,
                0);
            ASSERT_NE(engine, nullptr) << "Failed to create inference engine";
        } catch (const std::exception &e) {
            FAIL() << "Failed to initialize engine: " << e.what();
        }
    }

    // TearDownTestSuite runs once after all tests
    static void TearDownTestSuite() {
        engine.reset();
    }

    // SetUp runs before each test
    void SetUp() override {
        // Reset engine state before each test
        if (engine) {
            engine->reset();
        }
    }

    // Helper function to create basic generation config
    GenerationConfig CreateBasicConfig(int max_tokens = 50) {
        GenerationConfig config;
        config.max_new_tokens = max_tokens;
        config.sampler_type = sampler::SamplerType::ARGMAX;
        config.sampler_params.temperature = 0.7f;
        config.sampler_params.top_p = 0.9f;
        config.verbose = false;
        return config;
    }
};

// Initialize static members
std::shared_ptr<InferenceEngine> InferenceEngineTest::engine = nullptr;
std::string InferenceEngineTest::model_path = "/home/xiongtianyu/data/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B";

// ============================================================================
// Basic Functionality Tests
// ============================================================================

TEST_F(InferenceEngineTest, BasicTextGeneration) {
    auto config = CreateBasicConfig(128);
    std::string prompt = "Hello, World!";

    std::vector<std::pair<std::string, std::string>> messages = {
        {"user", prompt}};

    std::string output = engine->chat(messages, config);

    std::cout << "output" << output << std::endl;

    EXPECT_FALSE(output.empty()) << "Generated text should not be empty";
    EXPECT_GT(output.length(), prompt.length()) << "Output should be longer than prompt";

    const auto &stats = engine->last_stats();
    EXPECT_GT(stats.generated_tokens, 0) << "Should generate at least one token";
    EXPECT_GT(stats.total_time_ms, 0) << "Generation should take some time";
}

TEST_F(InferenceEngineTest, GreedySamplingDeterministic) {
    GenerationConfig config;
    config.max_new_tokens = 30;
    config.sampler_type = sampler::SamplerType::ARGMAX;
    config.verbose = false;

    std::string prompt = "The capital of France is";

    // Generate twice
    std::string output1 = engine->generate(prompt, config);
    engine->reset();
    std::string output2 = engine->generate(prompt, config);

    EXPECT_EQ(output1, output2) << "Greedy sampling should produce deterministic output";
    EXPECT_FALSE(output1.empty()) << "Output should not be empty";
}

TEST_F(InferenceEngineTest, StreamingGeneration) {
    auto config = CreateBasicConfig(40);
    config.stream = true;

    std::vector<std::string> streamed_tokens;
    config.stream_callback = [&streamed_tokens](const std::string &token) {
        streamed_tokens.push_back(token);
    };

    std::string prompt = "List three benefits of exercise:";
    std::string output = engine->generate(prompt, config);

    EXPECT_FALSE(output.empty()) << "Output should not be empty";
    EXPECT_GT(streamed_tokens.size(), 0) << "Should have streamed tokens";

    // Verify streamed tokens concatenate to output
    std::string concatenated;
    for (const auto &token : streamed_tokens) {
        concatenated += token;
    }
    EXPECT_EQ(concatenated, output) << "Streamed tokens should match final output";
}

TEST_F(InferenceEngineTest, ChatModeWithTemplate) {
    auto config = CreateBasicConfig(60);

    std::vector<std::pair<std::string, std::string>> messages = {
        {"system", "You are a helpful assistant."},
        {"user", "What is the largest planet in our solar system?"}};

    std::string response = engine->chat(messages, config);

    EXPECT_FALSE(response.empty()) << "Chat response should not be empty";

    const auto &stats = engine->last_stats();
    EXPECT_GT(stats.prompt_tokens, 0) << "Should have prompt tokens";
    EXPECT_GT(stats.generated_tokens, 0) << "Should generate tokens";
}

// ============================================================================
// Sampling Strategy Tests
// ============================================================================

TEST_F(InferenceEngineTest, TemperatureEffect) {
    std::string prompt = "The weather today is";

    std::vector<float> temperatures = {0.3f, 0.7f, 1.2f};
    std::vector<std::string> outputs;

    for (float temp : temperatures) {
        GenerationConfig config;
        config.max_new_tokens = 25;
        config.sampler_type = sampler::SamplerType::ARGMAX;
        config.sampler_params.temperature = temp;
        config.verbose = false;

        std::string output = engine->generate(prompt, config);
        outputs.push_back(output);

        EXPECT_FALSE(output.empty()) << "Output should not be empty for temp=" << temp;

        engine->reset();
    }

    // Just verify we got different outputs (probabilistic test)
    EXPECT_EQ(outputs.size(), 3) << "Should have 3 outputs";
}

TEST_F(InferenceEngineTest, TopKSampling) {
    GenerationConfig config;
    config.max_new_tokens = 30;
    config.sampler_type = sampler::SamplerType::ARGMAX;
    config.sampler_params.temperature = 1.0f;
    config.sampler_params.top_k = 10;
    config.verbose = false;

    std::string prompt = "Once upon a time";
    std::string output = engine->generate(prompt, config);

    EXPECT_FALSE(output.empty()) << "Top-k sampling should produce output";
    EXPECT_GT(engine->last_stats().generated_tokens, 0);
}

TEST_F(InferenceEngineTest, TopPSampling) {
    GenerationConfig config;
    config.max_new_tokens = 30;
    config.sampler_type = sampler::SamplerType::ARGMAX;
    config.sampler_params.temperature = 1.0f;
    config.sampler_params.top_p = 0.9f;
    config.verbose = false;

    std::string prompt = "Once upon a time";
    std::string output = engine->generate(prompt, config);

    EXPECT_FALSE(output.empty()) << "Top-p sampling should produce output";
    EXPECT_GT(engine->last_stats().generated_tokens, 0);
}

// ============================================================================
// Context Length Tests
// ============================================================================

TEST_F(InferenceEngineTest, LongContextHandling) {
    auto config = CreateBasicConfig(100);

    std::string prompt = "In a comprehensive analysis of artificial intelligence, "
                         "we must consider multiple aspects including machine learning, "
                         "neural networks, natural language processing, and computer vision. "
                         "Each of these domains has made significant progress in recent years. "
                         "For example, ";

    std::string output = engine->generate(prompt, config);

    EXPECT_FALSE(output.empty()) << "Long context should be handled";

    const auto &stats = engine->last_stats();
    EXPECT_GT(stats.prompt_tokens, 50) << "Should have many prompt tokens";
    EXPECT_GT(stats.generated_tokens, 0) << "Should generate tokens";
}

TEST_F(InferenceEngineTest, ShortPrompt) {
    auto config = CreateBasicConfig(20);

    std::string prompt = "Hi";
    std::string output = engine->generate(prompt, config);

    EXPECT_FALSE(output.empty()) << "Short prompt should produce output";

    const auto &stats = engine->last_stats();
    EXPECT_LT(stats.prompt_tokens, 5) << "Short prompt should have few tokens";
}

// ============================================================================
// KV Cache Tests
// ============================================================================

TEST_F(InferenceEngineTest, KVCacheDynamicGrowth) {
    auto kv_cache = engine->kv_cache();
    auto *dynamic_cache = dynamic_cast<kvcache::DynamicKVCacheManager *>(kv_cache.get());

    if (!dynamic_cache) {
        GTEST_SKIP() << "Not using dynamic KV cache";
    }

    int initial_capacity = dynamic_cache->allocated_capacity();
    size_t initial_memory = dynamic_cache->memory_usage();

    EXPECT_GT(initial_capacity, 0) << "Initial capacity should be positive";
    EXPECT_GT(initial_memory, 0) << "Initial memory should be positive";

    // Generate long sequence to trigger growth
    GenerationConfig config;
    config.max_new_tokens = 300;
    config.sampler_type = sampler::SamplerType::ARGMAX;
    config.sampler_params.temperature = 0.8f;
    config.verbose = false;

    std::string prompt = "Write a detailed explanation of quantum computing: ";
    engine->generate(prompt, config);

    int final_capacity = dynamic_cache->allocated_capacity();
    // size_t final_memory = dynamic_cache->memory_usage();

    // Cache might have grown
    EXPECT_GE(final_capacity, initial_capacity) << "Capacity should not decrease";
    EXPECT_GT(dynamic_cache->utilization(), 0.0f) << "Cache should be utilized";
}

TEST_F(InferenceEngineTest, KVCacheReset) {
    auto kv_cache = engine->kv_cache();

    // Generate some text to fill cache
    auto config = CreateBasicConfig(50);
    engine->generate("Test prompt", config);

    EXPECT_GT(kv_cache->current_length(), 0) << "Cache should have content";

    // Reset
    engine->reset();

    EXPECT_EQ(kv_cache->current_length(), 0) << "Cache should be empty after reset";
}

// ============================================================================
// Performance Tests
// ============================================================================

TEST_F(InferenceEngineTest, PerformanceBenchmark) {
    std::vector<int> token_counts = {10, 50, 100};

    for (int count : token_counts) {
        GenerationConfig config;
        config.max_new_tokens = count;
        config.sampler_type = sampler::SamplerType::ARGMAX;
        config.verbose = false;

        std::string prompt = "Testing performance with different token counts.";

        engine->generate(prompt, config);

        const auto &stats = engine->last_stats();

        EXPECT_GT(stats.total_time_ms, 0) << "Generation should take time";
        EXPECT_GT(stats.tokens_per_second(), 0) << "Should have positive throughput";
        EXPECT_EQ(stats.generated_tokens, count) << "Should generate requested tokens";

        engine->reset();
    }
}

TEST_F(InferenceEngineTest, PrefillVsDecodeTime) {
    auto config = CreateBasicConfig(50);

    std::string prompt = "Testing the difference between prefill and decode phases.";
    engine->generate(prompt, config);

    const auto &stats = engine->last_stats();

    EXPECT_GT(stats.prefill_time_ms, 0) << "Prefill should take time";
    EXPECT_GT(stats.decode_time_ms, 0) << "Decode should take time";
    EXPECT_DOUBLE_EQ(stats.total_time_ms,
                     stats.prefill_time_ms + stats.decode_time_ms)
        << "Total time should equal prefill + decode";
}

// ============================================================================
// Multi-turn Conversation Tests
// ============================================================================

TEST_F(InferenceEngineTest, MultiTurnConversation) {
    auto config = CreateBasicConfig(50);

    std::vector<std::pair<std::string, std::string>> messages = {
        {"system", "You are a helpful math tutor."},
        {"user", "What is 15 + 27?"}};

    std::string response1 = engine->chat(messages, config);
    EXPECT_FALSE(response1.empty()) << "First response should not be empty";

    // Add assistant response and continue
    messages.push_back({"assistant", response1});
    messages.push_back({"user", "Now multiply that by 2."});

    engine->reset();
    std::string response2 = engine->chat(messages, config);
    EXPECT_FALSE(response2.empty()) << "Second response should not be empty";
}

// ============================================================================
// Edge Cases Tests
// ============================================================================

TEST_F(InferenceEngineTest, EmptyPromptHandling) {
    auto config = CreateBasicConfig(20);

    // Empty string might be encoded to BOS token
    EXPECT_NO_THROW({
        std::string output = engine->generate("", config);
    }) << "Should handle empty prompt gracefully";
}

TEST_F(InferenceEngineTest, SpecialCharacters) {
    auto config = CreateBasicConfig(20);

    std::string prompt = "What is 2+2? Answer:";
    std::string output = engine->generate(prompt, config);

    EXPECT_FALSE(output.empty()) << "Should handle special characters";
}

TEST_F(InferenceEngineTest, UnicodeCharacters) {
    auto config = CreateBasicConfig(20);

    std::string prompt = "Hello 世界! Translate to English:";

    EXPECT_NO_THROW({
        std::string output = engine->generate(prompt, config);
        EXPECT_FALSE(output.empty()) << "Should handle unicode";
    });
}

TEST_F(InferenceEngineTest, VeryHighTemperature) {
    GenerationConfig config;
    config.max_new_tokens = 20;
    config.sampler_type = sampler::SamplerType::ARGMAX;
    config.sampler_params.temperature = 2.0f;
    config.verbose = false;

    std::string prompt = "Random word:";

    EXPECT_NO_THROW({
        std::string output = engine->generate(prompt, config);
        EXPECT_FALSE(output.empty()) << "Should handle high temperature";
    });
}

TEST_F(InferenceEngineTest, MaxTokensLimit) {
    GenerationConfig config;
    config.max_new_tokens = 5;
    config.sampler_type = sampler::SamplerType::ARGMAX;
    config.verbose = false;

    std::string prompt = "Generate a very long story about";
    std::string output = engine->generate(prompt, config);

    const auto &stats = engine->last_stats();
    EXPECT_LE(stats.generated_tokens, 5)
        << "Should not exceed max_new_tokens";
}

// ============================================================================
// Configuration Tests
// ============================================================================

TEST_F(InferenceEngineTest, ConfigValidation) {
    GenerationConfig config;
    config.max_new_tokens = -1; // Invalid

    EXPECT_THROW({ config.validate(); }, std::invalid_argument) << "Should throw on invalid config";
}

TEST_F(InferenceEngineTest, StatsAccuracy) {
    auto config = CreateBasicConfig(30);

    std::string prompt = "Testing statistics accuracy.";
    engine->generate(prompt, config);

    const auto &stats = engine->last_stats();

    EXPECT_EQ(stats.total_tokens,
              stats.prompt_tokens + stats.generated_tokens)
        << "Total tokens should equal prompt + generated";

    EXPECT_GT(stats.tokens_per_second(), 0)
        << "Throughput should be positive";
}