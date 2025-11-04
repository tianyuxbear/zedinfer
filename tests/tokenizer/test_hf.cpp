#include "frontend/tokenizer/hf_tokenizer.hpp"

#include <gtest/gtest.h>
#include <string>
#include <vector>

using namespace zedinfer::tokenizer;

const std::string tokenizer_json_path("/home/xiongtianyu/data/models/deepseek-ai/"
                                      "DeepSeek-R1-Distill-Qwen-1.5B/tokenizer.json");

class HFTokenizerTest : public ::testing::Test {
protected:
    std::shared_ptr<Tokenizer> tokenizer;

    void SetUp() override {
        tokenizer = HFTokenizer::create(tokenizer_json_path);
    }
};

// ============ 基础功能测试 ============
TEST_F(HFTokenizerTest, HelloTest) {
    std::string text = "Hello, world!";
    auto tokens = tokenizer->encode(text);

    EXPECT_GT(tokens.size(), 0) << "编码结果不应为空";

    std::string decoded = tokenizer->decode(tokens);
    EXPECT_EQ(decoded, text) << "解码结果应与原文一致";
}

TEST_F(HFTokenizerTest, EmptyTest) {
    std::string text = "";
    auto tokens = tokenizer->encode(text);

    // 只有BOS token（如果配置了add_bos_token=true）
    EXPECT_EQ(tokens.size(), 1);
    EXPECT_EQ(tokens[0], 151646) << "BOS token ID应为151646";
}

TEST_F(HFTokenizerTest, MathTest) {
    std::string text = "123 + 456 = 789";
    auto tokens = tokenizer->encode(text);
    std::string decoded = tokenizer->decode(tokens);

    EXPECT_EQ(decoded, text);
}

TEST_F(HFTokenizerTest, SpacesTest) {
    std::string text = "a  b   c"; // 多个空格
    auto tokens = tokenizer->encode(text);
    std::string decoded = tokenizer->decode(tokens);

    EXPECT_EQ(decoded, "a  b   c");
}

// ============ 问答模板测试 ============
TEST_F(HFTokenizerTest, PromptTest) {
    std::string prompt = "Who are you?";
    std::string expected_input = "<｜begin▁of▁sentence｜><｜User｜>Who are you?<｜Assistant｜><think>\n";

    std::vector<int> expected_ids = {
        151646, 151646, 151644, 15191, 525, 498, 30, 151645, 151648, 198};

    auto hf_tokenizer = dynamic_cast<HFTokenizer *>(tokenizer.get());
    ASSERT_NE(hf_tokenizer, nullptr);

    std::vector<std::pair<std::string, std::string>> messages = {
        {"user", prompt}};

    std::string actual_input = hf_tokenizer->apply_chat_template(messages, true);
    EXPECT_EQ(actual_input, expected_input)
        << "模板应用后不匹配";
}

TEST_F(HFTokenizerTest, EncodePromptTest) {
    std::string input_content = "<｜begin▁of▁sentence｜><｜User｜>Who are you?<｜Assistant｜><think>\n";

    std::vector<int> expected_ids = {
        151646, 151646, 151644, 15191, 525, 498, 30, 151645, 151648, 198};

    auto actual_ids = tokenizer->encode(input_content);

    EXPECT_EQ(actual_ids.size(), expected_ids.size())
        << "Token数量应匹配。实际: " << actual_ids.size()
        << ", 期望: " << expected_ids.size();

    for (size_t i = 0; i < std::min(actual_ids.size(), expected_ids.size()); i++) {
        EXPECT_EQ(actual_ids[i], expected_ids[i])
            << "位置 " << i << " token不匹配。实际: " << actual_ids[i]
            << ", 期望: " << expected_ids[i];
    }
}

TEST_F(HFTokenizerTest, DecodePromptTest) {
    std::vector<int> input_ids = {
        151646, 151646, 151644, 15191, 525, 498, 30, 151645, 151648, 198};

    std::string decoded = tokenizer->decode(input_ids);

    // 解码后应包含用户问题
    EXPECT_NE(decoded.find("Who are you?"), std::string::npos)
        << "解码结果应包含原始问题";
}

// ============ 边界情况测试 ============

TEST_F(HFTokenizerTest, LongTextTest) {
    std::string long_text = "The quick brown fox jumps over the lazy dog. ";
    for (int i = 0; i < 10; i++) {
        long_text += long_text; // 指数增长
    }

    auto tokens = tokenizer->encode(long_text);
    EXPECT_GT(tokens.size(), 100);

    // 解码应该不崩溃
    std::string decoded = tokenizer->decode(tokens);
    EXPECT_EQ(decoded, long_text);
}

TEST_F(HFTokenizerTest, SpecialCharactersTest) {
    std::string text = "@#$%^&*()_+-=[]{}|;':\",./<>?";
    auto tokens = tokenizer->encode(text);
    std::string decoded = tokenizer->decode(tokens);

    EXPECT_EQ(decoded, text);
}

// ============ 批量测试 ============

TEST_F(HFTokenizerTest, BatchEncodingTest) {
    std::vector<std::string> texts = {
        "Hello",
        "How are you?",
        "What is AI?",
        "1+1=2",
        "Good morning!"};

    for (const auto &text : texts) {
        auto tokens = tokenizer->encode(text);
        std::string decoded = tokenizer->decode(tokens);

        EXPECT_EQ(decoded, text) << "批量测试失败于: " << text;
    }
}

// ============ 性能测试 ============

TEST_F(HFTokenizerTest, PerformanceTest) {
    std::string text = "This is a performance test. ";
    for (int i = 0; i < 10; i++) {
        text += text;
    }

    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < 100; i++) {
        auto tokens = tokenizer->encode(text);
        tokenizer->decode(tokens);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    std::cout << "100次编码解码耗时: " << duration.count() << "ms" << std::endl;
    EXPECT_LT(duration.count(), 10000) << "性能测试超时";
}

// ========== 中文测试 ==========
TEST_F(HFTokenizerTest, PromptTest_ZH) {
    std::string prompt = "你好，世界！";
    std::string expected_input = "<｜begin▁of▁sentence｜><｜User｜>你好，世界！<｜Assistant｜><think>\n";

    std::vector<int> expected_ids = {
        151646, 151646, 151644, 108386, 3837, 99489, 6313, 151645, 151648,
        198};

    auto hf_tokenizer = dynamic_cast<HFTokenizer *>(tokenizer.get());
    ASSERT_NE(hf_tokenizer, nullptr);

    std::vector<std::pair<std::string, std::string>> messages = {
        {"user", prompt}};

    std::string actual_input = hf_tokenizer->apply_chat_template(messages, true);
    EXPECT_EQ(actual_input, expected_input)
        << "模板应用后不匹配";
}

TEST_F(HFTokenizerTest, EncodePromptTest_ZH) {
    std::string input_content = "<｜begin▁of▁sentence｜><｜User｜>你好，世界！<｜Assistant｜><think>\n";

    std::vector<int> expected_ids = {
        151646, 151646, 151644, 108386, 3837, 99489, 6313, 151645, 151648,
        198};

    auto actual_ids = tokenizer->encode(input_content);

    EXPECT_EQ(actual_ids.size(), expected_ids.size())
        << "Token数量应匹配。实际: " << actual_ids.size()
        << ", 期望: " << expected_ids.size();

    for (size_t i = 0; i < std::min(actual_ids.size(), expected_ids.size()); i++) {
        EXPECT_EQ(actual_ids[i], expected_ids[i])
            << "位置 " << i << " token不匹配。实际: " << actual_ids[i]
            << ", 期望: " << expected_ids[i];
    }
}

TEST_F(HFTokenizerTest, DecodePromptTest_ZH) {
    std::vector<int> input_ids = {
        151646, 151646, 151644, 108386, 3837, 99489, 6313, 151645, 151648,
        198};

    std::string decoded = tokenizer->decode(input_ids);

    // 解码后应包含用户问题
    EXPECT_NE(decoded.find("你好，世界！"), std::string::npos)
        << "解码结果应包含原始问题";
}

TEST_F(HFTokenizerTest, DecodeAnswerTest_ZH) {
    std::vector<int> input_ids = {
        151646, 151646, 151644, 108386, 3837, 99489, 6313, 151645, 151648, 198, 108386, 6313, 104198, 33464, 39350, 10911, 16, 3837, 104139, 109944, 99663, 101214, 101037, 94432, 151649, 271, 108386, 6313, 104198, 33464, 39350, 10911, 16, 3837, 104139, 109944, 99663, 101214, 101037, 11319, 151643};

    std::string decoded = tokenizer->decode(input_ids);

    std::cout << "decoded: \n"
              << decoded << std::endl;

    // 解码后应包含用户问题
    EXPECT_NE(decoded.find("你好，世界！"), std::string::npos)
        << "解码结果应包含原始问题";
}
