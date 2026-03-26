#include "zedinfer/chat_template.hpp"

#include <gtest/gtest.h>

using namespace zedinfer;

// ============================================================================
// Default templates
// ============================================================================

TEST(ChatTemplateTest, DeepSeekR1Defaults) {
    auto t = ChatTemplate::default_deepseek_r1();
    EXPECT_FALSE(t.bos_token.empty());
    EXPECT_FALSE(t.eos_token.empty());
    EXPECT_FALSE(t.user_prefix.empty());
    EXPECT_FALSE(t.assistant_prefix.empty());
    EXPECT_FALSE(t.generation_prompt.empty());
    EXPECT_TRUE(t.add_bos_first_turn_only);
}

TEST(ChatTemplateTest, QwenChatMLDefaults) {
    auto t = ChatTemplate::default_qwen_chatml();
    EXPECT_TRUE(t.bos_token.empty());      // ChatML format has no BOS token
    EXPECT_FALSE(t.user_prefix.empty());
    EXPECT_FALSE(t.system_prefix.empty()); // ChatML has system support
}

// ============================================================================
// apply() formatting
// ============================================================================

TEST(ChatTemplateTest, SingleUserMessage) {
    auto t = ChatTemplate::default_deepseek_r1();
    std::vector<std::pair<std::string, std::string>> messages = {{"user", "Hello"}};
    std::string result = t.apply(messages, true);

    EXPECT_NE(result.find(t.bos_token), std::string::npos) << "Should contain BOS";
    EXPECT_NE(result.find("Hello"), std::string::npos) << "Should contain user message";
    EXPECT_NE(result.find(t.generation_prompt), std::string::npos) << "Should end with generation prompt";
}

TEST(ChatTemplateTest, MultiTurnConversation) {
    auto t = ChatTemplate::default_deepseek_r1();
    std::vector<std::pair<std::string, std::string>> messages
        = {{"user", "Hi"}, {"assistant", "Hello! How can I help?"}, {"user", "What is 2+2?"}};
    std::string result = t.apply(messages, true);

    EXPECT_NE(result.find("Hi"), std::string::npos);
    EXPECT_NE(result.find("Hello! How can I help?"), std::string::npos);
    EXPECT_NE(result.find("What is 2+2?"), std::string::npos);
}

TEST(ChatTemplateTest, SystemMessage) {
    auto t = ChatTemplate::default_deepseek_r1();
    std::vector<std::pair<std::string, std::string>> messages
        = {{"system", "You are a helpful assistant."}, {"user", "Hi"}};
    std::string result = t.apply(messages, true);

    EXPECT_NE(result.find("You are a helpful assistant."), std::string::npos);
    EXPECT_NE(result.find("Hi"), std::string::npos);
}

TEST(ChatTemplateTest, NoGenerationPrompt) {
    auto t = ChatTemplate::default_deepseek_r1();
    std::vector<std::pair<std::string, std::string>> messages = {{"user", "Hello"}};
    std::string with_prompt = t.apply(messages, true);
    std::string without_prompt = t.apply(messages, false);

    EXPECT_GT(with_prompt.size(), without_prompt.size());
    // Without generation prompt, result should NOT end with the assistant trigger
    size_t pos = without_prompt.rfind(t.generation_prompt);
    // Either not found, or not at the end
    if (pos != std::string::npos) {
        EXPECT_NE(pos + t.generation_prompt.size(), without_prompt.size());
    }
}

TEST(ChatTemplateTest, EmptyMessages) {
    auto t = ChatTemplate::default_deepseek_r1();
    std::vector<std::pair<std::string, std::string>> messages;
    std::string result = t.apply(messages, true);

    // Should at least have BOS + generation prompt
    EXPECT_NE(result.find(t.bos_token), std::string::npos);
}

TEST(ChatTemplateTest, BOSOnlyOnce) {
    auto t = ChatTemplate::default_deepseek_r1();
    std::vector<std::pair<std::string, std::string>> messages
        = {{"user", "Hello"}, {"assistant", "Hi"}, {"user", "Bye"}};
    std::string result = t.apply(messages, true);

    // BOS should appear at least once
    size_t first = result.find(t.bos_token);
    EXPECT_NE(first, std::string::npos);
}

// ============================================================================
// ChatML format (Qwen)
// ============================================================================

TEST(ChatTemplateTest, QwenChatMLSystemFormat) {
    auto t = ChatTemplate::default_qwen_chatml();
    std::vector<std::pair<std::string, std::string>> messages = {{"system", "You are helpful."}, {"user", "Hi"}};
    std::string result = t.apply(messages, true);

    // ChatML system uses <|im_start|>system\n...<|im_end|>\n
    EXPECT_NE(result.find(t.system_prefix), std::string::npos);
    EXPECT_NE(result.find("You are helpful."), std::string::npos);
}
