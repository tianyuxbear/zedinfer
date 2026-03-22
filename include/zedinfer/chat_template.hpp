#pragma once

#include <string>

namespace zedinfer {

/**
 * Chat template defining how to format multi-turn conversations.
 * Loaded from model directory at engine init time.
 */
struct ChatTemplate {
    // Structural tokens
    std::string bos_token;
    std::string eos_token;
    std::string user_prefix;
    std::string user_suffix;
    std::string assistant_prefix;
    std::string assistant_suffix;

    // String appended after last user message to trigger generation
    std::string generation_prompt;

    // Display prefix prepended to model output (e.g., "<think> " for reasoning models)
    std::string output_prefix;

    // If true, BOS token is only added on the first turn
    bool add_bos_first_turn_only = true;

    // Load template from model directory, inferring from model_type with optional override
    static ChatTemplate load(const std::string &model_path,
                             const std::string &model_type);

    // Built-in default for DeepSeek-R1 distillation models
    static ChatTemplate default_deepseek_r1();

    // Built-in default for standard Qwen models (ChatML format)
    static ChatTemplate default_qwen_chatml();
};

} // namespace zedinfer
