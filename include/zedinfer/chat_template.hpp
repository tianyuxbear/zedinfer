#pragma once

#include <string>
#include <utility>
#include <vector>

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

    // System message formatting (empty = plain text fallback)
    std::string system_prefix;
    std::string system_suffix;

    // If true, BOS token is only added on the first turn
    bool add_bos_first_turn_only = true;

    // Load template from model directory, inferring from model_type with optional override
    static ChatTemplate load(const std::string& model_path, const std::string& model_type);

    // Built-in default for DeepSeek-R1 distillation models
    static ChatTemplate default_deepseek_r1();

    // Built-in default for standard Qwen models (ChatML format)
    static ChatTemplate default_qwen_chatml();

    // Format an OpenAI-style messages array into a prompt string.
    // Each pair is (role, content) where role is "system", "user", or "assistant".
    // If add_generation_prompt is true, appends generation_prompt at the end.
    std::string apply(const std::vector<std::pair<std::string, std::string>>& messages,
                      bool add_generation_prompt = true) const;
};

} // namespace zedinfer
