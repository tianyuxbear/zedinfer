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

    // Alternate generation prompt that disables an explicit <think> block
    // (Qwen3.5's enable_thinking=false variant). Empty for templates without
    // a closed-think variant — callers should fall back to generation_prompt
    // in that case.
    std::string generation_prompt_no_think;

    // Display prefix prepended to model output (e.g., "<think> " for reasoning models)
    std::string output_prefix;

    // Display prefix used when generation_prompt_no_think is selected.
    // For Qwen3.5 closed-think, the prompt already contains the
    // "<think>\n\n</think>\n\n" prefix, so this is empty.
    std::string output_prefix_no_think;

    // System message formatting (empty = plain text fallback)
    std::string system_prefix;
    std::string system_suffix;

    // If true, BOS token is only added on the first turn
    bool add_bos_first_turn_only = true;

    // Load template from model directory, inferring from model_type with optional override
    static ChatTemplate load(const std::string& model_path, const std::string& model_type);

    // Built-in default for DeepSeek-R1 distillation models
    static ChatTemplate default_deepseek_r1();

    // Built-in default for standard Qwen models (ChatML format).
    // Use for non-reasoning models (Qwen2). No closed-think variant — the
    // `<think>` token isn't part of the trained vocabulary semantics.
    static ChatTemplate default_qwen_chatml();

    // Built-in default for Qwen3 / Qwen3-MoE (reasoning models that emit
    // `<think>...</think>` blocks themselves; jinja's enable_thinking=false
    // injects the empty closed-think block to skip reasoning). Generation
    // prompt does NOT pre-inject `<think>` — the model emits it.
    static ChatTemplate default_qwen3_chatml();

    // Built-in default for Qwen3.5 (reasoning model: assistant turn must start
    // with `<think>\n`, otherwise the model drifts and hallucinates a fake
    // user message before responding).
    static ChatTemplate default_qwen3_5_chatml();

    // Format an OpenAI-style messages array into a prompt string.
    // Each pair is (role, content) where role is "system", "user", or "assistant".
    // If add_generation_prompt is true, appends generation_prompt at the end.
    std::string apply(const std::vector<std::pair<std::string, std::string>>& messages,
                      bool add_generation_prompt = true) const;
};

} // namespace zedinfer
