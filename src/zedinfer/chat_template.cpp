#include "zedinfer/chat_template.hpp"

#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <plog/Log.h>
#include <string>

namespace fs = std::filesystem;
using json = nlohmann::json;

namespace zedinfer {

// DeepSeek-R1 uses fullwidth Unicode delimiters in special token names.
// U+FF5C (fullwidth vertical line) and U+2581 (lower one eighth block).
static const std::string DS_SEP = "\xef\xbd\x9c"; // ｜
static const std::string DS_MID = "\xe2\x96\x81"; // ▁

static std::string ds_token(const std::string& name) {
    return "<" + DS_SEP + name + DS_SEP + ">";
}

static std::string ds_sentence_token(const std::string& action) {
    return "<" + DS_SEP + action + DS_MID + "of" + DS_MID + "sentence" + DS_SEP + ">";
}

ChatTemplate ChatTemplate::default_deepseek_r1() {
    ChatTemplate t;
    t.bos_token = ds_sentence_token("begin");
    t.eos_token = ds_sentence_token("end");
    t.user_prefix = ds_token("User");
    t.user_suffix = "";
    t.assistant_prefix = ds_token("Assistant");
    t.assistant_suffix = ds_sentence_token("end");
    t.generation_prompt = ds_token("Assistant") + "<think>\n";
    t.output_prefix = "<think> ";
    t.system_prefix = "";
    t.system_suffix = "";
    t.add_bos_first_turn_only = true;
    return t;
}

ChatTemplate ChatTemplate::default_qwen_chatml() {
    ChatTemplate t;
    t.bos_token = "";
    t.eos_token = "<|im_end|>";
    t.user_prefix = "<|im_start|>user\n";
    t.user_suffix = "<|im_end|>\n";
    t.assistant_prefix = "<|im_start|>assistant\n";
    t.assistant_suffix = "<|im_end|>\n";
    t.generation_prompt = "<|im_start|>assistant\n";
    t.output_prefix = "";
    t.system_prefix = "<|im_start|>system\n";
    t.system_suffix = "<|im_end|>\n";
    t.add_bos_first_turn_only = false;
    return t;
}

ChatTemplate ChatTemplate::default_qwen3_5_chatml() {
    // Qwen3.5's chat_template.jinja exposes two assistant-prompt variants:
    //
    //   enable_thinking=true  (jinja default):  "<|im_start|>assistant\n<think>\n"
    //                          → model produces a reasoning chain then the answer.
    //   enable_thinking=false:                  "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    //                          → empty think block, model emits a direct answer.
    //
    // Both variants are stored here; GenerationConfig::enable_thinking picks at
    // request time. We diverge from HF's jinja default by defaulting
    // enable_thinking=false (see GenerationConfig::enable_thinking): on
    // Qwen3.5-GPTQ-Int4 the empty-thinking-start state at the end of the
    // open-<think> prefill is a fragile region of the model — the decoder
    // hallucinates the prompt content for the first ~30 tokens of thinking
    // (e.g. "What is 2+2?" reasoned as "the prompt actually says 2+4"). The
    // closed-<think> prefill skips that region entirely. Pre-filling some
    // thinking content (e.g. "<think>\nLet me think.") recovers coherent
    // thinking but is workload-specific; for general use closed-<think> is
    // the safe default. Higher-precision weight sets (bf16/fp16) likely
    // tolerate the empty start cleanly — re-evaluate when bringing those up.
    //
    // See docs/debug/qwen3_5_sampler_and_thinking_drift.md for the full
    // investigation (verified token-by-token against HF apply_chat_template;
    // tested ARGMAX/sampling, GDN parallelization revert, prompt-suffix
    // sweep — none of which localize a zedinfer-side cause).
    ChatTemplate t = default_qwen_chatml();
    t.generation_prompt = "<|im_start|>assistant\n<think>\n";
    t.output_prefix = "<think>\n";
    t.generation_prompt_no_think = "<|im_start|>assistant\n<think>\n\n</think>\n\n";
    t.output_prefix_no_think = "";
    return t;
}

// Extract token string from either a plain string or AddedToken object {"content": "..."}
static std::string extract_token_string(const json& j, const std::string& key) {
    if (!j.contains(key)) {
        return "";
    }
    const auto& val = j[key];
    if (val.is_string()) {
        return val.get<std::string>();
    }
    if (val.is_object() && val.contains("content")) {
        return val["content"].get<std::string>();
    }
    return "";
}

// Detect DeepSeek-R1 format by checking if eos_token contains the fullwidth delimiter.
static bool is_deepseek_r1_format(const std::string& model_path) {
    fs::path tc_path = fs::path(model_path) / "tokenizer_config.json";
    if (!fs::exists(tc_path)) {
        return false;
    }

    try {
        std::ifstream file(tc_path);
        json j;
        file >> j;

        std::string eos = extract_token_string(j, "eos_token");
        return eos.find(DS_SEP) != std::string::npos;
    } catch (...) {}

    return false;
}

ChatTemplate ChatTemplate::load(const std::string& model_path, const std::string& model_type) {
    // Try optional chat_template.json override
    fs::path override_path = fs::path(model_path) / "chat_template.json";
    if (fs::exists(override_path)) {
        try {
            std::ifstream file(override_path);
            json j;
            file >> j;

            ChatTemplate t;
            t.bos_token = j.value("bos_token", "");
            t.eos_token = j.value("eos_token", "");
            t.user_prefix = j.value("user_prefix", "");
            t.user_suffix = j.value("user_suffix", "");
            t.assistant_prefix = j.value("assistant_prefix", "");
            t.assistant_suffix = j.value("assistant_suffix", "");
            t.generation_prompt = j.value("generation_prompt", "");
            t.generation_prompt_no_think = j.value("generation_prompt_no_think", "");
            t.output_prefix = j.value("output_prefix", "");
            t.output_prefix_no_think = j.value("output_prefix_no_think", "");
            t.add_bos_first_turn_only = j.value("add_bos_first_turn_only", true);
            t.system_prefix = j.value("system_prefix", "");
            t.system_suffix = j.value("system_suffix", "");

            LOGI << "[ChatTemplate] Loaded from " << override_path.string();
            return t;
        } catch (const std::exception& e) {
            LOGW << "[ChatTemplate] Failed to parse " << override_path.string() << ": " << e.what()
                 << "; falling back to model-type default";
        }
    }

    // Auto-detect: DeepSeek-R1 distillation vs standard Qwen. Qwen3 MoE uses the same
    // ChatML conversation format as dense Qwen3, so it routes through the same branch.
    if (model_type == "qwen2" || model_type == "qwen3" || model_type == "qwen3_moe") {
        if (is_deepseek_r1_format(model_path)) {
            LOGI << "[ChatTemplate] Detected DeepSeek-R1 format for model_type=" << model_type;
            return default_deepseek_r1();
        }
        LOGI << "[ChatTemplate] Using ChatML template for model_type=" << model_type;
        return default_qwen_chatml();
    }

    // Qwen3.5 / Qwen3.5-MoE are reasoning models — same ChatML conversation
    // shell as Qwen3, but the assistant turn must open with `<think>\n` (see
    // default_qwen3_5_chatml).
    if (model_type == "qwen3_5" || model_type == "qwen3_5_moe") {
        LOGI << "[ChatTemplate] Using Qwen3.5 reasoning ChatML template for model_type=" << model_type;
        return default_qwen3_5_chatml();
    }

    LOGW << "[ChatTemplate] Unknown model_type=" << model_type << "; using ChatML template as fallback";
    return default_qwen_chatml();
}

std::string ChatTemplate::apply(const std::vector<std::pair<std::string, std::string>>& messages,
                                bool add_generation_prompt) const {
    std::string result;

    // BOS token: always prepend once. add_bos_first_turn_only is irrelevant here
    // because apply() formats a complete conversation in one call (always "first turn").
    // The flag only matters for incremental per-turn formatting (InferenceSession::chat).
    if (!bos_token.empty()) {
        result += bos_token;
    }

    for (const auto& [role, content] : messages) {
        if (role == "system") {
            if (!system_prefix.empty()) {
                result += system_prefix + content + system_suffix;
            } else {
                result += content + "\n";
            }
        } else if (role == "user") {
            result += user_prefix + content + user_suffix;
        } else if (role == "assistant") {
            result += assistant_prefix + content + assistant_suffix;
        }
    }

    if (add_generation_prompt && !generation_prompt.empty()) {
        result += generation_prompt;
    }

    return result;
}

} // namespace zedinfer
