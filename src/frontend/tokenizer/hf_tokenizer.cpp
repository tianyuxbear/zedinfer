#include "frontend/tokenizer/hf_tokenizer.hpp"
#include "frontend/tokenizer/byte_level.hpp"
#include "utils/logging.hpp"

#include <cctype>
#include <climits>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <plog/Log.h>

using json = nlohmann::json;

namespace zedinfer::tokenizer {

std::shared_ptr<Tokenizer> HFTokenizer::create(const std::string& tokenizer_path) {
    auto tokenizer = std::make_unique<HFTokenizer>();
    tokenizer->load_from_file(tokenizer_path);
    return tokenizer;
}

void HFTokenizer::load_from_file(const std::string& tokenizer_path) {
    std::ifstream file(tokenizer_path);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open: " + tokenizer_path);
    }

    json tokenizer;
    file >> tokenizer;

    // Load vocab and merges from model section
    if (tokenizer.contains("model")) {
        auto model = tokenizer["model"];

        if (model.contains("vocab")) {
            for (auto& [token, id] : model["vocab"].items()) {
                int token_id = id.get<int>();
                vocab_[token] = token_id;
                id_to_token_[token_id] = token;
            }
        }

        if (model.contains("merges")) {
            int rank = 0;
            for (const auto& merge_entry : model["merges"]) {
                std::string key;
                if (merge_entry.is_string()) {
                    // Old format: "token1 token2"
                    key = merge_entry.get<std::string>();
                } else if (merge_entry.is_array() && merge_entry.size() == 2) {
                    // New format: ["token1", "token2"]
                    key = merge_entry[0].get<std::string>() + " " + merge_entry[1].get<std::string>();
                } else {
                    continue;
                }
                merges_[key] = rank++;
            }
        }
    }

    // Load pre-tokenizer regex pattern (Split type only)
    if (tokenizer.contains("pre_tokenizer")) {
        const auto& pre_tok = tokenizer["pre_tokenizer"];
        if (pre_tok.contains("pretokenizers") && pre_tok["pretokenizers"].is_array()) {
            for (const auto& item : pre_tok["pretokenizers"]) {
                if (item.contains("type") && item["type"] == "Split") {
                    if (item.contains("pattern") && item["pattern"].contains("Regex")) {
                        pattern_ = item["pattern"]["Regex"].get<std::string>();
                        UErrorCode status = U_ZERO_ERROR;
                        icu::UnicodeString uPattern = icu::UnicodeString::fromUTF8(pattern_);
                        pre_tokenize_regex_.reset(icu::RegexPattern::compile(uPattern, 0, status));
                        if (U_FAILURE(status)) {
                            LOG_ERROR_(utils::BOTH) << "[Tokenizer] Invalid regex pattern: " << u_errorName(status)
                                                    << " - Pattern: " << pattern_ << std::endl;
                            pre_tokenize_regex_.reset();
                        }
                    }
                    break;
                }
            }
        }
    }

    // Load added (special) tokens into lookup maps
    if (tokenizer.contains("added_tokens")) {
        for (const auto& token_info : tokenizer["added_tokens"]) {
            std::string token = token_info["content"].get<std::string>();
            int id = token_info["id"].get<int>();
            special_tokens_.token_to_id[token] = id;
            special_tokens_.id_to_token[id] = token;
        }
    }

    // Load config and resolve BOS/EOS from tokenizer_config.json (authoritative source)
    std::filesystem::path tokenizer_config_path(tokenizer_path);
    if (tokenizer_config_path.filename() == "tokenizer.json") {
        tokenizer_config_path.replace_filename("tokenizer_config.json");
        load_config_file(tokenizer_config_path.string());
    }
}

std::vector<int> HFTokenizer::encode(const std::string& text) {
    std::vector<int> result;

    if (config_.add_bos_token && special_tokens_.bos_token_id != -1) {
        result.push_back(special_tokens_.bos_token_id);
    }

    // Split text by special tokens
    std::vector<std::string> segments = split_by_special_tokens(text);
    for (const auto& segment : segments) {
        if (special_tokens_.is_special_token(segment)) {
            result.push_back(special_tokens_.token_to_id[segment]);
        } else {
            auto words = pre_tokenize(segment);
            for (const auto& word : words) {
                auto word_tokens = bpe_encode(word);
                result.insert(result.end(), word_tokens.begin(), word_tokens.end());
            }
        }
    }

    if (config_.add_eos_token && special_tokens_.eos_token_id != -1) {
        result.push_back(special_tokens_.eos_token_id);
    }

    // Truncate to model_max_length, preserving EOS if present
    if (result.size() > static_cast<size_t>(config_.model_max_length)) {
        bool has_eos = false;
        if (config_.add_eos_token && !result.empty() && result.back() == special_tokens_.eos_token_id) {
            result.pop_back();
            has_eos = true;
        }

        if (has_eos) {
            result.resize(config_.model_max_length - 1);
            result.push_back(special_tokens_.eos_token_id);
        } else {
            result.resize(config_.model_max_length);
        }
    }

    return result;
}

std::string HFTokenizer::decode(const std::vector<int>& tokens) {
    std::vector<std::pair<std::string, bool>> parts; // <content, needs byte decode>

    for (int token_id : tokens) {
        if (special_tokens_.is_special_token_id(token_id)) {
            const std::string& token_str = special_tokens_.id_to_token[token_id];
            if (token_id == special_tokens_.bos_token_id) {
                continue;
            }
            if (token_id == special_tokens_.eos_token_id) {
                parts.push_back({"\n", false});
                continue;
            }
            parts.push_back({token_str, false});
            continue;
        }

        auto it = id_to_token_.find(token_id);
        if (it != id_to_token_.end()) {
            parts.push_back({it->second, true});
        }
    }

    // Decode segments
    std::string final_result;
    for (const auto& [content, needs_byte_decode] : parts) {
        if (needs_byte_decode) {
            final_result += ByteLevel::unicode_to_bytes(content);
        } else {
            final_result += content;
        }
    }

    if (config_.clean_up_tokenization_spaces) {
        final_result = cleanup_spaces(final_result);
    }

    return final_result;
}

std::string HFTokenizer::apply_chat_template(const std::vector<std::pair<std::string, std::string>>& messages,
                                             const ChatTemplate& tmpl, bool add_generation_prompt) {
    std::string result;

    if (special_tokens_.bos_token_id != -1) {
        result += tmpl.bos_token;
    }

    for (const auto& [role, content] : messages) {
        if (role == "user") {
            result += tmpl.user_prefix + content + tmpl.user_suffix;
        } else if (role == "assistant") {
            result += tmpl.assistant_prefix + content + tmpl.assistant_suffix;
        }
    }

    if (add_generation_prompt) {
        result += tmpl.generation_prompt;
    }

    return result;
}

// ------------------ Private Helpers ------------------

void HFTokenizer::load_config_file(const std::string& tokenizer_config_path) {
    std::ifstream file(tokenizer_config_path);
    if (!file.is_open()) {
        LOG_WARNING_(utils::BOTH) << "Note: tokenizer_config.json not found; using default configuration." << std::endl;
        return;
    }

    json config;
    file >> config;

    if (config.contains("add_bos_token") && config["add_bos_token"].is_boolean()) {
        config_.add_bos_token = config["add_bos_token"].get<bool>();
    }
    if (config.contains("add_eos_token") && config["add_eos_token"].is_boolean()) {
        config_.add_eos_token = config["add_eos_token"].get<bool>();
    }
    if (config.contains("clean_up_tokenization_spaces") && config["clean_up_tokenization_spaces"].is_boolean()) {
        config_.clean_up_tokenization_spaces = config["clean_up_tokenization_spaces"].get<bool>();
    }
    if (config.contains("model_max_length") && config["model_max_length"].is_number()) {
        config_.model_max_length = config["model_max_length"].get<int>();
    }

    // Extract token string from either a plain string or AddedToken object {"content": "..."}
    auto extract_token = [](const json& j, const std::string& key) -> std::string {
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
    };

    // Resolve eos_token / bos_token strings to IDs via the special token map.
    // tokenizer_config.json is the authoritative source for which token is EOS/BOS.
    std::string eos_str = extract_token(config, "eos_token");
    std::string bos_str = extract_token(config, "bos_token");

    if (!eos_str.empty()) {
        auto it = special_tokens_.token_to_id.find(eos_str);
        if (it != special_tokens_.token_to_id.end()) {
            special_tokens_.eos_token_id = it->second;
        }
    }
    if (!bos_str.empty()) {
        auto it = special_tokens_.token_to_id.find(bos_str);
        if (it != special_tokens_.token_to_id.end()) {
            special_tokens_.bos_token_id = it->second;
        }
    }
}

std::vector<std::string> HFTokenizer::split_by_special_tokens(const std::string& text) {
    std::vector<std::string> result;
    if (special_tokens_.token_to_id.empty()) {
        result.push_back(text);
        return result;
    }

    // Build regex to match any special token
    std::string pattern = "(";
    bool first = true;
    for (const auto& [token, _] : special_tokens_.token_to_id) {
        if (!first) {
            pattern += "|";
        }
        pattern += regex_escape(token);
        first = false;
    }
    pattern += ")";

    UErrorCode status = U_ZERO_ERROR;
    icu::UnicodeString uText = icu::UnicodeString::fromUTF8(text);
    icu::UnicodeString uPattern = icu::UnicodeString::fromUTF8(pattern);

    std::unique_ptr<icu::RegexPattern> compiled_pattern(icu::RegexPattern::compile(uPattern, 0, status));

    if (U_FAILURE(status)) {
        LOG_ERROR_(utils::BOTH) << "[HFTokenizer] Special token regex compilation failed: " << u_errorName(status)
                                << std::endl;
        result.push_back(text);
        return result;
    }

    std::unique_ptr<icu::RegexMatcher> matcher(compiled_pattern->matcher(uText, status));

    if (U_FAILURE(status)) {
        result.push_back(text);
        return result;
    }

    int32_t last_end = 0;
    while (matcher->find(status) && U_SUCCESS(status)) {
        int32_t start = matcher->start(status);
        int32_t end = matcher->end(status);

        if (start > last_end) {
            icu::UnicodeString segment = uText.tempSubString(last_end, start - last_end);
            std::string utf8_segment;
            segment.toUTF8String(utf8_segment);
            if (!utf8_segment.empty()) {
                result.push_back(utf8_segment);
            }
        }

        icu::UnicodeString special_token = matcher->group(status);
        std::string utf8_special;
        special_token.toUTF8String(utf8_special);
        result.push_back(utf8_special);

        last_end = end;
    }

    if (last_end < uText.length()) {
        icu::UnicodeString segment = uText.tempSubString(last_end);
        std::string utf8_segment;
        segment.toUTF8String(utf8_segment);
        if (!utf8_segment.empty()) {
            result.push_back(utf8_segment);
        }
    }

    if (result.empty()) {
        result.push_back(text);
    }

    return result;
}

// Escape regex metacharacters
std::string HFTokenizer::regex_escape(const std::string& str) {
    static const std::string special_chars = R"(\.^$*+?()[]{}|\)";
    std::string result;
    result.reserve(str.length() * 2);

    for (char c : str) {
        if (special_chars.find(c) != std::string::npos) {
            result += '\\';
        }
        result += c;
    }
    return result;
}

std::vector<std::string> HFTokenizer::pre_tokenize(const std::string& text) {
    std::vector<std::string> words;
    if (text.empty()) {
        return words;
    }

    if (pattern_.empty() || !pre_tokenize_regex_) {
        throw std::runtime_error("Tokenizer not properly initialized: missing regex pattern or compiled regex object.");
    }

    UErrorCode status = U_ZERO_ERROR;
    icu::UnicodeString uText = icu::UnicodeString::fromUTF8(text);
    std::unique_ptr<icu::RegexMatcher> matcher(pre_tokenize_regex_->matcher(uText, status));

    if (U_FAILURE(status)) {
        LOG_ERROR_(utils::BOTH) << "[HFTokenizer] Matcher creation failed: " << u_errorName(status) << std::endl;
        words.push_back(text);
        return words;
    }

    while (matcher->find(status) && U_SUCCESS(status)) {
        icu::UnicodeString uMatch = matcher->group(status);
        std::string word;
        uMatch.toUTF8String(word);
        words.push_back(word);
    }

    if (U_FAILURE(status)) {
        LOG_ERROR_(utils::BOTH) << "[HFTokenizer] Matching error: " << u_errorName(status) << std::endl;
    }

    if (words.empty() && !text.empty()) {
        words.push_back(text);
    }

    return words;
}

std::vector<int> HFTokenizer::bpe_encode(const std::string& word) {
    if (word.empty()) {
        return {};
    }

    auto vocab_it = vocab_.find(word);
    if (vocab_it != vocab_.end()) {
        return {vocab_it->second};
    }

    // Apply byte-level encoding
    std::string byte_encoded_word = ByteLevel::bytes_to_unicode(word);

    // Split into Unicode characters
    std::vector<std::string> chars;
    icu::UnicodeString uWord = icu::UnicodeString::fromUTF8(byte_encoded_word);
    for (int32_t i = 0; i < uWord.length(); ++i) {
        icu::UnicodeString uChar = uWord.tempSubString(i, 1);
        std::string char_str;
        uChar.toUTF8String(char_str);
        chars.push_back(char_str);
    }

    // Perform BPE merges
    while (chars.size() > 1) {
        auto best_pair = find_best_mergeable_pair(chars);
        if (best_pair.first == -1) {
            break;
        }
        chars = merge_pair(chars, best_pair.first, best_pair.second);
    }

    // Map to token IDs
    std::vector<int> result;
    for (const auto& token : chars) {
        auto it = vocab_.find(token);
        if (it != vocab_.end()) {
            result.push_back(it->second);
        } else {
            LOG_ERROR_(utils::BOTH) << "Token not in vocab after BPE: [" << token << "]" << std::endl;
            if (special_tokens_.unk_token_id != -1) {
                result.push_back(special_tokens_.unk_token_id);
            } else {
                throw std::runtime_error("Token not in vocab and no UNK token: " + token);
            }
        }
    }

    return result;
}

std::string HFTokenizer::cleanup_spaces(const std::string& text) {
    UErrorCode status = U_ZERO_ERROR;
    icu::UnicodeString uText = icu::UnicodeString::fromUTF8(text);

    auto apply_regex_replace = [&](const char* pattern, const char* replacement) -> bool {
        status = U_ZERO_ERROR;
        icu::UnicodeString uPattern = icu::UnicodeString::fromUTF8(pattern);
        std::unique_ptr<icu::RegexPattern> compiled_pattern(icu::RegexPattern::compile(uPattern, 0, status));
        if (U_FAILURE(status)) {
            return false;
        }

        std::unique_ptr<icu::RegexMatcher> matcher(compiled_pattern->matcher(uText, status));
        if (U_FAILURE(status)) {
            return false;
        }

        icu::UnicodeString uReplacement = icu::UnicodeString::fromUTF8(replacement);
        uText = matcher->replaceAll(uReplacement, status);
        return U_SUCCESS(status);
    };

    // Remove spaces before punctuation
    apply_regex_replace(R"(\s+([.,!?;:]))", "$1");
    // Collapse multiple spaces
    apply_regex_replace(R"(\s{2,})", " ");

    std::string result;
    uText.toUTF8String(result);
    return result;
}

// Find the highest-priority mergeable pair
std::pair<int, int> HFTokenizer::find_best_mergeable_pair(const std::vector<std::string>& chars) {
    int best_rank = INT_MAX;
    int best_i = -1;

    for (size_t i = 0; i < chars.size() - 1; ++i) {
        std::string pair = chars[i] + " " + chars[i + 1];
        auto it = merges_.find(pair);
        if (it != merges_.end() && it->second < best_rank) {
            best_rank = it->second;
            best_i = static_cast<int>(i);
        }
    }

    return {best_i, best_i == -1 ? -1 : best_i + 1};
}

// Merge a pair of adjacent tokens
std::vector<std::string> HFTokenizer::merge_pair(const std::vector<std::string>& chars, int i, int j) {
    std::vector<std::string> result;
    for (size_t idx = 0; idx < chars.size(); ++idx) {
        if (static_cast<int>(idx) == i) {
            result.push_back(chars[i] + chars[j]);
        } else if (static_cast<int>(idx) != j) {
            result.push_back(chars[idx]);
        }
    }
    return result;
}

} // namespace zedinfer::tokenizer