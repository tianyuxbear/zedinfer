#include "frontend/tokenizer/hf_tokenizer.hpp"
#include "frontend/tokenizer/byte_level.hpp"
#include "utils/logging.hpp"

#include <algorithm>
#include <cctype>
#include <climits>
#include <filesystem>
#include <fstream>
#include <immintrin.h>
#include <iostream>
#include <nlohmann/json.hpp>
#include <plog/Log.h>
#include <thread>

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
    precompute_int_merges();
}

std::vector<int> HFTokenizer::encode(const std::string& text) {
    // Set threshold for parallel execution: 8KB
    const size_t PARALLEL_THRESHOLD = 8 * 1024;

    std::vector<int> core_tokens;

    if (text.size() < PARALLEL_THRESHOLD) {
        core_tokens = encode_core(text);
    } else {
        core_tokens = encode_parallel(text);
    }

    std::vector<int> result;
    // Reserve space for BOS and EOS tokens
    result.reserve(core_tokens.size() + 2);

    if (config_.add_bos_token && special_tokens_.bos_token_id != -1) {
        result.push_back(special_tokens_.bos_token_id);
    }

    result.insert(result.end(), core_tokens.begin(), core_tokens.end());

    if (config_.add_eos_token && special_tokens_.eos_token_id != -1) {
        result.push_back(special_tokens_.eos_token_id);
    }

    // Truncate to model_max_length, preserving EOS if present
    if (config_.model_max_length > 0 && result.size() > static_cast<size_t>(config_.model_max_length)) {
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
    try {
        file >> config;
    } catch (...) { return; }

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

    // Return early if the full word is already in the vocabulary
    auto vocab_it = vocab_.find(word);
    if (vocab_it != vocab_.end()) {
        return {vocab_it->second};
    }

    // Map raw bytes directly to Token IDs: Raw Byte -> Byte-Level Unicode -> Vocab ID
    std::vector<int> ids;
    ids.reserve(word.size());

    for (unsigned char byte : word) {
        const std::string& char_str = ByteLevel::byte_to_unicode(byte);

        auto it = vocab_.find(char_str);
        if (it != vocab_.end()) {
            ids.push_back(it->second);
        } else {
            // UNK
            if (special_tokens_.unk_token_id != -1) {
                ids.push_back(special_tokens_.unk_token_id);
            } else {
                throw std::runtime_error("Token not in vocab and no UNK token: " + char_str);
            }
        }
    }

    // Perform merges on integer sequences
    while (ids.size() > 1) {
        std::pair<int, MergeRule> result = find_best_mergeable_pair_int(ids);
        int best_idx = result.first;
        if (best_idx == -1) {
            break;
        }
        ids[best_idx] = result.second.new_id;
        for (size_t i = best_idx + 1; i < ids.size() - 1; ++i) { ids[i] = ids[i + 1]; }
        ids.pop_back();
    }

    return ids;
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

// Precompute integer merge rules
void HFTokenizer::precompute_int_merges() {
    if (merges_.empty() || vocab_.empty()) {
        return;
    }

    int_merges_.clear();
    int_merges_.reserve(merges_.size());

    for (const auto& entry : merges_) {
        const std::string& pair_str = entry.first;
        int rank = entry.second;

        size_t space_pos = pair_str.find(' ');
        if (space_pos == std::string::npos) {
            // Skip entries that are not valid BPE merge pairs (i.e., do not contain a space).
            continue;
        }

        std::string p1 = pair_str.substr(0, space_pos);
        std::string p2 = pair_str.substr(space_pos + 1);

        auto it1 = vocab_.find(p1);
        auto it2 = vocab_.find(p2);

        if (it1 != vocab_.end() && it2 != vocab_.end()) {
            int id1 = it1->second;
            int id2 = it2->second;
            std::string combined = p1 + p2;
            auto it_combined = vocab_.find(combined);

            if (it_combined != vocab_.end()) {
                int new_id = it_combined->second;
                int_merges_[get_pair_key(id1, id2)] = {rank, new_id};
            }
        }
    }
}

std::pair<int, HFTokenizer::MergeRule> HFTokenizer::find_best_mergeable_pair_int(const std::vector<int>& ids) const {
    int best_idx = -1;
    MergeRule best_rule = {2147483647, -1}; // INT_MAX

    if (ids.size() < 2) {
        return {best_idx, best_rule};
    }

    for (size_t i = 0; i < ids.size() - 1; ++i) {
        uint64_t key = get_pair_key(ids[i], ids[i + 1]);
        auto it = int_merges_.find(key);
        if (it != int_merges_.end()) {
            if (it->second.rank < best_rule.rank) {
                best_rule = it->second;
                best_idx = static_cast<int>(i);
            }
        }
    }

    return {best_idx, best_rule};
}


// Helper: SIMD-accelerated search for safe split points
__attribute__((target("avx2"))) inline void scan_boundaries_simd(const char* data, size_t len, size_t& best_p1,
                                                                 size_t& best_p2, size_t& best_p3) {
    __m256i v_newline = _mm256_set1_epi8('\n'); // Priority 1
    __m256i v_dot = _mm256_set1_epi8('.');      // Priority 2
    __m256i v_excl = _mm256_set1_epi8('!');
    __m256i v_ques = _mm256_set1_epi8('?');
    __m256i v_space = _mm256_set1_epi8(' '); // Priority 3
    __m256i v_tab = _mm256_set1_epi8('\t');

    for (size_t i = 0; i + 32 <= len; i += 32) {
        __m256i v_data = _mm256_loadu_si256((const __m256i*)(data + i));

        if (best_p1 == 0) {
            uint32_t m1 = _mm256_movemask_epi8(_mm256_cmpeq_epi8(v_data, v_newline));
            if (m1 != 0) {
                best_p1 = i + __builtin_ctz(m1) + 1;
                return;
            }
        }

        if (best_p2 == 0) {
            uint32_t m2 = _mm256_movemask_epi8(
                _mm256_or_si256(_mm256_or_si256(_mm256_cmpeq_epi8(v_data, v_dot), _mm256_cmpeq_epi8(v_data, v_excl)),
                                _mm256_cmpeq_epi8(v_data, v_ques)));
            if (m2 != 0) {
                best_p2 = i + __builtin_ctz(m2) + 1;
            }
        }

        if (best_p3 == 0) {
            uint32_t m3 = _mm256_movemask_epi8(
                _mm256_or_si256(_mm256_cmpeq_epi8(v_data, v_space), _mm256_cmpeq_epi8(v_data, v_tab)));
            if (m3 != 0) {
                best_p3 = i + __builtin_ctz(m3) + 1;
            }
        }
    }
}

size_t HFTokenizer::find_safe_split_point(const std::string& text, size_t target_pos) const {
    if (target_pos >= text.size()) {
        return text.size();
    }

    size_t search_limit = std::min(text.size(), target_pos + 5000);
    size_t len = search_limit - target_pos;

    size_t best_p1 = 0, best_p2 = 0, best_p3 = 0;

    scan_boundaries_simd(text.data() + target_pos, len, best_p1, best_p2, best_p3);

    if (best_p1 != 0) {
        return target_pos + best_p1;
    }
    if (best_p2 != 0) {
        return target_pos + best_p2;
    }
    if (best_p3 != 0) {
        return target_pos + best_p3;
    }

    return search_limit;
}

std::vector<int> HFTokenizer::encode_core(const std::string& text) {
    std::vector<int> result;

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

    return result;
}

std::vector<int> HFTokenizer::encode_parallel(const std::string& text) {
    int num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0) {
        num_threads = 1;
    }

    std::vector<size_t> boundaries;
    boundaries.push_back(0);

    size_t rough_chunk_size = text.size() / num_threads;
    size_t current_pos = 0;

    for (int i = 0; i < num_threads - 1; ++i) {
        current_pos += rough_chunk_size;
        if (current_pos >= text.size()) {
            break;
        }

        size_t safe_pos = find_safe_split_point(text, current_pos);
        if (safe_pos < text.size()) {
            boundaries.push_back(safe_pos);
            current_pos = safe_pos;
        } else {
            break;
        }
    }
    boundaries.push_back(text.size());

    int actual_chunks = boundaries.size() - 1;
    std::vector<std::vector<int>> partial_results(actual_chunks);

#pragma omp parallel for
    for (int i = 0; i < actual_chunks; ++i) {
        size_t start = boundaries[i];
        size_t len = boundaries[i + 1] - start;
        partial_results[i] = encode_core(text.substr(start, len));
    }

    std::vector<int> final_result;
    size_t total_tokens = 0;
    for (const auto& res : partial_results) { total_tokens += res.size(); }
    final_result.reserve(total_tokens);

    for (auto& res : partial_results) { final_result.insert(final_result.end(), res.begin(), res.end()); }

    return final_result;
}

} // namespace zedinfer::tokenizer