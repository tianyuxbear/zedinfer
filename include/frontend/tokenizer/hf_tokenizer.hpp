#pragma once

#include "base.hpp"

#include <unicode/regex.h>
#include <unordered_map>

namespace zedinfer::tokenizer {

class HFTokenizer : public Tokenizer {
public:
    static std::shared_ptr<Tokenizer> create(const std::string &tokenizer_path);

    // Core encode/decode interface.
    std::vector<int> encode(const std::string &text) override;
    std::string decode(const std::vector<int> &tokens) override;

    // Apply chat template to message list.
    std::string apply_chat_template(
        const std::vector<std::pair<std::string, std::string>> &messages,
        bool add_generation_prompt = false);

    // Vocabulary size.
    int get_vocab_size() const override { return vocab_.size(); }

    // Special token IDs.
    int get_bos_token_id() const override { return special_tokens_.bos_token_id; }
    int get_eos_token_id() const override { return special_tokens_.eos_token_id; }
    int get_pad_token_id() const override { return special_tokens_.pad_token_id; }
    int get_unk_token_id() const override { return special_tokens_.unk_token_id; }

    // Access tokenizer config.
    const Config &get_config() const override { return config_; }

    // Load tokenizer from file.
    void load_from_file(const std::string &tokenizer_path) override;

private:
    // Vocabulary and merges.
    std::unordered_map<std::string, int> vocab_;
    std::unordered_map<int, std::string> id_to_token_;
    std::unordered_map<std::string, int> merges_;

    // Pre-tokenization regex.
    std::string pattern_;
    std::unique_ptr<icu::RegexPattern> pre_tokenize_regex_;

    // Special tokens and config.
    SpecialTokens special_tokens_;
    Config config_;

    // Load config from JSON file.
    void load_config_file(const std::string &tokenizer_config_path);

    // Pre-tokenization: split text into chunks.
    std::vector<std::string> pre_tokenize(const std::string &text);
    std::vector<std::string> split_by_special_tokens(const std::string &text);
    std::string regex_escape(const std::string &str);

    // BPE encoding logic.
    std::vector<int> bpe_encode(const std::string &word);
    std::pair<int, int> find_best_mergeable_pair(const std::vector<std::string> &chars);
    std::vector<std::string> merge_pair(const std::vector<std::string> &chars, int i, int j);

    // Post-decode cleanup.
    std::string cleanup_spaces(const std::string &text);
};

} // namespace zedinfer::tokenizer