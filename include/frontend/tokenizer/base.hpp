#pragma once

#include <string>
#include <unordered_map>
#include <vector>

namespace neollm::tokenizer {

struct Config;

// Abstract base class for tokenizers.
class Tokenizer {
public:
    virtual ~Tokenizer() = default;

    // Encode text to token IDs.
    virtual std::vector<int> encode(const std::string &text) = 0;

    // Decode token IDs to text.
    virtual std::string decode(const std::vector<int> &tokens) = 0;

    // Return vocabulary size.
    virtual int get_vocab_size() const = 0;

    // Special token IDs; return -1 if not defined.
    virtual int get_bos_token_id() const { return -1; }
    virtual int get_eos_token_id() const { return -1; }
    virtual int get_pad_token_id() const { return -1; }
    virtual int get_unk_token_id() const { return -1; }

    // Access tokenizer config.
    virtual const Config &get_config() const = 0;

    // Load tokenizer from file.
    virtual void load_from_file(const std::string &path) = 0;
};

// Holds special token mappings and IDs.
struct SpecialTokens {
    std::unordered_map<std::string, int> token_to_id;
    std::unordered_map<int, std::string> id_to_token;

    int bos_token_id = -1;
    int eos_token_id = -1;
    int pad_token_id = -1;
    int unk_token_id = -1;

    bool is_special_token(const std::string &token) const {
        return token_to_id.count(token) > 0;
    }

    bool is_special_token_id(int id) const {
        return id_to_token.count(id) > 0;
    }
};

// Tokenizer configuration options.
struct Config {
    bool add_bos_token = true;
    bool add_eos_token = false;
    bool clean_up_tokenization_spaces = false;
    int model_max_length = 16384;
};

} // namespace neollm::tokenizer