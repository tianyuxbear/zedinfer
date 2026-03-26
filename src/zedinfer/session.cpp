#include "zedinfer/session.hpp"
#include "backend/kvcache/block_pool.hpp"
#include "zedinfer/engine.hpp"

#include <iomanip>
#include <random>
#include <sstream>

namespace zedinfer {

static std::string clean_output(const std::string& raw, const ChatTemplate& tmpl) {
    std::string text = raw;

    if (!tmpl.eos_token.empty()) {
        size_t pos = text.rfind(tmpl.eos_token);
        if (pos != std::string::npos && pos + tmpl.eos_token.size() == text.size()) {
            text.erase(pos);
        }
    }

    size_t end = text.find_last_not_of(" \t\n\r");
    if (end == std::string::npos) {
        return "";
    }
    return text.substr(0, end + 1);
}

std::string InferenceSession::generate_uuid() {
    static std::random_device rd;
    static std::mt19937_64 gen(rd());
    static std::uniform_int_distribution<uint64_t> dis;

    uint64_t high = dis(gen);
    uint64_t low = dis(gen);

    std::ostringstream oss;
    oss << std::hex << std::setfill('0') << std::setw(16) << high << std::setw(16) << low;

    return oss.str();
}

InferenceSession::InferenceSession(std::shared_ptr<InferenceEngine> engine, kvcache::SequenceBlockTable block_table,
                                   kvcache::BlockAllocator* allocator, const GenerationConfig& gen_config,
                                   const ChatTemplate& chat_template)
    : engine_(std::move(engine)),
      session_id_(generate_uuid()),
      block_table_(std::move(block_table)),
      allocator_(allocator),
      config_(gen_config),
      template_(chat_template),
      past_len_(0),
      is_first_turn_(true) {}

InferenceSession::~InferenceSession() {
    if (allocator_) {
        allocator_->release_sequence(block_table_);
    }
}

std::string InferenceSession::chat(const std::string& user_input) {
    std::string input;

    if (is_first_turn_) {
        input += template_.bos_token;
        is_first_turn_ = false;
    }

    input += template_.user_prefix + user_input + template_.user_suffix;
    input += template_.generation_prompt;

    chat_history_.push_back({"user", user_input});

    if (!template_.output_prefix.empty() && config_.stream && config_.stream_callback) {
        config_.stream_callback(template_.output_prefix);
    }

    // Generate using block table directly (no KVCache object needed)
    std::string raw_output = engine_->serving_loop().generate(block_table_, input, config_);

    std::string output = template_.output_prefix + clean_output(raw_output, template_);

    chat_history_.push_back({"assistant", raw_output});

    past_len_ = block_table_.seq_len;

    return output;
}

std::string InferenceSession::prepare_prompt(const std::string& user_input) {
    std::string input;
    if (is_first_turn_) {
        input += template_.bos_token;
    }
    input += template_.user_prefix + user_input + template_.user_suffix;
    input += template_.generation_prompt;
    return input;
}

void InferenceSession::complete_turn(const std::string& user_input, const std::string& raw_output) {
    is_first_turn_ = false;
    chat_history_.push_back({"user", user_input});
    chat_history_.push_back({"assistant", raw_output});
    past_len_ = block_table_.seq_len;
}

void InferenceSession::abort_turn() {
    // After a failed generation, the block table's seq_len may have advanced
    // (tokens were written to blocks during prefill/decode before the error).
    // Sync past_len_ so the next turn doesn't have a gap.
    past_len_ = block_table_.seq_len;
}

void InferenceSession::reset() {
    if (allocator_) {
        allocator_->free_sequence(block_table_);
        block_table_ = allocator_->allocate_sequence(256);
    }
    chat_history_.clear();
    past_len_ = 0;
    is_first_turn_ = true;
}

} // namespace zedinfer
