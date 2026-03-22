#include "zedinfer/session.hpp"
#include "zedinfer/engine.hpp"

#include <iomanip>
#include <random>
#include <sstream>

namespace zedinfer {

// Strip trailing stop token markers from model output.
// Uses template eos_token — no hardcoded token strings.
static std::string clean_output(const std::string &raw, const ChatTemplate &tmpl) {
    std::string text = raw;

    // Strip template's eos_token if it appears at the end
    if (!tmpl.eos_token.empty()) {
        size_t pos = text.rfind(tmpl.eos_token);
        if (pos != std::string::npos && pos + tmpl.eos_token.size() == text.size()) {
            text.erase(pos);
        }
    }

    // Trim trailing whitespace
    size_t end = text.find_last_not_of(" \t\n\r");
    if (end == std::string::npos) return "";
    return text.substr(0, end + 1);
}

// ============================================================================
// InferenceSession
// ============================================================================

std::string InferenceSession::generate_uuid() {
    static std::random_device rd;
    static std::mt19937_64 gen(rd());
    static std::uniform_int_distribution<uint64_t> dis;

    // Generate 128-bit UUID from two 64-bit random values
    uint64_t high = dis(gen);
    uint64_t low = dis(gen);

    std::ostringstream oss;
    oss << std::hex << std::setfill('0')
        << std::setw(16) << high
        << std::setw(16) << low;

    return oss.str();
}

InferenceSession::InferenceSession(
    std::shared_ptr<InferenceEngine> engine,
    kvcache::kvcache_t kvcache,
    const GenerationConfig &gen_config,
    const ChatTemplate &chat_template)
    : engine_(std::move(engine)), session_id_(generate_uuid()), kvcache_(std::move(kvcache)), config_(gen_config), template_(chat_template), past_len_(0), is_first_turn_(true) {
}

std::string InferenceSession::chat(const std::string &user_input) {
    std::string input;

    // Add BOS token on first turn
    if (is_first_turn_) {
        input += template_.bos_token;
        is_first_turn_ = false;
    }

    // Build prompt with chat template
    input += template_.user_prefix + user_input + template_.user_suffix;
    input += template_.generation_prompt;

    // Save user message to history
    chat_history_.push_back({"user", user_input});

    // Stream output_prefix (e.g., "<think> " for DeepSeek) before model generation
    if (!template_.output_prefix.empty() && config_.stream && config_.stream_callback) {
        config_.stream_callback(template_.output_prefix);
    }

    // Generate response using engine
    std::string raw_output = engine_->generate(*kvcache_, input, config_);

    // Clean output for display: strip stop tokens, prepend output_prefix
    std::string output = template_.output_prefix + clean_output(raw_output, template_);

    // Save assistant response to history
    chat_history_.push_back({"assistant", raw_output});

    // Update cached token count
    past_len_ = kvcache_->current_length();

    return output;
}

void InferenceSession::reset() {
    kvcache_->reset();
    chat_history_.clear();
    past_len_ = 0;
    is_first_turn_ = true;
}

} // namespace zedinfer