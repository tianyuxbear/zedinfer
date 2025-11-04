#include "neollm/session.hpp"
#include "neollm/engine.hpp"

#include <iomanip>
#include <random>
#include <sstream>

namespace neollm {

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
    const GenerationConfig &gen_config)
    : engine_(std::move(engine)), session_id_(generate_uuid()), kvcache_(std::move(kvcache)), config_(gen_config), past_len_(0), is_first_turn_(true) {
}

std::string InferenceSession::chat(const std::string &user_input) {
    std::string input;

    // Add BOS token on first turn
    if (is_first_turn_) {
        input += "<｜begin▁of▁sentence｜>";
        is_first_turn_ = false;
    }

    // Build prompt with chat template
    input += "<｜User｜>" + user_input;
    input += "<｜Assistant｜><think>\n";

    // Save user message to history
    chat_history_.push_back({"user", user_input});

    // Generate response using engine
    std::string output = "<think> ";
    output += engine_->generate(*kvcache_, input, config_);

    // Save assistant response to history
    chat_history_.push_back({"assistant", output});

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

} // namespace neollm