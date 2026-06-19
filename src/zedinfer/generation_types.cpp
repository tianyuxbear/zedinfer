#include "zedinfer/generation_types.hpp"

#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace zedinfer {

// ============================================================================
// GenerationConfig
// ============================================================================

void GenerationConfig::validate() const {
    if (max_new_tokens < 0) {
        throw std::invalid_argument("max_new_tokens must be non-negative (0 means unlimited)");
    }

    if (max_think_tokens < 0) {
        throw std::invalid_argument("max_think_tokens must be non-negative (0 disables the budget)");
    }

    if (stream && !stream_callback) {
        throw std::invalid_argument("stream_callback required when stream=true");
    }
}

std::string GenerationConfig::info() const {
    std::ostringstream oss;
    oss << "\n=== GenerationConfig: ===\n"
        << "  Mode: " << (gen_mode == GenerationMode::CHAT ? "CHAT" : "PING") << "\n"
        << "  Max tokens: " << (max_new_tokens == 0 ? std::string("unlimited") : std::to_string(max_new_tokens)) << "\n"
        << "  Enable thinking: " << std::boolalpha << enable_thinking << "\n"
        << "  Max think tokens: " << max_think_tokens << "\n"
        << "  Stream: " << std::boolalpha << stream << "\n"
        << "  Verbose: " << std::boolalpha << verbose;
    return oss.str();
}

int resolve_max_new_tokens(int requested_max_new_tokens, int used_context_tokens, int max_seq_len) {
    if (requested_max_new_tokens < 0) {
        throw std::invalid_argument("max_new_tokens must be non-negative (0 means unlimited)");
    }
    if (used_context_tokens < 0) {
        throw std::invalid_argument("used_context_tokens must be non-negative");
    }
    if (max_seq_len <= 0) {
        throw std::invalid_argument("max_seq_len must be positive");
    }

    const int remaining = max_seq_len - used_context_tokens;
    if (remaining <= 0) {
        throw std::invalid_argument("prompt leaves no room for generation: context="
                                    + std::to_string(used_context_tokens) + " tokens (max "
                                    + std::to_string(max_seq_len) + ")");
    }

    if (requested_max_new_tokens == 0 || requested_max_new_tokens > remaining) {
        return remaining;
    }
    return requested_max_new_tokens;
}

// ============================================================================
// GenerationStats
// ============================================================================

std::string GenerationStats::summary() const {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2);

    oss << "\n=== Generation Statistics ===\n"
        << "Tokens:\n"
        << "  Prompt:     " << prompt_tokens << "\n"
        << "  Generated:  " << generated_tokens << "\n"
        << "  Total:      " << total_tokens << "\n"
        << "Performance:\n"
        << "  Prefill:  " << prefill_time_ms << " ms (" << prefill_tokens_per_second() << " token/s)\n"
        << "  Decode:   " << decode_time_ms << " ms (" << decode_tokens_per_second() << " token/s)\n"
        << "  Total:    " << total_time_ms << " ms\n"
        << "=============================";

    return oss.str();
}

} // namespace zedinfer
