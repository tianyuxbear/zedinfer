#include "zedinfer/generation_types.hpp"

#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace zedinfer {

// ============================================================================
// GenerationConfig
// ============================================================================

void GenerationConfig::validate() const {
    if (max_new_tokens <= 0) {
        throw std::invalid_argument("max_new_tokens must be positive");
    }

    if (stream && !stream_callback) {
        throw std::invalid_argument("stream_callback required when stream=true");
    }
}

std::string GenerationConfig::info() const {
    std::ostringstream oss;
    oss << "\n=== GenerationConfig: ===\n"
        << "  Mode: " << (gen_mode == GenerationMode::CHAT ? "CHAT" : "PING") << "\n"
        << "  Max tokens: " << max_new_tokens << "\n"
        << "  Stream: " << std::boolalpha << stream << "\n"
        << "  Verbose: " << std::boolalpha << verbose;
    return oss.str();
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
        << "  Prefill:  " << prefill_time_ms << " ms ("
        << prefill_tokens_per_second() << " token/s)\n"
        << "  Decode:   " << decode_time_ms << " ms ("
        << decode_tokens_per_second() << " token/s)\n"
        << "  Total:    " << total_time_ms << " ms\n"
        << "=============================";

    return oss.str();
}

} // namespace zedinfer