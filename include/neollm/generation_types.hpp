#pragma once

#include <functional>
#include <string>

namespace neollm {

/**
 * Generation operation mode
 */
enum class GenerationMode {
    PING, // Single-turn generation without context
    CHAT  // Multi-turn conversation with history
};

/**
 * Configuration for text generation
 */
struct GenerationConfig {
    // Generation limits
    int max_new_tokens = 512;
    GenerationMode gen_mode = GenerationMode::PING;

    // Streaming output
    bool stream = false;
    std::function<void(const std::string &)> stream_callback = nullptr;

    // Diagnostics
    bool verbose = false;
    bool print_stats = false;

    /**
     * Validate configuration parameters
     * @throws std::invalid_argument if invalid
     */
    void validate() const;

    /**
     * Get human-readable configuration summary
     */
    std::string info() const;
};

/**
 * Performance metrics for generation
 */
struct GenerationStats {
    // Token counts
    int prompt_tokens = 0;
    int generated_tokens = 0;
    int total_tokens = 0;

    // Timing (milliseconds)
    double prefill_time_ms = 0.0;
    double decode_time_ms = 0.0;
    double total_time_ms = 0.0;

    /**
     * Compute prefill throughput (tokens/sec)
     */
    double prefill_tokens_per_second() const {
        return prefill_time_ms > 0.0 ? (prompt_tokens * 1000.0 / prefill_time_ms) : 0.0;
    }

    /**
     * Compute decode throughput (tokens/sec)
     */
    double decode_tokens_per_second() const {
        return decode_time_ms > 0.0 ? (generated_tokens * 1000.0 / decode_time_ms) : 0.0;
    }

    /**
     * Generate formatted statistics summary
     */
    std::string summary() const;
};

} // namespace neollm