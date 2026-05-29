#pragma once

#include <functional>
#include <string>
#include <vector>

namespace zedinfer {

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

    // For reasoning models (Qwen3.5 family) with a closed-think variant in
    // their chat template: enable_thinking=true uses generation_prompt (the
    // open-<think> form), enable_thinking=false uses generation_prompt_no_think
    // (closed-<think> empty block, direct-answer branch). Falls back to
    // generation_prompt if no closed-think variant exists.
    //
    // Default `false`: on Qwen3.5-GPTQ-Int4 the open-<think> branch hits a
    // fragile region of the model where the prefill state after `<think>\n`
    // (empty thinking start) causes the decoder to hallucinate the prompt
    // content for the first ~30 tokens of thinking. Closed-think skips that
    // region entirely and produces a clean direct answer. Higher-precision
    // weight sets (bf16/fp16) likely escape the fragility and may benefit
    // from enable_thinking=true; opt in explicitly when you trust the
    // weights to handle empty-thinking-start cleanly.
    bool enable_thinking = false;

    // Maximum tokens allowed inside an open <think>...</think> block before
    // the scheduler force-emits </think> to escape long-generation drift seen
    // on GPTQ-Int4 reasoning models. 0 disables the budget (model is free to
    // emit </think> whenever it wants). Only meaningful when enable_thinking
    // is true and the model emits explicit <think>/</think> tokens.
    int max_think_tokens = 128;

    // Streaming output
    bool stream = false;
    std::function<void(const std::string&)> stream_callback = nullptr;

    // Per-request sampling overrides (OpenAI-compatible).
    //   has_sampling_override : true if the HTTP layer (or any caller) set any
    //                           of the fields below; otherwise the engine uses
    //                           its default sampler verbatim.
    //   use_argmax            : route through ArgmaxSampler instead of
    //                           GeneralSampler. Set by the HTTP layer when the
    //                           client sends temperature == 0.
    //   temperature/top_k/top_p/repetition_penalty/seed : standard sampling
    //                           knobs. Applied to GeneralSampler before sample()
    //                           when use_argmax is false.
    //   stop_sequences        : string-level stop substrings checked by the
    //                           HTTP layer on the decoded output stream.
    bool         has_sampling_override = false;
    bool         use_argmax            = false;
    float        temperature           = 1.0f;
    int          top_k                 = 0;
    float        top_p                 = 1.0f;
    float        repetition_penalty    = 1.0f;
    unsigned int seed                  = 0;
    std::vector<std::string> stop_sequences;

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

} // namespace zedinfer