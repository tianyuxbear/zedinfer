#pragma once

#include "zedinfer/generation_types.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace zedinfer {

/**
 * Result of a single generation call.
 * Returned by InferenceEngine::generate_tokens() and consumed by generate().
 */
struct GenerationResult {
    std::vector<int> output_ids;
    GenerationStats stats;
};

/**
 * Phase of an inference request lifecycle.
 * Used by the scheduler to track request progress.
 */
enum class RequestPhase {
    QUEUED,   // waiting for scheduler admission
    PREFILL,  // processing prompt tokens
    DECODE,   // generating tokens one-by-one
    COMPLETE  // finished (EOS, max_tokens, or error)
};

/**
 * A single inference request.
 * Data definition for future scheduler integration.
 */
struct InferenceRequest {
    uint64_t request_id = 0;
    std::string session_id;
    std::vector<int> input_ids;
    GenerationConfig config;

    RequestPhase phase = RequestPhase::QUEUED;
    int generated_count = 0;
    int last_token = -1;

    std::vector<int> output_ids;
};

} // namespace zedinfer
