#pragma once

#include "backend/kvcache/base.hpp"
#include "neollm/generation_types.hpp"

#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace neollm {

// Forward declaration
class InferenceEngine;

/**
 * Inference session managing conversation state and KV cache
 * Each session maintains independent dialogue history and context
 */
class InferenceSession {
public:
    /**
     * Multi-turn chat interface
     * @param user_input User message
     * @return Assistant response
     */
    std::string chat(const std::string &user_input);

    /**
     * Reset session state (clear history and KV cache)
     */
    void reset();

    /**
     * Get unique session identifier
     */
    const std::string &session_id() const { return session_id_; }

    /**
     * Get conversation history [(role, content), ...]
     */
    const std::vector<std::pair<std::string, std::string>> &chat_history() const {
        return chat_history_;
    }

    /**
     * Get current KV cache length
     */
    int past_len() const { return past_len_; }

private:
    // Only InferenceEngine can create sessions
    friend class InferenceEngine;

    /**
     * Private constructor - use InferenceEngine::create_session()
     */
    InferenceSession(
        std::shared_ptr<InferenceEngine> engine,
        kvcache::kvcache_t kvcache,
        const GenerationConfig &gen_config);

    /**
     * Generate UUID for session identification
     */
    static std::string generate_uuid();

    // Core components
    std::shared_ptr<InferenceEngine> engine_; // Shared stateless engine
    std::string session_id_;                  // Unique session ID
    kvcache::kvcache_t kvcache_;              // Session-specific KV cache
    GenerationConfig config_;                 // Generation parameters

    // Conversation state
    std::vector<std::pair<std::string, std::string>> chat_history_; // [(role, content), ...]
    int past_len_;                                                  // Cached token count
    bool is_first_turn_;                                            // First turn flag for BOS token
};

} // namespace neollm