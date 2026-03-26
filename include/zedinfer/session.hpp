#pragma once

#include "backend/kvcache/block_pool.hpp"
#include "zedinfer/chat_template.hpp"
#include "zedinfer/generation_types.hpp"

#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace zedinfer {

// Forward declaration
class InferenceEngine;

namespace kvcache {
class BlockAllocator;
}

/**
 * Inference session managing conversation state and KV cache blocks.
 * Each session maintains independent dialogue history and block table.
 */
class InferenceSession {
public:
    ~InferenceSession();

    std::string chat(const std::string& user_input);
    void reset();

    const std::string& session_id() const { return session_id_; }
    const std::vector<std::pair<std::string, std::string>>& chat_history() const { return chat_history_; }
    int past_len() const { return past_len_; }

    // Async chat support (for HTTP handlers that use submit_async instead of blocking generate)
    std::string prepare_prompt(const std::string& user_input);
    kvcache::SequenceBlockTable& block_table() { return block_table_; }
    void complete_turn(const std::string& user_input, const std::string& raw_output);
    // Recover session state after a failed generation (sync past_len with block table)
    void abort_turn();
    // Check if server-side KV cache is consistent (detects session expiry/recreation)
    bool is_valid() const { return block_table_.num_layers > 0; }

    const std::string& output_prefix() const { return template_.output_prefix; }
    const ChatTemplate& chat_tmpl() const { return template_; }

private:
    friend class InferenceEngine;

    InferenceSession(std::shared_ptr<InferenceEngine> engine, kvcache::SequenceBlockTable block_table,
                     kvcache::BlockAllocator* allocator, const GenerationConfig& gen_config,
                     const ChatTemplate& chat_template);

    static std::string generate_uuid();

    // Core components
    std::shared_ptr<InferenceEngine> engine_;
    std::string session_id_;
    kvcache::SequenceBlockTable block_table_; // Per-session KV block ownership
    kvcache::BlockAllocator* allocator_;      // For freeing blocks on destroy/reset
    GenerationConfig config_;
    ChatTemplate template_;

    // Conversation state
    std::vector<std::pair<std::string, std::string>> chat_history_;
    int past_len_;
    bool is_first_turn_;
};

} // namespace zedinfer
