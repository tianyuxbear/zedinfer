#pragma once

#include "engine.hpp"

#include <memory>

namespace neollm {
class ChatSession {
private:
    std::unique_ptr<InferenceEngine> engine_;
    std::vector<std::pair<std::string, std::string>> message_history_;
    GenerationConfig gen_config_;
    bool is_first_turn;

public:
    ChatSession(std::unique_ptr<InferenceEngine> engine, const GenerationConfig &gen_config) {
        engine_ = std::move(engine);
        gen_config_ = gen_config;
        is_first_turn = true;
    }
    void chat(const std::string &user_input);
};
} // namespace neollm
