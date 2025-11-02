#include "neollm/chat.hpp"

namespace neollm {
void ChatSession::chat(const std::string &user_input) {
    std::string input, output;
    if (is_first_turn) {
        input += "<｜begin▁of▁sentence｜>";
        is_first_turn = false;
    }
    input += "<｜User｜>" + user_input;
    input += "<｜Assistant｜><think>\n";
    message_history_.push_back({"user", user_input});

    output += "<think> ";
    auto past_len = engine_->kv_cache()->current_length();
    output += engine_->generate(input, gen_config_, past_len);

    message_history_.push_back({"assistant", output});
}
} // namespace neollm