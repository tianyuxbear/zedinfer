#include "neollm/chat.hpp"

#include <iostream>

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
    std::vector<int> input_ids = engine_->tokenizer()->encode(input);

    auto past_len = engine_->kv_cache()->current_length();
    std::cout << "past_len" << past_len << std::endl;
    std::vector<int> output_ids = engine_->generate_tokens(input_ids, gen_config_, past_len);

    output += "<think>";
    output += engine_->tokenizer()->decode(output_ids);
    message_history_.push_back({"assistant", output});
}
} // namespace neollm