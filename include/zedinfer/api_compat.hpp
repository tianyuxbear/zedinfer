#pragma once

#include <nlohmann/json.hpp>

#include <string>

namespace zedinfer::api_compat {

using json = nlohmann::json;

enum class ResponseFormat {
    OpenAIChat,
    OpenAIResponses,
    Anthropic,
};

json convert_responses_to_chat(const json& body);
json convert_anthropic_to_chat(const json& body);

json chat_response_to_responses(const json& chat_response);
json chat_response_to_anthropic(const json& chat_response);

std::string build_tool_prompt(const json& tools);
std::string normalize_anthropic_stop_reason(const std::string& finish_reason, bool has_tool_calls);
void normalize_anthropic_billing_header(std::string& system_text);

} // namespace zedinfer::api_compat
