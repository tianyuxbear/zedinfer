#include "zedinfer/api_compat.hpp"

#include <gtest/gtest.h>

#include <stdexcept>

using zedinfer::api_compat::chat_response_to_anthropic;
using zedinfer::api_compat::chat_response_to_responses;
using zedinfer::api_compat::build_tool_prompt;
using zedinfer::api_compat::convert_anthropic_to_chat;
using zedinfer::api_compat::convert_responses_to_chat;
using zedinfer::api_compat::normalize_anthropic_billing_header;
using json = nlohmann::json;

TEST(ApiCompat, ConvertsResponsesStringInput) {
    json body = {{"instructions", "Be terse."}, {"input", "Hello"}, {"max_output_tokens", 7}};

    json chat = convert_responses_to_chat(body);

    ASSERT_TRUE(chat["messages"].is_array());
    ASSERT_EQ(chat["messages"].size(), 2);
    EXPECT_EQ(chat["messages"][0]["role"], "system");
    EXPECT_EQ(chat["messages"][0]["content"], "Be terse.");
    EXPECT_EQ(chat["messages"][1]["role"], "user");
    EXPECT_EQ(chat["messages"][1]["content"], "Hello");
    EXPECT_EQ(chat["max_tokens"], 7);
}

TEST(ApiCompat, MapsResponsesDeveloperInputToSystem) {
    json body = {{"input",
                  json::array({{{"type", "message"},
                                {"role", "developer"},
                                {"content", json::array({{{"type", "input_text"}, {"text", "Use tools when needed."}}})}},
                               {{"type", "message"},
                                {"role", "user"},
                                {"content", json::array({{{"type", "input_text"}, {"text", "List files."}}})}}})}};

    json chat = convert_responses_to_chat(body);

    ASSERT_EQ(chat["messages"].size(), 2);
    EXPECT_EQ(chat["messages"][0]["role"], "system");
    EXPECT_EQ(chat["messages"][0]["content"][0]["text"], "Use tools when needed.");
    EXPECT_EQ(chat["messages"][1]["role"], "user");
}

TEST(ApiCompat, ConvertsResponsesToolItems) {
    json body = {{"input",
                  json::array({{{"type", "function_call"},
                                {"call_id", "call_1"},
                                {"name", "get_weather"},
                                {"arguments", "{\"city\":\"Paris\"}"}},
                               {{"type", "function_call_output"}, {"call_id", "call_1"}, {"output", "sunny"}}})}};

    json chat = convert_responses_to_chat(body);

    ASSERT_EQ(chat["messages"].size(), 2);
    ASSERT_TRUE(chat["messages"][0]["tool_calls"].is_array());
    EXPECT_EQ(chat["messages"][0]["tool_calls"][0]["function"]["name"], "get_weather");
    EXPECT_EQ(chat["messages"][1]["role"], "tool");
    EXPECT_EQ(chat["messages"][1]["tool_call_id"], "call_1");
}

TEST(ApiCompat, ConvertsCodexResponsesFunctionToolsOnly) {
    json function_tool = {{"type", "function"},
                          {"name", "exec_command"},
                          {"description", "Run a command"},
                          {"strict", false},
                          {"parameters",
                           {{"type", "object"},
                            {"properties", {{"cmd", {{"type", "string"}}}}},
                            {"required", json::array({"cmd"})}}}};
    json namespace_tool = {{"type", "namespace"}, {"name", "multi_agent_v1"}, {"tools", json::array()}};
    json web_search_tool = {{"type", "web_search"}, {"external_web_access", true}};

    json body = {{"input",
                  json::array({{{"type", "message"},
                                {"role", "developer"},
                                {"content", json::array({{{"type", "input_text"}, {"text", "Use tools when needed."}}})}},
                               {{"type", "message"}, {"role", "user"}, {"content", "Current directory files?"}}})},
                 {"tools", json::array({function_tool, namespace_tool, web_search_tool})},
                 {"tool_choice", {{"type", "auto"}}}};

    json chat = convert_responses_to_chat(body);

    ASSERT_EQ(chat["messages"].size(), 2);
    EXPECT_EQ(chat["messages"][0]["role"], "system");
    EXPECT_EQ(chat["messages"][1]["content"][0]["text"], "Current directory files?");
    ASSERT_TRUE(chat.contains("tools"));
    ASSERT_EQ(chat["tools"].size(), 1);
    EXPECT_EQ(chat["tools"][0]["type"], "function");
    EXPECT_EQ(chat["tools"][0]["function"]["name"], "exec_command");
    EXPECT_FALSE(chat["tools"][0]["function"]["strict"]);
    EXPECT_EQ(chat["tool_choice"], "auto");
}

TEST(ApiCompat, ConvertsResponsesRequiredToolChoice) {
    json body = {{"input", "Run a command"}, {"tool_choice", {{"type", "tool"}, {"name", "exec_command"}}}};

    json chat = convert_responses_to_chat(body);

    EXPECT_EQ(chat["tool_choice"], "required");
}

TEST(ApiCompat, ConvertsResponsesReasoningItem) {
    json body = {{"input",
                  json::array({{{"type", "message"},
                                {"role", "assistant"},
                                {"content", json::array({{{"type", "output_text"}, {"text", "Checked files."}}})}},
                               {{"type", "reasoning"},
                                {"summary", json::array()},
                                {"content",
                                 json::array({{{"type", "reasoning_text"}, {"text", "Need to inspect the tree."}}})}}})}};

    json chat = convert_responses_to_chat(body);

    ASSERT_EQ(chat["messages"].size(), 1);
    EXPECT_EQ(chat["messages"][0]["content"][0]["text"], "Checked files.");
    EXPECT_EQ(chat["messages"][0]["reasoning_content"], "Need to inspect the tree.");
}

TEST(ApiCompat, RejectsMalformedResponsesReasoningItem) {
    json body = {{"input",
                  json::array({{{"type", "reasoning"}, {"summary", json::array()}, {"content", json::array()}}})}};

    EXPECT_THROW(convert_responses_to_chat(body), std::invalid_argument);
}

TEST(ApiCompat, ConvertsAnthropicContentAndTools) {
    json body = {{"system", "x-anthropic-billing-header: cc_version=1; cch=abcde;System"},
                 {"messages",
                  json::array({{{"role", "user"},
                                {"content",
                                 json::array({{{"type", "text"}, {"text", "Look"}},
                                              {{"type", "image"},
                                               {"source",
                                                {{"type", "base64"},
                                                 {"media_type", "image/png"},
                                                 {"data", "AAAA"}}}}})}}})},
                 {"tools",
                  json::array({{{"name", "calc"},
                                {"description", "Calculate"},
                                {"input_schema", {{"type", "object"}}}}})},
                 {"stop_sequences", json::array({"STOP"})},
                 {"thinking", {{"type", "enabled"}, {"budget_tokens", 512}}},
                 {"metadata", {{"user_id", "user-1"}}}};

    json chat = convert_anthropic_to_chat(body);

    EXPECT_EQ(chat["messages"][0]["role"], "system");
    EXPECT_NE(chat["messages"][0]["content"].get<std::string>().find("cch=fffff"), std::string::npos);
    EXPECT_EQ(chat["messages"][1]["content"][1]["type"], "image_url");
    EXPECT_EQ(chat["tools"][0]["function"]["name"], "calc");
    EXPECT_EQ(chat["stop"][0], "STOP");
    EXPECT_TRUE(chat["enable_thinking"]);
    EXPECT_EQ(chat["thinking_budget_tokens"], 512);
    EXPECT_EQ(chat["__metadata_user_id"], "user-1");
}

TEST(ApiCompat, SkipsAnthropicAssistantWithoutContent) {
    json body = {{"messages",
                  json::array({{{"role", "user"}, {"content", "Hello"}},
                               {{"role", "assistant"}},
                               {{"role", "user"}, {"content", "Continue"}}})}};

    json chat = convert_anthropic_to_chat(body);

    ASSERT_EQ(chat["messages"].size(), 2);
    EXPECT_EQ(chat["messages"][0]["role"], "user");
    EXPECT_EQ(chat["messages"][1]["content"], "Continue");
}

TEST(ApiCompat, BuildsToolPrompt) {
    json tools = json::array({{{"type", "function"},
                               {"function",
                                {{"name", "Bash"},
                                 {"description", "Run a shell command"},
                                 {"parameters",
                                  {{"type", "object"},
                                   {"properties", {{"command", {{"type", "string"}}}}},
                                   {"required", json::array({"command"})}}}}}}});

    std::string prompt = build_tool_prompt(tools);

    EXPECT_NE(prompt.find("<tool_call>"), std::string::npos);
    EXPECT_NE(prompt.find("Bash"), std::string::npos);
    EXPECT_NE(prompt.find("command"), std::string::npos);
}

TEST(ApiCompat, WrapsChatAsResponsesAndAnthropic) {
    json chat = {{"id", "chatcmpl-42"},
                 {"object", "chat.completion"},
                 {"created", 123},
                 {"model", "test-model"},
                 {"choices",
                  json::array({{{"index", 0},
                                {"message",
                                 {{"role", "assistant"},
                                  {"content", "Hi"},
                                  {"reasoning_content", "Thinking"}}},
                                {"finish_reason", "stop"}}})},
                 {"usage", {{"prompt_tokens", 2}, {"completion_tokens", 3}, {"total_tokens", 5}}}};

    json responses = chat_response_to_responses(chat);
    EXPECT_EQ(responses["id"], "resp_42");
    EXPECT_EQ(responses["output_text"], "Hi");
    ASSERT_GE(responses["output"].size(), 2);
    EXPECT_EQ(responses["usage"]["input_tokens"], 2);

    json anthropic = chat_response_to_anthropic(chat);
    EXPECT_EQ(anthropic["type"], "message");
    EXPECT_EQ(anthropic["content"][0]["type"], "thinking");
    EXPECT_EQ(anthropic["content"][1]["text"], "Hi");
    EXPECT_EQ(anthropic["usage"]["output_tokens"], 3);
}

TEST(ApiCompat, NormalizesAnthropicBillingHeader) {
    std::string header = "x-anthropic-billing-header: cc_version=1; cch=a1b2c;Prompt";
    normalize_anthropic_billing_header(header);
    EXPECT_NE(header.find("cch=fffff;"), std::string::npos);
}
