#include "zedinfer/api_compat.hpp"

#include <chrono>
#include <cstring>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

namespace zedinfer::api_compat {

namespace {

template <typename T> T json_value(const json& j, const char* key, T def) {
    if (!j.contains(key) || j.at(key).is_null()) {
        return def;
    }
    try {
        return j.at(key).get<T>();
    } catch (const std::exception&) { return def; }
}

bool is_array(const json& j, const char* key) {
    return j.contains(key) && j.at(key).is_array();
}

bool is_string(const json& j, const char* key) {
    return j.contains(key) && j.at(key).is_string();
}

int64_t now_epoch() {
    return std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch())
        .count();
}

std::string chat_id_suffix(const json& chat_response) {
    std::string id = json_value(chat_response, "id", std::string("chatcmpl-0"));
    auto pos = id.rfind('-');
    return pos == std::string::npos ? id : id.substr(pos + 1);
}

json assistant_message(const json& chat_response) {
    if (!is_array(chat_response, "choices") || chat_response.at("choices").empty()) {
        return json{{"role", "assistant"}, {"content", ""}};
    }
    const json& choice = chat_response.at("choices").at(0);
    if (choice.contains("message") && choice.at("message").is_object()) {
        return choice.at("message");
    }
    return json{{"role", "assistant"}, {"content", ""}};
}

std::string message_content_text(const json& msg) {
    if (!msg.contains("content") || msg.at("content").is_null()) {
        return "";
    }
    if (msg.at("content").is_string()) {
        return msg.at("content").get<std::string>();
    }
    if (!msg.at("content").is_array()) {
        return "";
    }
    std::string out;
    for (const auto& part : msg.at("content")) {
        if (part.is_object() && is_string(part, "text")) {
            out += part.at("text").get<std::string>();
        }
    }
    return out;
}

json parse_arguments_or_empty(const json& fn) {
    if (!is_string(fn, "arguments")) {
        return json::object();
    }
    try {
        return json::parse(fn.at("arguments").get<std::string>());
    } catch (const std::exception&) { return json::object(); }
}

std::string reserve_unique_tool_call_id(const std::string& raw_id, std::unordered_map<std::string, std::string>& latest,
                                        std::unordered_set<std::string>& used) {
    const std::string base = raw_id.empty() ? std::string("call") : raw_id;
    std::string id = base;
    int suffix = 2;
    while (used.find(id) != used.end()) { id = base + "_" + std::to_string(suffix++); }
    used.insert(id);
    latest[raw_id] = id;
    return id;
}

std::string latest_tool_call_id(const std::string& raw_id, const std::unordered_map<std::string, std::string>& latest) {
    auto it = latest.find(raw_id);
    return it == latest.end() ? raw_id : it->second;
}

} // namespace

json convert_responses_to_chat(const json& body) {
    if (!body.contains("input")) {
        throw std::invalid_argument("'input' is required");
    }
    if (!json_value(body, "previous_response_id", std::string()).empty()) {
        throw std::invalid_argument("zedinfer does not support 'previous_response_id'");
    }

    json chat_body = body;
    chat_body.erase("input");
    json messages = json::array();

    if (body.contains("instructions")) {
        messages.push_back({{"role", "system"}, {"content", json_value(body, "instructions", std::string())}});
        chat_body.erase("instructions");
    }

    const json& input = body.at("input");
    if (input.is_string()) {
        messages.push_back({{"role", "user"}, {"content", input}});
    } else if (input.is_array()) {
        for (json item : input) {
            const bool merge_prev = !messages.empty() && messages.back().value("role", "") == "assistant";
            if (is_string(item, "content")) {
                item["content"] = json::array({{{"type", "input_text"}, {"text", item.at("content")}}});
            }

            if (is_array(item, "content") && is_string(item, "role")
                && (item.at("role") == "user" || item.at("role") == "system" || item.at("role") == "developer")) {
                json content = json::array();
                for (const auto& part : item.at("content")) {
                    const std::string type = json_value(part, "type", std::string());
                    if (type == "input_text") {
                        if (!part.contains("text")) {
                            throw std::invalid_argument("'Input text' requires 'text'");
                        }
                        content.push_back({{"type", "text"}, {"text", part.at("text")}});
                    } else if (type == "input_image") {
                        if (!part.contains("image_url")) {
                            throw std::invalid_argument("'image_url' is required");
                        }
                        content.push_back({{"type", "image_url"}, {"image_url", {{"url", part.at("image_url")}}}});
                    } else if (type == "input_file") {
                        throw std::invalid_argument("'input_file' is not supported");
                    } else {
                        throw std::invalid_argument(
                            "'type' must be one of 'input_text', 'input_image', or 'input_file'");
                    }
                }
                item.erase("type");
                item.erase("status");
                if (item.at("role") == "developer") {
                    item["role"] = "system";
                }
                item["content"] = content;
                messages.push_back(item);
            } else if (is_string(item, "role") && item.at("role") == "assistant" && is_string(item, "type")
                       && item.at("type") == "message") {
                json content = json::array();
                if (is_string(item, "content")) {
                    content.push_back({{"type", "text"}, {"text", item.at("content")}});
                } else if (is_array(item, "content")) {
                    for (const auto& part : item.at("content")) {
                        const std::string type = json_value(part, "type", std::string());
                        if (type == "output_text" || type == "input_text") {
                            if (!is_string(part, "text")) {
                                throw std::invalid_argument("'Output text' requires 'text'");
                            }
                            content.push_back({{"type", "text"}, {"text", part.at("text")}});
                        } else if (type == "refusal") {
                            if (!is_string(part, "refusal")) {
                                throw std::invalid_argument("'Refusal' requires 'refusal'");
                            }
                            content.push_back({{"type", "refusal"}, {"refusal", part.at("refusal")}});
                        } else {
                            throw std::invalid_argument("'type' must be one of 'output_text' or 'refusal'");
                        }
                    }
                }
                if (merge_prev) {
                    if (!is_array(messages.back(), "content")) {
                        messages.back()["content"] = json::array();
                    }
                    auto& prev = messages.back()["content"];
                    prev.insert(prev.end(), content.begin(), content.end());
                } else {
                    item.erase("status");
                    item.erase("type");
                    item["content"] = content;
                    messages.push_back(item);
                }
            } else if (is_string(item, "arguments") && is_string(item, "call_id") && is_string(item, "name")
                       && is_string(item, "type") && item.at("type") == "function_call") {
                json tool_call = {{"id", item.at("call_id")},
                                  {"type", "function"},
                                  {"function", {{"name", item.at("name")}, {"arguments", item.at("arguments")}}}};
                if (merge_prev) {
                    if (!is_array(messages.back(), "tool_calls")) {
                        messages.back()["tool_calls"] = json::array();
                    }
                    messages.back()["tool_calls"].push_back(tool_call);
                } else {
                    messages.push_back(
                        {{"role", "assistant"}, {"content", ""}, {"tool_calls", json::array({tool_call})}});
                }
            } else if (is_string(item, "call_id") && item.contains("output") && is_string(item, "type")
                       && item.at("type") == "function_call_output") {
                if (item.at("output").is_string()) {
                    messages.push_back(
                        {{"role", "tool"}, {"tool_call_id", item.at("call_id")}, {"content", item.at("output")}});
                } else if (item.at("output").is_array()) {
                    json outputs = item.at("output");
                    for (auto& output : outputs) {
                        if (!output.is_object() || output.value("type", "") != "input_text") {
                            throw std::invalid_argument("Output of tool call should be 'input_text'");
                        }
                        output["type"] = "text";
                    }
                    messages.push_back({{"role", "tool"}, {"tool_call_id", item.at("call_id")}, {"content", outputs}});
                }
            } else if (is_array(item, "summary") && is_string(item, "type") && item.at("type") == "reasoning") {
                if (!is_array(item, "content")) {
                    throw std::invalid_argument("item['content'] is not an array");
                }
                if (item.at("content").empty()) {
                    throw std::invalid_argument("item['content'] is empty");
                }
                if (!is_string(item.at("content").at(0), "text")) {
                    throw std::invalid_argument("item['content']['text'] is not a string");
                }
                std::string reasoning;
                for (const auto& part : item.at("content")) {
                    if (is_string(part, "text")) {
                        reasoning += part.at("text").get<std::string>();
                    }
                }
                if (merge_prev) {
                    messages.back()["reasoning_content"] = reasoning;
                } else {
                    messages.push_back({{"role", "assistant"}, {"content", ""}, {"reasoning_content", reasoning}});
                }
            } else {
                throw std::invalid_argument("Cannot determine type of 'item'");
            }
        }
    } else {
        throw std::invalid_argument("'input' must be a string or array of objects");
    }

    chat_body["messages"] = messages;

    if (body.contains("tools")) {
        if (!body.at("tools").is_array()) {
            throw std::invalid_argument("'tools' must be an array of objects");
        }
        json tools = json::array();
        for (json tool : body.at("tools")) {
            const std::string type = json_value(tool, "type", std::string());
            if (type != "function") {
                continue;
            }
            tool.erase("type");
            if (!tool.contains("strict")) {
                tool["strict"] = true;
            }
            tools.push_back({{"type", "function"}, {"function", tool}});
        }
        chat_body.erase("tools");
        if (!tools.empty()) {
            chat_body["tools"] = tools;
        }
    }
    if (body.contains("max_output_tokens")) {
        chat_body.erase("max_output_tokens");
        chat_body["max_tokens"] = body.at("max_output_tokens");
    }
    if (body.contains("tool_choice") && body.at("tool_choice").is_object()) {
        const std::string type = json_value(body.at("tool_choice"), "type", std::string());
        if (type == "auto") {
            chat_body["tool_choice"] = "auto";
        } else if (type == "none") {
            chat_body["tool_choice"] = "none";
        } else if (type == "any" || type == "required" || type == "tool") {
            chat_body["tool_choice"] = "required";
        } else if (type == "function" && is_string(body.at("tool_choice"), "name")) {
            chat_body["tool_choice"]
                = {{"type", "function"}, {"function", {{"name", body.at("tool_choice").at("name")}}}};
        }
    }

    return chat_body;
}

void normalize_anthropic_billing_header(std::string& system_text) {
    constexpr const char* prefix = "x-anthropic-billing-header:";
    if (system_text.rfind(prefix, 0) != 0) {
        return;
    }
    const size_t cch = system_text.find("cch=", std::strlen(prefix));
    if (cch == std::string::npos) {
        return;
    }
    const size_t value = cch + 4;
    if (value + 5 < system_text.size() && system_text[value + 5] == ';') {
        for (size_t i = 0; i < 5; ++i) { system_text[value + i] = 'f'; }
    }
}

json convert_anthropic_to_chat(const json& body) {
    json out;
    json messages = json::array();
    std::unordered_map<std::string, std::string> latest_tool_call_ids;
    std::unordered_set<std::string> used_tool_call_ids;

    json system = body.contains("system") ? body.at("system") : json();
    if (!system.is_null()) {
        std::string text;
        if (system.is_string()) {
            text = system.get<std::string>();
            normalize_anthropic_billing_header(text);
        } else if (system.is_array()) {
            for (const auto& block : system) {
                if (json_value(block, "type", std::string()) == "text") {
                    std::string part = json_value(block, "text", std::string());
                    normalize_anthropic_billing_header(part);
                    text += part;
                }
            }
        }
        messages.push_back({{"role", "system"}, {"content", text}});
    }

    if (!body.contains("messages")) {
        throw std::invalid_argument("'messages' is required");
    }
    if (body.at("messages").is_array()) {
        for (const auto& msg : body.at("messages")) {
            std::string role = json_value(msg, "role", std::string());
            if (!msg.contains("content")) {
                if (role == "assistant") {
                    continue;
                }
                messages.push_back(msg);
                continue;
            }
            if (msg.at("content").is_string() || !msg.at("content").is_array()) {
                messages.push_back(msg);
                continue;
            }

            json converted = json::array();
            json tool_calls = json::array();
            json tool_results = json::array();
            std::string reasoning;

            for (const auto& block : msg.at("content")) {
                const std::string type = json_value(block, "type", std::string());
                if (type == "text") {
                    converted.push_back(block);
                } else if (type == "thinking") {
                    reasoning += json_value(block, "thinking", std::string());
                } else if (type == "image") {
                    const json source = block.contains("source") ? block.at("source") : json::object();
                    const std::string stype = json_value(source, "type", std::string());
                    if (stype == "base64") {
                        std::ostringstream ss;
                        ss << "data:" << json_value(source, "media_type", std::string("image/jpeg")) << ";base64,"
                           << json_value(source, "data", std::string());
                        converted.push_back({{"type", "image_url"}, {"image_url", {{"url", ss.str()}}}});
                    } else if (stype == "url") {
                        converted.push_back({{"type", "image_url"},
                                             {"image_url", {{"url", json_value(source, "url", std::string())}}}});
                    }
                } else if (type == "tool_use") {
                    const std::string tool_id = reserve_unique_tool_call_id(json_value(block, "id", std::string()),
                                                                            latest_tool_call_ids, used_tool_call_ids);
                    tool_calls.push_back({{"id", tool_id},
                                          {"type", "function"},
                                          {"function",
                                           {{"name", json_value(block, "name", std::string())},
                                            {"arguments", json_value(block, "input", json::object()).dump()}}}});
                } else if (type == "tool_result") {
                    std::string result_text;
                    json content = block.contains("content") ? block.at("content") : json();
                    if (content.is_string()) {
                        result_text = content.get<std::string>();
                    } else if (content.is_array()) {
                        for (const auto& c : content) {
                            if (json_value(c, "type", std::string()) == "text") {
                                result_text += json_value(c, "text", std::string());
                            }
                        }
                    }
                    tool_results.push_back(
                        {{"role", "tool"},
                         {"tool_call_id",
                          latest_tool_call_id(json_value(block, "tool_use_id", std::string()), latest_tool_call_ids)},
                         {"content", result_text}});
                }
            }

            if (!converted.empty() || !tool_calls.empty() || !reasoning.empty()) {
                json m = {{"role", role}};
                m["content"] = converted.empty() ? json("") : converted;
                if (!tool_calls.empty()) {
                    m["tool_calls"] = tool_calls;
                }
                if (!reasoning.empty()) {
                    m["reasoning_content"] = reasoning;
                }
                messages.push_back(m);
            }
            for (const auto& tool_result : tool_results) { messages.push_back(tool_result); }
        }
    }

    out["messages"] = messages;
    if (body.contains("tools") && body.at("tools").is_array()) {
        json tools = json::array();
        for (const auto& tool : body.at("tools")) {
            tools.push_back(
                {{"type", "function"},
                 {"function",
                  {{"name", json_value(tool, "name", std::string())},
                   {"description", json_value(tool, "description", std::string())},
                   {"parameters", tool.contains("input_schema") ? tool.at("input_schema") : json::object()}}}});
        }
        out["tools"] = tools;
    }
    if (body.contains("tool_choice") && body.at("tool_choice").is_object()) {
        const std::string type = json_value(body.at("tool_choice"), "type", std::string());
        if (type == "auto") {
            out["tool_choice"] = "auto";
        } else if (type == "any" || type == "tool") {
            out["tool_choice"] = "required";
        }
    }
    if (body.contains("stop_sequences")) {
        out["stop"] = body.at("stop_sequences");
    }
    out["max_tokens"] = body.contains("max_tokens") ? body.at("max_tokens") : json(4096);
    for (const auto& key : {"temperature", "top_p", "top_k", "stream", "session_id"}) {
        if (body.contains(key)) {
            out[key] = body.at(key);
        }
    }
    if (body.contains("thinking") && body.at("thinking").is_object()
        && json_value(body.at("thinking"), "type", std::string()) == "enabled") {
        out["enable_thinking"] = true;
        out["thinking_budget_tokens"] = json_value(body.at("thinking"), "budget_tokens", 10000);
    }
    if (body.contains("metadata") && body.at("metadata").is_object()) {
        std::string user_id = json_value(body.at("metadata"), "user_id", std::string());
        if (!user_id.empty()) {
            out["__metadata_user_id"] = user_id;
        }
    }
    return out;
}

std::string build_tool_prompt(const json& tools) {
    if (!tools.is_array() || tools.empty()) {
        return "";
    }

    std::ostringstream ss;
    ss << "You have access to external tools for filesystem, shell, search, code editing, and other agent tasks. "
          "Use a tool whenever the user asks about current files, directories, command output, repository state, "
          "or wants changes made. Do not claim you cannot inspect the environment when a suitable tool is "
          "available.\n\n"
          "When calling a tool, output only one or more tool blocks in this exact format:\n"
          "<tool_call>\n"
          "{\"name\":\"tool_name\",\"arguments\":{}}\n"
          "</tool_call>\n\n"
          "Available tools:\n";

    bool any = false;
    for (const auto& tool : tools) {
        json fn = json::object();
        if (tool.is_object() && json_value(tool, "type", std::string()) == "function" && tool.contains("function")
            && tool.at("function").is_object()) {
            fn = tool.at("function");
        } else if (tool.is_object() && is_string(tool, "name")) {
            fn = tool;
        }

        const std::string name = json_value(fn, "name", std::string());
        if (name.empty()) {
            continue;
        }
        any = true;

        ss << "- " << name;
        const std::string description = json_value(fn, "description", std::string());
        if (!description.empty()) {
            ss << ": " << description;
        }
        ss << "\n";

        json parameters = fn.contains("parameters") ? fn.at("parameters")
                                                    : (fn.contains("input_schema") ? fn.at("input_schema") : json());
        if (!parameters.is_null()) {
            ss << "  Arguments JSON schema: " << parameters.dump() << "\n";
        }
    }

    return any ? ss.str() : std::string();
}

json chat_response_to_responses(const json& chat_response) {
    const json msg = assistant_message(chat_response);
    const std::string suffix = chat_id_suffix(chat_response);
    const std::string resp_id = "resp_" + suffix;
    const std::string msg_id = "msg_" + suffix;
    json output = json::array();

    if (is_string(msg, "reasoning_content") && !msg.at("reasoning_content").get<std::string>().empty()) {
        output.push_back(
            {{"id", "rs_" + suffix},
             {"summary", json::array()},
             {"type", "reasoning"},
             {"content", json::array({{{"type", "reasoning_text"}, {"text", msg.at("reasoning_content")}}})},
             {"encrypted_content", ""},
             {"status", "completed"}});
    }

    const std::string text = message_content_text(msg);
    if (!text.empty()) {
        output.push_back({{"id", msg_id},
                          {"type", "message"},
                          {"role", "assistant"},
                          {"status", "completed"},
                          {"content", json::array({{{"type", "output_text"},
                                                    {"annotations", json::array()},
                                                    {"logprobs", json::array()},
                                                    {"text", text}}})}});
    }
    if (is_array(msg, "tool_calls")) {
        for (const auto& call : msg.at("tool_calls")) {
            const json fn = call.contains("function") ? call.at("function") : json::object();
            output.push_back({{"type", "function_call"},
                              {"status", "completed"},
                              {"arguments", json_value(fn, "arguments", std::string("{}"))},
                              {"call_id", "fc_" + json_value(call, "id", std::string())},
                              {"name", json_value(fn, "name", std::string())}});
        }
    }

    const json usage = chat_response.contains("usage") ? chat_response.at("usage") : json::object();
    const int64_t input_tokens = json_value(usage, "prompt_tokens", 0);
    const int64_t output_tokens = json_value(usage, "completion_tokens", 0);
    const int64_t epoch = chat_response.contains("created") ? chat_response.at("created").get<int64_t>() : now_epoch();
    return {{"id", resp_id},
            {"object", "response"},
            {"created_at", epoch},
            {"completed_at", epoch},
            {"status", "completed"},
            {"model", json_value(chat_response, "model", std::string())},
            {"output", output},
            {"output_text", text},
            {"usage",
             {{"input_tokens", input_tokens},
              {"output_tokens", output_tokens},
              {"total_tokens", input_tokens + output_tokens},
              {"input_tokens_details", {{"cached_tokens", 0}}}}}};
}

std::string normalize_anthropic_stop_reason(const std::string& finish_reason, bool has_tool_calls) {
    if (has_tool_calls || finish_reason == "tool_calls") {
        return "tool_use";
    }
    if (finish_reason == "length") {
        return "max_tokens";
    }
    return "end_turn";
}

json chat_response_to_anthropic(const json& chat_response) {
    const json msg = assistant_message(chat_response);
    json content = json::array();
    if (is_string(msg, "reasoning_content") && !msg.at("reasoning_content").get<std::string>().empty()) {
        content.push_back({{"type", "thinking"}, {"thinking", msg.at("reasoning_content")}, {"signature", ""}});
    }
    const std::string text = message_content_text(msg);
    if (!text.empty()) {
        content.push_back({{"type", "text"}, {"text", text}});
    }
    bool has_tool_calls = false;
    if (is_array(msg, "tool_calls")) {
        for (const auto& call : msg.at("tool_calls")) {
            const json fn = call.contains("function") ? call.at("function") : json::object();
            content.push_back({{"type", "tool_use"},
                               {"id", json_value(call, "id", std::string())},
                               {"name", json_value(fn, "name", std::string())},
                               {"input", parse_arguments_or_empty(fn)}});
            has_tool_calls = true;
        }
    }
    if (content.empty()) {
        content.push_back({{"type", "text"}, {"text", ""}});
    }

    std::string finish_reason = "stop";
    if (is_array(chat_response, "choices") && !chat_response.at("choices").empty()
        && chat_response.at("choices").at(0).contains("finish_reason")
        && !chat_response.at("choices").at(0).at("finish_reason").is_null()) {
        finish_reason = chat_response.at("choices").at(0).at("finish_reason").get<std::string>();
    }
    const json usage = chat_response.contains("usage") ? chat_response.at("usage") : json::object();
    return {{"id", json_value(chat_response, "id", std::string())},
            {"type", "message"},
            {"role", "assistant"},
            {"content", content},
            {"model", json_value(chat_response, "model", std::string())},
            {"stop_reason", normalize_anthropic_stop_reason(finish_reason, has_tool_calls)},
            {"stop_sequence", nullptr},
            {"usage",
             {{"cache_read_input_tokens", 0},
              {"input_tokens", json_value(usage, "prompt_tokens", 0)},
              {"output_tokens", json_value(usage, "completion_tokens", 0)}}}};
}

} // namespace zedinfer::api_compat
