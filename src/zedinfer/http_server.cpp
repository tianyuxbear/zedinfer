#include "zedinfer/http_server.hpp"
#include "utils/logging.hpp"
#include "zedinfer/api_compat.hpp"
#include "zedinfer/chat_template_jinja.hpp"
#include "zedinfer/engine.hpp"
#include "zedinfer/multimodal_positions.hpp"
#include "zedinfer/request.hpp"

#include <nlohmann/json.hpp>
#include <plog/Log.h>

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <mutex>
#include <queue>
#include <random>
#include <sstream>

using json = nlohmann::json;
namespace fs = std::filesystem;

namespace zedinfer {

// ============================================================================
// Thread-safe token queue for SSE streaming
// ============================================================================

namespace {

template <typename T> class TokenQueue {
public:
    void push(T item) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            queue_.push(std::move(item));
        }
        cv_.notify_one();
    }

    bool try_pop(T& item, std::chrono::milliseconds timeout) {
        std::unique_lock<std::mutex> lock(mutex_);
        if (!cv_.wait_for(lock, timeout, [this] { return !queue_.empty(); })) {
            return false;
        }
        item = std::move(queue_.front());
        queue_.pop();
        return true;
    }

private:
    std::queue<T> queue_;
    std::mutex mutex_;
    std::condition_variable cv_;
};

// Parse an OpenAI-style chat message (string content OR content-parts array)
// into a ChatMessageMM. Sets `had_images` to true when at least one image part
// was found, so the caller can warn about the vision encoder not yet running.
//
// content parts schema supported (matches OpenAI / Qwen3.5 spec):
//   {"type": "text",      "text": "..."}
//   {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
//   {"type": "image_url", "image_url": "data:image/png;base64,..."}    // shorthand
//   {"type": "image",     "image": "data:image/png;base64,..."}        // Qwen3.5 style
//
// Unknown part types are ignored. A message with no recognizable parts becomes
// an empty string content.
static std::string normalized_chat_role(const json& msg) {
    std::string role = msg.value("role", "");
    return role == "developer" ? "system" : role;
}

static nlohmann::ordered_json to_ordered_json(const json& value) {
    return nlohmann::ordered_json::parse(value.dump());
}

static ChatMessageMM parse_openai_message(const json& msg, bool& had_images) {
    ChatMessageMM out;
    out.role = normalized_chat_role(msg);
    if (msg.contains("reasoning_content") && msg["reasoning_content"].is_string()) {
        out.reasoning_content = msg["reasoning_content"].get<std::string>();
    }
    if (msg.contains("name") && msg["name"].is_string()) {
        out.name = msg["name"].get<std::string>();
    }
    if (msg.contains("tool_call_id") && msg["tool_call_id"].is_string()) {
        out.tool_call_id = msg["tool_call_id"].get<std::string>();
    }
    if (msg.contains("tool_calls") && msg["tool_calls"].is_array()) {
        out.tool_calls = to_ordered_json(msg["tool_calls"]);
    }
    if (!msg.contains("content")) {
        out.content = std::string();
        return out;
    }
    const auto& c = msg["content"];
    if (c.is_string()) {
        out.content = c.get<std::string>();
        return out;
    }
    if (!c.is_array()) {
        out.content = std::string();
        return out;
    }
    std::vector<ContentPart> parts;
    for (const auto& p : c) {
        if (!p.is_object()) {
            continue;
        }
        std::string ptype = p.value("type", "");
        if (ptype == "text" && p.contains("text") && p["text"].is_string()) {
            parts.push_back(TextPart{p["text"].get<std::string>()});
        } else if (ptype == "image_url" && p.contains("image_url")) {
            const auto& iu = p["image_url"];
            std::string url;
            if (iu.is_string()) {
                url = iu.get<std::string>();
            } else if (iu.is_object() && iu.contains("url") && iu["url"].is_string()) {
                url = iu["url"].get<std::string>();
            }
            if (!url.empty()) {
                parts.push_back(ImagePart{std::move(url)});
                had_images = true;
            }
        } else if (ptype == "image" && p.contains("image") && p["image"].is_string()) {
            parts.push_back(ImagePart{p["image"].get<std::string>()});
            had_images = true;
        }
    }
    out.content = std::move(parts);
    return out;
}

// Flatten a content (string or array) into a single text string for the legacy
// chat_template path which only understands plain text. Image parts are
// dropped silently (with a single caller-side warning when images were seen).
static std::string flatten_message_content(const json& msg) {
    if (!msg.contains("content")) {
        return "";
    }
    const auto& c = msg["content"];
    if (c.is_string()) {
        return c.get<std::string>();
    }
    if (!c.is_array()) {
        return "";
    }
    std::string flat;
    for (const auto& p : c) {
        if (p.is_object() && p.value("type", "") == "text" && p.contains("text") && p["text"].is_string()) {
            flat += p["text"].get<std::string>();
        }
    }
    return flat;
}

// ----------------------------------------------------------------------------
// Qwen3 tool-call output parsing.
// Qwen3 / Qwen3-Coder emit tool calls as:
//
//   <tool_call>
//   {"name": "get_weather", "arguments": {"city": "Beijing"}}
//   </tool_call>
//
// (sometimes with `parameters` instead of `arguments`). We extract every block
// and reshape into OpenAI's standard tool_calls schema. Anything outside the
// blocks stays as content text. Malformed JSON inside a block is preserved
// verbatim as inline text so we do not silently swallow data.
struct ToolCallParseResult {
    std::string content; // text outside any <tool_call> block
    json tool_calls;     // array; each entry {id, type:"function", function:{name,arguments}}
    bool has_tool_calls = false;
};

struct ParsedToolCall {
    std::string name;
    json arguments;
};

static std::string trim_ascii_ws(std::string s) {
    auto issp = [](char c) { return c == ' ' || c == '\t' || c == '\n' || c == '\r'; };
    while (!s.empty() && issp(s.front())) { s.erase(s.begin()); }
    while (!s.empty() && issp(s.back())) { s.pop_back(); }
    return s;
}

// Try to parse Qwen3-Coder XML-style payload between <tool_call>...</tool_call>:
//
//   <function=get_weather>
//   <parameter=city>
//   Beijing
//   </parameter>
//   </function>
//
// Returns true and fills name/args_json on success; args_json is an
// ordered object whose values are parsed as JSON when possible, else strings.
static bool parse_xml_tool_call_inner(const std::string& inner, std::string& name, json& args_json) {
    size_t fn_open = inner.find("<function=");
    if (fn_open == std::string::npos) {
        return false;
    }
    size_t fn_name_start = fn_open + 10; // strlen("<function=")
    size_t fn_name_end = inner.find('>', fn_name_start);
    if (fn_name_end == std::string::npos) {
        return false;
    }
    name = trim_ascii_ws(inner.substr(fn_name_start, fn_name_end - fn_name_start));
    if (name.empty()) {
        return false;
    }
    size_t fn_close = inner.find("</function>", fn_name_end);
    if (fn_close == std::string::npos) {
        return false;
    }
    std::string body = inner.substr(fn_name_end + 1, fn_close - (fn_name_end + 1));
    args_json = json::object();
    size_t pos = 0;
    while (pos < body.size()) {
        size_t p_open = body.find("<parameter=", pos);
        if (p_open == std::string::npos) {
            break;
        }
        size_t p_name_start = p_open + 11; // strlen("<parameter=")
        size_t p_name_end = body.find('>', p_name_start);
        if (p_name_end == std::string::npos) {
            break;
        }
        std::string key = trim_ascii_ws(body.substr(p_name_start, p_name_end - p_name_start));
        size_t p_close = body.find("</parameter>", p_name_end);
        if (p_close == std::string::npos) {
            break;
        }
        std::string raw_value = trim_ascii_ws(body.substr(p_name_end + 1, p_close - (p_name_end + 1)));
        // Try parsing as JSON literal first (numbers, bools, arrays, nested objects);
        // fall back to the raw string when not valid JSON.
        try {
            args_json[key] = json::parse(raw_value);
        } catch (const std::exception&) { args_json[key] = raw_value; }
        pos = p_close + 12; // strlen("</parameter>")
    }
    return true;
}

static bool parse_json_tool_call_object(const json& value, ParsedToolCall& call) {
    if (!value.is_object() || !value.contains("name") || !value["name"].is_string()) {
        return false;
    }
    call.name = value["name"].get<std::string>();
    call.arguments = value.contains("arguments")
                       ? value["arguments"]
                       : (value.contains("parameters") ? value["parameters"] : json::object());
    return !call.name.empty();
}

static std::vector<ParsedToolCall> parse_tool_call_payload(const std::string& inner) {
    std::vector<ParsedToolCall> calls;
    try {
        const json value = json::parse(inner);
        if (value.is_object()) {
            ParsedToolCall call;
            if (parse_json_tool_call_object(value, call)) {
                calls.push_back(std::move(call));
            }
        } else if (value.is_array()) {
            for (const auto& item : value) {
                ParsedToolCall call;
                if (parse_json_tool_call_object(item, call)) {
                    calls.push_back(std::move(call));
                }
            }
        }
        if (!calls.empty()) {
            return calls;
        }
    } catch (const std::exception&) {
        // Not JSON; try Qwen3-Coder XML style below.
    }

    ParsedToolCall call;
    if (parse_xml_tool_call_inner(inner, call.name, call.arguments)) {
        calls.push_back(std::move(call));
    }
    return calls;
}

static std::string random_id_fragment(size_t length = 32) {
    static constexpr char alphabet[] = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
    std::random_device rd;
    std::mt19937 generator(rd());
    std::string result(length, '0');
    for (char& c : result) { c = alphabet[generator() % (sizeof(alphabet) - 1)]; }
    return result;
}

static std::string generate_tool_call_id() {
    return "call_" + random_id_fragment();
}

static json make_openai_tool_call(const ParsedToolCall& parsed) {
    json call;
    call["id"] = generate_tool_call_id();
    call["type"] = "function";
    json fn;
    fn["name"] = parsed.name;
    fn["arguments"] = parsed.arguments.is_string() ? parsed.arguments.get<std::string>() : parsed.arguments.dump();
    call["function"] = std::move(fn);
    return call;
}

struct ToolCallTag {
    const char* open;
    const char* close;
    size_t open_len;
    size_t close_len;
};

static constexpr ToolCallTag kToolCallTags[]
    = {{"<tool_call>", "</tool_call>", 11, 12}, {"<function_call>", "</function_call>", 15, 16}};

static size_t max_tool_call_marker_len() {
    size_t max_len = 0;
    for (const auto& tag : kToolCallTags) { max_len = std::max(max_len, std::max(tag.open_len, tag.close_len)); }
    return max_len;
}

static const ToolCallTag* find_next_tool_call_tag(const std::string& text, size_t pos, size_t& open_pos) {
    const ToolCallTag* found = nullptr;
    open_pos = std::string::npos;
    for (const auto& tag : kToolCallTags) {
        const size_t p = text.find(tag.open, pos);
        if (p != std::string::npos && (open_pos == std::string::npos || p < open_pos)) {
            open_pos = p;
            found = &tag;
        }
    }
    return found;
}

static ToolCallParseResult parse_qwen3_tool_calls(const std::string& text) {
    ToolCallParseResult out;
    out.tool_calls = json::array();
    size_t pos = 0;
    while (pos < text.size()) {
        size_t open = std::string::npos;
        const ToolCallTag* tag = find_next_tool_call_tag(text, pos, open);
        if (tag == nullptr) {
            out.content.append(text, pos, text.size() - pos);
            break;
        }
        out.content.append(text, pos, open - pos);
        size_t close = text.find(tag->close, open + tag->open_len);
        if (close == std::string::npos) {
            // Unterminated <tool_call>: keep the rest as plain text (the model
            // got cut off mid-emission).
            out.content.append(text, open, text.size() - open);
            break;
        }
        std::string inner = trim_ascii_ws(text.substr(open + tag->open_len, close - (open + tag->open_len)));
        std::vector<ParsedToolCall> parsed_calls = parse_tool_call_payload(inner);
        if (!parsed_calls.empty()) {
            for (const auto& parsed : parsed_calls) { out.tool_calls.push_back(make_openai_tool_call(parsed)); }
            out.has_tool_calls = true;
        } else {
            out.content.append(text, open, close + tag->close_len - open);
        }
        pos = close + tag->close_len;
    }
    return out;
}

// Reasoning-mode tag literals. Qwen3.5 / DeepSeek-R1 emit these as exactly the
// substrings "<think>" / "</think>" after token decoding (each is a single
// special token id in those vocabs). We surface anything between them via the
// OpenAI o1-style `reasoning_content` field so clients can fold thinking out
// of the main reply.
static constexpr const char* kReasoningOpenTag = "<think>";
static constexpr const char* kReasoningCloseTag = "</think>";
static constexpr size_t kReasoningOpenLen = 7;  // strlen("<think>")
static constexpr size_t kReasoningCloseLen = 8; // strlen("</think>")

struct ReasoningSplit {
    std::string reasoning;
    std::string content;
};

// Walk text from left to right, accumulating into `reasoning` while inside an
// open <think>...</think> block, and into `content` otherwise. `initial_in_think`
// lets callers seed the state from a session prefix (e.g. DeepSeek-R1's
// "<think> " output_prefix opens a block before the model has emitted anything).
// Unterminated <think> blocks at end-of-text are treated as fully reasoning.
static ReasoningSplit split_reasoning(const std::string& text, bool initial_in_think = false) {
    ReasoningSplit out;
    bool in_think = initial_in_think;
    size_t pos = 0;
    while (pos < text.size()) {
        if (in_think) {
            size_t end = text.find(kReasoningCloseTag, pos);
            if (end == std::string::npos) {
                out.reasoning.append(text, pos, text.size() - pos);
                pos = text.size();
            } else {
                out.reasoning.append(text, pos, end - pos);
                pos = end + kReasoningCloseLen;
                in_think = false;
            }
        } else {
            size_t open = text.find(kReasoningOpenTag, pos);
            if (open == std::string::npos) {
                out.content.append(text, pos, text.size() - pos);
                pos = text.size();
            } else {
                out.content.append(text, pos, open - pos);
                pos = open + kReasoningOpenLen;
                in_think = true;
            }
        }
    }
    return out;
}

static size_t valid_utf8_length(const std::string& s) {
    size_t i = 0, last_good = 0;
    while (i < s.size()) {
        uint8_t c = static_cast<uint8_t>(s[i]);
        int n = 0;
        if ((c & 0x80) == 0) {
            n = 1;
        } else if ((c & 0xE0) == 0xC0) {
            n = 2;
        } else if ((c & 0xF0) == 0xE0) {
            n = 3;
        } else if ((c & 0xF8) == 0xF0) {
            n = 4;
        } else {
            i++;
            continue;
        }
        if (i + n > s.size()) {
            break;
        }
        bool ok = true;
        for (int j = 1; j < n; j++) {
            if ((static_cast<uint8_t>(s[i + j]) & 0xC0) != 0x80) {
                ok = false;
                break;
            }
        }
        if (!ok) {
            i++;
            continue;
        }
        i += n;
        last_good = i;
    }
    return last_good;
}

static constexpr const char* kCompatFormatField = "__zedinfer_compat_format";

static api_compat::ResponseFormat parse_compat_format(json& body) {
    api_compat::ResponseFormat format = api_compat::ResponseFormat::OpenAIChat;
    if (body.contains(kCompatFormatField) && body[kCompatFormatField].is_string()) {
        const std::string value = body[kCompatFormatField].get<std::string>();
        if (value == "responses") {
            format = api_compat::ResponseFormat::OpenAIResponses;
        } else if (value == "anthropic") {
            format = api_compat::ResponseFormat::Anthropic;
        }
        body.erase(kCompatFormatField);
    }
    return format;
}

static std::string compat_suffix_from_chat_id(const std::string& id) {
    size_t pos = id.rfind('-');
    return pos == std::string::npos ? id : id.substr(pos + 1);
}

static void inject_tool_prompt_message(json& body) {
    if (!body.contains("messages") || !body["messages"].is_array() || !body.contains("tools")) {
        return;
    }
    std::string tool_prompt = api_compat::build_tool_prompt(body["tools"]);
    if (tool_prompt.empty()) {
        return;
    }
    body["messages"].insert(body["messages"].begin(), json{{"role", "system"}, {"content", tool_prompt}});
}

} // anonymous namespace

// ============================================================================
// Helpers
// ============================================================================

std::string HttpServer::generate_request_id() {
    uint64_t id = request_counter_.fetch_add(1, std::memory_order_relaxed);
    return "chatcmpl-" + std::to_string(id);
}

void HttpServer::send_error(httplib::Response& res, int status, const std::string& message, const std::string& type,
                            const std::string& code) {
    json err;
    err["error"]["message"] = message;
    err["error"]["type"] = type;
    if (!code.empty()) {
        err["error"]["code"] = code;
    }
    res.status = status;
    res.set_content(err.dump(), "application/json");
}

std::string HttpServer::resolve_web_root() {
    for (const auto& c : {"web", "../web"}) {
        if (fs::exists(c) && fs::is_directory(c)) {
            LOGI << "[HttpServer] Web root: " << fs::absolute(c).string();
            return c;
        }
    }
    LOGW << "[HttpServer] Web root not found";
    return "";
}

std::string HttpServer::guess_content_type(const std::string& filename) {
    auto pos = filename.rfind('.');
    if (pos == std::string::npos) {
        return "application/octet-stream";
    }
    std::string ext = filename.substr(pos);
    if (ext == ".html") {
        return "text/html";
    }
    if (ext == ".svg") {
        return "image/svg+xml";
    }
    if (ext == ".png") {
        return "image/png";
    }
    if (ext == ".jpg" || ext == ".jpeg") {
        return "image/jpeg";
    }
    if (ext == ".css") {
        return "text/css";
    }
    if (ext == ".js") {
        return "application/javascript";
    }
    if (ext == ".json") {
        return "application/json";
    }
    if (ext == ".ico") {
        return "image/x-icon";
    }
    return "application/octet-stream";
}

void HttpServer::cache_static_files() {
    if (web_root_.empty()) {
        return;
    }

    for (auto& entry : fs::recursive_directory_iterator(web_root_)) {
        if (!entry.is_regular_file()) {
            continue;
        }
        std::string rel = "/" + fs::relative(entry.path(), web_root_).string();
        std::ifstream file(entry.path(), std::ios::binary);
        if (!file.is_open()) {
            continue;
        }
        std::ostringstream ss;
        ss << file.rdbuf();
        file_cache_[rel] = {ss.str(), guess_content_type(entry.path().filename().string())};
    }
    LOGI << "[HttpServer] Cached " << file_cache_.size() << " static files";
}

// ============================================================================
// Session Management
// ============================================================================

InferenceSession* HttpServer::acquire_session(const std::string& session_id, bool enable_thinking) {
    std::lock_guard<std::mutex> lock(sessions_mutex_);

    auto it = sessions_.find(session_id);
    if (it != sessions_.end()) {
        bool expected = false;
        if (!it->second->busy.compare_exchange_strong(expected, true)) {
            return nullptr;
        }
        it->second->session->set_enable_thinking(enable_thinking);
        it->second->last_access = std::chrono::steady_clock::now();
        return it->second->session.get();
    }

    cleanup_idle_sessions();

    GenerationConfig config;
    config.max_new_tokens = config_.default_max_tokens;
    config.enable_thinking = enable_thinking;
    auto session = engine_->create_session(config);
    auto* ptr = session.get();

    auto entry = std::make_unique<SessionEntry>();
    entry->session = std::move(session);
    entry->last_access = std::chrono::steady_clock::now();
    entry->busy = true;

    LOGI << "[HttpServer] Created session " << session_id << " (total: " << sessions_.size() + 1 << ")";

    sessions_[session_id] = std::move(entry);
    return ptr;
}

void HttpServer::unlock_session(const std::string& session_id) {
    std::lock_guard<std::mutex> lock(sessions_mutex_);
    auto it = sessions_.find(session_id);
    if (it != sessions_.end()) {
        it->second->busy = false;
        // A DELETE that arrived mid-use deferred reclamation to us.
        if (it->second->pending_delete) {
            LOGI << "[HttpServer] Reclaiming deferred-delete session " << session_id;
            sessions_.erase(it);
        }
    }
}

void HttpServer::delete_session(const std::string& session_id) {
    std::lock_guard<std::mutex> lock(sessions_mutex_);
    auto it = sessions_.find(session_id);
    if (it != sessions_.end()) {
        // Never erase a busy session: a streaming response holds a raw
        // InferenceSession* into this entry. Defer reclamation to
        // unlock_session() so the in-flight turn finishes against live memory.
        if (it->second->busy) {
            it->second->pending_delete = true;
            LOGI << "[HttpServer] Deferring delete of busy session " << session_id;
            return;
        }
        LOGI << "[HttpServer] Deleted session " << session_id;
        sessions_.erase(it);
    }
}

void HttpServer::cleanup_idle_sessions() {
    auto now = std::chrono::steady_clock::now();
    auto timeout = std::chrono::seconds(config_.session_idle_timeout);

    for (auto it = sessions_.begin(); it != sessions_.end();) {
        if (!it->second->busy && (now - it->second->last_access > timeout)) {
            LOGI << "[HttpServer] Removing idle session " << it->first;
            it = sessions_.erase(it);
        } else {
            ++it;
        }
    }

    while (static_cast<int>(sessions_.size()) >= config_.max_sessions) {
        auto oldest = sessions_.end();
        for (auto it = sessions_.begin(); it != sessions_.end(); ++it) {
            if (!it->second->busy
                && (oldest == sessions_.end() || it->second->last_access < oldest->second->last_access)) {
                oldest = it;
            }
        }
        if (oldest == sessions_.end()) {
            break;
        }
        LOGI << "[HttpServer] Evicting session " << oldest->first << " (at capacity)";
        sessions_.erase(oldest);
    }
}

// ============================================================================
// Constructor and lifecycle
// ============================================================================

HttpServer::HttpServer(ServerConfig config, std::shared_ptr<InferenceEngine> engine)
    : config_(std::move(config)), engine_(std::move(engine)) {
    display_model_name_ = !config_.served_model_name.empty() ? config_.served_model_name : engine_->model_name();
    web_root_ = resolve_web_root();
    cache_static_files();
    server_.set_payload_max_length(10 * 1024 * 1024);

    // CORS: allow browser clients from any origin to call the API. Permissive
    // because this is a single-user local-deployment service; tighten by editing
    // these headers if you front the server with a reverse proxy that enforces
    // its own origin policy.
    server_.set_default_headers({
        {"Access-Control-Allow-Origin", "*"},
        {"Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS"},
        {"Access-Control-Allow-Headers", "Content-Type, Authorization, X-Api-Key, Anthropic-Version, Anthropic-Beta"},
        {"Access-Control-Max-Age", "86400"},
    });
    // CORS preflight: respond 204 to any OPTIONS request without invoking handlers.
    server_.Options("(.*)", [](const httplib::Request&, httplib::Response& res) { res.status = 204; });

    // Bearer-token auth: gate /v1/*, /tokenize, /detokenize when api_key is set.
    // /health and static files stay public so browsers / monitors can probe the
    // server without a credential. CORS preflight (OPTIONS) is always allowed.
    if (!config_.api_key.empty()) {
        server_.set_pre_routing_handler([this](const httplib::Request& req, httplib::Response& res) {
            if (req.method == "OPTIONS") {
                return httplib::Server::HandlerResponse::Unhandled;
            }
            const std::string& path = req.path;
            bool needs_auth = (path.rfind("/v1/", 0) == 0) || (path == "/responses") || (path == "/tokenize")
                           || (path == "/detokenize");
            if (!needs_auth) {
                return httplib::Server::HandlerResponse::Unhandled;
            }
            std::string auth = req.get_header_value("Authorization");
            if (auth.empty()) {
                auth = req.get_header_value("X-Api-Key");
            }
            const std::string prefix = "Bearer ";
            if (auth.rfind(prefix, 0) == 0) {
                auth = auth.substr(prefix.size());
            }
            if (auth.empty()) {
                send_error(res, 401, "Missing or malformed Authorization header (expected 'Bearer <api_key>')",
                           "authentication_error", "invalid_api_key");
                return httplib::Server::HandlerResponse::Handled;
            }
            if (auth != config_.api_key) {
                send_error(res, 401, "Invalid API key", "authentication_error", "invalid_api_key");
                return httplib::Server::HandlerResponse::Handled;
            }
            return httplib::Server::HandlerResponse::Unhandled;
        });
    }

    // HTTP access logger
    server_.set_logger([](const httplib::Request& req, const httplib::Response& res) {
        // Only log API calls, not static files
        if (req.path.find("/v1/") == 0 || req.path == "/health") {
            LOGI << "[HTTP] " << req.method << " " << req.path << " → " << res.status;
        }
    });

    // API routes
    server_.Post("/v1/chat/completions",
                 [this](const httplib::Request& req, httplib::Response& res) { handle_chat_completions(req, res); });
    server_.Post("/v1/responses",
                 [this](const httplib::Request& req, httplib::Response& res) { handle_responses(req, res); });
    server_.Post("/responses",
                 [this](const httplib::Request& req, httplib::Response& res) { handle_responses(req, res); });
    server_.Post("/v1/messages",
                 [this](const httplib::Request& req, httplib::Response& res) { handle_anthropic_messages(req, res); });
    server_.Post("/v1/messages/count_tokens", [this](const httplib::Request& req, httplib::Response& res) {
        handle_anthropic_count_tokens(req, res);
    });
    server_.Get("/v1/models", [this](const httplib::Request& req, httplib::Response& res) { handle_models(req, res); });
    server_.Get("/health", [this](const httplib::Request& req, httplib::Response& res) { handle_health(req, res); });
    server_.Delete("/v1/sessions/(.*)",
                   [this](const httplib::Request& req, httplib::Response& res) { handle_delete_session(req, res); });
    server_.Post("/tokenize",
                 [this](const httplib::Request& req, httplib::Response& res) { handle_tokenize(req, res); });
    server_.Post("/detokenize",
                 [this](const httplib::Request& req, httplib::Response& res) { handle_detokenize(req, res); });

    // Static files from cache
    server_.Get("/(.*)", [this](const httplib::Request& req, httplib::Response& res) {
        std::string path = req.matches[1].str();
        if (path.empty() || path == "/") {
            path = "/index.html";
        } else if (path[0] != '/') {
            path = "/" + path;
        }

        auto it = file_cache_.find(path);
        if (it != file_cache_.end()) {
            res.set_content(it->second.first, it->second.second);
        } else {
            res.status = 404;
        }
    });
}

void HttpServer::start() {
    LOGI << "[HttpServer] Starting on " << config_.host << ":" << config_.port;
    server_.listen(config_.host, config_.port);
}

void HttpServer::stop() {
    server_.stop();
}

// ============================================================================
// GET /v1/models
// ============================================================================

void HttpServer::handle_models(const httplib::Request&, httplib::Response& res) {
    auto epoch
        = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count();

    json response;
    response["object"] = "list";
    const bool has_vision = engine_->has_vision();
    json model = {
        {"id", display_model_name_},
        {"object", "model"},
        {"created", epoch},
        {"capabilities", {{"vision", has_vision}, {"multimodal", has_vision}}},
        {"input_modalities", has_vision ? json::array({"text", "image"}) : json::array({"text"})},
    };
    response["data"] = json::array({model});
    res.set_content(response.dump(), "application/json");
}

// ============================================================================
// GET /health
// ============================================================================

void HttpServer::handle_health(const httplib::Request&, httplib::Response& res) {
    json response;
    response["status"] = "ok";
    response["model"] = display_model_name_;
    response["active_requests"] = engine_->serving_loop().active_count();
    response["pending_requests"] = engine_->serving_loop().pending_count();

    auto* pool = engine_->block_pool();
    if (pool) {
        response["block_pool"]
            = {{"total_blocks", pool->total_blocks()},
               {"free_blocks", pool->free_blocks()},
               {"utilization", 1.0 - static_cast<double>(pool->free_blocks()) / pool->total_blocks()}};
    }
    {
        std::lock_guard<std::mutex> lock(sessions_mutex_);
        response["active_sessions"] = sessions_.size();
    }
    res.set_content(response.dump(), "application/json");
}

// ============================================================================
// DELETE /v1/sessions/:id
// ============================================================================

void HttpServer::handle_delete_session(const httplib::Request& req, httplib::Response& res) {
    std::string session_id = req.matches[1];
    delete_session(session_id);
    res.set_content("{\"deleted\":true}", "application/json");
}

// ============================================================================
// POST /tokenize and /detokenize (vLLM-style extensions, non-OpenAI standard)
// ============================================================================

void HttpServer::handle_tokenize(const httplib::Request& req, httplib::Response& res) {
    json body;
    try {
        body = json::parse(req.body);
    } catch (const json::parse_error& e) {
        send_error(res, 400, std::string("Invalid JSON: ") + e.what(), "invalid_request_error", "invalid_json");
        return;
    }

    // Accept either {"prompt": "..."} or {"text": "..."}; vLLM uses "prompt".
    std::string text;
    if (body.contains("prompt") && body["prompt"].is_string()) {
        text = body["prompt"].get<std::string>();
    } else if (body.contains("text") && body["text"].is_string()) {
        text = body["text"].get<std::string>();
    } else {
        send_error(res, 400, "Missing 'prompt' or 'text' field", "invalid_request_error", "missing_field");
        return;
    }

    std::vector<int> ids;
    try {
        ids = engine_->tokenizer().encode(text);
    } catch (const std::exception& e) {
        send_error(res, 500, std::string("Tokenization failed: ") + e.what(), "internal_error", "tokenize_failed");
        return;
    }

    json response;
    response["count"] = ids.size();
    response["max_model_len"] = engine_->exec_config().max_seq_len;
    response["tokens"] = ids;
    res.set_content(response.dump(), "application/json");
}

void HttpServer::handle_detokenize(const httplib::Request& req, httplib::Response& res) {
    json body;
    try {
        body = json::parse(req.body);
    } catch (const json::parse_error& e) {
        send_error(res, 400, std::string("Invalid JSON: ") + e.what(), "invalid_request_error", "invalid_json");
        return;
    }

    if (!body.contains("tokens") || !body["tokens"].is_array()) {
        send_error(res, 400, "Missing or invalid 'tokens' array", "invalid_request_error", "missing_field");
        return;
    }

    std::vector<int> ids;
    ids.reserve(body["tokens"].size());
    for (const auto& tok : body["tokens"]) {
        if (!tok.is_number_integer()) {
            send_error(res, 400, "'tokens' must be an array of integers", "invalid_request_error", "invalid_token");
            return;
        }
        ids.push_back(tok.get<int>());
    }

    std::string text;
    try {
        text = engine_->tokenizer().decode(ids);
    } catch (const std::exception& e) {
        send_error(res, 500, std::string("Detokenization failed: ") + e.what(), "internal_error", "detokenize_failed");
        return;
    }

    json response;
    response["prompt"] = text;
    res.set_content(response.dump(), "application/json");
}

// ============================================================================
// POST /v1/responses and /responses (OpenAI Responses-compatible)
// ============================================================================

void HttpServer::handle_responses(const httplib::Request& req, httplib::Response& res) {
    json body;
    try {
        body = json::parse(req.body);
        body = api_compat::convert_responses_to_chat(body);
    } catch (const json::parse_error& e) {
        send_error(res, 400, std::string("Invalid JSON: ") + e.what(), "invalid_request_error", "invalid_json");
        return;
    } catch (const std::exception& e) {
        send_error(res, 400, e.what(), "invalid_request_error", "invalid_request");
        return;
    }

    body[kCompatFormatField] = "responses";
    httplib::Request chat_req = req;
    chat_req.path = "/v1/chat/completions";
    chat_req.body = body.dump();
    handle_chat_completions(chat_req, res);
}

// ============================================================================
// POST /v1/messages (Anthropic Messages-compatible)
// ============================================================================

void HttpServer::handle_anthropic_messages(const httplib::Request& req, httplib::Response& res) {
    json body;
    try {
        body = json::parse(req.body);
        body = api_compat::convert_anthropic_to_chat(body);
    } catch (const json::parse_error& e) {
        send_error(res, 400, std::string("Invalid JSON: ") + e.what(), "invalid_request_error", "invalid_json");
        return;
    } catch (const std::exception& e) {
        send_error(res, 400, e.what(), "invalid_request_error", "invalid_request");
        return;
    }

    body[kCompatFormatField] = "anthropic";
    httplib::Request chat_req = req;
    chat_req.path = "/v1/chat/completions";
    chat_req.body = body.dump();
    handle_chat_completions(chat_req, res);
}

// ============================================================================
// POST /v1/messages/count_tokens (Anthropic-compatible)
// ============================================================================

void HttpServer::handle_anthropic_count_tokens(const httplib::Request& req, httplib::Response& res) {
    json body;
    try {
        body = json::parse(req.body);
        body = api_compat::convert_anthropic_to_chat(body);
    } catch (const json::parse_error& e) {
        send_error(res, 400, std::string("Invalid JSON: ") + e.what(), "invalid_request_error", "invalid_json");
        return;
    } catch (const std::exception& e) {
        send_error(res, 400, e.what(), "invalid_request_error", "invalid_request");
        return;
    }

    bool enable_thinking = body.value("enable_thinking", config_.default_enable_thinking);
    bool any_image_part = false;
    std::string prompt;
    const ChatTemplateJinja* jinja_tpl = engine_->chat_template_jinja();

    nlohmann::ordered_json tools_ordered;
    bool has_tools = false;
    if (body.contains("tools") && body["tools"].is_array() && !body["tools"].empty()) {
        try {
            nlohmann::ordered_json ordered_body = nlohmann::ordered_json::parse(body.dump());
            tools_ordered = ordered_body["tools"];
            has_tools = true;
        } catch (const std::exception&) {}
    }
    if (has_tools && !jinja_tpl) {
        inject_tool_prompt_message(body);
    }

    if (jinja_tpl) {
        std::vector<ChatMessageMM> messages;
        messages.reserve(body["messages"].size());
        for (const auto& msg : body["messages"]) { messages.push_back(parse_openai_message(msg, any_image_part)); }
        prompt = jinja_tpl->render(messages, /*add_generation_prompt=*/true, enable_thinking,
                                   has_tools ? &tools_ordered : nullptr);
    } else {
        std::vector<std::pair<std::string, std::string>> messages;
        messages.reserve(body["messages"].size());
        for (const auto& msg : body["messages"]) {
            messages.emplace_back(normalized_chat_role(msg), flatten_message_content(msg));
        }
        prompt = engine_->chat_template().apply(messages);
    }

    try {
        auto ids = engine_->tokenizer().encode(prompt);
        res.set_content(json{{"input_tokens", ids.size()}}.dump(), "application/json");
    } catch (const std::exception& e) {
        send_error(res, 500, std::string("Token counting failed: ") + e.what(), "internal_error", "tokenize_failed");
    }
}

// ============================================================================
// POST /v1/chat/completions
// ============================================================================

void HttpServer::handle_chat_completions(const httplib::Request& req, httplib::Response& res) {
    // 1. Parse JSON
    json body;
    try {
        body = json::parse(req.body);
    } catch (const json::parse_error& e) {
        send_error(res, 400, std::string("Invalid JSON: ") + e.what(), "invalid_request_error", "invalid_json");
        return;
    }

    if (!body.contains("messages") || !body["messages"].is_array() || body["messages"].empty()) {
        send_error(res, 400, "Missing or empty 'messages' array", "invalid_request_error", "missing_field");
        return;
    }

    api_compat::ResponseFormat compat_format = parse_compat_format(body);

    if (body.contains("n") && body["n"].is_number_integer() && body["n"].get<int>() != 1) {
        send_error(res, 400, "Only n=1 is supported", "invalid_request_error", "unsupported_parameter");
        return;
    }
    if (!body.contains("max_tokens") && body.contains("max_completion_tokens")) {
        body["max_tokens"] = body["max_completion_tokens"];
    }

    // 2. Extract parameters
    bool stream = body.value("stream", false);
    int requested_max_tokens = config_.default_max_tokens;
    if (body.contains("max_tokens") && !body["max_tokens"].is_null()) {
        if (!body["max_tokens"].is_number_integer()) {
            send_error(res, 400, "'max_tokens' must be an integer", "invalid_request_error", "invalid_request");
            return;
        }
        requested_max_tokens = body["max_tokens"].get<int>();
    }
    if (requested_max_tokens < 0) {
        send_error(res, 400, "'max_tokens' must be non-negative (0 means unlimited)", "invalid_request_error",
                   "invalid_request");
        return;
    }
    std::string session_id = body.value("session_id", std::string(""));
    // Non-OpenAI extension: opt into Qwen3.5 thinking-mode prompt rendering.
    // If omitted, use the server CLI default so clients that cannot send the
    // extension field can still select the prompt variant at startup.
    bool enable_thinking = body.value("enable_thinking", config_.default_enable_thinking);

    // Pre-scan messages for image parts. When the request carries images the
    // KV cache from a prior session turn cannot be reused (vision embeddings
    // change the prefix), so we forcibly downgrade to stateless full-prefill
    // and warn — preserves the OpenAI contract without crashing in the
    // session-vs-multimodal corner.
    bool request_has_images = false;
    for (const auto& msg : body["messages"]) {
        if (!msg.contains("content") || !msg["content"].is_array()) {
            continue;
        }
        for (const auto& p : msg["content"]) {
            if (p.is_object() && (p.value("type", "") == "image_url" || p.value("type", "") == "image")) {
                request_has_images = true;
                break;
            }
        }
        if (request_has_images) {
            break;
        }
    }
    if (request_has_images && !session_id.empty()) {
        LOGW << "[HttpServer] Session " << session_id
             << " has image input; ignoring session_id and using stateless full prefill "
                "(image embeddings invalidate cached prefix KV).";
        session_id.clear();
    }

    // 3. Extract last user message. Multimodal `content` arrays are flattened
    // to their text parts so logging and session-continuation can use a plain
    // string. Image parts are skipped here (the vision encoder owns them).
    std::string last_user_message;
    for (auto it = body["messages"].rbegin(); it != body["messages"].rend(); ++it) {
        if ((*it).value("role", "") == "user") {
            last_user_message = flatten_message_content(*it);
            break;
        }
    }

    // 4. Session concurrency check + lock
    SessionLock session_lock;
    InferenceSession* session = nullptr;
    if (!session_id.empty()) {
        session = acquire_session(session_id, enable_thinking);
        if (session == nullptr) {
            send_error(res, 409, "Session is busy with another request", "conflict_error", "session_busy");
            return;
        }
        session_lock = SessionLock(this, session_id);
    }

    // 4b. response_format (simplified JSON mode). When the client sets
    //   {"response_format": {"type": "json_object"}} or
    //   {"response_format": {"type": "json_schema", "json_schema": {...}}}
    // we inject a system-side instruction telling the model to reply with valid
    // JSON. This is the prompt-injection variant — NOT constrained decoding;
    // accuracy depends on the model honoring the directive (>95% on Qwen3.5).
    //
    // We prepend a fresh system message here, then coalesce all system/developer
    // messages below before rendering. Some HF Jinja templates, including
    // Qwen3.5's, reject multiple system messages even though OpenAI-compatible
    // clients can produce them via Responses instructions + system/developer
    // input items.
    if (body.contains("response_format") && body["response_format"].is_object()) {
        const auto& rf = body["response_format"];
        std::string ftype = rf.value("type", "");
        if (ftype == "json_object" || ftype == "json_schema") {
            std::string instruction
                = "You must respond with valid JSON only. Do not include any text outside the JSON object.";
            if (ftype == "json_schema" && rf.contains("json_schema")) {
                instruction += " The JSON must conform to this schema: " + rf["json_schema"].dump();
            }
            json sys_msg = {{"role", "system"}, {"content", instruction}};
            body["messages"].insert(body["messages"].begin(), sys_msg);
        }
    }

    // 5. Build prompt — session vs stateless. Two template paths:
    //   (a) Jinja path: when the engine has a compiled ChatTemplateJinja
    //       (Qwen3.5 family) we go through it so multimodal `content` arrays
    //       with image_url parts can be rendered. The Jinja template itself
    //       expands image parts into <|image_pad|> placeholders.
    //   (b) Legacy path: data-driven ChatTemplate for everyone else. Image
    //       parts in the request are dropped here (text-only models cannot
    //       consume them anyway); a single warning is emitted.
    const ChatTemplateJinja* jinja_tpl = engine_->chat_template_jinja();
    bool any_image_part = false;

    // OpenAI `tools` array. Forwarded to the Jinja chat template so the model
    // sees function signatures in its context. We re-parse the raw request
    // body as ordered_json so that property order inside each function
    // schema (parameters, required, etc.) is preserved — minja uses
    // ordered_json under the hood and HF chat templates iterate JSON schema
    // properties in insertion order.
    nlohmann::ordered_json tools_ordered;
    bool has_tools = false;
    if (body.contains("tools") && body["tools"].is_array() && !body["tools"].empty()) {
        try {
            nlohmann::ordered_json ordered_body = nlohmann::ordered_json::parse(req.body);
            if (ordered_body.contains("tools") && ordered_body["tools"].is_array() && !ordered_body["tools"].empty()) {
                tools_ordered = ordered_body["tools"];
                has_tools = true;
            }
        } catch (const std::exception& e) {
            LOGW << "[HttpServer] Failed to re-parse request body for ordered tools: " << e.what();
        }
    }
    if (has_tools && !jinja_tpl) {
        inject_tool_prompt_message(body);
    }
    api_compat::coalesce_system_messages(body["messages"]);

    auto build_prompt_from_messages = [&]() -> std::string {
        if (jinja_tpl) {
            std::vector<ChatMessageMM> mm_messages;
            mm_messages.reserve(body["messages"].size());
            for (const auto& msg : body["messages"]) {
                mm_messages.push_back(parse_openai_message(msg, any_image_part));
            }
            return jinja_tpl->render(mm_messages, /*add_generation_prompt=*/true,
                                     /*enable_thinking=*/enable_thinking, has_tools ? &tools_ordered : nullptr);
        }
        std::vector<std::pair<std::string, std::string>> messages;
        messages.reserve(body["messages"].size());
        for (const auto& msg : body["messages"]) {
            // Detect images for the warning; legacy template ignores them.
            if (msg.contains("content") && msg["content"].is_array()) {
                for (const auto& p : msg["content"]) {
                    if (p.is_object() && (p.value("type", "") == "image_url" || p.value("type", "") == "image")) {
                        any_image_part = true;
                        break;
                    }
                }
            }
            messages.emplace_back(normalized_chat_role(msg), flatten_message_content(msg));
        }
        return engine_->chat_template().apply(messages);
    };

    std::vector<int> input_ids;
    bool use_session = false;

    // Track whether the rendered prompt ends inside an open <think> block so
    // that streaming reasoning_content routing knows the first decoded token is
    // already thinking content (DeepSeek-R1 always, Qwen3.5 enable_thinking=true).
    bool prompt_opens_think = false;
    auto detect_open_think = [](const std::string& prompt) {
        size_t open = prompt.rfind(kReasoningOpenTag);
        if (open == std::string::npos) {
            return false;
        }
        size_t close = prompt.rfind(kReasoningCloseTag);
        return close == std::string::npos || close < open;
    };
    auto tokenize_prompt = [&](const std::string& prompt, std::vector<int>& ids) {
        try {
            ids = engine_->tokenizer().encode(prompt);
            return true;
        } catch (const std::exception& e) {
            send_error(res, 400, std::string("Failed to tokenize prompt: ") + e.what(), "invalid_request_error",
                       "prompt_tokenize_failed");
            return false;
        }
    };

    try {
        std::string prompt;
        if (!session_id.empty()) {
            // Check if session KV cache is still valid (not expired/recreated)
            if (session->is_valid() && session->past_len() > 0) {
                // Session alive with history — only prefill new message
                prompt = session->prepare_prompt(last_user_message);
                use_session = true;
            } else {
                // Session is fresh (new or recreated after expiry) — full prefill
                prompt = build_prompt_from_messages();
                use_session = true;
                LOGI << "[HttpServer] Session " << session_id << " fresh start, full prefill";
            }
        } else {
            // Stateless mode
            prompt = build_prompt_from_messages();
        }
        prompt_opens_think = detect_open_think(prompt);
        if (!tokenize_prompt(prompt, input_ids)) {
            return;
        }
    } catch (const std::exception& e) {
        send_error(res, 400, std::string("Failed to render prompt: ") + e.what(), "invalid_request_error",
                   "prompt_render_failed");
        return;
    }
    // Vision pipeline. The Jinja chat template renders ONE <|image_pad|>
    // placeholder per image part; the vision tower outputs num_image_tokens
    // (= patches / spatial_merge^2) rows per image. Before scattering we must
    // expand each single placeholder into num_image_tokens copies so the
    // tokenized input_ids have a matching count of <|image_pad|> positions.
    tensor_t multimodal_input_embeds;
    std::vector<int32_t> multimodal_pos_ids_thw;
    int32_t multimodal_mrope_delta = 0;
    if (any_image_part && engine_->has_vision()) {
        // 1. Encode every image to its embedding tensor — chunks come out in
        // encounter order across all messages.
        std::vector<EncodedImage> encoded_images;
        encoded_images.reserve(4);
        for (const auto& msg : body["messages"]) {
            if (!msg.contains("content") || !msg["content"].is_array()) {
                continue;
            }
            for (const auto& part : msg["content"]) {
                if (!part.is_object()) {
                    continue;
                }
                const std::string ptype = part.value("type", "");
                std::string uri;
                if (ptype == "image_url" && part.contains("image_url")) {
                    const auto& iu = part["image_url"];
                    if (iu.is_string()) {
                        uri = iu.get<std::string>();
                    } else if (iu.is_object() && iu.contains("url") && iu["url"].is_string()) {
                        uri = iu["url"].get<std::string>();
                    }
                } else if (ptype == "image" && part.contains("image") && part["image"].is_string()) {
                    uri = part["image"].get<std::string>();
                }
                if (uri.empty()) {
                    continue;
                }
                try {
                    encoded_images.push_back(engine_->encode_image_data_uri(uri));
                } catch (const std::exception& e) {
                    send_error(res, 400, std::string("Failed to decode / encode image: ") + e.what(),
                               "invalid_request_error", "invalid_image_data");
                    LOGW << "[HttpServer] Image encode failed: " << e.what();
                    return;
                }
            }
        }

        if (!encoded_images.empty()) {
            // 2. Re-render the prompt and expand each <|image_pad|> to num_image_tokens copies.
            const int pad_id = engine_->image_pad_token_id();
            const std::string pad_str = "<|image_pad|>";
            std::string prompt;
            try {
                prompt = build_prompt_from_messages();
            } catch (const std::exception& e) {
                send_error(res, 400, std::string("Failed to render prompt: ") + e.what(), "invalid_request_error",
                           "prompt_render_failed");
                return;
            }
            std::string expanded;
            expanded.reserve(prompt.size() + encoded_images.size() * 4096);
            size_t pos = 0, img_idx = 0;
            while (pos < prompt.size() && img_idx < encoded_images.size()) {
                const size_t hit = prompt.find(pad_str, pos);
                if (hit == std::string::npos) {
                    break;
                }
                expanded.append(prompt, pos, hit - pos);
                const int n_tok = static_cast<int>(encoded_images[img_idx].num_tokens());
                for (int k = 0; k < n_tok; ++k) { expanded.append(pad_str); }
                pos = hit + pad_str.size();
                ++img_idx;
            }
            expanded.append(prompt, pos, prompt.size() - pos);
            if (img_idx != encoded_images.size()) {
                send_error(res, 400,
                           "Image count does not match rendered <|image_pad|> placeholders: encoded "
                               + std::to_string(encoded_images.size()) + " image(s), rendered "
                               + std::to_string(img_idx) + " placeholder(s)",
                           "invalid_request_error", "image_placeholder_mismatch");
                return;
            }

            // 3. Re-tokenize the expanded prompt and reset prompt_opens_think
            //    from the expanded form (image placeholder expansion happens
            //    after the assistant <think> marker so this is identical).
            prompt_opens_think = detect_open_think(expanded);
            if (!tokenize_prompt(expanded, input_ids)) {
                return;
            }

            std::vector<tensor_t> image_chunks;
            std::vector<ImageTokenGrid> image_grids;
            image_chunks.reserve(encoded_images.size());
            image_grids.reserve(encoded_images.size());
            for (const auto& encoded : encoded_images) {
                image_chunks.push_back(encoded.embeds);
                image_grids.push_back(ImageTokenGrid{encoded.grid_t, encoded.grid_h, encoded.grid_w});
            }
            try {
                auto positions = build_multimodal_position_ids(input_ids, pad_id, image_grids);
                multimodal_pos_ids_thw = std::move(positions.pos_ids_thw);
                multimodal_mrope_delta = positions.mrope_position_delta;
            } catch (const std::exception& e) {
                send_error(res, 400, std::string("Failed to build multimodal position ids: ") + e.what(),
                           "invalid_request_error", "multimodal_positions_failed");
                LOGW << "[HttpServer] build_multimodal_position_ids failed: " << e.what();
                return;
            }

            try {
                multimodal_input_embeds = engine_->build_multimodal_input_embeds(input_ids, image_chunks);
                LOGI.printf("[HttpServer] Multimodal: %zu image chunk(s), %zu prompt tokens, pad_id=%d",
                            image_chunks.size(), input_ids.size(), pad_id);
            } catch (const std::exception& e) {
                send_error(res, 500, std::string("Failed to build multimodal input embeddings: ") + e.what(),
                           "internal_error", "multimodal_embeds_failed");
                LOGW << "[HttpServer] build_multimodal_input_embeds failed: " << e.what();
                return;
            }
        }
    } else if (any_image_part) {
        LOGW << "[HttpServer] Multimodal request received but the loaded model has no vision tower; "
                "image data ignored.";
    }

    // 6. Validate length and resolve default/unlimited max_tokens.
    int max_seq_len = engine_->exec_config().max_seq_len;
    const int prompt_token_count = static_cast<int>(input_ids.size());
    const int prefix_token_count = (use_session && session != nullptr) ? session->block_table().seq_len : 0;
    const int used_context_tokens = prefix_token_count + prompt_token_count;
    if (used_context_tokens > max_seq_len) {
        send_error(res, 400,
                   "Prompt too long: " + std::to_string(used_context_tokens) + " tokens (max "
                       + std::to_string(max_seq_len) + ")",
                   "invalid_request_error", "prompt_too_long");
        return;
    }
    int max_tokens = 0;
    try {
        max_tokens = resolve_max_new_tokens(requested_max_tokens, used_context_tokens, max_seq_len);
    } catch (const std::exception& e) {
        send_error(res, 400, e.what(), "invalid_request_error", "prompt_too_long");
        return;
    }

    // 7. Log request with content
    std::string request_id = generate_request_id();
    std::string msg_preview = last_user_message.substr(0, 100);
    if (last_user_message.size() > 100) {
        msg_preview += "...";
    }
    LOG_INFO_(utils::BOTH) << "[Request] " << request_id << " | session=" << (session_id.empty() ? "none" : session_id)
                           << " | tokens=" << input_ids.size() << " | max=" << max_tokens
                           << " | stream=" << (stream ? "Y" : "N") << " | user: " << msg_preview;

    // 8. Build inference request
    auto cancel_flag = std::make_shared<std::atomic<bool>>(false);

    auto inference_req = std::make_unique<InferenceRequest>();
    inference_req->input_ids = std::move(input_ids);
    inference_req->config.max_new_tokens = max_tokens;
    inference_req->config.enable_thinking = enable_thinking;
    inference_req->config.max_think_tokens = config_.default_max_think_tokens;
    inference_req->arrival_time = std::chrono::steady_clock::now();
    inference_req->cancelled = cancel_flag;
    // Attach vision-pre-baked input_embeds when present; the serving loop
    // forwards this tensor to hybrid_transformer_forward, which bypasses the
    // text embed_tokens lookup and uses our tensor (with image embeddings
    // already scattered) as the layer-0 hidden state.
    if (multimodal_input_embeds) {
        inference_req->set_input_embeds(multimodal_input_embeds);
        inference_req->set_pos_ids_thw_host(std::move(multimodal_pos_ids_thw), inference_req->input_ids.size(),
                                            multimodal_mrope_delta);
    }

    // OpenAI-compatible sampling overrides. Setting has_sampling_override
    // tells the scheduler to bypass the engine's default sampler and route
    // through the argmax/general samplers based on these fields.
    auto& gen_cfg = inference_req->config;
    bool any_sampling_field = false;
    if (body.contains("temperature") && body["temperature"].is_number()) {
        gen_cfg.temperature = body["temperature"].get<float>();
        any_sampling_field = true;
    }
    if (body.contains("top_p") && body["top_p"].is_number()) {
        gen_cfg.top_p = body["top_p"].get<float>();
        any_sampling_field = true;
    }
    if (body.contains("top_k") && body["top_k"].is_number_integer()) {
        gen_cfg.top_k = body["top_k"].get<int>();
        any_sampling_field = true;
    }
    if (body.contains("repetition_penalty") && body["repetition_penalty"].is_number()) {
        gen_cfg.repetition_penalty = body["repetition_penalty"].get<float>();
        any_sampling_field = true;
    }
    if (body.contains("seed") && body["seed"].is_number_integer()) {
        gen_cfg.seed = static_cast<unsigned int>(body["seed"].get<int64_t>());
        any_sampling_field = true;
    }
    // OpenAI `stop` may be either a string or array of strings. Stored on the
    // request; the HTTP layer enforces it on the decoded stream (the scheduler
    // operates at token-id granularity and would not match arbitrary substrings).
    if (body.contains("stop")) {
        const auto& s = body["stop"];
        if (s.is_string()) {
            gen_cfg.stop_sequences.push_back(s.get<std::string>());
            any_sampling_field = true;
        } else if (s.is_array()) {
            for (const auto& v : s) {
                if (v.is_string()) {
                    gen_cfg.stop_sequences.push_back(v.get<std::string>());
                }
            }
            if (!gen_cfg.stop_sequences.empty()) {
                any_sampling_field = true;
            }
        }
    }
    gen_cfg.has_sampling_override = any_sampling_field;
    // OpenAI greedy semantics: temperature == 0 maps to argmax.
    if (any_sampling_field && gen_cfg.temperature <= 0.0f) {
        gen_cfg.use_argmax = true;
    }
    // Snapshot stop_sequences for HTTP-layer post-processing; the inference_req
    // is about to be moved into the scheduler and must not be touched after.
    std::vector<std::string> stop_sequences_snapshot = gen_cfg.stop_sequences;
    if (use_session && session) {
        inference_req->borrow_block_table(session->block_table());
    }

    std::shared_ptr<TokenQueue<std::string>> token_queue;
    if (stream) {
        token_queue = std::make_shared<TokenQueue<std::string>>();
        inference_req->config.stream = true;
        inference_req->stream_callback = [token_queue, cancel_flag](const std::string& tok) {
            if (!cancel_flag->load()) {
                token_queue->push(tok);
            }
        };
    }

    // 9. Submit
    std::future<GenerationResult> future;
    try {
        future = engine_->serving_loop().submit_async(std::move(inference_req));
    } catch (const std::runtime_error&) {
        send_error(res, 503, "Server overloaded, queue full", "server_error", "queue_full");
        return;
    }

    auto t_start = std::chrono::steady_clock::now();

    // 10. Respond
    if (stream) {
        // --- SSE streaming ---
        struct Ctx {
            std::future<GenerationResult> future;
            std::shared_ptr<TokenQueue<std::string>> queue;
            std::shared_ptr<std::atomic<bool>> cancel_flag;
            std::string req_id, model, user_msg, output_prefix;
            int64_t epoch;
            int prompt_tokens = 0;
            int timeout_sec;
            InferenceSession* session;
            SessionLock lock;
            std::chrono::steady_clock::time_point t_start;
            bool sent_role = false;
            bool sent_prefix = false;
            std::string utf8_buf, full_output;
            std::chrono::steady_clock::time_point last_activity;
            // Reasoning-mode state machine. in_thinking flips on <think> and
            // back on </think>; output emitted while in_thinking==true lands in
            // delta.reasoning_content, otherwise delta.content. reasoning_carry
            // holds a short tail of bytes that *might* be the start of a
            // <think>/</think> tag — they are withheld from emission until the
            // next chunk arrives so we never split a tag across deltas.
            bool in_thinking = false;
            std::string reasoning_carry;
            // Stop-sequence filter. When stop_sequences is non-empty, the
            // flush pipeline scans the cumulative byte stream for any stop
            // string, truncates the output there, and sets stop_hit so the
            // outer loop emits finish_reason="stop" + [DONE]. stop_pending
            // holds the trailing bytes that might be a partial prefix of any
            // stop string (so a stop can be detected even when split across
            // two SSE chunks).
            std::vector<std::string> stop_sequences;
            std::string stop_pending;
            bool stop_hit = false;
            // Tool-call streaming state machine. Recognises Qwen3-style
            // <tool_call>...</tool_call> blocks, accumulates the inner text,
            // parses it (JSON or XML), and emits delta.tool_calls chunks.
            // tool_carry holds a short tail that might be a partial prefix of
            // <tool_call> / </tool_call>.
            bool in_tool_call = false;
            std::string tool_buf;
            std::string tool_carry;
            std::string tool_open_marker;
            std::string tool_close_marker;
            int tool_id_counter = 0;
            bool has_emitted_tool_call = false;
            api_compat::ResponseFormat format = api_compat::ResponseFormat::OpenAIChat;
            std::string resp_id, resp_msg_id, resp_reasoning_id;
            bool responses_reasoning_started = false;
            bool responses_text_started = false;
            bool anthropic_thinking_started = false;
            bool anthropic_text_started = false;
            int anthropic_tool_count = 0;
            std::string stream_content_text;
            std::string stream_reasoning_text;
            json stream_tool_calls = json::array();
        };

        auto ctx = std::make_shared<Ctx>();
        ctx->future = std::move(future);
        ctx->queue = std::move(token_queue);
        ctx->cancel_flag = cancel_flag;
        ctx->req_id = request_id;
        ctx->model = display_model_name_;
        ctx->format = compat_format;
        {
            std::string suffix = compat_suffix_from_chat_id(request_id);
            ctx->resp_id = "resp_" + suffix;
            ctx->resp_msg_id = "msg_" + suffix;
            ctx->resp_reasoning_id = "rs_" + suffix;
        }
        ctx->user_msg = last_user_message;
        ctx->output_prefix = session ? session->output_prefix() : "";
        ctx->epoch
            = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch())
                  .count();
        ctx->prompt_tokens = prompt_token_count;
        ctx->timeout_sec = config_.request_timeout_sec;
        ctx->session = session;
        ctx->lock = std::move(session_lock); // transfer lock ownership to ctx
        ctx->t_start = t_start;
        ctx->last_activity = std::chrono::steady_clock::now();
        // Seed reasoning state from the rendered prompt: if the chat template's
        // generation_prompt left a <think> open, the first decoded token is
        // already reasoning content.
        ctx->in_thinking = prompt_opens_think;
        // Hand the stop-sequence list over to the SSE state so the streaming
        // path can enforce it in lock-step with the blocking path.
        ctx->stop_sequences = stop_sequences_snapshot;
        res.set_header("Cache-Control", "no-cache");
        res.set_header("Connection", "keep-alive");

        res.set_chunked_content_provider(
            "text/event-stream",
            [ctx](size_t, httplib::DataSink& sink) -> bool {
                try {
                    auto sse = [&sink](const json& j) {
                        std::string d = "data: " + j.dump() + "\n\n";
                        sink.write(d.data(), d.size());
                    };
                    auto sse_event = [&sink](const std::string& event, const json& data) {
                        std::string d = "event: " + event + "\ndata: " + data.dump() + "\n\n";
                        sink.write(d.data(), d.size());
                    };
                    auto base = [&ctx]() {
                        return json{{"id", ctx->req_id},
                                    {"object", "chat.completion.chunk"},
                                    {"created", ctx->epoch},
                                    {"model", ctx->model}};
                    };
                    auto responses_failed = [&ctx, &sse_event](const std::string& message, const std::string& code) {
                        json response = {{"id", ctx->resp_id},
                                         {"object", "response"},
                                         {"created_at", ctx->epoch},
                                         {"status", "failed"},
                                         {"model", ctx->model},
                                         {"output", json::array()},
                                         {"error", {{"code", code}, {"message", message}}}};
                        sse_event("response.failed", {{"type", "response.failed"}, {"response", response}});
                    };
                    // Emit one delta payload — `is_reasoning` selects between
                    // delta.content (normal text) and delta.reasoning_content
                    // (text inside <think>...</think>). full_output keeps the
                    // raw concatenation including tags, so post-stream stop_seq
                    // logic and session storage stay consistent.
                    auto send_chunk = [&ctx, &sse, &sse_event, &base](const std::string& text, bool is_reasoning) {
                        if (text.empty()) {
                            return;
                        }
                        ctx->full_output += text;
                        if (is_reasoning) {
                            ctx->stream_reasoning_text += text;
                        } else {
                            ctx->stream_content_text += text;
                        }
                        if (ctx->format == api_compat::ResponseFormat::Anthropic) {
                            if (is_reasoning) {
                                if (!ctx->anthropic_thinking_started) {
                                    sse_event("content_block_start",
                                              {{"type", "content_block_start"},
                                               {"index", 0},
                                               {"content_block", {{"type", "thinking"}, {"thinking", ""}}}});
                                    ctx->anthropic_thinking_started = true;
                                }
                                sse_event("content_block_delta",
                                          {{"type", "content_block_delta"},
                                           {"index", 0},
                                           {"delta", {{"type", "thinking_delta"}, {"thinking", text}}}});
                            } else {
                                const int index = ctx->anthropic_thinking_started ? 1 : 0;
                                if (!ctx->anthropic_text_started) {
                                    sse_event("content_block_start",
                                              {{"type", "content_block_start"},
                                               {"index", index},
                                               {"content_block", {{"type", "text"}, {"text", ""}}}});
                                    ctx->anthropic_text_started = true;
                                }
                                sse_event("content_block_delta", {{"type", "content_block_delta"},
                                                                  {"index", index},
                                                                  {"delta", {{"type", "text_delta"}, {"text", text}}}});
                            }
                            return;
                        }
                        if (ctx->format == api_compat::ResponseFormat::OpenAIResponses) {
                            if (is_reasoning) {
                                if (!ctx->responses_reasoning_started) {
                                    sse_event("response.output_item.added", {{"type", "response.output_item.added"},
                                                                             {"item",
                                                                              {{"id", ctx->resp_reasoning_id},
                                                                               {"summary", json::array()},
                                                                               {"type", "reasoning"},
                                                                               {"content", json::array()},
                                                                               {"encrypted_content", ""},
                                                                               {"status", "in_progress"}}}});
                                    ctx->responses_reasoning_started = true;
                                }
                                sse_event("response.reasoning_text.delta", {{"type", "response.reasoning_text.delta"},
                                                                            {"item_id", ctx->resp_reasoning_id},
                                                                            {"delta", text}});
                            } else {
                                if (!ctx->responses_text_started) {
                                    sse_event("response.output_item.added", {{"type", "response.output_item.added"},
                                                                             {"item",
                                                                              {{"id", ctx->resp_msg_id},
                                                                               {"type", "message"},
                                                                               {"role", "assistant"},
                                                                               {"status", "in_progress"},
                                                                               {"content", json::array()}}}});
                                    sse_event("response.content_part.added",
                                              {{"type", "response.content_part.added"},
                                               {"item_id", ctx->resp_msg_id},
                                               {"part", {{"type", "output_text"}, {"text", ""}}}});
                                    ctx->responses_text_started = true;
                                }
                                sse_event("response.output_text.delta", {{"type", "response.output_text.delta"},
                                                                         {"item_id", ctx->resp_msg_id},
                                                                         {"delta", text}});
                            }
                            return;
                        }
                        json delta;
                        if (is_reasoning) {
                            delta["reasoning_content"] = text;
                        } else {
                            delta["content"] = text;
                        }
                        json c = base();
                        c["choices"] = json::array({{{"index", 0}, {"delta", delta}, {"finish_reason", nullptr}}});
                        sse(c);
                    };
                    // Route `chunk` into reasoning / content deltas honoring the
                    // running ctx->in_thinking state and the <think>/</think>
                    // tag boundaries. To avoid splitting a tag across two SSE
                    // chunks, hold back any trailing suffix that *might* be a
                    // tag prefix into ctx->reasoning_carry; the next call
                    // prepends it. On `force`, the carry is flushed verbatim.
                    auto route_reasoning = [&ctx, &send_chunk](std::string chunk, bool force) {
                        chunk = ctx->reasoning_carry + chunk;
                        ctx->reasoning_carry.clear();
                        if (!force) {
                            const size_t maxK = std::max(kReasoningOpenLen, kReasoningCloseLen) - 1;
                            size_t safe_end = chunk.size();
                            for (size_t k = std::min(chunk.size(), maxK); k > 0; --k) {
                                std::string suffix = chunk.substr(chunk.size() - k);
                                bool match = (std::string(kReasoningOpenTag).compare(0, k, suffix) == 0)
                                          || (std::string(kReasoningCloseTag).compare(0, k, suffix) == 0);
                                if (match) {
                                    safe_end = chunk.size() - k;
                                    break;
                                }
                            }
                            ctx->reasoning_carry = chunk.substr(safe_end);
                            chunk.resize(safe_end);
                        }
                        size_t pos = 0;
                        while (pos < chunk.size()) {
                            if (ctx->in_thinking) {
                                size_t end = chunk.find(kReasoningCloseTag, pos);
                                if (end == std::string::npos) {
                                    send_chunk(chunk.substr(pos), true);
                                    pos = chunk.size();
                                } else {
                                    send_chunk(chunk.substr(pos, end - pos), true);
                                    pos = end + kReasoningCloseLen;
                                    ctx->in_thinking = false;
                                }
                            } else {
                                size_t open = chunk.find(kReasoningOpenTag, pos);
                                if (open == std::string::npos) {
                                    send_chunk(chunk.substr(pos), false);
                                    pos = chunk.size();
                                } else {
                                    send_chunk(chunk.substr(pos, open - pos), false);
                                    pos = open + kReasoningOpenLen;
                                    ctx->in_thinking = true;
                                }
                            }
                        }
                    };
                    // Emit a single Qwen3-style tool_call as a delta.tool_calls
                    // SSE chunk. Increments tool_id_counter; flips
                    // has_emitted_tool_call so the completion chunk uses
                    // finish_reason="tool_calls".
                    auto emit_tool_call_delta = [&ctx, &sse, &sse_event, &base](const std::string& fn_name,
                                                                                const json& fn_args) {
                        if (fn_name.empty()) {
                            return;
                        }
                        json call;
                        call["index"] = ctx->tool_id_counter++;
                        call["id"] = generate_tool_call_id();
                        call["type"] = "function";
                        json fn;
                        fn["name"] = fn_name;
                        fn["arguments"] = fn_args.is_string() ? fn_args.get<std::string>() : fn_args.dump();
                        call["function"] = std::move(fn);
                        ctx->stream_tool_calls.push_back(call);
                        ctx->has_emitted_tool_call = true;

                        if (ctx->format == api_compat::ResponseFormat::Anthropic) {
                            const int index = (ctx->anthropic_thinking_started ? 1 : 0)
                                            + (ctx->anthropic_text_started ? 1 : 0) + ctx->anthropic_tool_count++;
                            sse_event("content_block_start", {{"type", "content_block_start"},
                                                              {"index", index},
                                                              {"content_block",
                                                               {{"type", "tool_use"},
                                                                {"id", call["id"]},
                                                                {"name", fn_name},
                                                                {"input", json::object()}}}});
                            sse_event("content_block_delta", {{"type", "content_block_delta"},
                                                              {"index", index},
                                                              {"delta",
                                                               {{"type", "input_json_delta"},
                                                                {"partial_json", call["function"]["arguments"]}}}});
                            return;
                        }
                        if (ctx->format == api_compat::ResponseFormat::OpenAIResponses) {
                            const std::string fc_id = "fc_" + call["id"].get<std::string>();
                            sse_event("response.output_item.added", {{"type", "response.output_item.added"},
                                                                     {"item",
                                                                      {{"type", "function_call"},
                                                                       {"status", "in_progress"},
                                                                       {"arguments", ""},
                                                                       {"call_id", fc_id},
                                                                       {"name", fn_name}}}});
                            sse_event("response.function_call_arguments.delta",
                                      {{"type", "response.function_call_arguments.delta"},
                                       {"item_id", fc_id},
                                       {"delta", call["function"]["arguments"]}});
                            return;
                        }
                        json delta;
                        delta["tool_calls"] = json::array({call});
                        json c = base();
                        c["choices"] = json::array({{{"index", 0}, {"delta", delta}, {"finish_reason", nullptr}}});
                        sse(c);
                    };
                    // Split `chunk` at <tool_call>...</tool_call> boundaries. Text
                    // outside tool_call blocks forwards to route_reasoning; text
                    // inside accumulates in ctx->tool_buf and on closing tag is
                    // parsed (JSON-then-XML) and emitted via emit_tool_call_delta.
                    auto route_tool = [&ctx, &route_reasoning, &emit_tool_call_delta](std::string chunk, bool force) {
                        chunk = ctx->tool_carry + chunk;
                        ctx->tool_carry.clear();
                        if (!force) {
                            size_t maxK = max_tool_call_marker_len();
                            maxK = maxK > 0 ? maxK - 1 : 0;
                            size_t safe_end = chunk.size();
                            for (size_t k = std::min(chunk.size(), maxK); k > 0; --k) {
                                std::string suffix = chunk.substr(chunk.size() - k);
                                bool match = false;
                                for (const auto& tag : kToolCallTags) {
                                    if (std::string(tag.open).compare(0, k, suffix) == 0
                                        || std::string(tag.close).compare(0, k, suffix) == 0) {
                                        match = true;
                                        break;
                                    }
                                }
                                if (match) {
                                    safe_end = chunk.size() - k;
                                    break;
                                }
                            }
                            ctx->tool_carry = chunk.substr(safe_end);
                            chunk.resize(safe_end);
                        }
                        size_t pos = 0;
                        while (pos < chunk.size()) {
                            if (ctx->in_tool_call) {
                                const std::string close_marker = ctx->tool_close_marker.empty()
                                                                   ? std::string("</tool_call>")
                                                                   : ctx->tool_close_marker;
                                size_t end = chunk.find(close_marker, pos);
                                if (end == std::string::npos) {
                                    ctx->tool_buf.append(chunk, pos, chunk.size() - pos);
                                    pos = chunk.size();
                                } else {
                                    ctx->tool_buf.append(chunk, pos, end - pos);
                                    pos = end + close_marker.size();
                                    ctx->in_tool_call = false;
                                    std::string trimmed = trim_ascii_ws(ctx->tool_buf);
                                    std::string open_marker = ctx->tool_open_marker.empty() ? std::string("<tool_call>")
                                                                                            : ctx->tool_open_marker;
                                    ctx->tool_buf.clear();
                                    ctx->tool_open_marker.clear();
                                    ctx->tool_close_marker.clear();
                                    std::vector<ParsedToolCall> parsed_calls = parse_tool_call_payload(trimmed);
                                    if (!parsed_calls.empty()) {
                                        for (const auto& parsed : parsed_calls) {
                                            emit_tool_call_delta(parsed.name, parsed.arguments);
                                        }
                                    } else {
                                        route_reasoning(open_marker + trimmed + close_marker, force);
                                    }
                                }
                            } else {
                                size_t open = std::string::npos;
                                const ToolCallTag* tag = find_next_tool_call_tag(chunk, pos, open);
                                if (tag == nullptr) {
                                    route_reasoning(chunk.substr(pos), force);
                                    pos = chunk.size();
                                } else {
                                    if (open > pos) {
                                        route_reasoning(chunk.substr(pos, open - pos), false);
                                    }
                                    pos = open + tag->open_len;
                                    ctx->in_tool_call = true;
                                    ctx->tool_buf.clear();
                                    ctx->tool_open_marker = tag->open;
                                    ctx->tool_close_marker = tag->close;
                                }
                            }
                        }
                        if (force && ctx->in_tool_call) {
                            route_reasoning(ctx->tool_open_marker + ctx->tool_buf, true);
                            ctx->in_tool_call = false;
                            ctx->tool_buf.clear();
                            ctx->tool_open_marker.clear();
                            ctx->tool_close_marker.clear();
                        }
                    };
                    // Stop-sequence filter. Runs first in the flush pipeline so a
                    // user-supplied stop terminates the stream regardless of
                    // whether the bytes would have ended up in content,
                    // reasoning_content, or a tool_call. Sets ctx->stop_hit and
                    // signals scheduler cancellation; the SSE loop sees stop_hit
                    // and stops emitting further chunks.
                    auto stop_filter = [&ctx](std::string text, bool force) -> std::string {
                        if (ctx->stop_sequences.empty() || ctx->stop_hit) {
                            return text;
                        }
                        std::string combined = ctx->stop_pending + text;
                        ctx->stop_pending.clear();
                        size_t earliest = std::string::npos;
                        for (const auto& s : ctx->stop_sequences) {
                            if (s.empty()) {
                                continue;
                            }
                            size_t p = combined.find(s);
                            if (p != std::string::npos && (earliest == std::string::npos || p < earliest)) {
                                earliest = p;
                            }
                        }
                        if (earliest != std::string::npos) {
                            ctx->stop_hit = true;
                            if (ctx->cancel_flag) {
                                ctx->cancel_flag->store(true);
                            }
                            return combined.substr(0, earliest);
                        }
                        if (!force) {
                            size_t maxK = 0;
                            for (const auto& s : ctx->stop_sequences) {
                                if (s.size() > maxK) {
                                    maxK = s.size();
                                }
                            }
                            if (maxK > 0) {
                                maxK--;
                                size_t carry = std::min(combined.size(), maxK);
                                ctx->stop_pending = combined.substr(combined.size() - carry);
                                combined.resize(combined.size() - carry);
                            }
                        }
                        return combined;
                    };
                    auto flush = [&ctx, &route_tool, &stop_filter](bool force) {
                        if (ctx->stop_hit) {
                            return;
                        }
                        const bool has_carry
                            = !ctx->reasoning_carry.empty() || !ctx->tool_carry.empty() || !ctx->stop_pending.empty();
                        if (ctx->utf8_buf.empty() && !(force && has_carry)) {
                            return;
                        }
                        size_t n = force ? ctx->utf8_buf.size() : valid_utf8_length(ctx->utf8_buf);
                        std::string segment;
                        if (n > 0) {
                            segment = ctx->utf8_buf.substr(0, n);
                            ctx->utf8_buf.erase(0, n);
                        }
                        std::string filtered = stop_filter(std::move(segment), force);
                        if (!filtered.empty() || force) {
                            route_tool(std::move(filtered), force);
                        }
                    };

                    // Send role delta
                    if (!ctx->sent_role) {
                        ctx->sent_role = true;
                        if (ctx->format == api_compat::ResponseFormat::Anthropic) {
                            sse_event("message_start", {{"type", "message_start"},
                                                        {"message",
                                                         {{"id", ctx->req_id},
                                                          {"type", "message"},
                                                          {"role", "assistant"},
                                                          {"content", json::array()},
                                                          {"model", ctx->model},
                                                          {"stop_reason", nullptr},
                                                          {"stop_sequence", nullptr},
                                                          {"usage",
                                                           {{"cache_read_input_tokens", 0},
                                                            {"input_tokens", ctx->prompt_tokens},
                                                            {"output_tokens", 0}}}}}});
                        } else if (ctx->format == api_compat::ResponseFormat::OpenAIResponses) {
                            json response
                                = {{"id", ctx->resp_id},      {"object", "response"}, {"created_at", ctx->epoch},
                                   {"status", "in_progress"}, {"model", ctx->model},  {"output", json::array()}};
                            sse_event("response.created", {{"type", "response.created"}, {"response", response}});
                            sse_event("response.in_progress",
                                      {{"type", "response.in_progress"}, {"response", response}});
                        } else {
                            json c = base();
                            c["choices"] = json::array(
                                {{{"index", 0}, {"delta", {{"role", "assistant"}}}, {"finish_reason", nullptr}}});
                            sse(c);
                        }
                    }

                    // Route the session's output_prefix (DeepSeek-R1 prepends
                    // "<think> ", Qwen3.5 thinking-mode may set similar) through
                    // the reasoning state machine so an opening <think> in the
                    // prefix flips into reasoning_content mode for subsequent
                    // model output. The prefix itself is sent on first activation.
                    if (!ctx->sent_prefix && !ctx->output_prefix.empty()) {
                        ctx->sent_prefix = true;
                        route_reasoning(ctx->output_prefix, false);
                    }

                    std::string tok;
                    bool got = false;
                    while (ctx->queue->try_pop(tok, std::chrono::milliseconds(50))) {
                        ctx->utf8_buf += tok;
                        flush(false);
                        got = true;
                    }
                    if (got) {
                        ctx->last_activity = std::chrono::steady_clock::now();
                    }

                    // Timeout
                    auto idle = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now()
                                                                                 - ctx->last_activity)
                                    .count();
                    if (idle >= ctx->timeout_sec) {
                        if (ctx->session) {
                            ctx->session->abort_turn();
                        }
                        if (ctx->format == api_compat::ResponseFormat::OpenAIResponses) {
                            responses_failed("Request timed out", "timeout_error");
                        } else {
                            sse({{"error", {{"message", "Request timed out"}, {"type", "timeout_error"}}}});
                            sink.write("data: [DONE]\n\n", 15);
                        }
                        sink.done();
                        ctx->lock.unlock();
                        return false;
                    }

                    // Completion
                    if (ctx->future.wait_for(std::chrono::milliseconds(0)) == std::future_status::ready) {
                        while (ctx->queue->try_pop(tok, std::chrono::milliseconds(0))) { ctx->utf8_buf += tok; }
                        flush(true);

                        try {
                            auto result = ctx->future.get();
                            if (ctx->session) {
                                ctx->session->complete_turn(ctx->user_msg, ctx->full_output);
                            }

                            auto elapsed = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now()
                                                                                     - ctx->t_start)
                                               .count();
                            int gen_tokens = static_cast<int>(result.output_ids.size());
                            std::string out_preview = ctx->full_output.substr(0, 100);
                            if (ctx->full_output.size() > 100) {
                                out_preview += "...";
                            }
                            LOG_INFO_(utils::BOTH)
                                << "[Response] " << ctx->req_id << " | tokens=" << gen_tokens << " | "
                                << static_cast<int>(elapsed) << "ms" << " | " << std::fixed << std::setprecision(1)
                                << (elapsed > 0 ? gen_tokens * 1000.0 / elapsed : 0) << " tok/s"
                                << " | assistant: " << out_preview;

                            // Resolve finish_reason. Priority:
                            //   stop_hit         -> "stop"          (client stop sequence terminated us)
                            //   tool_call emitted -> "tool_calls"   (model produced at least one tool call)
                            //   else             -> scheduler-set    ("stop" on EOS, "length" on max_tokens)
                            std::string fr = result.finish_reason;
                            if (ctx->has_emitted_tool_call) {
                                fr = "tool_calls";
                            }
                            if (ctx->stop_hit) {
                                fr = "stop";
                            }
                            json message = {{"role", "assistant"}};
                            if (!ctx->stream_tool_calls.empty()) {
                                message["content"]
                                    = ctx->stream_content_text.empty() ? json(nullptr) : json(ctx->stream_content_text);
                                message["tool_calls"] = ctx->stream_tool_calls;
                            } else {
                                message["content"] = ctx->stream_content_text;
                            }
                            if (!ctx->stream_reasoning_text.empty()) {
                                message["reasoning_content"] = ctx->stream_reasoning_text;
                            }
                            json chat_response = {
                                {"id", ctx->req_id},
                                {"object", "chat.completion"},
                                {"created", ctx->epoch},
                                {"model", ctx->model},
                                {"choices", json::array({{{"index", 0}, {"message", message}, {"finish_reason", fr}}})},
                                {"usage",
                                 {{"prompt_tokens", result.stats.prompt_tokens},
                                  {"completion_tokens", result.stats.generated_tokens},
                                  {"total_tokens", result.stats.total_tokens}}}};

                            if (ctx->format == api_compat::ResponseFormat::OpenAIChat) {
                                json c = base();
                                c["choices"]
                                    = json::array({{{"index", 0}, {"delta", json::object()}, {"finish_reason", fr}}});
                                c["usage"] = chat_response["usage"];
                                sse(c);
                            } else if (ctx->format == api_compat::ResponseFormat::Anthropic) {
                                if (!ctx->anthropic_thinking_started && !ctx->anthropic_text_started
                                    && ctx->anthropic_tool_count == 0) {
                                    ctx->anthropic_text_started = true;
                                    sse_event("content_block_start",
                                              {{"type", "content_block_start"},
                                               {"index", 0},
                                               {"content_block", {{"type", "text"}, {"text", ""}}}});
                                }
                                if (ctx->anthropic_thinking_started) {
                                    sse_event("content_block_delta",
                                              {{"type", "content_block_delta"},
                                               {"index", 0},
                                               {"delta", {{"type", "signature_delta"}, {"signature", ""}}}});
                                    sse_event("content_block_stop", {{"type", "content_block_stop"}, {"index", 0}});
                                }
                                if (ctx->anthropic_text_started) {
                                    const int index = ctx->anthropic_thinking_started ? 1 : 0;
                                    sse_event("content_block_stop", {{"type", "content_block_stop"}, {"index", index}});
                                }
                                const int tool_base
                                    = (ctx->anthropic_thinking_started ? 1 : 0) + (ctx->anthropic_text_started ? 1 : 0);
                                for (int i = 0; i < ctx->anthropic_tool_count; ++i) {
                                    sse_event("content_block_stop",
                                              {{"type", "content_block_stop"}, {"index", tool_base + i}});
                                }
                                const std::string stop_reason
                                    = api_compat::normalize_anthropic_stop_reason(fr, !ctx->stream_tool_calls.empty());
                                sse_event("message_delta",
                                          {{"type", "message_delta"},
                                           {"delta", {{"stop_reason", stop_reason}, {"stop_sequence", nullptr}}},
                                           {"usage", {{"output_tokens", result.stats.generated_tokens}}}});
                                sse_event("message_stop", {{"type", "message_stop"}});
                            } else {
                                json response = api_compat::chat_response_to_responses(chat_response);
                                for (const auto& item : response["output"]) {
                                    const std::string type = item.value("type", "");
                                    if (type == "message" && item.contains("content") && !item["content"].empty()) {
                                        const json part = item["content"][0];
                                        sse_event("response.output_text.done",
                                                  {{"type", "response.output_text.done"},
                                                   {"item_id", item["id"]},
                                                   {"text", part.value("text", std::string())}});
                                        sse_event("response.content_part.done", {{"type", "response.content_part.done"},
                                                                                 {"item_id", item["id"]},
                                                                                 {"part", part}});
                                    }
                                    sse_event("response.output_item.done",
                                              {{"type", "response.output_item.done"}, {"item", item}});
                                }
                                sse_event("response.completed",
                                          {{"type", "response.completed"}, {"response", response}});
                            }
                        } catch (const std::exception& e) {
                            LOGE << "[Response] " << ctx->req_id << " error: " << e.what();
                            if (ctx->session) {
                                ctx->session->abort_turn();
                            }
                            if (ctx->format == api_compat::ResponseFormat::OpenAIResponses) {
                                responses_failed(e.what(), "internal_error");
                            } else {
                                sse({{"error", {{"message", e.what()}, {"type", "internal_error"}}}});
                            }
                        }
                        if (ctx->format == api_compat::ResponseFormat::OpenAIChat) {
                            sink.write("data: [DONE]\n\n", 15);
                        }
                        sink.done();
                        ctx->lock.unlock();
                        return false;
                    }
                    return true;

                } catch (const std::exception& e) {
                    ctx->cancel_flag->store(true);
                    if (ctx->session) {
                        ctx->session->abort_turn();
                    }
                    LOGW << "[Response] " << ctx->req_id << " stream aborted: " << e.what();
                    try {
                        if (ctx->format == api_compat::ResponseFormat::OpenAIResponses) {
                            json response = {{"id", ctx->resp_id},
                                             {"object", "response"},
                                             {"created_at", ctx->epoch},
                                             {"status", "failed"},
                                             {"model", ctx->model},
                                             {"output", json::array()},
                                             {"error", {{"code", "internal_error"}, {"message", e.what()}}}};
                            json event = {{"type", "response.failed"}, {"response", response}};
                            std::string d = "event: response.failed\ndata: " + event.dump() + "\n\n";
                            sink.write(d.data(), d.size());
                        } else {
                            sink.write("data: [DONE]\n\n", 15);
                        }
                    } catch (...) {}
                    sink.done();
                    ctx->lock.unlock();
                    return false;
                }
            },
            // on_close: cancel generation when client disconnects
            [ctx](bool success) {
                if (!success) {
                    ctx->cancel_flag->store(true);
                    if (ctx->session) {
                        ctx->session->abort_turn();
                    }
                    LOGI << "[Request] " << ctx->req_id << " client disconnected";
                }
                ctx->lock.unlock();
            });

    } else {
        // --- Blocking response ---
        if (future.wait_for(std::chrono::seconds(config_.request_timeout_sec)) == std::future_status::timeout) {
            send_error(res, 408, "Request timed out", "timeout_error", "request_timeout");
            LOGW << "[Response] " << request_id << " timed out";
            return;
        }

        GenerationResult result;
        try {
            result = future.get();
        } catch (const std::exception& e) {
            if (session) {
                session->abort_turn();
            }
            send_error(res, 500, std::string("Generation failed: ") + e.what(), "internal_error", "generation_failed");
            LOGE << "[Response] " << request_id << " error: " << e.what();
            return;
        }

        std::string output_text = engine_->tokenizer().decode(result.output_ids);
        if (session) {
            session->complete_turn(last_user_message, output_text);
        }
        // Prepend output_prefix for display (e.g. DeepSeek-R1 "<think> ")
        std::string output_prefix = session ? session->output_prefix() : "";
        if (!output_prefix.empty()) {
            output_text = output_prefix + output_text;
        }
        // Apply OpenAI-compatible stop sequences: scan decoded output for the
        // earliest occurrence of any stop string and truncate there. The stop
        // sequence itself is NOT included in the response, per OpenAI spec.
        // finish_reason is forced to "stop" since we hit a user-defined boundary.
        if (!stop_sequences_snapshot.empty()) {
            size_t earliest = output_text.size();
            for (const auto& s : stop_sequences_snapshot) {
                if (s.empty()) {
                    continue;
                }
                size_t p = output_text.find(s);
                if (p != std::string::npos && p < earliest) {
                    earliest = p;
                }
            }
            if (earliest < output_text.size()) {
                output_text = output_text.substr(0, earliest);
                result.finish_reason = "stop";
            }
        }

        // Log completion with output preview
        auto elapsed = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_start).count();
        std::string out_preview = output_text.substr(0, 100);
        if (output_text.size() > 100) {
            out_preview += "...";
        }
        LOG_INFO_(utils::BOTH) << "[Response] " << request_id << " | tokens=" << result.stats.generated_tokens << " | "
                               << static_cast<int>(elapsed) << "ms" << " | " << std::fixed << std::setprecision(1)
                               << (elapsed > 0 ? result.stats.generated_tokens * 1000.0 / elapsed : 0) << " tok/s"
                               << " | assistant: " << out_preview;

        auto epoch
            = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch())
                  .count();

        // Split reasoning (<think>...</think>) from final content. Initial-state
        // sources, in priority order:
        //   1. Session output_prefix opens a <think> block (DeepSeek-R1 session path).
        //   2. Stateless / fresh-session heuristic: when the chat template renders
        //      its generation_prompt ending with an open <think> tag (Qwen3.5
        //      enable_thinking=true, DeepSeek-R1 always), the model's first
        //      output token is already inside the thinking block. In that case
        //      output_text contains a </think> with no preceding <think>; treat
        //      everything before the first </think> as reasoning.
        bool initial_in_think = !output_prefix.empty() && (output_prefix.find(kReasoningOpenTag) != std::string::npos)
                             && (output_prefix.find(kReasoningCloseTag) == std::string::npos);
        if (!initial_in_think && output_text.find(kReasoningCloseTag) != std::string::npos
            && output_text.find(kReasoningOpenTag) == std::string::npos) {
            initial_in_think = true;
        }
        ReasoningSplit split = split_reasoning(output_text, initial_in_think);

        // Extract Qwen3-style <tool_call>...</tool_call> blocks from the
        // post-reasoning content. tool_calls (if any) get surfaced through the
        // OpenAI tool_calls field and the finish_reason is upgraded to
        // "tool_calls" so clients can dispatch.
        ToolCallParseResult tc = parse_qwen3_tool_calls(split.content);

        json message = {{"role", "assistant"}};
        if (tc.has_tool_calls) {
            // OpenAI spec: content may be null when only tool_calls are
            // emitted; otherwise include the leftover text.
            if (tc.content.empty()) {
                message["content"] = nullptr;
            } else {
                message["content"] = tc.content;
            }
            message["tool_calls"] = std::move(tc.tool_calls);
            result.finish_reason = "tool_calls";
        } else {
            message["content"] = split.content;
        }
        if (!split.reasoning.empty()) {
            message["reasoning_content"] = split.reasoning;
        }

        json response;
        response["id"] = request_id;
        response["object"] = "chat.completion";
        response["created"] = epoch;
        response["model"] = display_model_name_;
        response["choices"]
            = json::array({{{"index", 0}, {"message", std::move(message)}, {"finish_reason", result.finish_reason}}});
        response["usage"] = {{"prompt_tokens", result.stats.prompt_tokens},
                             {"completion_tokens", result.stats.generated_tokens},
                             {"total_tokens", result.stats.total_tokens}};
        if (compat_format == api_compat::ResponseFormat::OpenAIResponses) {
            response = api_compat::chat_response_to_responses(response);
        } else if (compat_format == api_compat::ResponseFormat::Anthropic) {
            response = api_compat::chat_response_to_anthropic(response);
        }
        res.set_content(response.dump(), "application/json");
    }
}

} // namespace zedinfer
