#include "zedinfer/http_server.hpp"
#include "zedinfer/engine.hpp"
#include "zedinfer/request.hpp"
#include "zedinfer/session.hpp"

#include <nlohmann/json.hpp>
#include <plog/Log.h>

#include <chrono>
#include <condition_variable>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <queue>
#include <sstream>

using json = nlohmann::json;
namespace fs = std::filesystem;

namespace zedinfer {

// ============================================================================
// Thread-safe token queue for SSE streaming
// ============================================================================

namespace {

template <typename T>
class TokenQueue {
public:
    void push(T item) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            queue_.push(std::move(item));
        }
        cv_.notify_one();
    }

    bool try_pop(T &item, std::chrono::milliseconds timeout) {
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

struct StreamState {
    std::future<GenerationResult> future;
    std::shared_ptr<TokenQueue<std::string>> token_queue;
    std::string request_id;
    std::string model_name;
    int64_t epoch;
    bool sent_role = false;
    std::string utf8_buffer;  // accumulates partial UTF-8 sequences
    std::chrono::steady_clock::time_point last_activity;  // for timeout detection
    int timeout_sec = 300;
    // Session support: update session state after generation completes
    InferenceSession *session = nullptr;
    std::string user_message;
    std::string accumulated_output;  // accumulate full output for session state
};

// Return the length of the longest valid UTF-8 prefix in `s`.
// Scans forward character by character. Returns position after the last
// complete, valid UTF-8 character. Trailing incomplete sequences are excluded.
static size_t valid_utf8_length(const std::string &s) {
    size_t len = s.size();
    size_t i = 0;
    size_t last_good = 0;  // end of last complete valid character

    while (i < len) {
        uint8_t c = static_cast<uint8_t>(s[i]);
        int char_len = 0;

        if ((c & 0x80) == 0)      char_len = 1; // 0xxxxxxx  ASCII
        else if ((c & 0xE0) == 0xC0) char_len = 2; // 110xxxxx
        else if ((c & 0xF0) == 0xE0) char_len = 3; // 1110xxxx
        else if ((c & 0xF8) == 0xF0) char_len = 4; // 11110xxx
        else {
            // Orphan continuation byte or invalid — skip it
            i++;
            continue;
        }

        // Check if we have enough bytes for the full character
        if (i + char_len > len) {
            // Incomplete sequence at end — stop here
            break;
        }

        // Verify all continuation bytes are 10xxxxxx
        bool valid = true;
        for (int j = 1; j < char_len; j++) {
            if ((static_cast<uint8_t>(s[i + j]) & 0xC0) != 0x80) {
                valid = false;
                break;
            }
        }

        if (!valid) {
            // Invalid sequence — skip the leading byte
            i++;
            continue;
        }

        i += char_len;
        last_good = i;
    }

    return last_good;
}

} // anonymous namespace

// ============================================================================
// Session Management
// ============================================================================

InferenceSession *HttpServer::get_or_create_session(const std::string &session_id) {
    std::lock_guard<std::mutex> lock(sessions_mutex_);

    auto it = sessions_.find(session_id);
    if (it != sessions_.end()) {
        return it->second.get();
    }

    // Create new session
    GenerationConfig config;
    config.max_new_tokens = 1024;
    auto session = engine_->create_session(config);
    auto *ptr = session.get();

    LOGI << "[HttpServer] Created session " << session_id
         << " (total sessions: " << sessions_.size() + 1 << ")";

    sessions_[session_id] = std::move(session);
    return ptr;
}

// ============================================================================
// Helpers
// ============================================================================

std::string HttpServer::generate_request_id() {
    uint64_t id = request_counter_.fetch_add(1, std::memory_order_relaxed);
    return "chatcmpl-" + std::to_string(id);
}

void HttpServer::send_error(httplib::Response &res, int status,
                            const std::string &message, const std::string &type,
                            const std::string &code) {
    json err;
    err["error"]["message"] = message;
    err["error"]["type"] = type;
    if (!code.empty()) {
        err["error"]["code"] = code;
    }
    res.status = status;
    res.set_content(err.dump(), "application/json");
}

void HttpServer::load_web_ui() {
    std::vector<std::string> search_paths = {
        "web/index.html",
        "../web/index.html",
    };

    for (const auto &path : search_paths) {
        if (fs::exists(path)) {
            std::ifstream file(path);
            if (file.is_open()) {
                std::ostringstream ss;
                ss << file.rdbuf();
                web_ui_html_ = ss.str();
                LOGI << "[HttpServer] Loaded web UI from " << path;
                return;
            }
        }
    }
    web_ui_html_ = "<html><body><h1>ZedInfer</h1><p>Web UI not found. "
                    "Place web/index.html in the project directory.</p></body></html>";
    LOGW << "[HttpServer] Web UI not found, using fallback";
}

// ============================================================================
// Constructor and lifecycle
// ============================================================================

HttpServer::HttpServer(ServerConfig config, std::shared_ptr<InferenceEngine> engine)
    : config_(std::move(config)), engine_(std::move(engine)) {

    load_web_ui();

    server_.set_payload_max_length(10 * 1024 * 1024);  // 10MB max request body

    server_.Post("/v1/chat/completions",
        [this](const httplib::Request &req, httplib::Response &res) {
            handle_chat_completions(req, res);
        });

    server_.Get("/v1/models",
        [this](const httplib::Request &req, httplib::Response &res) {
            handle_models(req, res);
        });

    server_.Get("/health",
        [this](const httplib::Request &req, httplib::Response &res) {
            handle_health(req, res);
        });

    server_.Get("/",
        [this](const httplib::Request &, httplib::Response &res) {
            res.set_content(web_ui_html_, "text/html");
        });

    // Serve static files (images, etc.) from web/ directory
    server_.Get("/images/(.*)", [](const httplib::Request &req, httplib::Response &res) {
        std::string filename = req.matches[1];
        // Prevent directory traversal
        if (filename.find("..") != std::string::npos) {
            res.status = 403;
            return;
        }
        std::vector<std::string> search_paths = {
            "web/images/" + filename,
            "../web/images/" + filename,
        };
        for (const auto &path : search_paths) {
            if (fs::exists(path)) {
                std::ifstream file(path, std::ios::binary);
                if (file.is_open()) {
                    std::ostringstream ss;
                    ss << file.rdbuf();
                    // Determine content type
                    std::string content_type = "application/octet-stream";
                    if (filename.size() > 4 && filename.substr(filename.size() - 4) == ".svg")
                        content_type = "image/svg+xml";
                    else if (filename.size() > 4 && filename.substr(filename.size() - 4) == ".png")
                        content_type = "image/png";
                    res.set_content(ss.str(), content_type);
                    return;
                }
            }
        }
        res.status = 404;
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

void HttpServer::handle_models(const httplib::Request &, httplib::Response &res) {
    auto now = std::chrono::system_clock::now();
    auto epoch = std::chrono::duration_cast<std::chrono::seconds>(
        now.time_since_epoch()).count();

    json response;
    response["object"] = "list";
    response["data"] = json::array({
        {{"id", engine_->model_name()},
         {"object", "model"},
         {"created", epoch}}
    });
    res.set_content(response.dump(), "application/json");
}

// ============================================================================
// GET /health
// ============================================================================

void HttpServer::handle_health(const httplib::Request &, httplib::Response &res) {
    json response;
    response["status"] = "ok";
    response["model"] = engine_->model_name();
    response["active_requests"] = engine_->active_count();
    response["pending_requests"] = engine_->pending_count();

    auto *pool = engine_->block_pool();
    if (pool) {
        response["block_pool"]["total_blocks"] = pool->total_blocks();
        response["block_pool"]["free_blocks"] = pool->free_blocks();
        response["block_pool"]["utilization"] =
            1.0 - static_cast<double>(pool->free_blocks()) / pool->total_blocks();
    }
    res.set_content(response.dump(), "application/json");
}

// ============================================================================
// POST /v1/chat/completions
// ============================================================================

void HttpServer::handle_chat_completions(const httplib::Request &req,
                                          httplib::Response &res) {
    // 1. Parse JSON
    json body;
    try {
        body = json::parse(req.body);
    } catch (const json::parse_error &e) {
        send_error(res, 400, std::string("Invalid JSON: ") + e.what(),
                   "invalid_request_error", "invalid_json");
        return;
    }

    // 2. Validate messages
    if (!body.contains("messages") || !body["messages"].is_array() ||
        body["messages"].empty()) {
        send_error(res, 400, "Missing or empty 'messages' array",
                   "invalid_request_error", "missing_field");
        return;
    }

    // 3. Extract parameters
    bool stream = body.value("stream", false);
    int max_tokens = body.value("max_tokens", 512);
    std::string session_id = body.value("session_id", std::string(""));

    // 4. Extract last user message
    std::string last_user_message;
    for (auto it = body["messages"].rbegin(); it != body["messages"].rend(); ++it) {
        if ((*it).value("role", "") == "user") {
            last_user_message = (*it).value("content", "");
            break;
        }
    }

    // 5. Build prompt and tokenize — two paths: session vs stateless
    std::vector<int> input_ids;
    InferenceSession *session = nullptr;

    if (!session_id.empty()) {
        // SESSION MODE: only prefill the new user message, reuse KV cache
        session = get_or_create_session(session_id);
        std::string prompt = session->prepare_prompt(last_user_message);
        input_ids = engine_->tokenizer().encode(prompt);
    } else {
        // STATELESS MODE: format and prefill the entire conversation
        std::vector<std::pair<std::string, std::string>> messages;
        for (const auto &msg : body["messages"]) {
            messages.emplace_back(msg.value("role", ""), msg.value("content", ""));
        }
        std::string prompt = engine_->chat_template().apply(messages);
        input_ids = engine_->tokenizer().encode(prompt);
    }

    // 5b. Validate prompt length
    int max_seq_len = engine_->exec_config().max_seq_len;
    if (static_cast<int>(input_ids.size()) > max_seq_len) {
        send_error(res, 400,
                   "Prompt too long: " + std::to_string(input_ids.size()) +
                   " tokens exceeds max_seq_len " + std::to_string(max_seq_len),
                   "invalid_request_error", "prompt_too_long");
        return;
    }

    // 6. Branch: streaming vs non-streaming
    if (stream) {
        handle_chat_completions_stream(req, res, std::move(input_ids), max_tokens,
                                        session, last_user_message);
        return;
    }

    // --- Non-streaming path ---
    std::string request_id = generate_request_id();

    auto inference_req = std::make_unique<InferenceRequest>();
    inference_req->input_ids = std::move(input_ids);
    inference_req->config.max_new_tokens = max_tokens;
    inference_req->arrival_time = std::chrono::steady_clock::now();
    if (session) {
        inference_req->block_table_ref = &session->block_table();
    }

    // 7. Submit
    std::future<GenerationResult> future;
    try {
        future = engine_->submit_async(std::move(inference_req));
    } catch (const std::runtime_error &) {
        send_error(res, 503, "Server overloaded, queue full",
                   "server_error", "queue_full");
        return;
    }

    // 8. Wait for result with timeout
    if (future.wait_for(std::chrono::seconds(config_.request_timeout_sec)) ==
        std::future_status::timeout) {
        send_error(res, 408, "Request timed out",
                   "timeout_error", "request_timeout");
        return;
    }

    GenerationResult result;
    try {
        result = future.get();
    } catch (const std::exception &e) {
        send_error(res, 500, std::string("Generation failed: ") + e.what(),
                   "internal_error", "generation_failed");
        return;
    }

    // 9. Decode output and update session
    std::string output_text = engine_->tokenizer().decode(result.output_ids);
    if (session) {
        session->complete_turn(last_user_message, output_text);
    }

    // 10. Format response
    auto now = std::chrono::system_clock::now();
    auto epoch = std::chrono::duration_cast<std::chrono::seconds>(
        now.time_since_epoch()).count();

    json response;
    response["id"] = request_id;
    response["object"] = "chat.completion";
    response["created"] = epoch;
    response["model"] = engine_->model_name();
    response["choices"] = json::array({
        {{"index", 0},
         {"message", {{"role", "assistant"}, {"content", output_text}}},
         {"finish_reason", "stop"}}
    });
    response["usage"] = {
        {"prompt_tokens", result.stats.prompt_tokens},
        {"completion_tokens", result.stats.generated_tokens},
        {"total_tokens", result.stats.total_tokens}
    };

    res.set_content(response.dump(), "application/json");
}

// ============================================================================
// POST /v1/chat/completions (streaming SSE)
// ============================================================================

void HttpServer::handle_chat_completions_stream(
    const httplib::Request &, httplib::Response &res,
    std::vector<int> input_ids, int max_tokens,
    InferenceSession *session, std::string user_message) {

    std::string request_id = generate_request_id();
    std::string model_name = engine_->model_name();

    auto now = std::chrono::system_clock::now();
    auto epoch = std::chrono::duration_cast<std::chrono::seconds>(
        now.time_since_epoch()).count();

    auto token_queue = std::make_shared<TokenQueue<std::string>>();

    // Build request with stream callback
    auto inference_req = std::make_unique<InferenceRequest>();
    inference_req->input_ids = std::move(input_ids);
    inference_req->config.max_new_tokens = max_tokens;
    // IMPORTANT: config.stream MUST be true — the scheduler checks
    // (config.stream && stream_callback) at scheduler.cpp:193,221.
    // Safe because submit_async() does not call validate().
    inference_req->config.stream = true;
    inference_req->arrival_time = std::chrono::steady_clock::now();
    inference_req->stream_callback = [token_queue](const std::string &token) {
        token_queue->push(token);
    };
    if (session) {
        inference_req->block_table_ref = &session->block_table();
    }

    // Submit
    std::future<GenerationResult> future;
    try {
        future = engine_->submit_async(std::move(inference_req));
    } catch (const std::runtime_error &) {
        send_error(res, 503, "Server overloaded, queue full",
                   "server_error", "queue_full");
        return;
    }

    // SSE headers and chunked content provider
    res.set_header("Cache-Control", "no-cache");
    res.set_header("Connection", "keep-alive");

    auto state = std::make_shared<StreamState>();
    state->future = std::move(future);
    state->token_queue = token_queue;
    state->request_id = request_id;
    state->model_name = model_name;
    state->epoch = epoch;
    state->last_activity = std::chrono::steady_clock::now();
    state->timeout_sec = config_.request_timeout_sec;
    state->session = session;
    state->user_message = std::move(user_message);

    // Helper to build a base SSE chunk JSON
    auto make_chunk = [](const std::string &id, const std::string &model, int64_t created) {
        json chunk;
        chunk["id"] = id;
        chunk["object"] = "chat.completion.chunk";
        chunk["created"] = created;
        chunk["model"] = model;
        return chunk;
    };

    res.set_chunked_content_provider(
        "text/event-stream",
        [state, make_chunk](size_t /*offset*/, httplib::DataSink &sink) -> bool {
          try {
            // Send initial role chunk once
            if (!state->sent_role) {
                state->sent_role = true;
                json chunk = make_chunk(state->request_id, state->model_name, state->epoch);
                chunk["choices"] = json::array({
                    {{"index", 0},
                     {"delta", {{"role", "assistant"}}},
                     {"finish_reason", nullptr}}
                });
                std::string data = "data: " + chunk.dump() + "\n\n";
                sink.write(data.data(), data.size());
            }

            // Helper: flush valid UTF-8 from buffer as SSE chunk
            auto flush_utf8 = [&state, &make_chunk, &sink](bool force_all) {
                if (state->utf8_buffer.empty()) return;
                size_t valid = force_all ? state->utf8_buffer.size()
                                         : valid_utf8_length(state->utf8_buffer);
                if (valid > 0) {
                    std::string to_send = state->utf8_buffer.substr(0, valid);
                    state->utf8_buffer.erase(0, valid);
                    state->accumulated_output += to_send;
                    json chunk = make_chunk(state->request_id, state->model_name, state->epoch);
                    chunk["choices"] = json::array({
                        {{"index", 0},
                         {"delta", {{"content", to_send}}},
                         {"finish_reason", nullptr}}
                    });
                    std::string data = "data: " + chunk.dump() + "\n\n";
                    sink.write(data.data(), data.size());
                }
            };

            // Pop tokens from queue, buffer for UTF-8 safety
            std::string token;
            bool got_token = false;
            while (state->token_queue->try_pop(token, std::chrono::milliseconds(50))) {
                state->utf8_buffer += token;
                flush_utf8(false);
                got_token = true;
            }

            // Update activity timestamp if we got tokens
            if (got_token) {
                state->last_activity = std::chrono::steady_clock::now();
            }

            // Timeout: if no tokens and future not ready for too long, abort
            auto idle_sec = std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::steady_clock::now() - state->last_activity).count();
            if (idle_sec >= state->timeout_sec) {
                json err_chunk;
                err_chunk["error"] = {{"message", "Request timed out waiting for generation"},
                                       {"type", "timeout_error"}};
                std::string data = "data: " + err_chunk.dump() + "\n\n";
                sink.write(data.data(), data.size());
                std::string done = "data: [DONE]\n\n";
                sink.write(done.data(), done.size());
                sink.done();
                return false;
            }

            // Check if generation is complete
            if (state->future.wait_for(std::chrono::milliseconds(0)) ==
                std::future_status::ready) {

                // Drain remaining tokens
                while (state->token_queue->try_pop(token, std::chrono::milliseconds(0))) {
                    state->utf8_buffer += token;
                }
                // Force-flush everything (generation done, no more bytes coming)
                flush_utf8(true);

                // Final chunk with finish_reason and usage
                try {
                    auto result = state->future.get();

                    // Update session state after successful generation
                    if (state->session) {
                        state->session->complete_turn(
                            state->user_message, state->accumulated_output);
                    }

                    json final_chunk = make_chunk(state->request_id, state->model_name, state->epoch);
                    final_chunk["choices"] = json::array({
                        {{"index", 0},
                         {"delta", json::object()},
                         {"finish_reason", "stop"}}
                    });
                    final_chunk["usage"] = {
                        {"prompt_tokens", result.stats.prompt_tokens},
                        {"completion_tokens", result.stats.generated_tokens},
                        {"total_tokens", result.stats.total_tokens}
                    };
                    std::string data = "data: " + final_chunk.dump() + "\n\n";
                    sink.write(data.data(), data.size());
                } catch (const std::exception &e) {
                    // Generation failed — send error as SSE event
                    json err_chunk;
                    err_chunk["error"] = {{"message", e.what()}, {"type", "internal_error"}};
                    std::string data = "data: " + err_chunk.dump() + "\n\n";
                    sink.write(data.data(), data.size());
                }

                // [DONE]
                std::string done = "data: [DONE]\n\n";
                sink.write(done.data(), done.size());

                sink.done();
                return false;
            }

            return true;  // continue streaming

          } catch (const std::exception &e) {
            // Safety net: if any exception slips through (e.g. JSON UTF-8 error),
            // send error and terminate stream gracefully instead of crashing.
            try {
                std::string err_data = "data: {\"error\":{\"message\":\"" +
                    std::string(e.what()) + "\"}}\n\n";
                sink.write(err_data.data(), err_data.size());
                std::string done = "data: [DONE]\n\n";
                sink.write(done.data(), done.size());
            } catch (...) {}
            sink.done();
            return false;
          }
        });
}

} // namespace zedinfer
