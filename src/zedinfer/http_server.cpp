#include "zedinfer/http_server.hpp"
#include "zedinfer/engine.hpp"
#include "zedinfer/request.hpp"
#include "utils/logging.hpp"

#include <nlohmann/json.hpp>
#include <plog/Log.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
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
        if (!cv_.wait_for(lock, timeout, [this] { return !queue_.empty(); }))
            return false;
        item = std::move(queue_.front());
        queue_.pop();
        return true;
    }

private:
    std::queue<T> queue_;
    std::mutex mutex_;
    std::condition_variable cv_;
};

static size_t valid_utf8_length(const std::string &s) {
    size_t i = 0, last_good = 0;
    while (i < s.size()) {
        uint8_t c = static_cast<uint8_t>(s[i]);
        int n = 0;
        if ((c & 0x80) == 0)         n = 1;
        else if ((c & 0xE0) == 0xC0) n = 2;
        else if ((c & 0xF0) == 0xE0) n = 3;
        else if ((c & 0xF8) == 0xF0) n = 4;
        else { i++; continue; }
        if (i + n > s.size()) break;
        bool ok = true;
        for (int j = 1; j < n; j++)
            if ((static_cast<uint8_t>(s[i + j]) & 0xC0) != 0x80) { ok = false; break; }
        if (!ok) { i++; continue; }
        i += n;
        last_good = i;
    }
    return last_good;
}

} // anonymous namespace

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
    if (!code.empty()) err["error"]["code"] = code;
    res.status = status;
    res.set_content(err.dump(), "application/json");
}

std::string HttpServer::resolve_web_root() {
    for (const auto &c : {"web", "../web"}) {
        if (fs::exists(c) && fs::is_directory(c)) {
            LOGI << "[HttpServer] Web root: " << fs::absolute(c).string();
            return c;
        }
    }
    LOGW << "[HttpServer] Web root not found";
    return "";
}

std::string HttpServer::guess_content_type(const std::string &filename) {
    auto pos = filename.rfind('.');
    if (pos == std::string::npos) return "application/octet-stream";
    std::string ext = filename.substr(pos);
    if (ext == ".html") return "text/html";
    if (ext == ".svg")  return "image/svg+xml";
    if (ext == ".png")  return "image/png";
    if (ext == ".jpg" || ext == ".jpeg") return "image/jpeg";
    if (ext == ".css")  return "text/css";
    if (ext == ".js")   return "application/javascript";
    if (ext == ".json") return "application/json";
    if (ext == ".ico")  return "image/x-icon";
    return "application/octet-stream";
}

void HttpServer::cache_static_files() {
    if (web_root_.empty()) return;

    for (auto &entry : fs::recursive_directory_iterator(web_root_)) {
        if (!entry.is_regular_file()) continue;
        std::string rel = "/" + fs::relative(entry.path(), web_root_).string();
        std::ifstream file(entry.path(), std::ios::binary);
        if (!file.is_open()) continue;
        std::ostringstream ss;
        ss << file.rdbuf();
        file_cache_[rel] = {ss.str(), guess_content_type(entry.path().filename().string())};
    }
    LOGI << "[HttpServer] Cached " << file_cache_.size() << " static files";
}

// ============================================================================
// Session Management
// ============================================================================

InferenceSession *HttpServer::get_or_create_session(const std::string &session_id) {
    std::lock_guard<std::mutex> lock(sessions_mutex_);

    auto it = sessions_.find(session_id);
    if (it != sessions_.end()) {
        it->second->last_access = std::chrono::steady_clock::now();
        return it->second->session.get();
    }

    cleanup_idle_sessions();

    GenerationConfig config;
    config.max_new_tokens = 1024;
    auto session = engine_->create_session(config);
    auto *ptr = session.get();

    auto entry = std::make_unique<SessionEntry>();
    entry->session = std::move(session);
    entry->last_access = std::chrono::steady_clock::now();

    LOGI << "[HttpServer] Created session " << session_id
         << " (total: " << sessions_.size() + 1 << ")";

    sessions_[session_id] = std::move(entry);
    return ptr;
}

bool HttpServer::try_lock_session(const std::string &session_id) {
    std::lock_guard<std::mutex> lock(sessions_mutex_);
    auto it = sessions_.find(session_id);
    if (it == sessions_.end()) return true; // will be created fresh
    bool expected = false;
    return it->second->busy.compare_exchange_strong(expected, true);
}

void HttpServer::unlock_session(const std::string &session_id) {
    std::lock_guard<std::mutex> lock(sessions_mutex_);
    auto it = sessions_.find(session_id);
    if (it != sessions_.end()) it->second->busy = false;
}

void HttpServer::delete_session(const std::string &session_id) {
    std::lock_guard<std::mutex> lock(sessions_mutex_);
    auto it = sessions_.find(session_id);
    if (it != sessions_.end()) {
        LOGI << "[HttpServer] Deleted session " << session_id;
        sessions_.erase(it);
    }
}

void HttpServer::cleanup_idle_sessions() {
    auto now = std::chrono::steady_clock::now();
    auto timeout = std::chrono::seconds(config_.session_idle_timeout);

    for (auto it = sessions_.begin(); it != sessions_.end(); ) {
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
            if (!it->second->busy &&
                (oldest == sessions_.end() || it->second->last_access < oldest->second->last_access))
                oldest = it;
        }
        if (oldest == sessions_.end()) break;
        LOGI << "[HttpServer] Evicting session " << oldest->first << " (at capacity)";
        sessions_.erase(oldest);
    }
}

// ============================================================================
// Constructor and lifecycle
// ============================================================================

HttpServer::HttpServer(ServerConfig config, std::shared_ptr<InferenceEngine> engine)
    : config_(std::move(config)), engine_(std::move(engine)) {

    web_root_ = resolve_web_root();
    cache_static_files();
    server_.set_payload_max_length(10 * 1024 * 1024);

    // HTTP access logger
    server_.set_logger([](const httplib::Request &req, const httplib::Response &res) {
        // Only log API calls, not static files
        if (req.path.find("/v1/") == 0 || req.path == "/health") {
            LOGI << "[HTTP] " << req.method << " " << req.path
                 << " → " << res.status;
        }
    });

    // API routes
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
    server_.Delete("/v1/sessions/(.*)",
        [this](const httplib::Request &req, httplib::Response &res) {
            handle_delete_session(req, res);
        });

    // Static files from cache
    server_.Get("/(.*)", [this](const httplib::Request &req, httplib::Response &res) {
        std::string path = req.matches[1].str();
        if (path.empty() || path == "/") path = "/index.html";
        else if (path[0] != '/') path = "/" + path;

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

void HttpServer::handle_models(const httplib::Request &, httplib::Response &res) {
    auto epoch = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();

    json response;
    response["object"] = "list";
    response["data"] = json::array({{
        {"id", engine_->model_name()},
        {"object", "model"},
        {"created", epoch}
    }});
    res.set_content(response.dump(), "application/json");
}

// ============================================================================
// GET /health
// ============================================================================

void HttpServer::handle_health(const httplib::Request &, httplib::Response &res) {
    json response;
    response["status"] = "ok";
    response["model"] = engine_->model_name();
    response["active_requests"] = engine_->serving_loop().active_count();
    response["pending_requests"] = engine_->serving_loop().pending_count();

    auto *pool = engine_->block_pool();
    if (pool) {
        response["block_pool"] = {
            {"total_blocks", pool->total_blocks()},
            {"free_blocks", pool->free_blocks()},
            {"utilization", 1.0 - static_cast<double>(pool->free_blocks()) / pool->total_blocks()}
        };
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

void HttpServer::handle_delete_session(const httplib::Request &req, httplib::Response &res) {
    std::string session_id = req.matches[1];
    delete_session(session_id);
    res.set_content("{\"deleted\":true}", "application/json");
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

    if (!body.contains("messages") || !body["messages"].is_array() ||
        body["messages"].empty()) {
        send_error(res, 400, "Missing or empty 'messages' array",
                   "invalid_request_error", "missing_field");
        return;
    }

    // 2. Extract parameters
    bool stream = body.value("stream", false);
    int max_tokens = body.value("max_tokens", 512);
    std::string session_id = body.value("session_id", std::string(""));

    // 3. Extract last user message
    std::string last_user_message;
    for (auto it = body["messages"].rbegin(); it != body["messages"].rend(); ++it) {
        if ((*it).value("role", "") == "user") {
            last_user_message = (*it).value("content", "");
            break;
        }
    }

    // 4. Session concurrency check + lock
    SessionLock session_lock;
    if (!session_id.empty()) {
        if (!try_lock_session(session_id)) {
            send_error(res, 409, "Session is busy with another request",
                       "conflict_error", "session_busy");
            return;
        }
        session_lock = SessionLock(this, session_id);
    }

    // 5. Build prompt — session vs stateless
    std::vector<int> input_ids;
    InferenceSession *session = nullptr;
    bool use_session = false;

    if (!session_id.empty()) {
        session = get_or_create_session(session_id);

        // Check if session KV cache is still valid (not expired/recreated)
        if (session->is_valid() && session->past_len() > 0) {
            // Session alive with history — only prefill new message
            input_ids = engine_->tokenizer().encode(session->prepare_prompt(last_user_message));
            use_session = true;
        } else {
            // Session is fresh (new or recreated after expiry) — full prefill
            std::vector<std::pair<std::string, std::string>> messages;
            for (const auto &msg : body["messages"])
                messages.emplace_back(msg.value("role", ""), msg.value("content", ""));
            input_ids = engine_->tokenizer().encode(engine_->chat_template().apply(messages));
            use_session = true;
            LOGI << "[HttpServer] Session " << session_id << " fresh start, full prefill";
        }
    } else {
        // Stateless mode
        std::vector<std::pair<std::string, std::string>> messages;
        for (const auto &msg : body["messages"])
            messages.emplace_back(msg.value("role", ""), msg.value("content", ""));
        input_ids = engine_->tokenizer().encode(engine_->chat_template().apply(messages));
    }

    // 6. Validate length
    int max_seq_len = engine_->exec_config().max_seq_len;
    if (static_cast<int>(input_ids.size()) > max_seq_len) {
        send_error(res, 400,
                   "Prompt too long: " + std::to_string(input_ids.size()) +
                   " tokens (max " + std::to_string(max_seq_len) + ")",
                   "invalid_request_error", "prompt_too_long");
        return;
    }

    // 7. Log request with content
    std::string request_id = generate_request_id();
    std::string msg_preview = last_user_message.substr(0, 100);
    if (last_user_message.size() > 100) msg_preview += "...";
    LOG_INFO_(utils::BOTH) << "[Request] " << request_id
         << " | session=" << (session_id.empty() ? "none" : session_id)
         << " | tokens=" << input_ids.size()
         << " | max=" << max_tokens
         << " | stream=" << (stream ? "Y" : "N")
         << " | user: " << msg_preview;

    // 8. Build inference request
    auto cancel_flag = std::make_shared<std::atomic<bool>>(false);

    auto inference_req = std::make_unique<InferenceRequest>();
    inference_req->input_ids = std::move(input_ids);
    inference_req->config.max_new_tokens = max_tokens;
    inference_req->arrival_time = std::chrono::steady_clock::now();
    inference_req->cancelled = cancel_flag;
    if (use_session && session) inference_req->borrow_block_table(session->block_table());

    std::shared_ptr<TokenQueue<std::string>> token_queue;
    if (stream) {
        token_queue = std::make_shared<TokenQueue<std::string>>();
        inference_req->config.stream = true;
        inference_req->stream_callback = [token_queue, cancel_flag](const std::string &tok) {
            if (!cancel_flag->load()) token_queue->push(tok);
        };
    }

    // 9. Submit
    std::future<GenerationResult> future;
    try {
        future = engine_->serving_loop().submit_async(std::move(inference_req));
    } catch (const std::runtime_error &) {
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
            int timeout_sec;
            InferenceSession *session;
            SessionLock lock;
            std::chrono::steady_clock::time_point t_start;
            bool sent_role = false;
            bool sent_prefix = false;
            std::string utf8_buf, full_output;
            std::chrono::steady_clock::time_point last_activity;
        };

        auto ctx = std::make_shared<Ctx>();
        ctx->future = std::move(future);
        ctx->queue = std::move(token_queue);
        ctx->cancel_flag = cancel_flag;
        ctx->req_id = request_id;
        ctx->model = engine_->model_name();
        ctx->user_msg = last_user_message;
        ctx->output_prefix = session ? session->output_prefix() : "";
        ctx->epoch = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        ctx->timeout_sec = config_.request_timeout_sec;
        ctx->session = session;
        ctx->lock = std::move(session_lock); // transfer lock ownership to ctx
        ctx->t_start = t_start;
        ctx->last_activity = std::chrono::steady_clock::now();

        res.set_header("Cache-Control", "no-cache");
        res.set_header("Connection", "keep-alive");

        res.set_chunked_content_provider(
            "text/event-stream",
            [ctx](size_t, httplib::DataSink &sink) -> bool {
              try {
                auto sse = [&sink](const json &j) {
                    std::string d = "data: " + j.dump() + "\n\n";
                    sink.write(d.data(), d.size());
                };
                auto base = [&ctx]() {
                    return json{{"id",ctx->req_id},{"object","chat.completion.chunk"},
                                {"created",ctx->epoch},{"model",ctx->model}};
                };
                auto send_content = [&ctx, &sse, &base](const std::string &text) {
                    ctx->full_output += text;
                    json c = base();
                    c["choices"] = json::array({{{"index",0},
                        {"delta",{{"content",text}}},{"finish_reason",nullptr}}});
                    sse(c);
                };
                auto flush = [&ctx, &send_content](bool force) {
                    if (ctx->utf8_buf.empty()) return;
                    size_t n = force ? ctx->utf8_buf.size() : valid_utf8_length(ctx->utf8_buf);
                    if (n > 0) {
                        std::string t = ctx->utf8_buf.substr(0, n);
                        ctx->utf8_buf.erase(0, n);
                        send_content(t);
                    }
                };

                // Send role delta
                if (!ctx->sent_role) {
                    ctx->sent_role = true;
                    json c = base();
                    c["choices"] = json::array({{{"index",0},
                        {"delta",{{"role","assistant"}}},{"finish_reason",nullptr}}});
                    sse(c);
                }

                // Send output_prefix (e.g. "<think> " for DeepSeek-R1) as first content
                // Note: sent to client for display but NOT included in full_output
                // (consistent with CLI session.cpp which stores raw model output)
                if (!ctx->sent_prefix && !ctx->output_prefix.empty()) {
                    ctx->sent_prefix = true;
                    json c = base();
                    c["choices"] = json::array({{{"index",0},
                        {"delta",{{"content",ctx->output_prefix}}},{"finish_reason",nullptr}}});
                    sse(c);
                }

                std::string tok;
                bool got = false;
                while (ctx->queue->try_pop(tok, std::chrono::milliseconds(50))) {
                    ctx->utf8_buf += tok;
                    flush(false);
                    got = true;
                }
                if (got) ctx->last_activity = std::chrono::steady_clock::now();

                // Timeout
                auto idle = std::chrono::duration_cast<std::chrono::seconds>(
                    std::chrono::steady_clock::now() - ctx->last_activity).count();
                if (idle >= ctx->timeout_sec) {
                    if (ctx->session) ctx->session->abort_turn();
                    sse({{"error",{{"message","Request timed out"},{"type","timeout_error"}}}});
                    sink.write("data: [DONE]\n\n", 15);
                    sink.done();
                    ctx->lock.unlock();
                    return false;
                }

                // Completion
                if (ctx->future.wait_for(std::chrono::milliseconds(0)) == std::future_status::ready) {
                    while (ctx->queue->try_pop(tok, std::chrono::milliseconds(0)))
                        ctx->utf8_buf += tok;
                    flush(true);

                    try {
                        auto result = ctx->future.get();
                        if (ctx->session)
                            ctx->session->complete_turn(ctx->user_msg, ctx->full_output);

                        auto elapsed = std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - ctx->t_start).count();
                        int gen_tokens = static_cast<int>(result.output_ids.size());
                        std::string out_preview = ctx->full_output.substr(0, 100);
                        if (ctx->full_output.size() > 100) out_preview += "...";
                        LOG_INFO_(utils::BOTH) << "[Response] " << ctx->req_id
                             << " | tokens=" << gen_tokens
                             << " | " << static_cast<int>(elapsed) << "ms"
                             << " | " << std::fixed << std::setprecision(1)
                             << (elapsed > 0 ? gen_tokens * 1000.0 / elapsed : 0) << " tok/s"
                             << " | assistant: " << out_preview;

                        json c = base();
                        c["choices"] = json::array({{{"index",0},
                            {"delta",json::object()},{"finish_reason","stop"}}});
                        c["usage"] = {{"prompt_tokens",result.stats.prompt_tokens},
                            {"completion_tokens",result.stats.generated_tokens},
                            {"total_tokens",result.stats.total_tokens}};
                        sse(c);
                    } catch (const std::exception &e) {
                        LOGE << "[Response] " << ctx->req_id << " error: " << e.what();
                        if (ctx->session) ctx->session->abort_turn();
                        sse({{"error",{{"message",e.what()},{"type","internal_error"}}}});
                    }
                    sink.write("data: [DONE]\n\n", 15);
                    sink.done();
                    ctx->lock.unlock();
                    return false;
                }
                return true;

              } catch (const std::exception &e) {
                ctx->cancel_flag->store(true);
                if (ctx->session) ctx->session->abort_turn();
                LOGW << "[Response] " << ctx->req_id << " stream aborted: " << e.what();
                try { sink.write("data: [DONE]\n\n", 15); } catch (...) {}
                sink.done();
                ctx->lock.unlock();
                return false;
              }
            },
            // on_close: cancel generation when client disconnects
            [ctx](bool success) {
                if (!success) {
                    ctx->cancel_flag->store(true);
                    if (ctx->session) ctx->session->abort_turn();
                    LOGI << "[Request] " << ctx->req_id << " client disconnected";
                }
                ctx->lock.unlock();
            });

    } else {
        // --- Blocking response ---
        if (future.wait_for(std::chrono::seconds(config_.request_timeout_sec)) ==
            std::future_status::timeout) {
            send_error(res, 408, "Request timed out", "timeout_error", "request_timeout");
            LOGW << "[Response] " << request_id << " timed out";
            return;
        }

        GenerationResult result;
        try {
            result = future.get();
        } catch (const std::exception &e) {
            if (session) session->abort_turn();
            send_error(res, 500, std::string("Generation failed: ") + e.what(),
                       "internal_error", "generation_failed");
            LOGE << "[Response] " << request_id << " error: " << e.what();
            return;
        }

        std::string output_text = engine_->tokenizer().decode(result.output_ids);
        if (session) session->complete_turn(last_user_message, output_text);
        // Prepend output_prefix for display (e.g. DeepSeek-R1 "<think> ")
        std::string output_prefix = session ? session->output_prefix() : "";
        if (!output_prefix.empty()) output_text = output_prefix + output_text;

        // Log completion with output preview
        auto elapsed = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - t_start).count();
        std::string out_preview = output_text.substr(0, 100);
        if (output_text.size() > 100) out_preview += "...";
        LOG_INFO_(utils::BOTH) << "[Response] " << request_id
             << " | tokens=" << result.stats.generated_tokens
             << " | " << static_cast<int>(elapsed) << "ms"
             << " | " << std::fixed << std::setprecision(1)
             << (elapsed > 0 ? result.stats.generated_tokens * 1000.0 / elapsed : 0) << " tok/s"
             << " | assistant: " << out_preview;

        auto epoch = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();

        json response;
        response["id"] = request_id;
        response["object"] = "chat.completion";
        response["created"] = epoch;
        response["model"] = engine_->model_name();
        response["choices"] = json::array({{
            {"index", 0},
            {"message", {{"role", "assistant"}, {"content", output_text}}},
            {"finish_reason", "stop"}
        }});
        response["usage"] = {
            {"prompt_tokens", result.stats.prompt_tokens},
            {"completion_tokens", result.stats.generated_tokens},
            {"total_tokens", result.stats.total_tokens}
        };
        res.set_content(response.dump(), "application/json");
    }
}

} // namespace zedinfer
