#pragma once

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#include <httplib.h>
#pragma GCC diagnostic pop

#include "zedinfer/session.hpp"

#include <atomic>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace zedinfer {

class InferenceEngine;

struct ServerConfig {
    std::string host = "0.0.0.0";
    int port = 8080;
    int request_timeout_sec = 300;
    int max_sessions = 100;
    int session_idle_timeout = 1800;
    // Override the model id surfaced in /v1/models, /health, and chat
    // completion responses. Empty -> use engine.model_name() (the raw path).
    // Useful for impersonating an OpenAI model id from existing clients.
    std::string served_model_name;
    // Bearer token expected in Authorization header on protected endpoints.
    // Empty -> auth disabled (backward compatible). When set, /v1/* and
    // /tokenize / /detokenize require "Authorization: Bearer <api_key>".
    // /health and static files are always public.
    std::string api_key;
};

class HttpServer {
public:
    HttpServer(ServerConfig config, std::shared_ptr<InferenceEngine> engine);

    void start();
    void stop();

private:
    ServerConfig config_;
    std::shared_ptr<InferenceEngine> engine_;
    httplib::Server server_;
    std::atomic<uint64_t> request_counter_{0};
    std::string web_root_;
    // Model id used in API responses; resolved once at construction from
    // ServerConfig.served_model_name override, falling back to engine.model_name().
    std::string display_model_name_;

    // Static file cache
    std::unordered_map<std::string, std::pair<std::string, std::string>> file_cache_;

    // Session management
    struct SessionEntry {
        std::unique_ptr<InferenceSession> session;
        std::chrono::steady_clock::time_point last_access;
        std::atomic<bool> busy{false};
        // Set by delete_session() when a DELETE arrives while the session is
        // busy (e.g. mid-stream). The entry is kept alive until unlock_session()
        // observes the flag and reclaims it, so an in-flight streaming response
        // that still holds a raw InferenceSession* is never freed underneath it.
        bool pending_delete = false;
    };
    std::unordered_map<std::string, std::unique_ptr<SessionEntry>> sessions_;
    std::mutex sessions_mutex_;

    // RAII session lock — movable, auto-unlocks on destruction
    class SessionLock {
    public:
        SessionLock() = default;
        SessionLock(HttpServer* server, std::string session_id) : server_(server), session_id_(std::move(session_id)) {}
        ~SessionLock() { unlock(); }
        SessionLock(SessionLock&& o) noexcept : server_(o.server_), session_id_(std::move(o.session_id_)) {
            o.server_ = nullptr;
        }
        SessionLock& operator=(SessionLock&& o) noexcept {
            unlock();
            server_ = o.server_;
            session_id_ = std::move(o.session_id_);
            o.server_ = nullptr;
            return *this;
        }
        SessionLock(const SessionLock&) = delete;
        SessionLock& operator=(const SessionLock&) = delete;
        void unlock() {
            if (server_ && !session_id_.empty()) {
                server_->unlock_session(session_id_);
                server_ = nullptr;
            }
        }
        const std::string& id() const { return session_id_; }

    private:
        HttpServer* server_ = nullptr;
        std::string session_id_;
    };

    // Atomically get/create and mark a session busy. Returns nullptr when an
    // existing session is already serving another request.
    InferenceSession* acquire_session(const std::string& session_id);
    void unlock_session(const std::string& session_id);
    void delete_session(const std::string& session_id);
    void cleanup_idle_sessions();

    // Route handlers
    void handle_chat_completions(const httplib::Request& req, httplib::Response& res);
    void handle_models(const httplib::Request& req, httplib::Response& res);
    void handle_health(const httplib::Request& req, httplib::Response& res);
    void handle_delete_session(const httplib::Request& req, httplib::Response& res);
    void handle_tokenize(const httplib::Request& req, httplib::Response& res);
    void handle_detokenize(const httplib::Request& req, httplib::Response& res);

    // Helpers
    std::string generate_request_id();
    void send_error(httplib::Response& res, int status, const std::string& message, const std::string& type,
                    const std::string& code = "");
    std::string resolve_web_root();
    void cache_static_files();
    std::string guess_content_type(const std::string& filename);
};

} // namespace zedinfer
