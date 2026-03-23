#pragma once

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#include "httplib.h"
#pragma GCC diagnostic pop

#include <atomic>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace zedinfer {

class InferenceEngine;
}

#include "zedinfer/session.hpp"

namespace zedinfer {

struct ServerConfig {
    std::string host = "0.0.0.0";
    int port = 8080;
    int request_timeout_sec = 300;
};

class HttpServer {
public:
    HttpServer(ServerConfig config, std::shared_ptr<InferenceEngine> engine);

    void start();  // blocking
    void stop();   // non-blocking, safe from signal handler

private:
    ServerConfig config_;
    std::shared_ptr<InferenceEngine> engine_;
    httplib::Server server_;
    std::atomic<uint64_t> request_counter_{0};
    std::string web_ui_html_;

    // Session management
    std::unordered_map<std::string, std::unique_ptr<InferenceSession>> sessions_;
    std::mutex sessions_mutex_;

    InferenceSession *get_or_create_session(const std::string &session_id);

    // Route handlers
    void handle_chat_completions(const httplib::Request &req, httplib::Response &res);
    void handle_chat_completions_stream(const httplib::Request &req, httplib::Response &res,
                                         std::vector<int> input_ids, int max_tokens,
                                         InferenceSession *session = nullptr,
                                         std::string user_message = "");
    void handle_models(const httplib::Request &req, httplib::Response &res);
    void handle_health(const httplib::Request &req, httplib::Response &res);

    // Helpers
    std::string generate_request_id();
    void send_error(httplib::Response &res, int status,
                    const std::string &message, const std::string &type,
                    const std::string &code = "");
    void load_web_ui();
};

} // namespace zedinfer
