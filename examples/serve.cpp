#include "backend/device/device.hpp"
#include "utils/logging.hpp"
#include "utils/system_info.hpp"
#include "zedinfer.h"
#include "zedinfer/engine.hpp"
#include "zedinfer/http_server.hpp"
#include "zedinfer/scheduler.hpp"

#include <argparse/argparse.hpp>
#include <csignal>
#include <cstdio>
#include <iostream>
#include <memory>
#include <string>
#include <thread>

using namespace zedinfer;

// Global pointers for signal handler (C++ signal() requires C-linkage function pointer,
// cannot capture locals — standard pattern used by llama.cpp)
static HttpServer *g_server = nullptr;
static std::shared_ptr<InferenceEngine> g_engine = nullptr;

static void signal_handler(int) {
    if (g_server) g_server->stop();
    if (g_engine) g_engine->serving_loop().stop();
}

int main(int argc, char *argv[]) {
    utils::initLoggerWithOverwrite(plog::info, "logs/serve.log");
    LOG_VERBOSE_(utils::BOTH) << utils::get_runtime_info();

    argparse::ArgumentParser program("ZedInfer Server");

    program.add_argument("model_path")
        .help("Path to the model directory");

    program.add_argument("--host")
        .help("Host to bind to")
        .default_value(std::string("0.0.0.0"));

    program.add_argument("--port")
        .help("Port to listen on")
        .default_value(8080)
        .scan<'i', int>();

    program.add_argument("--nvidia")
        .help("Use NVIDIA GPU backend")
        .default_value(false)
        .implicit_value(true);

    program.add_argument("--max-batch-tokens")
        .help("Maximum tokens per batch")
        .default_value(2048)
        .scan<'i', int>();

    program.add_argument("--max-batch-requests")
        .help("Maximum concurrent requests")
        .default_value(64)
        .scan<'i', int>();

    program.add_argument("--gpu-memory-utilization")
        .help("Fraction of GPU memory for KV cache (0.0-1.0)")
        .default_value(0.9f)
        .scan<'g', float>();

    try {
        program.parse_args(argc, argv);
    } catch (const std::exception &err) {
        std::cerr << err.what() << std::endl;
        std::cerr << program;
        return 1;
    }

    auto model_path = program.get<std::string>("model_path");
    auto host = program.get<std::string>("--host");
    int port = program.get<int>("--port");
    bool use_nvidia = program.get<bool>("--nvidia");

    zedinferDeviceType_t device_type =
        use_nvidia ? ZEDINFER_DEVICE_NVIDIA : ZEDINFER_DEVICE_CPU;
    device::Device device(device_type, 0);

    // Build scheduler config from CLI args
    SchedulerConfig sched_config;
    sched_config.max_batch_tokens = program.get<int>("--max-batch-tokens");
    sched_config.max_batch_requests = program.get<int>("--max-batch-requests");
    sched_config.gpu_memory_utilization = program.get<float>("--gpu-memory-utilization");

    // Create engine
    auto engine = InferenceEngine::create(model_path, device, sched_config);
    g_engine = engine;

    // Start engine serving loop on a dedicated thread
    std::thread engine_thread([&engine] {
        engine->serving_loop().run_serving();
    });

    // Create HTTP server
    ServerConfig server_config;
    server_config.host = host;
    server_config.port = port;

    HttpServer server(server_config, engine);
    g_server = &server;

    // Register signal handlers
    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);

    printf("\n========================================\n");
    printf("  ZedInfer Server\n");
    printf("  Model: %s\n", engine->model_name().c_str());
    printf("  Listening: http://%s:%d\n", host.c_str(), port);
    printf("  Web UI: http://%s:%d/\n", host.c_str(), port);
    printf("  API: http://%s:%d/v1/chat/completions\n", host.c_str(), port);
    printf("========================================\n\n");

    server.start();  // blocks until stop()

    // Cleanup
    engine->serving_loop().stop();
    engine_thread.join();
    g_server = nullptr;
    g_engine = nullptr;

    printf("Server stopped.\n");
    return 0;
}
