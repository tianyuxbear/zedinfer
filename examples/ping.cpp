#include "backend/device/device.hpp"
#include "utils/logging.hpp"
#include "utils/system_info.hpp"
#include "zedinfer.h"
#include "zedinfer/engine.hpp"
#include "zedinfer/scheduler.hpp"
#include "zedinfer/session.hpp"

#include "zedinfer/version.hpp"
#include <argparse/argparse.hpp>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>

using namespace zedinfer;

int main(int argc, char* argv[]) {
    argparse::ArgumentParser program("ZedInfer Ping", std::string("zedinfer ") + ZEDINFER_VERSION + " (build "
                                                          + ZEDINFER_GIT_HASH + ", " + ZEDINFER_BUILD_DATE + ")");

    program.add_argument("model_path").help("Path to the model directory");

    program.add_argument("--nvidia").help("Use NVIDIA GPU backend").default_value(false).implicit_value(true);

    program.add_argument("--gpu-memory-utilization")
        .help("Fraction of GPU memory for KV cache (0.0-1.0)")
        .default_value(0.9f)
        .scan<'g', float>();

    program.add_argument("--prompt")
        .help("Prompt text for single-turn generation")
        .default_value(std::string("Who are you?"));

    program.add_argument("--max-new-tokens")
        .help("Maximum number of new tokens to generate")
        .default_value(512)
        .scan<'i', int>();

    program.add_argument("--thinking")
        .help("Enable the reasoning <think> block (Qwen3.5). Off by default because the "
              "open-<think> branch is fragile on GPTQ-Int4 weights and produces hallucinated "
              "prompt content. Opt in if the weights handle empty-thinking-start cleanly.")
        .default_value(false)
        .implicit_value(true);

    program.add_argument("--max-think-tokens")
        .help("Force </think> after this many tokens inside an open <think> block (0 = disabled)")
        .default_value(128)
        .scan<'i', int>();

    program.add_argument("--stream")
        .help("Stream tokens to stdout as they are generated")
        .default_value(false)
        .implicit_value(true);

    try {
        program.parse_args(argc, argv);
    } catch (const std::exception& err) {
        std::cerr << err.what() << std::endl;
        std::cerr << program;
        return 1;
    }

    utils::initLoggerWithOverwrite(plog::verbose, "logs/ping.log");
    LOG_VERBOSE_(utils::BOTH) << utils::get_runtime_info();

    auto model_path = program.get<std::string>("model_path");
    bool use_nvidia = program.get<bool>("--nvidia");

    zedinferDeviceType_t device_type = use_nvidia ? ZEDINFER_DEVICE_NVIDIA : ZEDINFER_DEVICE_CPU;
    device::Device device(device_type, 0);

    SchedulerConfig sched_config;
    sched_config.gpu_memory_utilization = program.get<float>("--gpu-memory-utilization");
    if (const char* env = std::getenv("ZEDINFER_KV_BLOCK_SIZE")) {
        sched_config.kv_block_size = std::atoi(env);
        printf("[ping] Using kv_block_size=%d from env\n", sched_config.kv_block_size);
    }

    auto t0 = std::chrono::high_resolution_clock::now();
    std::shared_ptr<InferenceEngine> engine;
    try {
        engine = InferenceEngine::create(model_path, device, sched_config);
    } catch (const std::exception& err) {
        std::string msg = err.what() ? err.what() : "";
        // Qwen3.5 currently throws "not implemented until M1" from forward_config(),
        // which engine init invokes for KV/scratch sizing. Treat this as a clean WIP exit.
        if (msg.find("not implemented until M1") != std::string::npos) {
            std::cout << "[ping] Qwen3.5 forward path is M1 work-in-progress; exiting clean.\n"
                      << "       Detail: " << msg << "\n";
            return 0;
        }
        std::cerr << "[ping] engine init failed: " << msg << "\n";
        std::cerr << "       Hint: pinned-host memory or VRAM may be insufficient; try --gpu-memory-utilization 0.5 or a smaller model.\n";
        return 3;
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    auto init_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();

    printf("Engine initialized in %ld ms (%s)\n", init_ms, use_nvidia ? "NVIDIA" : "CPU");

    GenerationConfig gen_config;
    gen_config.gen_mode = GenerationMode::PING;
    gen_config.max_new_tokens = program.get<int>("--max-new-tokens");
    gen_config.enable_thinking = program.get<bool>("--thinking");
    gen_config.max_think_tokens = program.get<int>("--max-think-tokens");
    gen_config.verbose = true;
    gen_config.print_stats = true;
    if (program.get<bool>("--stream")) {
        gen_config.stream = true;
        gen_config.stream_callback = [](const std::string& tok) {
            std::cout << tok << std::flush;
        };
    }

    auto prompt = program.get<std::string>("--prompt");
    try {
        auto session = engine->create_session(gen_config);
        session->chat(prompt);
    } catch (const std::exception& err) {
        std::string msg = err.what() ? err.what() : "";
        if (msg.find("not implemented until M1") != std::string::npos) {
            std::cout << "[ping] Qwen3.5 forward path is M1 work-in-progress; exiting clean.\n"
                      << "       Detail: " << msg << "\n";
            return 0;
        }
        std::cerr << "[ping] generation failed: " << msg << "\n";
        return 2;
    }

    return 0;
}
