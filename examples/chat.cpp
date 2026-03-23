#include "backend/device/device.hpp"
#include "utils/logging.hpp"
#include "utils/system_info.hpp"
#include "zedinfer.h"
#include "zedinfer/engine.hpp"
#include "zedinfer/scheduler.hpp"
#include "zedinfer/session.hpp"

#include <argparse/argparse.hpp>
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <readline/readline.h>
#include <string>

using namespace zedinfer;

static void print_welcome() {
    printf("\n");
    printf("========================================\n");
    printf("  ZedInfer Chat\n");
    printf("========================================\n");
    printf("\n");
    printf("Commands:\n");
    printf("  exit, quit, q      Exit\n");
    printf("  reset, clear       Clear conversation\n");
    printf("  help               Show this message\n");
    printf("\n");
}

int main(int argc, char *argv[]) {
    utils::initLoggerWithOverwrite(plog::verbose, "logs/chat.log");
    LOG_VERBOSE_(utils::BOTH) << utils::get_runtime_info();

    argparse::ArgumentParser program("ZedInfer Chat");

    program.add_argument("model_path")
        .help("Path to the model directory");

    program.add_argument("--nvidia")
        .help("Use NVIDIA GPU backend")
        .default_value(false)
        .implicit_value(true);

    program.add_argument("--gpu-memory-utilization")
        .help("Fraction of GPU memory for KV cache (0.0-1.0)")
        .default_value(0.9f)
        .scan<'g', float>();

    program.add_argument("--max-tokens")
        .help("Maximum tokens per response")
        .default_value(16384)
        .scan<'i', int>();

    try {
        program.parse_args(argc, argv);
    } catch (const std::exception &err) {
        std::cerr << err.what() << std::endl;
        std::cerr << program;
        return 1;
    }

    auto model_path = program.get<std::string>("model_path");
    bool use_nvidia = program.get<bool>("--nvidia");
    int max_tokens = program.get<int>("--max-tokens");

    zedinferDeviceType_t device_type =
        use_nvidia ? ZEDINFER_DEVICE_NVIDIA : ZEDINFER_DEVICE_CPU;
    device::Device device(device_type, 0);

    SchedulerConfig sched_config;
    sched_config.gpu_memory_utilization = program.get<float>("--gpu-memory-utilization");

    auto engine = InferenceEngine::create(model_path, device, sched_config);

    GenerationConfig gen_config;
    gen_config.gen_mode = GenerationMode::CHAT;
    gen_config.max_new_tokens = max_tokens;
    gen_config.verbose = true;
    gen_config.print_stats = true;
    gen_config.stream = true;
    gen_config.stream_callback = [](const std::string &token_text) {
        std::cout << token_text << std::flush;
    };

    auto session = engine->create_session(gen_config);

    print_welcome();
    std::setlocale(LC_ALL, "en_US.UTF-8");

    while (true) {
        char *input = readline("\n\033[1;32mUser:\033[0m ");
        if (!input) break; // EOF (Ctrl+D)

        std::string user_input(input);
        free(input);

        // Trim
        user_input.erase(0, user_input.find_first_not_of(" \t\n\r"));
        user_input.erase(user_input.find_last_not_of(" \t\n\r") + 1);
        if (user_input.empty()) continue;

        // Commands
        std::string cmd = user_input;
        std::transform(cmd.begin(), cmd.end(), cmd.begin(), ::tolower);

        if (cmd == "exit" || cmd == "quit" || cmd == "q") break;
        if (cmd == "reset" || cmd == "clear" || cmd == "cls") {
            session->reset();
            printf("Conversation cleared.\n");
            continue;
        }
        if (cmd == "help") { print_welcome(); continue; }

        // Generate
        try {
            std::cout << "\033[1;34mAssistant:\033[0m ";
            session->chat(user_input);
            std::cout << "\n";
        } catch (const std::exception &e) {
            std::cerr << "Error: " << e.what() << std::endl;
        }
    }

    printf("\nGoodbye!\n");
    return 0;
}
