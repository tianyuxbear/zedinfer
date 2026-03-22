#include "backend/device/device.hpp"
#include "utils/logging.hpp"
#include "utils/system_info.hpp"
#include "zedinfer.h"
#include "zedinfer/engine.hpp"
#include "zedinfer/session.hpp"

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <readline/readline.h>
#include <string>

namespace fs = std::filesystem;
using namespace zedinfer;

std::string get_usage_message(const char *program_name) {
    std::ostringstream oss;

    oss << "\n";
    oss << "📖 Usage:\n";
    oss << "   " << program_name << " <model_path>\n";
    oss << "\n";
    oss << "📝 Arguments:\n";
    oss << "   model_path     Path to the model directory\n";
    oss << "                  • Supports absolute paths\n";
    oss << "                  • Supports relative paths\n";
    oss << "\n";
    oss << "💡 Examples:\n";
    oss << "   Relative path:\n";
    oss << "     $ " << program_name << " ./models/llama-7b\n";
    oss << "\n";
    oss << "   Absolute path:\n";
    oss << "     $ " << program_name << " /home/user/models/llama-7b\n";
    oss << "\n";

    return oss.str();
}

std::string get_welcome_message() {
    std::ostringstream oss;

    oss << "\n";
    oss << "╔══════════════════════════════════════════════════════════════╗\n";
    oss << "║                                                              ║\n";
    oss << "║             🚀 Welcome to ZedInfer Inference Engine          ║\n";
    oss << "║                                                              ║\n";
    oss << "╚══════════════════════════════════════════════════════════════╝\n";
    oss << "\n";
    oss << "💬 Ready for conversation! Type your message and press Enter.\n";
    oss << "\n";
    oss << "📝 Commands:\n";
    oss << "   exit, quit, q        Exit the program\n";
    oss << "   reset, clear, cls    Clear conversation history\n";
    oss << "   help                 Show this help message\n";
    oss << "\n";

    return oss.str();
}

int main(int argc, char *argv[]) {
    utils::initLoggerWithOverwrite(plog::verbose, "logs/chat.log");

    // Validate argument count
    if (argc != 2) {
        PLOG_ERROR_(utils::BOTH) << "Error: Invalid number of arguments";
        PLOG_VERBOSE_(utils::BOTH) << get_usage_message(argv[0]);
        return 1;
    }

    PLOG_VERBOSE_(utils::BOTH) << utils::get_runtime_info();

    // Parse and resolve model path
    std::string model_path_arg = argv[1];
    fs::path model_path_fs;

    try {
        model_path_fs = fs::absolute(model_path_arg);
    } catch (const fs::filesystem_error &e) {
        PLOG_ERROR_(utils::BOTH) << "Error: Invalid path: " << e.what();
        return 1;
    }

    // Validate model path exists
    if (!fs::exists(model_path_fs)) {
        PLOG_ERROR_(utils::BOTH) << "Error: Model path does not exist: " << model_path_fs;
        return 1;
    }

    // Validate model path is directory
    if (!fs::is_directory(model_path_fs)) {
        PLOG_ERROR_(utils::BOTH) << "Error: Model path is not a directory: " << model_path_fs;
        return 1;
    }

    std::string model_path = model_path_fs.string();

    // Initialize inference engine
    device::Device device(ZEDINFER_DEVICE_NVIDIA, 0);
    std::shared_ptr<zedinfer::InferenceEngine>
        engine = zedinfer::InferenceEngine::create(model_path, device);

    // Configure generation parameters
    zedinfer::GenerationConfig gen_config;
    gen_config.gen_mode = zedinfer::GenerationMode::CHAT;
    gen_config.max_new_tokens = 16384;
    gen_config.verbose = true;
    gen_config.print_stats = true;
    gen_config.stream = true;
    gen_config.stream_callback = [](const std::string &token_text) {
        std::cout << token_text << std::flush; // Stream tokens in real-time
    };

    // Create chat session
    auto session = engine->create_session(gen_config);

    PLOG_VERBOSE_(utils::BOTH) << get_welcome_message();

    // Set UTF-8 locale for proper Chinese character handling
    std::setlocale(LC_ALL, "en_US.UTF-8");

    // Main conversation loop
    std::string user_input;
    while (true) {
        // Display user prompt with readline
        char *input = readline("\n👨‍💻 \033[1;32mUser:\033[0m ");

        if (!input) {
            PLOGI << "Exiting..."; // Handle EOF (Ctrl+D)
            break;
        }

        // Convert to std::string
        std::string user_input(input);

        // Trim whitespace
        user_input.erase(0, user_input.find_first_not_of(" \t\n\r"));
        user_input.erase(user_input.find_last_not_of(" \t\n\r") + 1);

        // Skip empty input
        if (user_input.empty()) {
            continue;
        }

        // Convert to lowercase for command comparison
        std::string lower_input = user_input;
        std::transform(lower_input.begin(), lower_input.end(),
                       lower_input.begin(), ::tolower);

        // Handle exit commands
        if (lower_input == "exit" || lower_input == "quit" || lower_input == "q") {
            PLOGI << "Exiting...";
            break;
        }

        // Handle clear/reset commands
        if (lower_input == "reset" || lower_input == "clear" || lower_input == "cls") {
            session->reset();
            PLOG_VERBOSE_(utils::BOTH) << "✨ Chat history cleared.";
            continue;
        }

        // Handle help command
        if (lower_input == "help") {
            PLOG_VERBOSE_(utils::BOTH) << get_welcome_message();
            continue;
        }

        // Process chat message
        try {
            std::cout << "🤖 \033[1;34mAssistant:\033[0m ";
            session->chat(user_input);
            std::cout << "\n";
        } catch (const std::exception &e) {
            PLOG_ERROR_(utils::BOTH) << "Error: " << e.what();
        }
    }

    PLOG_VERBOSE_(utils::BOTH) << "\n👋 Goodbye! Thanks for using ZedInfer.";
    return 0;
}