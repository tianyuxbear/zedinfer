#include "frontend/sampler/sampler.hpp"
#include "neollm/chat.hpp"

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>

namespace fs = std::filesystem;

void print_usage(const char *program_name) {
    std::cerr << "Usage: " << program_name << " <model_path>\n";
    std::cerr << "\nArguments:\n";
    std::cerr << "  model_path    Path to the model directory (absolute or relative)\n";
    std::cerr << "\nExample:\n";
    std::cerr << "  " << program_name << " ./models/llama-7b\n";
    std::cerr << "  " << program_name << " /home/user/models/llama-7b\n";
}

void print_welcome() {
    std::cout << "\n========================================\n";
    std::cout << "  Welcome to NeoLLM Chat\n";
    std::cout << "========================================\n";
    std::cout << "Commands:\n";
    std::cout << "  /exit, /quit, /q  - Exit the chat\n";
    std::cout << "  /clear, /reset    - Clear conversation history\n";
    std::cout << "  /help             - Show this help message\n";
    std::cout << "========================================\n\n";
}

int main(int argc, char *argv[]) {
    // 检查参数数量
    if (argc != 2) {
        std::cerr << "Error: Invalid number of arguments\n\n";
        print_usage(argv[0]);
        return 1;
    }

    // 获取模型路径参数
    std::string model_path_arg = argv[1];

    // 处理相对路径和绝对路径
    fs::path model_path_fs;
    try {
        model_path_fs = fs::absolute(model_path_arg); // 转换为绝对路径
    } catch (const fs::filesystem_error &e) {
        std::cerr << "Error: Invalid path: " << e.what() << "\n";
        return 1;
    }

    // 验证路径是否存在
    if (!fs::exists(model_path_fs)) {
        std::cerr << "Error: Model path does not exist: " << model_path_fs << "\n";
        return 1;
    }

    // 验证是否为目录
    if (!fs::is_directory(model_path_fs)) {
        std::cerr << "Error: Model path is not a directory: " << model_path_fs << "\n";
        return 1;
    }

    std::string model_path = model_path_fs.string();

    std::cout << "[Info] Model path: " << model_path << "\n";
    std::cout << "[Info] Loading model...\n";

    // 初始化推理引擎
    std::unique_ptr<neollm::InferenceEngine> engine = neollm::InferenceEngine::create(model_path, NEOLLM_DEVICE_CPU, 0);

    // 配置生成参数
    neollm::GenerationConfig gen_config;
    gen_config.sampler_type = neollm::sampler::SamplerType::ARGMAX;
    gen_config.max_new_tokens = 16384;
    gen_config.max_seq_len = 16384;
    gen_config.verbose = false;
    gen_config.print_stats = true;
    gen_config.stream = true;
    gen_config.stream_callback = [](const std::string &token_text) {
        std::cout << token_text << std::flush; // 实时输出每个token
    };

    // 创建对话会话
    neollm::ChatSession chat_session(std::move(engine), gen_config);

    std::cout << "[Info] Model loaded successfully!\n";
    print_welcome();

    // 多轮对话循环
    std::string user_input;
    while (true) {
        // 显示提示符
        std::cout << "\n\033[1;32mYou:\033[0m ";

        // 读取用户输入
        if (!std::getline(std::cin, user_input)) {
            // EOF (Ctrl+D)
            std::cout << "\n[Info] Exiting...\n";
            break;
        }

        // 去除首尾空白
        user_input.erase(0, user_input.find_first_not_of(" \t\n\r"));
        user_input.erase(user_input.find_last_not_of(" \t\n\r") + 1);

        // 处理空输入
        if (user_input.empty()) {
            continue;
        }

        // 处理命令
        if (user_input == "/exit" || user_input == "/quit" || user_input == "/q") {
            std::cout << "[Info] Exiting...\n";
            break;
        } else if (user_input == "/clear" || user_input == "/reset") {
            // chat_session.reset();
            std::cout << "[Info] Conversation history cleared.\n";
            continue;
        } else if (user_input == "/help") {
            print_welcome();
            continue;
        }

        // 进行对话
        try {
            std::cout << "\033[1;34mAssistant:\033[0m " << "<think> ";
            chat_session.chat(user_input);
            std::cout << "\n"; // 换行

        } catch (const std::exception &e) {
            std::cerr << "\n[Error] " << e.what() << "\n";
        }
    }

    std::cout << "\nGoodbye!\n";
    return 0;
}