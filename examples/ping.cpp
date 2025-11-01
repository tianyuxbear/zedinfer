#include "neollm.h"
#include "neollm/engine.hpp"
#include <chrono>
#include <iostream>
#include <memory>
#include <string>
static const std::string model_path = "/home/xiongtianyu/data/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B";

using namespace neollm;

int main() {
    auto start = std::chrono::high_resolution_clock::now();
    std::unique_ptr<InferenceEngine> engine = InferenceEngine::create(model_path, NEOLLM_DEVICE_CPU, 0);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "InferenceEngine::create took " << duration.count() << " ms\n";

    GenerationConfig gen_config;
    gen_config.max_new_tokens = 128;
    gen_config.verbose = true;
    gen_config.print_stats = true;
    gen_config.stream = true;
    gen_config.stream_callback = [](const std::string &token_text) {
        std::cout << token_text << std::flush; // 实时输出每个token
    };

    std::string prompt = "Who are you?";

    std::vector<std::pair<std::string, std::string>> messages = {
        {"user", prompt}};

    std::string output = engine->chat(messages, gen_config);

    // std::cout << "output: \n"
    //           << output << std::endl;
}
