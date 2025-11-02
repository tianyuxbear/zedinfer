#include "neollm.h"
#include "neollm/engine.hpp"
#include "plog/Severity.h"
#include "utils/logger.hpp"

#include <chrono>
#include <memory>
#include <string>

static const std::string model_path = "/mnt/hdd/shared/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B";

using namespace neollm;

int main() {
    initLoggerWithOverwrite(plog::verbose, "logs/ping.log");
    LOG_VERBOSE_(BOTH) << get_runtime_info();

    auto start = std::chrono::high_resolution_clock::now();

    std::unique_ptr<InferenceEngine> engine = InferenceEngine::create(model_path, NEOLLM_DEVICE_CPU, 0);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    LOGI << "InferenceEngine::create took " << duration.count() << " ms\n";

    GenerationConfig gen_config;
    gen_config.max_new_tokens = 128;
    gen_config.verbose = true;
    gen_config.print_stats = true;

    std::string prompt = "Who are you?";

    std::vector<std::pair<std::string, std::string>> messages = {
        {"user", prompt}};

    std::string output = engine->chat(messages, gen_config);
}
