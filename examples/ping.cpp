#include "backend/device/device.hpp"
#include "plog/Severity.h"
#include "utils/logging.hpp"
#include "utils/system_info.hpp"
#include "zedinfer.h"
#include "zedinfer/engine.hpp"
#include "zedinfer/session.hpp"

#include <chrono>
#include <memory>
#include <string>

// For server1
// static const std::string model_path = "/mnt/hdd0/shared/models/deepseek-ai/DeepSeek-R1-0528-Qwen3-8B";
// For server2/server3
// static const std::string model_path = "/mnt/hdd/shared/models/deepseek-ai/DeepSeek-R1-0528-Qwen3-8B";
// For NVIDIA server
static const std::string model_path = "/home/scratch.trt_llm_data_ci/llm-models/DeepSeek-R1-Distill-Qwen-1.5B";
using namespace zedinfer;

int main() {
    utils::initLoggerWithOverwrite(plog::verbose, "logs/ping.log");
    LOG_VERBOSE_(utils::BOTH) << utils::get_runtime_info();

    auto start = std::chrono::high_resolution_clock::now();

    device::Device device(ZEDINFER_DEVICE_NVIDIA, 0);
    std::shared_ptr<InferenceEngine> engine = InferenceEngine::create(model_path, device);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    LOGI << "InferenceEngine::create took " << duration.count() << " ms\n";

    GenerationConfig gen_config;
    gen_config.max_new_tokens = 16384;
    gen_config.verbose = true;
    gen_config.print_stats = true;

    std::string prompt = "Who are you?";

    auto session = engine->create_session(gen_config);

    session->chat(prompt);
}
