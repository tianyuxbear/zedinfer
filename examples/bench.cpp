#include "backend/device/device.hpp"
#include "plog/Severity.h"
#include "utils/logging.hpp"
#include "utils/system_info.hpp"
#include "zedinfer.h"
#include "zedinfer/engine.hpp"
#include "zedinfer/session.hpp"

#include <argparse/argparse.hpp>
#include <chrono>
#include <iostream>
#include <memory>
#include <string>

using namespace zedinfer;

int main(int argc, char *argv[]) {
    // Initialize logger
    utils::initLoggerWithOverwrite(plog::verbose, "logs/bench.log");
    LOG_VERBOSE_(utils::BOTH) << utils::get_runtime_info();

    // 1. Argument Parsing Setup
    argparse::ArgumentParser program("ZedInfer Benchmarks");

    program.add_argument("model_path")
        .help("Path to the model directory");

    program.add_argument("-p", "--prefill-len")
        .help("Sequence length for prefill phase")
        .default_value(128)
        .scan<'i', int>(); // Use scan to enforce integer parsing

    program.add_argument("-d", "--decode-len")
        .help("Number of tokens to decode")
        .default_value(128)
        .scan<'i', int>();

    program.add_argument("-r", "--rounds")
        .help("Number of benchmark rounds")
        .default_value(3)
        .scan<'i', int>();

    program.add_argument("--nvidia")
        .help("Use NVIDIA GPU backend")
        .default_value(false)
        .implicit_value(true); // Treated as flag

    try {
        program.parse_args(argc, argv);
    } catch (const std::exception &err) {
        std::cerr << err.what() << std::endl;
        std::cerr << program;
        return 1;
    }

    // 2. Retrieve Arguments
    auto model_path = program.get<std::string>("model_path");
    auto prefill_len = program.get<int>("--prefill-len");
    auto decode_len = program.get<int>("--decode-len");
    auto rounds = program.get<int>("--rounds");
    bool use_nvidia = program.get<bool>("--nvidia"); // Correctly retrieve nvidia flag

    // 3. Device Initialization
    zedinferDeviceType_t device_type = use_nvidia ? ZEDINFER_DEVICE_NVIDIA : ZEDINFER_DEVICE_CPU;
    device::Device device(device_type, 0);

    LOGI << "Initializing engine with device: " << (use_nvidia ? "NVIDIA" : "CPU");
    auto start = std::chrono::high_resolution_clock::now();

    // Pass the actual model_path instead of hardcoded string
    std::shared_ptr<InferenceEngine> engine = InferenceEngine::create(model_path, device);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    LOGI << "Engine initialized in " << duration.count() << " ms";

    // 4. Warmup Phase (Critical for accurate benchmarking)
    LOGI << "Starting warmup...";
    engine->warmup(prefill_len, decode_len);

    // 5. Benchmarking Loop
    double total_prefill_time = 0.0;
    double total_decode_time = 0.0;

    std::cout << "\n[ZedInfer] Running " << rounds << " rounds of profiling..." << std::endl;

    for (int i = 0; i < rounds; ++i) {
        auto res = engine->profile(prefill_len, decode_len);
        total_prefill_time += res.first;
        total_decode_time += res.second;
        // Optional: Log per-round progress
        LOGI << "Round " << i + 1 << ": prefill=" << res.first << "ms, decode=" << res.second << "ms";
    }

    // 6. Report Metrics
    double avg_prefill_time = total_prefill_time / rounds;
    double avg_decode_time = total_decode_time / rounds;

    // Calculate Throughput (Tokens Per Second)
    double prefill_tps = (prefill_len / avg_prefill_time) * 1000.0;
    double decode_tps = (decode_len / avg_decode_time) * 1000.0;

    printf("\n================ Performance Report ================\n");
    printf("Config : prefill=%d, decode=%d, rounds=%d\n", prefill_len, decode_len, rounds);
    printf("Prefill: %8.2f ms | %8.2f tok/s\n", avg_prefill_time, prefill_tps);
    printf("Decode : %8.2f ms | %8.2f tok/s\n", avg_decode_time, decode_tps);
    printf("====================================================\n");

    return 0;
}