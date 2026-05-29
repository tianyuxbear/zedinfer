#include "backend/device/device.hpp"
#include "utils/logging.hpp"
#include "utils/random.hpp"
#include "utils/system_info.hpp"
#include "zedinfer.h"
#include "zedinfer/engine.hpp"
#include "zedinfer/request.hpp"
#include "zedinfer/scheduler.hpp"

#include "zedinfer/version.hpp"
#include <argparse/argparse.hpp>
#include <chrono>
#include <cstdio>
#include <future>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

using namespace zedinfer;

int main(int argc, char* argv[]) {
    argparse::ArgumentParser program("ZedInfer Batch Benchmark", std::string("zedinfer ") + ZEDINFER_VERSION
                                                                     + " (build " + ZEDINFER_GIT_HASH + ", "
                                                                     + ZEDINFER_BUILD_DATE + ")");

    program.add_argument("model_path").help("Path to the model directory");

    program.add_argument("-b", "--batch-size").help("Number of concurrent requests").default_value(4).scan<'i', int>();

    program.add_argument("-p", "--prefill-len")
        .help("Prompt length per request (tokens)")
        .default_value(128)
        .scan<'i', int>();

    program.add_argument("-d", "--decode-len")
        .help("Max tokens to generate per request")
        .default_value(128)
        .scan<'i', int>();

    program.add_argument("-r", "--rounds").help("Number of benchmark rounds").default_value(1).scan<'i', int>();

    program.add_argument("--nvidia").help("Use NVIDIA GPU backend").default_value(false).implicit_value(true);

    program.add_argument("--gpu-memory-utilization")
        .help("Fraction of GPU memory for KV cache (0.0-1.0)")
        .default_value(0.9f)
        .scan<'g', float>();

    try {
        program.parse_args(argc, argv);
    } catch (const std::exception& err) {
        std::cerr << err.what() << std::endl;
        std::cerr << program;
        return 1;
    }

    utils::initLoggerWithOverwrite(plog::verbose, "logs/batch_bench.log");
    LOG_VERBOSE_(utils::BOTH) << utils::get_runtime_info();

    auto model_path = program.get<std::string>("model_path");
    int batch_size = program.get<int>("--batch-size");
    int prefill_len = program.get<int>("--prefill-len");
    int decode_len = program.get<int>("--decode-len");
    int rounds = program.get<int>("--rounds");
    bool use_nvidia = program.get<bool>("--nvidia");

    zedinferDeviceType_t device_type = use_nvidia ? ZEDINFER_DEVICE_NVIDIA : ZEDINFER_DEVICE_CPU;
    device::Device device(device_type, 0);

    SchedulerConfig sched_config;
    sched_config.gpu_memory_utilization = program.get<float>("--gpu-memory-utilization");

    auto engine = InferenceEngine::create(model_path, device, sched_config);

    // Also run single-request baseline for comparison
    printf("\n============ Single-Request Baseline ============\n");
    {
        auto res = engine->profiler().profile(prefill_len, decode_len);
        double prefill_tps = prefill_len / res.first * 1000.0;
        double decode_tps = decode_len / res.second * 1000.0;
        printf("Prefill: %8.2f ms | %8.2f tok/s\n", res.first, prefill_tps);
        printf("Decode : %8.2f ms | %8.2f tok/s  (per request)\n", res.second, decode_tps);
    }

    // Batch benchmark
    printf("\n============ Batch Benchmark ============\n");
    printf("Config: batch_size=%d, prefill=%d, decode=%d, rounds=%d\n", batch_size, prefill_len, decode_len, rounds);

    // Random token range for input
    int min_id = 100, max_id = 30000;

    // Fixed seed so the random prompts (and therefore the whole batched run) are
    // reproducible across invocations — required for before/after A/B validation
    // and for stable throughput comparisons.
    utils::set_seed(12345);

    double total_time_ms = 0.0;
    int total_generated_tokens = 0;

    for (int r = 0; r < rounds; ++r) {
        // Create batch of requests with random prompts
        std::vector<std::future<GenerationResult>> futures;

        auto t0 = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < batch_size; ++i) {
            auto req = std::make_unique<InferenceRequest>();

            // Generate random prompt tokens
            req->input_ids.resize(prefill_len);
            for (auto& t : req->input_ids) { t = utils::randint(min_id, max_id); }

            req->config.max_new_tokens = decode_len;
            req->config.verbose = false;
            req->arrival_time = std::chrono::steady_clock::now();

            futures.push_back(engine->serving_loop().submit_async(std::move(req)));
        }

        // Run engine loop until all requests complete
        engine->serving_loop().run_loop();

        auto t1 = std::chrono::high_resolution_clock::now();
        double round_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        // Collect results
        int round_tokens = 0;
        for (auto& f : futures) {
            auto result = f.get();
            round_tokens += static_cast<int>(result.output_ids.size());
        }

        total_time_ms += round_ms;
        total_generated_tokens += round_tokens;

        printf("Round %d: %d requests, %d tokens generated, %.2f ms\n", r + 1, batch_size, round_tokens, round_ms);
    }

    // Summary
    double avg_time_ms = total_time_ms / rounds;
    double avg_tokens = static_cast<double>(total_generated_tokens) / rounds;
    double throughput_tps = avg_tokens / avg_time_ms * 1000.0;

    double avg_per_request_ms = avg_time_ms / batch_size;

    printf("\n============ Batch Performance Report ============\n");
    printf("Batch size     : %d requests\n", batch_size);
    printf("Avg total time : %8.2f ms\n", avg_time_ms);
    printf("Avg tokens/batch: %8.0f\n", avg_tokens);
    printf("Throughput     : %8.2f tok/s  (total, all requests)\n", throughput_tps);
    printf("Per-request    : %8.2f ms  (avg wall time per request)\n", avg_per_request_ms);
    printf("===================================================\n");

    return 0;
}
