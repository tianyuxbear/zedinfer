#pragma once

#include "backend/kvcache/block_pool.hpp"
#include "zedinfer/batch_context.hpp"
#include "zedinfer/request.hpp"
#include "zedinfer/scheduler.hpp"

#include <future>
#include <memory>
#include <string>
#include <vector>

namespace zedinfer {

class InferenceEngine;

/**
 * Serving loop: owns the Scheduler and drives synchronous/batched generation.
 * Separated from InferenceEngine to isolate serving logic from resource management.
 */
class ServingLoop {
public:
    explicit ServingLoop(std::shared_ptr<InferenceEngine> engine);

    // Synchronous generation (session-based, used by chat/ping)
    std::string generate(kvcache::SequenceBlockTable &block_table,
                         const std::string &prompt,
                         const GenerationConfig &config);

    GenerationResult generate_tokens(kvcache::SequenceBlockTable &block_table,
                                     const std::vector<int> &input_ids,
                                     const GenerationConfig &config);

    // Async batch mode (used by HTTP API and batch_bench)
    std::future<GenerationResult> submit_async(std::unique_ptr<InferenceRequest> request);
    bool step();
    void run_loop();

private:
    std::shared_ptr<InferenceEngine> engine_;
    Scheduler scheduler_;

    std::unique_ptr<InferenceRequest> build_request(
        const std::vector<int> &input_ids,
        const GenerationConfig &config);
};

} // namespace zedinfer
