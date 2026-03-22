#include "zedinfer/serving_loop.hpp"
#include "zedinfer/engine.hpp"
#include "frontend/models/forward_config.hpp"
#include "frontend/models/paged_forward_context.hpp"
#include "utils/logging.hpp"

#include <chrono>
#include <plog/Log.h>

namespace zedinfer {

ServingLoop::ServingLoop(std::shared_ptr<InferenceEngine> engine)
    : engine_(std::move(engine)) {
    if (engine_->block_allocator()) {
        scheduler_.set_block_allocator(engine_->block_allocator());
    }
}

// ============================================================================
// Synchronous Generation
// ============================================================================

std::string ServingLoop::generate(
    kvcache::SequenceBlockTable &block_table,
    const std::string &prompt,
    const GenerationConfig &config) {

    config.validate();

    if (config.verbose) {
        if (config.gen_mode == GenerationMode::PING) {
            LOG_INFO_(utils::BOTH) << "\n=== [Inference] Prompt: ===\n" << prompt;
        } else {
            LOGI << "\n=== [Inference] Prompt: ===\n" << prompt;
        }
        LOGI << "[Inference] Encoding prompt...";
    }

    auto input_ids = engine_->tokenizer().encode(prompt);

    if (config.verbose) {
        LOGI << "[Inference] Prompt tokens: " << input_ids.size();
        LOGI << "[Inference] Generating...";
    }

    auto result = generate_tokens(block_table, input_ids, config);
    std::string output = engine_->tokenizer().decode(result.output_ids);

    if (config.verbose) {
        LOGI << "[Inference] Generation complete";
        if (config.gen_mode == GenerationMode::PING) {
            LOG_INFO_(utils::BOTH) << "\n=== [Inference] Generated ===\n" << output;
        } else {
            LOGI << "\n=== [Inference] Generated ===\n" << output;
        }
    }

    if (config.print_stats) {
        LOGI << result.stats.summary();
    }

    return output;
}

GenerationResult ServingLoop::generate_tokens(
    kvcache::SequenceBlockTable &block_table,
    const std::vector<int> &input_ids,
    const GenerationConfig &config) {

    auto request = build_request(input_ids, config);
    scheduler_.submit(std::move(request));
    return scheduler_.run_one(
        engine_->model(), block_table, *engine_->block_pool(),
        engine_->exec_config(),
        engine_->sampler(), engine_->tokenizer(),
        engine_->stop_token_ids());
}

std::unique_ptr<InferenceRequest> ServingLoop::build_request(
    const std::vector<int> &input_ids,
    const GenerationConfig &config) {

    auto req = std::make_unique<InferenceRequest>();
    req->input_ids = input_ids;
    req->config = config;
    req->stream_callback = config.stream ? config.stream_callback : nullptr;
    req->arrival_time = std::chrono::steady_clock::now();
    return req;
}

// ============================================================================
// Batch Mode
// ============================================================================

std::future<GenerationResult> ServingLoop::submit_async(
    std::unique_ptr<InferenceRequest> request) {
    auto future = request->result_promise.get_future();
    scheduler_.submit(std::move(request));
    return future;
}

bool ServingLoop::step() {
    auto batch = scheduler_.schedule();
    if (batch.empty()) return false;

    auto batch_ctx = batch.build_context();

    model::PagedForwardContext ctx(batch_ctx, *engine_->block_allocator());
    tensor_t logits = model::transformer_forward(
        engine_->model().forward_config(), ctx, engine_->exec_config());

    scheduler_.process_results(batch, logits,
        engine_->sampler(), engine_->tokenizer(),
        engine_->stop_token_ids());

    return true;
}

void ServingLoop::run_loop() {
    while (scheduler_.has_work()) {
        step();
    }
}

} // namespace zedinfer
