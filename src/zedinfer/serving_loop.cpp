#include "zedinfer/serving_loop.hpp"
#include "zedinfer/engine.hpp"
#include "frontend/models/forward_config.hpp"
#include "frontend/models/paged_forward_context.hpp"
#include "utils/logging.hpp"

#include <chrono>
#include <plog/Log.h>

namespace zedinfer {

ServingLoop::ServingLoop(std::shared_ptr<InferenceEngine> engine,
                         SchedulerConfig sched_config)
    : engine_(std::move(engine)),
      scheduler_(std::move(sched_config)) {
    if (engine_->block_allocator()) {
        scheduler_.set_block_allocator(engine_->block_allocator());
    }
    if (engine_->prefix_cache()) {
        scheduler_.set_prefix_cache(engine_->prefix_cache());
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

    // Build request borrowing the session's block table
    auto request = build_request(input_ids, config);
    request->block_table_ref = &block_table;
    auto future = submit_async(std::move(request));

    while (future.wait_for(std::chrono::milliseconds(0)) != std::future_status::ready) {
        step();
    }

    return future.get();
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
    // Wake the serving loop if it's waiting
    work_cv_.notify_one();
    return future;
}

bool ServingLoop::step() {
    auto batch = scheduler_.schedule();
    if (batch.empty()) return false;

    auto batch_ctx = batch.build_context();

    try {
        auto t0 = std::chrono::high_resolution_clock::now();

        model::PagedForwardContext ctx(batch_ctx, *engine_->block_allocator());
        tensor_t logits = model::transformer_forward(
            engine_->model().forward_config(), ctx, engine_->exec_config());

        auto t1 = std::chrono::high_resolution_clock::now();
        double step_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        // Track timing: prefill or decode based on batch composition
        bool has_prefill = !batch.prefill_requests.empty();
        bool has_decode = !batch.decode_requests.empty();

        if (has_prefill) {
            for (auto *req : batch.prefill_requests) {
                req->stats.prefill_time_ms += step_ms;
                req->stats.total_time_ms += step_ms;
            }
        }
        if (has_decode) {
            for (auto *req : batch.decode_requests) {
                req->stats.decode_time_ms += step_ms / batch.decode_requests.size();
                req->stats.total_time_ms += step_ms / batch.decode_requests.size();
            }
        }

        scheduler_.process_results(batch, logits,
            engine_->sampler(), engine_->tokenizer(),
            engine_->stop_token_ids());
    } catch (const std::exception &e) {
        LOGE << "[ServingLoop] Forward pass failed: " << e.what();
        // Fail all requests in this batch gracefully instead of crashing
        fail_batch(batch, std::string("Forward pass error: ") + e.what());
    } catch (...) {
        LOGE << "[ServingLoop] Forward pass failed with unknown error";
        fail_batch(batch, "Unknown forward pass error");
    }

    return true;
}

void ServingLoop::run_loop() {
    while (scheduler_.has_work()) {
        try {
            step();
        } catch (const std::exception &e) {
            LOGE << "[ServingLoop] Unexpected error in run_loop: " << e.what();
        }
    }
}

void ServingLoop::run_serving() {
    running_ = true;
    LOGI << "[ServingLoop] Started";

    while (running_) {
        // Process all pending work
        while (scheduler_.has_work() && running_) {
            try {
                step();
            } catch (const std::exception &e) {
                LOGE << "[ServingLoop] Unexpected error in run_serving: " << e.what();
            }
        }

        // No work — wait for new submissions or stop signal
        if (running_) {
            std::unique_lock<std::mutex> lock(work_mutex_);
            work_cv_.wait(lock, [this] {
                return scheduler_.has_work() || !running_;
            });
        }
    }

    // Drain remaining work before exit
    while (scheduler_.has_work()) {
        try {
            step();
        } catch (...) {}
    }

    LOGI << "[ServingLoop] Stopped";
}

void ServingLoop::stop() {
    running_ = false;
    work_cv_.notify_one();
}

void ServingLoop::fail_batch(ScheduledBatch &batch, const std::string &error_msg) {
    // Fail all requests in the batch by setting an exception on their promises.
    // This ensures HTTP handlers get an error instead of hanging forever.
    auto fail_request = [&](InferenceRequest *req) {
        if (!req) return;
        req->phase = RequestPhase::COMPLETE;
        // Free owned blocks
        if (engine_->block_allocator() && !req->block_table_ref &&
            req->block_table.num_layers > 0) {
            engine_->block_allocator()->free_sequence(req->block_table);
        }
        try {
            req->result_promise.set_exception(
                std::make_exception_ptr(std::runtime_error(error_msg)));
        } catch (...) {}
    };

    for (auto *req : batch.decode_requests) fail_request(req);
    for (auto *req : batch.prefill_requests) fail_request(req);

    // Remove completed requests from scheduler's active list
    // (scheduler expects process_results to clean up, but we're bypassing it)
    scheduler_.cleanup_failed_requests();
}

} // namespace zedinfer
