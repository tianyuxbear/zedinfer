#pragma once

#include "backend/kvcache/block_pool.hpp"
#include "zedinfer/batch_context.hpp"
#include "zedinfer/request.hpp"
#include "zedinfer/scheduler.hpp"

#include <atomic>
#include <condition_variable>
#include <future>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace zedinfer {

class InferenceEngine;

/**
 * Serving loop: owns the Scheduler and drives synchronous/batched generation.
 *
 * For HTTP API: submit_async() from HTTP threads, run_serving() on engine thread.
 * run_serving() sleeps when idle, wakes on new submissions.
 */
/**
 * Holds a non-owning pointer to the engine that owns this loop (engine outlives
 * its own unique_ptr members). Must not take a shared_ptr back, or InferenceEngine
 * would form a cycle with itself and never destruct.
 */
class ServingLoop {
public:
    explicit ServingLoop(InferenceEngine& engine, SchedulerConfig sched_config = {});

    // Synchronous generation (session-based, used by chat/ping)
    std::string generate(kvcache::SequenceBlockTable& block_table, const std::string& prompt,
                         const GenerationConfig& config);

    GenerationResult generate_tokens(kvcache::SequenceBlockTable& block_table, const std::vector<int>& input_ids,
                                     const GenerationConfig& config);

    // Async batch mode
    std::future<GenerationResult> submit_async(std::unique_ptr<InferenceRequest> request);

    // Single iteration (non-blocking). Returns true if work was done.
    bool step();

    // Run until all current work is done (offline batch mode, e.g. batch_bench).
    void run_loop();

    // Run continuously until stop() is called (serving mode, e.g. HTTP API).
    // Sleeps when idle, wakes on new submissions.
    void run_serving();

    // Signal the serving loop to stop. Safe to call from any thread.
    void stop();

    // Scheduler status (for health endpoint)
    int pending_count() const { return scheduler_.pending_count(); }
    int active_count() const { return scheduler_.active_count(); }

private:
    InferenceEngine* engine_;
    // Cached from SchedulerConfig at construction (the config itself is
    // moved into scheduler_). True when --mtp / ZEDINFER_MTP_SPEC enabled
    // MTP speculative decoding for this engine.
    bool mtp_enabled_ = false;
    Scheduler scheduler_;

    // Thread synchronization for serving mode
    std::mutex work_mutex_;
    std::condition_variable work_cv_;
    std::atomic<bool> running_{false};

    std::unique_ptr<InferenceRequest> build_request(const std::vector<int>& input_ids, const GenerationConfig& config);

    // Fail all requests in a batch with an error (used when forward pass throws)
    void fail_batch(ScheduledBatch& batch, const std::string& error_msg);
};

} // namespace zedinfer
