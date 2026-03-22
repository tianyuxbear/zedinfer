#include "zedinfer/scheduler.hpp"
#include "frontend/models/base.hpp"
#include "frontend/sampler/sampler.hpp"
#include "frontend/tokenizer/base.hpp"
#include "backend/kvcache/base.hpp"
#include "utils/logging.hpp"

#include <chrono>
#include <plog/Log.h>
#include <stdexcept>

namespace zedinfer {

static bool is_stop_token(int token_id, const std::vector<int> &stop_ids) {
    for (int stop_id : stop_ids) {
        if (token_id == stop_id) return true;
    }
    return false;
}

Scheduler::Scheduler(SchedulerConfig config) : config_(config) {}

void Scheduler::submit(std::unique_ptr<InferenceRequest> request) {
    if (static_cast<int>(waiting_queue_.size()) >= config_.max_queue_size) {
        throw std::runtime_error("[Scheduler] Queue full (max_queue_size=" +
                                 std::to_string(config_.max_queue_size) + ")");
    }
    request->request_id = next_request_id_++;
    request->phase = RequestPhase::QUEUED;
    waiting_queue_.push_back(std::move(request));
}

bool Scheduler::has_work() const {
    return !waiting_queue_.empty() || active_request_ != nullptr;
}

int Scheduler::pending_count() const {
    return static_cast<int>(waiting_queue_.size());
}

int Scheduler::active_count() const {
    return active_request_ != nullptr ? 1 : 0;
}

GenerationResult Scheduler::run_one(
    model::Model &model,
    kvcache::KVCache &kvcache,
    const ExecutorConfig &exec_config,
    sampler::Sampler &sampler,
    tokenizer::Tokenizer &tokenizer,
    const std::vector<int> &stop_token_ids) {

    if (waiting_queue_.empty()) {
        throw std::runtime_error("[Scheduler] No pending requests");
    }

    // Pop front request
    auto request_ptr = std::move(waiting_queue_.front());
    waiting_queue_.pop_front();
    InferenceRequest &req = *request_ptr;
    active_request_ = &req;

    GenerationStats &stats = req.stats;
    stats.prompt_tokens = req.input_ids.size();
    req.output_ids.reserve(req.config.max_new_tokens);

    // Prefill
    req.phase = RequestPhase::PREFILL;

    auto t0 = std::chrono::high_resolution_clock::now();

    int past_len = kvcache.current_length();
    tensor_t logits = model.forward(req.input_ids, past_len, kvcache, exec_config);
    int next_token = sampler.sample(logits);

    auto t1 = std::chrono::high_resolution_clock::now();
    stats.prefill_time_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    stats.total_time_ms += stats.prefill_time_ms;

    req.output_ids.push_back(next_token);
    req.last_token = next_token;
    req.generated_count = 1;

    if (is_stop_token(next_token, stop_token_ids)) {
        req.phase = RequestPhase::COMPLETE;
        stats.generated_tokens = req.output_ids.size();
        stats.total_tokens = stats.prompt_tokens + stats.generated_tokens;
        active_request_ = nullptr;
        return GenerationResult{std::move(req.output_ids), stats};
    }

    if (req.config.stream && req.stream_callback) {
        req.stream_callback(tokenizer.decode({next_token}));
    }

    // Decode
    req.phase = RequestPhase::DECODE;
    past_len += req.input_ids.size();

    for (int i = 1; i < req.config.max_new_tokens; ++i) {
        auto s0 = std::chrono::high_resolution_clock::now();

        logits = model.forward({req.last_token}, past_len, kvcache, exec_config);
        next_token = sampler.sample(logits);

        auto s1 = std::chrono::high_resolution_clock::now();
        double step_ms = std::chrono::duration<double, std::milli>(s1 - s0).count();
        stats.decode_time_ms += step_ms;
        stats.total_time_ms += step_ms;

        req.output_ids.push_back(next_token);
        req.last_token = next_token;
        req.generated_count++;
        past_len++;

        if (is_stop_token(next_token, stop_token_ids)) break;

        if (req.config.stream && req.stream_callback) {
            req.stream_callback(tokenizer.decode({next_token}));
        }

        if (past_len >= static_cast<int>(tokenizer.get_config().model_max_length)) {
            if (req.config.verbose) {
                LOGI << "[Scheduler] Reached max sequence length";
            }
            break;
        }
    }

    req.phase = RequestPhase::COMPLETE;
    stats.generated_tokens = req.output_ids.size();
    stats.total_tokens = stats.prompt_tokens + stats.generated_tokens;

    GenerationResult result{std::move(req.output_ids), stats};
    active_request_ = nullptr;
    return result;
}

} // namespace zedinfer
