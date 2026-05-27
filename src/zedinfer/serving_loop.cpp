#include "zedinfer/serving_loop.hpp"
#include "backend/device/runtime_api.hpp"
#include "backend/ops/ops.hpp"
#include "frontend/models/decode_scratch.hpp"
#include "frontend/models/forward_config.hpp"
#include "frontend/models/hybrid_transformer_forward.hpp"
#include "frontend/models/mtp_module.hpp"
#include "frontend/models/qwen3_5.hpp"
#include "frontend/models/qwen3_5_moe.hpp"
#include "frontend/models/paged_forward_context.hpp"
#include "utils/logging.hpp"
#include "zedinfer/engine.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <plog/Log.h>

namespace zedinfer {

ServingLoop::ServingLoop(InferenceEngine& engine, SchedulerConfig sched_config)
    : engine_(&engine), scheduler_(std::move(sched_config)) {
    if (engine_->block_allocator()) {
        scheduler_.set_block_allocator(engine_->block_allocator());
    }
    if (engine_->prefix_cache()) {
        scheduler_.set_prefix_cache(engine_->prefix_cache());
    }
    // Wire SSMStatePool for hybrid models (Qwen3.5 / Qwen3.5-MoE). Non-hybrid
    // models return nullptr from ssm_state_pool() and the scheduler keeps the
    // pre-existing single-pool admission semantics.
    if (auto* pool = engine_->ssm_state_pool()) {
        scheduler_.set_ssm_state_pool(pool);
    }
    // Pass <think> / </think> ids (Qwen3.5 family) so the scheduler can drive
    // the per-request thinking-budget force-emit. Returns -1 for models without
    // these special tokens, in which case the scheduler treats the budget as
    // disabled. The "\n\n" id (resolved via tokenizer.encode) anchors the
    // post-</think> state after a force-close.
    scheduler_.set_think_token_ids(engine_->think_open_token_id(), engine_->think_close_token_id(),
                                   engine_->double_newline_token_id());
}

// ============================================================================
// Synchronous Generation
// ============================================================================

std::string ServingLoop::generate(kvcache::SequenceBlockTable& block_table, const std::string& prompt,
                                  const GenerationConfig& config) {
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

GenerationResult ServingLoop::generate_tokens(kvcache::SequenceBlockTable& block_table,
                                              const std::vector<int>& input_ids, const GenerationConfig& config) {
    // Build request borrowing the session's block table
    auto request = build_request(input_ids, config);
    request->borrow_block_table(block_table);
    auto future = submit_async(std::move(request));

    while (future.wait_for(std::chrono::milliseconds(0)) != std::future_status::ready) { step(); }

    return future.get();
}

std::unique_ptr<InferenceRequest> ServingLoop::build_request(const std::vector<int>& input_ids,
                                                             const GenerationConfig& config) {
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

std::future<GenerationResult> ServingLoop::submit_async(std::unique_ptr<InferenceRequest> request) {
    auto future = request->result_promise.get_future();
    scheduler_.submit(std::move(request));
    // Wake the serving loop if it's waiting
    work_cv_.notify_one();
    return future;
}

bool ServingLoop::step() {
    auto batch = scheduler_.schedule();
    if (batch.empty()) {
        return false;
    }

    auto batch_ctx = batch.build_context();

    try {
        auto t0 = std::chrono::high_resolution_clock::now();

        model::PagedForwardContext ctx(batch_ctx, *engine_->block_allocator());
        // Use decode scratch for single-token decode (no prefill in batch)
        bool is_pure_decode = batch.prefill_requests.empty() && batch.decode_requests.size() == 1;
        auto* scratch = is_pure_decode ? engine_->decode_scratch() : nullptr;

        // Hybrid Qwen3.5 dispatch: SSU + paged-attn mixed path needs the
        // request's SSM slot index, image embeds, and pos_ids_thw. Single-user
        // M1 smoke assumes one request per batch — pick whichever phase is
        // present. Multi-request batched hybrid is M5 territory.
        tensor_t logits;
        tensor_t mtp_hidden_last;  // captured iff ZEDINFER_MTP_DEBUG=1, see below
        const model::Qwen3_5MoeModel* mtp_owner = nullptr;
        if (auto* hybrid_model = dynamic_cast<const model::Qwen3_5Model*>(&engine_->model())) {
            InferenceRequest* req = !batch.decode_requests.empty()
                                       ? batch.decode_requests[0]
                                       : (!batch.prefill_requests.empty() ? batch.prefill_requests[0] : nullptr);
            if (!req) {
                throw std::runtime_error("[ServingLoop] hybrid model: empty batch (no request to forward)");
            }
            // MoE variant carries an ExpertPool + shared-expert config the dense
            // forward config doesn't populate; downcast first so the right
            // is_moe / num_experts / expert_pool fields are set.
            const auto* moe_model = dynamic_cast<const model::Qwen3_5MoeModel*>(hybrid_model);
            model::HybridForwardConfig hcfg = moe_model ? moe_model->hybrid_forward_config_moe()
                                                          : hybrid_model->hybrid_forward_config();
            // MTP smoke: capture main's pre-final-norm residual when the env
            // knob is set AND this model ships an MTP head. Caller (just
            // below the sampler) will invoke MTPModule::forward(hidden_last,
            // sampled_token) and log the speculative next-next-token. Pure
            // observability — no effect on the live generation path.
            const bool mtp_debug_env = std::getenv("ZEDINFER_MTP_DEBUG") != nullptr;
            tensor_t* hidden_out_ptr = nullptr;
            if (mtp_debug_env && moe_model && moe_model->mtp_module()
                && moe_model->mtp_module()->ready()) {
                hidden_out_ptr = &mtp_hidden_last;
                mtp_owner      = moe_model;
            }
            logits = model::hybrid_transformer_forward(hcfg, ctx, *req, engine_->exec_config(), scratch,
                                                         req->image_embeds(), hidden_out_ptr);
        } else {
            logits = model::transformer_forward(engine_->model().forward_config(), ctx, engine_->exec_config(),
                                                  scratch);
        }

        auto t1 = std::chrono::high_resolution_clock::now();
        double step_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        // Track timing: prefill or decode based on batch composition
        bool has_prefill = !batch.prefill_requests.empty();
        bool has_decode = !batch.decode_requests.empty();

        if (has_prefill) {
            for (auto* req : batch.prefill_requests) {
                req->stats.prefill_time_ms += step_ms;
                req->stats.total_time_ms += step_ms;
            }
        }
        if (has_decode) {
            for (auto* req : batch.decode_requests) {
                req->stats.decode_time_ms += step_ms / batch.decode_requests.size();
                req->stats.total_time_ms += step_ms / batch.decode_requests.size();
            }
        }

        scheduler_.process_results(batch, logits, engine_->sampler(), engine_->tokenizer(), engine_->stop_token_ids());

        // MTP smoke (Stage B.1): if main captured hidden_last for us, invoke
        // MTPModule::forward(last-pos hidden, main's just-sampled token) and
        // log MTP's top-1 next-next-token guess. Observability only — does
        // not change the live decode trajectory. Gated on ZEDINFER_MTP_DEBUG
        // (set in the upstream `hidden_out_ptr` branch) so the hot path
        // pays nothing in normal serving.
        if (mtp_hidden_last && mtp_owner) {
            try {
                InferenceRequest* req = !batch.decode_requests.empty()
                                           ? batch.decode_requests[0]
                                           : (!batch.prefill_requests.empty() ? batch.prefill_requests[0]
                                                                              : nullptr);
                if (req) {
                    const auto& w = mtp_owner->weights();
                    auto embed_w  = w.has_tensor("embed_tokens.weight")
                                        ? w.get_tensor("embed_tokens.weight") : nullptr;
                    auto lmhead_w = w.has_tensor("lm_head.weight")
                                        ? w.get_tensor("lm_head.weight") : nullptr;
                    if (embed_w && lmhead_w) {
                        const size_t N = mtp_hidden_last->shape()[0];
                        auto last_row = mtp_hidden_last->slice(0, N - 1, N);

                        auto mtp_logits = mtp_owner->mtp_module()->forward(
                            last_row, req->last_token, embed_w, lmhead_w, engine_->exec_config());

                        // Top-1 via ops::argmax (small extra D2H copy).
                        const size_t vocab = mtp_logits->shape()[1];
                        auto last_view = mtp_logits->view({vocab});
                        auto idx_dev = Tensor::create({1}, ZEDINFER_DTYPE_I64,
                                                      engine_->exec_config().device_type,
                                                      engine_->exec_config().device_id);
                        auto val_dev = Tensor::create({1}, engine_->exec_config().data_type,
                                                      engine_->exec_config().device_type,
                                                      engine_->exec_config().device_id);
                        ops::argmax(idx_dev, val_dev, last_view);
                        int64_t mtp_top1 = -1;
                        auto* api = device::getRuntimeAPI(engine_->exec_config().device_type);
                        api->memcpy_sync(&mtp_top1, idx_dev->data(), sizeof(int64_t),
                                         ZEDINFER_MEMCPY_D2H);
                        fprintf(stderr, "[MTP-debug] main_token=%d  mtp_top1=%lld\n",
                                req->last_token, (long long)mtp_top1);
                    }
                }
            } catch (const std::exception& e) {
                LOGW << "[ServingLoop] MTP-debug branch raised: " << e.what();
            }
        }
    } catch (const std::exception& e) {
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
        } catch (const std::exception& e) { LOGE << "[ServingLoop] Unexpected error in run_loop: " << e.what(); }
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
            } catch (const std::exception& e) { LOGE << "[ServingLoop] Unexpected error in run_serving: " << e.what(); }
        }

        // No work — wait for new submissions or stop signal
        if (running_) {
            std::unique_lock<std::mutex> lock(work_mutex_);
            work_cv_.wait(lock, [this] { return scheduler_.has_work() || !running_; });
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

void ServingLoop::fail_batch(ScheduledBatch& batch, const std::string& error_msg) {
    // Fail all requests in the batch by setting an exception on their promises.
    // This ensures HTTP handlers get an error instead of hanging forever.
    auto fail_request = [&](InferenceRequest* req) {
        if (!req) {
            return;
        }
        req->phase = RequestPhase::COMPLETE;
        // Free owned blocks
        if (engine_->block_allocator() && req->owns_block_table() && req->block_table().num_layers > 0) {
            engine_->block_allocator()->free_sequence(req->block_table());
        }
        try {
            req->result_promise.set_exception(std::make_exception_ptr(std::runtime_error(error_msg)));
        } catch (...) {}
    };

    for (auto* req : batch.decode_requests) { fail_request(req); }
    for (auto* req : batch.prefill_requests) { fail_request(req); }

    // Remove completed requests from scheduler's active list
    // (scheduler expects process_results to clean up, but we're bypassing it)
    scheduler_.cleanup_failed_requests();
}

} // namespace zedinfer
