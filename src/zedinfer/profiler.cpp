#include "zedinfer/profiler.hpp"
#include "zedinfer/engine.hpp"
#include "backend/kvcache/block_pool.hpp"
#include "frontend/models/forward_config.hpp"
#include "frontend/models/decode_scratch.hpp"
#include "frontend/models/paged_forward_context.hpp"
#include "utils/logging.hpp"
#include "utils/random.hpp"
#include "utils/types.hpp"

#include <chrono>
#include <plog/Log.h>

namespace zedinfer {

Profiler::Profiler(std::shared_ptr<InferenceEngine> engine)
    : engine_(std::move(engine)) {}

void Profiler::warmup(size_t prefill_len, size_t decode_steps) {
    LOGI << "[Profiler] Warming up with prefill_len=" << prefill_len
         << ", decode_steps=" << decode_steps;

    auto t0 = std::chrono::high_resolution_clock::now();

    auto *allocator = engine_->block_allocator();
    auto *pool = engine_->block_pool();
    auto *scratch = engine_->decode_scratch();
    const auto &mc = engine_->model().config();
    auto fwd_cfg = engine_->model().forward_config();

    int estimated_tokens = static_cast<int>(prefill_len + decode_steps);
    auto block_table = allocator->allocate_sequence(estimated_tokens);

    int min_id = 100, max_id = mc.vocab_size - 100;
    std::vector<int> dummy(prefill_len);
    for (auto &t : dummy) t = utils::randint(min_id, max_id);

    // Prefill (no scratch — N > 1)
    int past_len = 0;
    allocator->ensure_blocks(block_table, past_len + static_cast<int>(prefill_len));
    {
        model::PagedForwardContext ctx(dummy, past_len, block_table, *pool);
        auto logits = model::transformer_forward(fwd_cfg, ctx, engine_->exec_config());
        int next = engine_->sampler().sample(logits);
        block_table.seq_len += static_cast<int>(prefill_len);
        past_len += static_cast<int>(prefill_len);

        // Decode (with scratch — N == 1)
        for (size_t i = 1; i < decode_steps; ++i) {
            std::vector<int> tok = {next};
            allocator->ensure_blocks(block_table, past_len + 1);
            model::PagedForwardContext dctx(tok, past_len, block_table, *pool);
            logits = model::transformer_forward(fwd_cfg, dctx, engine_->exec_config(), scratch);
            next = engine_->sampler().sample(logits);
            block_table.seq_len += 1;
            past_len++;
        }
    }

    allocator->free_sequence(block_table);

    auto t1 = std::chrono::high_resolution_clock::now();
    LOGI << "[Profiler] Warmup complete in "
         << std::chrono::duration<double, std::milli>(t1 - t0).count() << " ms";
}

std::pair<double, double> Profiler::profile(size_t prefill_len, size_t decode_steps) {
    LOGI << "[Profiler] Profiling with prefill_len=" << prefill_len
         << ", decode_steps=" << decode_steps;

    auto *allocator = engine_->block_allocator();
    auto *pool = engine_->block_pool();
    auto *scratch = engine_->decode_scratch();
    const auto &mc = engine_->model().config();
    auto fwd_cfg = engine_->model().forward_config();

    int estimated_tokens = static_cast<int>(prefill_len + decode_steps);
    auto block_table = allocator->allocate_sequence(estimated_tokens);

    int min_id = 100, max_id = mc.vocab_size - 100;
    std::vector<int> dummy(prefill_len);
    for (auto &t : dummy) t = utils::randint(min_id, max_id);

    // Prefill (no scratch)
    int past_len = 0;
    allocator->ensure_blocks(block_table, past_len + static_cast<int>(prefill_len));

    auto p0 = std::chrono::high_resolution_clock::now();
    tensor_t logits;
    int next;
    {
        model::PagedForwardContext ctx(dummy, past_len, block_table, *pool);
        logits = model::transformer_forward(fwd_cfg, ctx, engine_->exec_config());
        next = engine_->sampler().sample(logits);
    }
    auto p1 = std::chrono::high_resolution_clock::now();

    block_table.seq_len += static_cast<int>(prefill_len);
    past_len += static_cast<int>(prefill_len);

    // Decode (with scratch)
    auto d0 = std::chrono::high_resolution_clock::now();
    for (size_t i = 1; i < decode_steps; ++i) {
        std::vector<int> tok = {next};
        allocator->ensure_blocks(block_table, past_len + 1);
        model::PagedForwardContext ctx(tok, past_len, block_table, *pool);
        logits = model::transformer_forward(fwd_cfg, ctx, engine_->exec_config(), scratch);
        next = engine_->sampler().sample(logits);
        block_table.seq_len += 1;
        past_len++;
    }
    auto d1 = std::chrono::high_resolution_clock::now();

    allocator->free_sequence(block_table);

    return {
        std::chrono::duration<double, std::milli>(p1 - p0).count(),
        std::chrono::duration<double, std::milli>(d1 - d0).count()};
}

} // namespace zedinfer
