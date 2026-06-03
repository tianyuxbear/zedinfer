#include "zedinfer/profiler.hpp"
#include "backend/kvcache/block_pool.hpp"
#include "frontend/models/decode_scratch.hpp"
#include "frontend/models/forward_config.hpp"
#include "frontend/models/hybrid_transformer_forward.hpp"
#include "frontend/models/paged_forward_context.hpp"
#include "frontend/models/qwen3_5.hpp"
#include "frontend/models/qwen3_5_moe.hpp"
#include "frontend/models/ssm_state_pool.hpp"
#include "utils/logging.hpp"
#include "utils/random.hpp"
#include "utils/types.hpp"
#include "zedinfer/engine.hpp"
#include "zedinfer/request.hpp"

#include <chrono>
#include <plog/Log.h>
#include <stdexcept>
#include <vector>

namespace zedinfer {

namespace {

class ScopedSequence {
public:
    ScopedSequence(kvcache::BlockAllocator* allocator, int estimated_tokens)
        : allocator_(allocator), table_(allocator->allocate_sequence(estimated_tokens)) {}

    ~ScopedSequence() {
        if (allocator_) {
            allocator_->free_sequence(table_);
        }
    }

    ScopedSequence(const ScopedSequence&) = delete;
    ScopedSequence& operator=(const ScopedSequence&) = delete;

    kvcache::SequenceBlockTable& table() { return table_; }

private:
    kvcache::BlockAllocator* allocator_ = nullptr;
    kvcache::SequenceBlockTable table_;
};

class ScopedSSMSlot {
public:
    explicit ScopedSSMSlot(model::SSMStatePool* pool) : pool_(pool) {
        if (!pool_) {
            throw std::runtime_error("[Profiler] hybrid model has no SSMStatePool");
        }
        slot_ = pool_->acquire_slot();
        pool_->reset_slot(slot_);
    }

    ~ScopedSSMSlot() {
        if (pool_ && slot_ >= 0) {
            pool_->release_slot(slot_);
        }
    }

    ScopedSSMSlot(const ScopedSSMSlot&) = delete;
    ScopedSSMSlot& operator=(const ScopedSSMSlot&) = delete;

    int slot() const { return slot_; }

private:
    model::SSMStatePool* pool_ = nullptr;
    int slot_ = -1;
};

model::HybridForwardConfig make_hybrid_config(const model::Qwen3_5Model& model) {
    if (auto* moe_model = dynamic_cast<const model::Qwen3_5MoeModel*>(&model)) {
        return moe_model->hybrid_forward_config_moe();
    }
    return model.hybrid_forward_config();
}

std::vector<int> make_dummy_tokens(size_t num_tokens, int vocab_size) {
    int min_id = 100, max_id = vocab_size - 100;
    std::vector<int> dummy(num_tokens);
    for (auto& t : dummy) { t = utils::randint(min_id, max_id); }
    return dummy;
}

template <typename ForwardFn>
void run_warmup_sequence(InferenceEngine& engine, size_t prefill_len, size_t decode_steps, ForwardFn&& forward) {
    auto* allocator = engine.block_allocator();
    auto* pool = engine.block_pool();
    auto* scratch = engine.decode_scratch();
    const auto& mc = engine.model().config();

    int estimated_tokens = static_cast<int>(prefill_len + decode_steps);
    ScopedSequence seq(allocator, estimated_tokens);

    auto dummy = make_dummy_tokens(prefill_len, mc.vocab_size);

    int past_len = 0;
    allocator->ensure_blocks(seq.table(), past_len + static_cast<int>(prefill_len));
    {
        model::PagedForwardContext ctx(dummy, past_len, seq.table(), *pool);
        auto logits = forward(ctx, nullptr);
        int next = engine.sampler().sample(logits);
        seq.table().seq_len += static_cast<int>(prefill_len);
        past_len += static_cast<int>(prefill_len);

        for (size_t i = 1; i < decode_steps; ++i) {
            std::vector<int> tok = {next};
            allocator->ensure_blocks(seq.table(), past_len + 1);
            model::PagedForwardContext dctx(tok, past_len, seq.table(), *pool);
            logits = forward(dctx, scratch);
            next = engine.sampler().sample(logits);
            seq.table().seq_len += 1;
            past_len++;
        }
    }
}

template <typename ForwardFn>
std::pair<double, double> run_profile_sequence(InferenceEngine& engine, size_t prefill_len, size_t decode_steps,
                                               ForwardFn&& forward) {
    auto* allocator = engine.block_allocator();
    auto* pool = engine.block_pool();
    auto* scratch = engine.decode_scratch();
    const auto& mc = engine.model().config();

    int estimated_tokens = static_cast<int>(prefill_len + decode_steps);
    ScopedSequence seq(allocator, estimated_tokens);

    auto dummy = make_dummy_tokens(prefill_len, mc.vocab_size);

    int past_len = 0;
    allocator->ensure_blocks(seq.table(), past_len + static_cast<int>(prefill_len));

    auto p0 = std::chrono::high_resolution_clock::now();
    tensor_t logits;
    int next;
    {
        model::PagedForwardContext ctx(dummy, past_len, seq.table(), *pool);
        logits = forward(ctx, nullptr);
        next = engine.sampler().sample(logits);
    }
    auto p1 = std::chrono::high_resolution_clock::now();

    seq.table().seq_len += static_cast<int>(prefill_len);
    past_len += static_cast<int>(prefill_len);

    auto d0 = std::chrono::high_resolution_clock::now();
    for (size_t i = 1; i < decode_steps; ++i) {
        std::vector<int> tok = {next};
        allocator->ensure_blocks(seq.table(), past_len + 1);
        model::PagedForwardContext ctx(tok, past_len, seq.table(), *pool);
        logits = forward(ctx, scratch);
        next = engine.sampler().sample(logits);
        seq.table().seq_len += 1;
        past_len++;
    }
    auto d1 = std::chrono::high_resolution_clock::now();

    return {std::chrono::duration<double, std::milli>(p1 - p0).count(),
            std::chrono::duration<double, std::milli>(d1 - d0).count()};
}

} // namespace

Profiler::Profiler(InferenceEngine& engine) : engine_(&engine) {}

void Profiler::warmup(size_t prefill_len, size_t decode_steps) {
    LOGI << "[Profiler] Warming up with prefill_len=" << prefill_len << ", decode_steps=" << decode_steps;

    auto t0 = std::chrono::high_resolution_clock::now();

    if (auto* hybrid_model = dynamic_cast<const model::Qwen3_5Model*>(&engine_->model())) {
        auto hcfg = make_hybrid_config(*hybrid_model);
        ScopedSSMSlot ssm_slot(engine_->ssm_state_pool());
        InferenceRequest req;
        req.set_ssm_slot_idx(ssm_slot.slot());
        auto forward = [&](model::PagedForwardContext& ctx, model::DecodeScratch* scratch) {
            return model::hybrid_transformer_forward(hcfg, ctx, req, engine_->exec_config(), scratch);
        };
        run_warmup_sequence(*engine_, prefill_len, decode_steps, forward);
    } else {
        auto fwd_cfg = engine_->model().forward_config();
        auto forward = [&](model::PagedForwardContext& ctx, model::DecodeScratch* scratch) {
            return model::transformer_forward(fwd_cfg, ctx, engine_->exec_config(), scratch);
        };
        run_warmup_sequence(*engine_, prefill_len, decode_steps, forward);
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    LOGI << "[Profiler] Warmup complete in " << std::chrono::duration<double, std::milli>(t1 - t0).count() << " ms";
}

std::pair<double, double> Profiler::profile(size_t prefill_len, size_t decode_steps) {
    LOGI << "[Profiler] Profiling with prefill_len=" << prefill_len << ", decode_steps=" << decode_steps;

    if (auto* hybrid_model = dynamic_cast<const model::Qwen3_5Model*>(&engine_->model())) {
        auto hcfg = make_hybrid_config(*hybrid_model);
        ScopedSSMSlot ssm_slot(engine_->ssm_state_pool());
        InferenceRequest req;
        req.set_ssm_slot_idx(ssm_slot.slot());
        auto forward = [&](model::PagedForwardContext& ctx, model::DecodeScratch* scratch) {
            return model::hybrid_transformer_forward(hcfg, ctx, req, engine_->exec_config(), scratch);
        };
        return run_profile_sequence(*engine_, prefill_len, decode_steps, forward);
    } else {
        auto fwd_cfg = engine_->model().forward_config();
        auto forward = [&](model::PagedForwardContext& ctx, model::DecodeScratch* scratch) {
            return model::transformer_forward(fwd_cfg, ctx, engine_->exec_config(), scratch);
        };
        return run_profile_sequence(*engine_, prefill_len, decode_steps, forward);
    }
}

} // namespace zedinfer
