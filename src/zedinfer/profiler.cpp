#include "zedinfer/profiler.hpp"
#include "zedinfer/engine.hpp"
#include "backend/kvcache/dynamic.hpp"
#include "frontend/models/forward_config.hpp"
#include "frontend/models/forward_context.hpp"
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

    const auto &mc = engine_->model().config();

    kvcache::DynamicKVCacheConfig kv_config;
    kv_config.num_layers = mc.num_hidden_layers;
    kv_config.num_kv_heads = mc.num_key_value_heads;
    kv_config.head_dim = mc.hidden_size / mc.num_attention_heads;
    kv_config.device_type = engine_->exec_config().device_type;
    kv_config.device_id = engine_->exec_config().device_id;
    kv_config.dtype = utils::str_to_dtype(mc.torch_dtype);
    kv_config.initial_capacity = prefill_len + decode_steps;
    kv_config.model_max_seq_len = engine_->tokenizer().get_config().model_max_length;

    auto tmp_kv = kvcache::DynamicKVCache::create(kv_config);
    auto fwd_cfg = engine_->model().forward_config();

    int min_id = 100, max_id = mc.vocab_size - 100;
    std::vector<int> dummy(prefill_len);
    for (auto &t : dummy) t = utils::randint(min_id, max_id);

    int past_len = 0;
    {
        model::ContiguousForwardContext ctx(dummy, past_len, *tmp_kv);
        auto logits = model::transformer_forward(fwd_cfg, ctx, engine_->exec_config());
        int next = engine_->sampler().sample(logits);
        past_len += prefill_len;

        for (size_t i = 1; i < decode_steps; ++i) {
            std::vector<int> tok = {next};
            model::ContiguousForwardContext dctx(tok, past_len, *tmp_kv);
            logits = model::transformer_forward(fwd_cfg, dctx, engine_->exec_config());
            next = engine_->sampler().sample(logits);
            past_len++;
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    LOGI << "[Profiler] Warmup complete in "
         << std::chrono::duration<double, std::milli>(t1 - t0).count() << " ms";
}

std::pair<double, double> Profiler::profile(size_t prefill_len, size_t decode_steps) {
    LOGI << "[Profiler] Profiling with prefill_len=" << prefill_len
         << ", decode_steps=" << decode_steps;

    const auto &mc = engine_->model().config();

    kvcache::DynamicKVCacheConfig kv_config;
    kv_config.num_layers = mc.num_hidden_layers;
    kv_config.num_kv_heads = mc.num_key_value_heads;
    kv_config.head_dim = mc.hidden_size / mc.num_attention_heads;
    kv_config.device_type = engine_->exec_config().device_type;
    kv_config.device_id = engine_->exec_config().device_id;
    kv_config.dtype = utils::str_to_dtype(mc.torch_dtype);
    kv_config.initial_capacity = prefill_len + decode_steps;
    kv_config.model_max_seq_len = engine_->tokenizer().get_config().model_max_length;

    auto tmp_kv = kvcache::DynamicKVCache::create(kv_config);
    auto fwd_cfg = engine_->model().forward_config();

    int min_id = 100, max_id = mc.vocab_size - 100;
    std::vector<int> dummy(prefill_len);
    for (auto &t : dummy) t = utils::randint(min_id, max_id);

    auto p0 = std::chrono::high_resolution_clock::now();
    int past_len = 0;
    tensor_t logits;
    int next;
    {
        model::ContiguousForwardContext ctx(dummy, past_len, *tmp_kv);
        logits = model::transformer_forward(fwd_cfg, ctx, engine_->exec_config());
        next = engine_->sampler().sample(logits);
    }
    auto p1 = std::chrono::high_resolution_clock::now();

    past_len += prefill_len;

    auto d0 = std::chrono::high_resolution_clock::now();
    for (size_t i = 1; i < decode_steps; ++i) {
        std::vector<int> tok = {next};
        model::ContiguousForwardContext ctx(tok, past_len, *tmp_kv);
        logits = model::transformer_forward(fwd_cfg, ctx, engine_->exec_config());
        next = engine_->sampler().sample(logits);
        past_len++;
    }
    auto d1 = std::chrono::high_resolution_clock::now();

    return {
        std::chrono::duration<double, std::milli>(p1 - p0).count(),
        std::chrono::duration<double, std::milli>(d1 - d0).count()};
}

} // namespace zedinfer
