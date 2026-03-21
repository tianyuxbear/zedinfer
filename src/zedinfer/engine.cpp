#include "zedinfer/engine.hpp"
#include "backend/kvcache/dynamic.hpp"
#include "frontend/sampler/sampler.hpp"
#include "frontend/tokenizer/hf_tokenizer.hpp"
#include "utils/logging.hpp"
#include "utils/random.hpp"
#include "utils/types.hpp"
#include "zedinfer/session.hpp"

#include <chrono>
#include <plog/Log.h>
#include <stdexcept>

namespace zedinfer {

InferenceEngine::InferenceEngine(
    std::shared_ptr<model::Model> model,
    std::shared_ptr<tokenizer::Tokenizer> tokenizer,
    std::shared_ptr<sampler::Sampler> sampler,
    device::Device device,
    ExecutorConfig exec_config)
    : model_(std::move(model)),
      tokenizer_(std::move(tokenizer)),
      sampler_(std::move(sampler)),
      device_(device),
      exec_config_(exec_config) {}

std::shared_ptr<InferenceEngine> InferenceEngine::create(
    const std::string &model_path,
    device::Device device) {

    if (model_path.empty()) {
        throw std::invalid_argument("Model path cannot be empty");
    }

    LOGI << "[Engine] Loading model from: " << model_path;

    auto model = model::Model::parse(model_path, device.type());
    if (!model) {
        throw std::runtime_error("Failed to parse model from: " + model_path);
    }
    LOGI << "[Engine] Model loaded: " << model->model_type();

    auto tokenizer = tokenizer::HFTokenizer::create(model_path + "/tokenizer.json");
    if (!tokenizer) {
        throw std::runtime_error("Failed to load tokenizer");
    }

    ExecutorConfig exec_config;
    exec_config.device_type = device.type();
    exec_config.device_id = device.id();
    exec_config.data_type = utils::str_to_dtype(model->config().torch_dtype);
    exec_config.max_seq_len = tokenizer->get_config().model_max_length;

    auto sampler = sampler::createSampler(exec_config, sampler::SamplerType::ARGMAX);

    auto engine = std::shared_ptr<InferenceEngine>(
        new InferenceEngine(
            std::move(model),
            std::move(tokenizer),
            std::move(sampler),
            device,
            exec_config));

    LOGI << "[Engine] Initialization complete";

    LOG_VERBOSE_(utils::BOTH) << "[Engine] Performing warmup...";
    engine->warmup();
    LOG_VERBOSE_(utils::BOTH) << "[Engine] Ready";

    return engine;
}

std::unique_ptr<InferenceSession> InferenceEngine::create_session(
    const GenerationConfig &config) {

    kvcache::DynamicKVCacheConfig kv_config;
    const auto &mc = model_->config();

    kv_config.num_layers = mc.num_hidden_layers;
    kv_config.num_kv_heads = mc.num_key_value_heads;
    kv_config.head_dim = mc.hidden_size / mc.num_attention_heads;
    kv_config.device_type = device_.type();
    kv_config.device_id = device_.id();
    kv_config.dtype = utils::str_to_dtype(mc.torch_dtype);
    kv_config.model_max_seq_len = tokenizer_->get_config().model_max_length;

    auto kv_cache = kvcache::DynamicKVCache::create_dynamic_kvcache(kv_config);

    LOGI << "[Session] KV cache initialized: "
         << "capacity=" << kv_cache->allocated_capacity()
         << ", memory=" << kv_cache->memory_usage() / (1024.0 * 1024.0) << " MB";

    return std::unique_ptr<InferenceSession>(
        new InferenceSession(shared_from_this(), std::move(kv_cache), config));
}

std::string InferenceEngine::generate(
    kvcache::KVCache &kvcache,
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

    auto input_ids = tokenizer_->encode(prompt);

    if (config.verbose) {
        LOGI << "[Inference] Prompt tokens: " << input_ids.size();
        LOGI << "[Inference] Generating...";
    }

    auto output_ids = generate_tokens(kvcache, input_ids, config);
    std::string output = tokenizer_->decode(output_ids);

    if (config.verbose) {
        LOGI << "[Inference] Generation complete";
        if (config.gen_mode == GenerationMode::PING) {
            LOG_INFO_(utils::BOTH) << "\n=== [Inference] Generated ===\n" << output;
        } else {
            LOGI << "\n=== [Inference] Generated ===\n" << output;
        }
    }

    if (config.print_stats) {
        LOGI << last_stats_.summary();
    }

    return output;
}

std::vector<int> InferenceEngine::generate_tokens(
    kvcache::KVCache &kvcache,
    const std::vector<int> &input_ids,
    const GenerationConfig &config) {

    last_stats_ = GenerationStats();
    last_stats_.prompt_tokens = input_ids.size();

    std::vector<int> generated_ids;
    generated_ids.reserve(config.max_new_tokens);

    // Prefill
    auto t0 = std::chrono::high_resolution_clock::now();

    int past_len = kvcache.current_length();
    tensor_t logits = model_->forward(input_ids, past_len, kvcache, exec_config_);
    int next_token = sampler_->sample(logits);

    auto t1 = std::chrono::high_resolution_clock::now();
    update_stats_prefill(
        std::chrono::duration<double, std::milli>(t1 - t0).count(),
        input_ids.size());

    generated_ids.push_back(next_token);

    if (config.stream && config.stream_callback) {
        config.stream_callback(tokenizer_->decode({next_token}));
    }

    if (should_stop(next_token)) {
        last_stats_.generated_tokens = generated_ids.size();
        last_stats_.total_tokens = last_stats_.prompt_tokens + last_stats_.generated_tokens;
        return generated_ids;
    }

    // Decode
    past_len += input_ids.size();

    for (int i = 1; i < config.max_new_tokens; ++i) {
        auto s0 = std::chrono::high_resolution_clock::now();

        logits = model_->forward({next_token}, past_len, kvcache, exec_config_);
        next_token = sampler_->sample(logits);

        auto s1 = std::chrono::high_resolution_clock::now();
        update_stats_decode(
            std::chrono::duration<double, std::milli>(s1 - s0).count());

        generated_ids.push_back(next_token);
        past_len++;

        if (config.stream && config.stream_callback) {
            config.stream_callback(tokenizer_->decode({next_token}));
        }

        if (should_stop(next_token)) break;

        if (past_len >= static_cast<int>(tokenizer_->get_config().model_max_length)) {
            if (config.verbose) {
                LOGI << "[Inference] Reached max sequence length";
            }
            break;
        }
    }

    last_stats_.generated_tokens = generated_ids.size();
    last_stats_.total_tokens = last_stats_.prompt_tokens + last_stats_.generated_tokens;
    return generated_ids;
}

bool InferenceEngine::should_stop(int token_id) const {
    return token_id == tokenizer_->get_eos_token_id();
}

void InferenceEngine::update_stats_prefill(double time_ms, int num_tokens) {
    last_stats_.prompt_tokens = num_tokens;
    last_stats_.prefill_time_ms = time_ms;
    last_stats_.total_time_ms += time_ms;
}

void InferenceEngine::update_stats_decode(double time_ms) {
    last_stats_.decode_time_ms += time_ms;
    last_stats_.total_time_ms += time_ms;
}

void InferenceEngine::warmup(size_t prefill_len, size_t decode_steps) {
    LOGI << "[Engine] Warming up with prefill_len=" << prefill_len
         << ", decode_steps=" << decode_steps;

    auto t0 = std::chrono::high_resolution_clock::now();

    kvcache::DynamicKVCacheConfig kv_config;
    const auto &mc = model_->config();

    kv_config.num_layers = mc.num_hidden_layers;
    kv_config.num_kv_heads = mc.num_key_value_heads;
    kv_config.head_dim = mc.hidden_size / mc.num_attention_heads;
    kv_config.device_type = device_.type();
    kv_config.device_id = device_.id();
    kv_config.dtype = utils::str_to_dtype(mc.torch_dtype);
    kv_config.initial_capacity = prefill_len + decode_steps;
    kv_config.model_max_seq_len = tokenizer_->get_config().model_max_length;

    auto tmp_kv = kvcache::DynamicKVCache::create_dynamic_kvcache(kv_config);

    int min_id = 100, max_id = mc.vocab_size - 100;
    std::vector<int> dummy(prefill_len);
    for (auto &t : dummy) t = utils::randint(min_id, max_id);

    int past_len = 0;
    auto logits = model_->forward(dummy, past_len, *tmp_kv, exec_config_);
    int next = sampler_->sample(logits);
    past_len += prefill_len;

    for (size_t i = 1; i < decode_steps; ++i) {
        logits = model_->forward({next}, past_len, *tmp_kv, exec_config_);
        next = sampler_->sample(logits);
        past_len++;
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    LOGI << "[Engine] Warmup complete in "
         << std::chrono::duration<double, std::milli>(t1 - t0).count() << " ms";
}

std::pair<double, double> InferenceEngine::profile(size_t prefill_len, size_t decode_steps) {
    LOGI << "[Engine] Profiling with prefill_len=" << prefill_len
         << ", decode_steps=" << decode_steps;

    kvcache::DynamicKVCacheConfig kv_config;
    const auto &mc = model_->config();

    kv_config.num_layers = mc.num_hidden_layers;
    kv_config.num_kv_heads = mc.num_key_value_heads;
    kv_config.head_dim = mc.hidden_size / mc.num_attention_heads;
    kv_config.device_type = device_.type();
    kv_config.device_id = device_.id();
    kv_config.dtype = utils::str_to_dtype(mc.torch_dtype);
    kv_config.initial_capacity = prefill_len + decode_steps;
    kv_config.model_max_seq_len = tokenizer_->get_config().model_max_length;

    auto tmp_kv = kvcache::DynamicKVCache::create_dynamic_kvcache(kv_config);

    int min_id = 100, max_id = mc.vocab_size - 100;
    std::vector<int> dummy(prefill_len);
    for (auto &t : dummy) t = utils::randint(min_id, max_id);

    auto p0 = std::chrono::high_resolution_clock::now();
    int past_len = 0;
    auto logits = model_->forward(dummy, past_len, *tmp_kv, exec_config_);
    int next = sampler_->sample(logits);
    auto p1 = std::chrono::high_resolution_clock::now();

    past_len += prefill_len;

    auto d0 = std::chrono::high_resolution_clock::now();
    for (size_t i = 1; i < decode_steps; ++i) {
        logits = model_->forward({next}, past_len, *tmp_kv, exec_config_);
        next = sampler_->sample(logits);
        past_len++;
    }
    auto d1 = std::chrono::high_resolution_clock::now();

    return {
        std::chrono::duration<double, std::milli>(p1 - p0).count(),
        std::chrono::duration<double, std::milli>(d1 - d0).count()};
}

} // namespace zedinfer
