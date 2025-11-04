#include "neollm/engine.hpp"
#include "backend/device/device.hpp"
#include "backend/kvcache/base.hpp"
#include "backend/kvcache/dynamic.hpp"
#include "frontend/graph/builder.hpp"
#include "frontend/sampler/sampler.hpp"
#include "frontend/tokenizer/hf_tokenizer.hpp"
#include "neollm/activation.hpp"
#include "neollm/session.hpp"
#include "utils/logging.hpp"
#include "utils/types.hpp"

#include <chrono>
#include <memory>
#include <plog/Log.h>
#include <stdexcept>

namespace neollm {

// ============================================================================
// InferenceEngine
// ============================================================================

InferenceEngine::InferenceEngine(
    std::shared_ptr<model::Model> model,
    graph::compute_graph_t graph,
    std::shared_ptr<GraphExecutor> executor,
    std::shared_ptr<tokenizer::Tokenizer> tokenizer,
    std::shared_ptr<sampler::Sampler> sampler,
    device::Device device)
    : model_(std::move(model)),
      graph_(std::move(graph)),
      executor_(std::move(executor)),
      tokenizer_(std::move(tokenizer)),
      sampler_(std::move(sampler)),
      device_(device) {}

std::shared_ptr<InferenceEngine> InferenceEngine::create(
    const std::string &model_path,
    device::Device device,
    size_t max_prefill_len) {

    if (model_path.empty()) {
        throw std::invalid_argument("Model path cannot be empty");
    }

    LOGI << "[Engine] Loading model from: " << model_path;

    // Load model configuration and weights
    auto model = model::Model::parse(model_path, device.type());
    if (!model) {
        throw std::runtime_error("Failed to parse model from: " + model_path);
    }
    LOGI << "[Engine] Model loaded: " << model->model_type();

    // Load tokenizer
    auto tokenizer = tokenizer::HFTokenizer::create(model_path + "/tokenizer.json");
    if (!tokenizer) {
        throw std::runtime_error("Failed to load tokenizer");
    }

    // Build computation graph based on model architecture
    LOGI << "[Engine] Building computation graph...";

    auto builder = graph::GraphBuilder::create(model->model_type());
    auto graph = builder->build(model.get());

    LOGI << "[Engine] Computation graph built";

    // Configure and create executor
    ExecutorConfig exec_config;
    exec_config.device_type = device.type();
    exec_config.device_id = device.id();
    exec_config.data_type = utils::str_to_dtype(model->config().torch_dtype);
    exec_config.max_prefill_len = max_prefill_len;
    exec_config.max_seq_len = tokenizer->get_config().model_max_length;

    auto executor = GraphExecutor::create(graph, exec_config);
    LOGI << "[Engine] Executor created";

    // Create sampler (default: argmax)
    auto sampler = sampler::createSampler(sampler::SamplerType::ARGMAX);

    // Construct engine instance
    auto engine = std::shared_ptr<InferenceEngine>(
        new InferenceEngine(
            std::move(model),
            std::move(graph),
            std::move(executor),
            std::move(tokenizer),
            std::move(sampler),
            device));

    LOGI << "[Engine] Initialization complete";

    // Warmup engine with reasonable defaults
    LOG_VERBOSE_(utils::BOTH) << "[Engine] Performing warmup...";
    engine->warmup(std::min(max_prefill_len, size_t(32)), 8);
    LOG_VERBOSE_(utils::BOTH) << "[Engine] Ready";

    return engine;
}

std::unique_ptr<InferenceSession> InferenceEngine::create_session(
    const GenerationConfig &config) {

    // Configure KV cache for this session
    kvcache::DynamicKVCacheConfig kv_config;
    const auto &model_config = model_->config();

    kv_config.num_layers = model_config.num_hidden_layers;
    kv_config.num_kv_heads = model_config.num_key_value_heads;
    kv_config.head_dim = model_config.hidden_size / model_config.num_attention_heads;
    kv_config.device_type = device_.type();
    kv_config.device_id = device_.id();
    kv_config.dtype = utils::str_to_dtype(model_config.torch_dtype);
    kv_config.model_max_seq_len = tokenizer_->get_config().model_max_length;

    // Create independent KV cache for session
    auto kv_cache = kvcache::DynamicKVCache::create_dynamic_kvcache(kv_config);

    LOGI << "[Session] KV cache initialized: "
         << "capacity=" << kv_cache->allocated_capacity()
         << ", memory=" << kv_cache->memory_usage() / (1024.0 * 1024.0) << " MB";

    // Create session with shared engine reference
    return std::unique_ptr<InferenceSession>(
        new InferenceSession(shared_from_this(), std::move(kv_cache), config));
}

std::string InferenceEngine::generate(
    kvcache::KVCache &kvcache,
    const std::string &prompt,
    const GenerationConfig &config) {

    config.validate();

    // Log prompt if verbose
    if (config.verbose) {
        if (config.gen_mode == GenerationMode::PING) {
            LOG_INFO_(utils::BOTH) << "\n=== [Inference] Prompt: ===\n"
                                   << prompt;
        } else {
            LOGI << "\n=== [Inference] Prompt: ===\n"
                 << prompt;
        }
        LOGI << "[Inference] Encoding prompt...";
    }

    // Tokenize input
    auto input_ids = tokenizer_->encode(prompt);

    if (config.verbose) {
        LOGI << "[Inference] Prompt tokens: " << input_ids.size();
        LOGI << "[Inference] Generating...";
    }

    // Generate output tokens
    auto output_ids = generate_tokens(kvcache, input_ids, config);

    // Decode to text
    std::string output = tokenizer_->decode(output_ids);

    // Log output if verbose
    if (config.verbose) {
        LOGI << "[Inference] Generation complete";
        if (config.gen_mode == GenerationMode::PING) {
            LOG_INFO_(utils::BOTH) << "\n=== [Inference] Generated ===\n"
                                   << output;
        } else {
            LOGI << "\n=== [Inference] Generated ===\n"
                 << output;
        }
    }

    // Print statistics if requested
    if (config.print_stats) {
        LOGI << last_stats_.summary();
    }

    return output;
}

std::vector<int> InferenceEngine::generate_tokens(
    kvcache::KVCache &kvcache,
    const std::vector<int> &input_ids,
    const GenerationConfig &config) {

    // Initialize statistics
    last_stats_ = GenerationStats();
    last_stats_.prompt_tokens = input_ids.size();

    std::vector<int> generated_ids;
    generated_ids.reserve(config.max_new_tokens);

    // ===== Prefill Phase =====
    // Process all input tokens in parallel
    auto prefill_start = std::chrono::high_resolution_clock::now();

    int past_len = kvcache.current_length();
    tensor_t logits = executor_->forward(kvcache, input_ids, past_len);
    int next_token = sampler_->sample(logits);

    auto prefill_end = std::chrono::high_resolution_clock::now();
    double prefill_time = std::chrono::duration<double, std::milli>(
                              prefill_end - prefill_start)
                              .count();
    update_stats_prefill(prefill_time, input_ids.size());

    generated_ids.push_back(next_token);

    // Stream first token if enabled
    if (config.stream && config.stream_callback) {
        std::string token_text = tokenizer_->decode({next_token});
        config.stream_callback(token_text);
    }

    // Check for early termination
    if (should_stop(next_token)) {
        last_stats_.generated_tokens = generated_ids.size();
        last_stats_.total_tokens = last_stats_.prompt_tokens + last_stats_.generated_tokens;
        return generated_ids;
    }

    // ===== Decode Phase =====
    // Generate tokens autoregressively one at a time
    past_len += input_ids.size();

    for (int i = 1; i < config.max_new_tokens; ++i) {
        auto step_start = std::chrono::high_resolution_clock::now();

        // Forward pass for single token
        logits = executor_->forward(kvcache, {next_token}, past_len);
        next_token = sampler_->sample(logits);

        auto step_end = std::chrono::high_resolution_clock::now();
        double step_time = std::chrono::duration<double, std::milli>(
                               step_end - step_start)
                               .count();
        update_stats_decode(step_time);

        generated_ids.push_back(next_token);
        past_len++;

        // Stream token if enabled
        if (config.stream && config.stream_callback) {
            std::string token_text = tokenizer_->decode({next_token});
            config.stream_callback(token_text);
        }

        // Check stopping conditions
        if (should_stop(next_token)) {
            break;
        }

        // Check sequence length limit
        if (past_len >= tokenizer_->get_config().model_max_length) {
            if (config.verbose) {
                LOGI << "[Inference] Reached max sequence length";
            }
            break;
        }
    }

    // Finalize statistics
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

    auto warmup_start = std::chrono::high_resolution_clock::now();

    // Create temporary KV cache for warmup
    kvcache::DynamicKVCacheConfig kv_config;
    const auto &model_config = model_->config();

    kv_config.num_layers = model_config.num_hidden_layers;
    kv_config.num_kv_heads = model_config.num_key_value_heads;
    kv_config.head_dim = model_config.hidden_size / model_config.num_attention_heads;
    kv_config.device_type = device_.type();
    kv_config.device_id = device_.id();
    kv_config.dtype = utils::str_to_dtype(model_config.torch_dtype);
    kv_config.initial_capacity = prefill_len + decode_steps;
    kv_config.model_max_seq_len = tokenizer_->get_config().model_max_length;

    auto temp_kvcache = kvcache::DynamicKVCache::create_dynamic_kvcache(kv_config);

    // Generate dummy input tokens
    std::vector<int> dummy_input(prefill_len, 1); // Use token_id=1 as dummy

    // Warmup prefill phase
    int past_len = 0;
    tensor_t logits = executor_->forward(*temp_kvcache, dummy_input, past_len);
    int next_token = sampler_->sample(logits);

    past_len += prefill_len;

    // Warmup decode phase
    for (size_t i = 1; i < decode_steps; ++i) {
        logits = executor_->forward(*temp_kvcache, {next_token}, past_len);
        next_token = sampler_->sample(logits);
        past_len++;
    }

    auto warmup_end = std::chrono::high_resolution_clock::now();
    double warmup_time = std::chrono::duration<double, std::milli>(
                             warmup_end - warmup_start)
                             .count();

    LOGI << "[Engine] Warmup complete in " << warmup_time << " ms";
}

} // namespace neollm