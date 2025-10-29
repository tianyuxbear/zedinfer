#include "neollm/engine.hpp"
#include "frontend/sampler/sampler.hpp"
#include "frontend/tokenizer/hf_tokenizer.hpp"
#include "graph/builder.hpp"
#include "neollm.h"
#include "utils/types.hpp"
#include <iostream>
#include <sstream>

namespace neollm {

// ============================================================================
// GenerationConfig Implementation
// ============================================================================

void GenerationConfig::validate() const {
    if (max_new_tokens <= 0) {
        throw std::invalid_argument("max_new_tokens must be positive");
    }
    if (max_seq_len <= 0) {
        throw std::invalid_argument("max_seq_len must be positive");
    }
    if (!sampler_params.validate()) {
        throw std::invalid_argument("Invalid sampler parameters");
    }
}

std::string GenerationConfig::info() const {
    std::ostringstream oss;
    oss << "GenerationConfig:\n"
        << "  max_new_tokens: " << max_new_tokens << "\n"
        << "  max_seq_len: " << max_seq_len << "\n"
        << "  sampler_type: " << sampler::to_string(sampler_type) << "\n"
        << "  sampler_params: " << sampler_params.info() << "\n"
        << "  stream: " << (stream ? "true" : "false") << "\n"
        << "  verbose: " << (verbose ? "true" : "false");
    return oss.str();
}

// ============================================================================
// GenerationStats Implementation
// ============================================================================

std::string GenerationStats::summary() const {
    std::ostringstream oss;
    oss << "\n=== Generation Statistics ===\n"
        << "Prompt tokens: " << prompt_tokens << "\n"
        << "Generated tokens: " << generated_tokens << "\n"
        << "Total tokens: " << total_tokens << "\n"
        << "Prefill time: " << prefill_time_ms << " ms "
        << "(" << prefill_tokens_per_second() << " tokens/s)\n"
        << "Decode time: " << decode_time_ms << " ms "
        << "(" << decode_tokens_per_second() << " tokens/s)\n"
        << "Total time: " << total_time_ms << " ms\n"
        << "============================";
    return oss.str();
}

// ============================================================================
// InferenceEngine Implementation
// ============================================================================

InferenceEngine::InferenceEngine(
    std::unique_ptr<model::Model> model,
    std::unique_ptr<tokenizer::Tokenizer> tokenizer,
    graph::compute_graph_t graph,
    kvcache::kvcache_t kv_cache,
    std::unique_ptr<graph::GraphExecutor> executor,
    const GenerationConfig &gen_config)
    : model_(std::move(model)),
      tokenizer_(std::move(tokenizer)),
      graph_(graph),
      kv_cache_(kv_cache),
      executor_(std::move(executor)),
      gen_config_(gen_config) {

    device_type_ = kv_cache_->config().device_type;
    device_id_ = kv_cache_->config().device_id;
    dtype_ = kv_cache_->config().dtype;
}

std::shared_ptr<InferenceEngine> InferenceEngine::create(
    const std::string &model_path,
    NeollmDeviceType_t device_type,
    int device_id) {

    return InferenceEngineBuilder()
        .set_model_path(model_path)
        .set_device(device_type, device_id)
        .build();
}

std::string InferenceEngine::chat(
    const std::vector<std::pair<std::string, std::string>> &messages,
    const GenerationConfig &config) {

    // Apply chat template
    auto *hf_tokenizer = dynamic_cast<tokenizer::HFTokenizer *>(tokenizer_.get());
    if (!hf_tokenizer) {
        throw std::runtime_error("Chat mode requires HFTokenizer");
    }

    std::string formatted_prompt = hf_tokenizer->apply_chat_template(messages, true);

    if (config.verbose) {
        std::cout << "[Chat] Formatted prompt:\n"
                  << formatted_prompt << std::endl;
    }

    return generate(formatted_prompt, config);
}

std::string InferenceEngine::generate(
    const std::string &prompt,
    const GenerationConfig &config) {

    config.validate();

    if (config.verbose) {
        std::cout << "\n[Inference] Encoding prompt..." << std::endl;
    }

    // Encode prompt
    auto input_ids = tokenizer_->encode(prompt);

    if (config.verbose) {
        std::cout << "[Inference] Prompt tokens: " << input_ids.size() << std::endl;
        std::cout << "[Inference] Generating..." << std::endl;
    }

    // Generate tokens
    auto output_ids = generate_tokens(input_ids, config);

    // Decode output
    std::string output = tokenizer_->decode(output_ids);

    if (config.verbose) {
        std::cout << "[Inference] Generation complete!" << std::endl;
    }

    if (config.print_stats) {
        std::cout << last_stats_.summary() << std::endl;
    }

    return output;
}

std::vector<int> InferenceEngine::generate_tokens(
    const std::vector<int> &input_ids,
    const GenerationConfig &config) {

    // Reset stats
    last_stats_ = GenerationStats();
    last_stats_.prompt_tokens = input_ids.size();

    // Create sampler
    auto sampler = create_sampler(config);

    // Reset KV cache
    reset();

    std::vector<int> generated_ids;
    generated_ids.reserve(config.max_new_tokens);

    // Prefill phase
    auto prefill_start = std::chrono::high_resolution_clock::now();

    tensor_t logits = executor_->forward(input_ids, 0);
    int next_token = sampler->sample(logits);

    auto prefill_end = std::chrono::high_resolution_clock::now();
    double prefill_time = std::chrono::duration<double, std::milli>(
                              prefill_end - prefill_start)
                              .count();
    update_stats_prefill(prefill_time, input_ids.size());

    generated_ids.push_back(next_token);

    if (config.stream && config.stream_callback) {
        std::string token_text = tokenizer_->decode({next_token});
        config.stream_callback(token_text);
    }

    if (should_stop(next_token, config)) {
        last_stats_.generated_tokens = generated_ids.size();
        last_stats_.total_tokens = last_stats_.prompt_tokens + last_stats_.generated_tokens;
        return generated_ids;
    }

    // Decode phase
    int past_len = input_ids.size();
    for (int i = 1; i < config.max_new_tokens; ++i) {
        auto step_start = std::chrono::high_resolution_clock::now();

        // Forward pass
        logits = executor_->forward({next_token}, past_len);
        next_token = sampler->sample(logits);

        auto step_end = std::chrono::high_resolution_clock::now();
        double step_time = std::chrono::duration<double, std::milli>(
                               step_end - step_start)
                               .count();
        update_stats_decode(step_time);

        generated_ids.push_back(next_token);
        past_len++;

        if (config.stream && config.stream_callback) {
            std::string token_text = tokenizer_->decode({next_token});
            config.stream_callback(token_text);
        }

        if (config.verbose && (i % 10 == 0)) {
            std::cout << "[Inference] Generated " << i << " tokens..." << std::endl;
        }

        if (should_stop(next_token, config)) {
            break;
        }

        if (past_len >= config.max_seq_len) {
            if (config.verbose) {
                std::cout << "[Inference] Reached max sequence length" << std::endl;
            }
            break;
        }
    }

    last_stats_.generated_tokens = generated_ids.size();
    last_stats_.total_tokens = last_stats_.prompt_tokens + last_stats_.generated_tokens;

    return generated_ids;
}

void InferenceEngine::reset() {
    kv_cache_->reset();
}

void InferenceEngine::print_info() const {
    std::cout << "\n=== Inference Engine Info ===\n";
    std::cout << "Model: " << model_->model_type() << "\n";
    std::cout << "Parameters: " << model_->num_parameters() << "\n";
    std::cout << "Vocabulary size: " << tokenizer_->get_vocab_size() << "\n";

    const char *device_str = (device_type_ == NEOLLM_DEVICE_CPU) ? "CPU" : "CUDA";
    std::cout << "Device: " << device_str << ":" << device_id_ << "\n";

    const char *dtype_str = (dtype_ == NEOLLM_DTYPE_BF16) ? "BF16" : (dtype_ == NEOLLM_DTYPE_F16) ? "FP16"
                                                                                                  : "FP32";
    std::cout << "Data type: " << dtype_str << "\n";

    std::cout << "KV Cache capacity: " << kv_cache_->allocated_capacity() << "\n";
    std::cout << "KV Cache memory: " << kv_cache_->memory_usage() / (1024.0 * 1024.0)
              << " MB\n";
    std::cout << "============================\n"
              << std::endl;
}

bool InferenceEngine::should_stop(int token_id, const GenerationConfig &config) const {
    // Check EOS token
    if (token_id == tokenizer_->get_eos_token_id()) {
        return true;
    }

    // Check additional stop tokens
    for (int stop_id : config.stop_token_ids) {
        if (token_id == stop_id) {
            return true;
        }
    }

    return false;
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

std::shared_ptr<sampler::Sampler> InferenceEngine::create_sampler(
    const GenerationConfig &config) const {
    return sampler::createSampler(config.sampler_type, config.sampler_params);
}

// ============================================================================
// InferenceEngineBuilder Implementation
// ============================================================================

std::unique_ptr<InferenceEngine> InferenceEngineBuilder::build() {
    if (model_path_.empty()) {
        throw std::invalid_argument("Model path not set");
    }

    std::cout << "[Builder] Loading model from: " << model_path_ << std::endl;

    // Load model
    auto model = model::Model::parse(model_path_, NEOLLM_DEVICE_CPU);
    if (!model) {
        throw std::runtime_error("Failed to parse model");
    }

    std::cout << "[Builder] Model loaded: " << model->model_type() << std::endl;

    // Load tokenizer
    auto tokenizer = tokenizer::HFTokenizer::create(model_path_ + "/tokenizer.json");
    if (!tokenizer) {
        throw std::runtime_error("Failed to load tokenizer");
    }

    std::cout << "[Builder] Tokenizer loaded, vocab size: "
              << tokenizer->get_vocab_size() << std::endl;

    // Setup KV cache config
    kvcache::DynamicKVCacheConfig kv_config;
    const auto &model_config = model->config();
    kv_config.num_layers = model_config.num_hidden_layers;
    kv_config.num_kv_heads = model_config.num_key_value_heads;
    kv_config.head_dim = model_config.hidden_size / model_config.num_attention_heads;
    kv_config.initial_capacity = 256;
    kv_config.model_max_seq_len = std::min(8192ul, model_config.max_position_embeddings);
    kv_config.device_type = device_type_;
    kv_config.device_id = device_id_;
    kv_config.dtype = utils::str_to_dtype(model_config.torch_dtype);

    // Create KV cache
    auto kv_cache = kvcache::DynamicKVCacheManager::create_dynamic_kvcache(kv_config);

    std::cout << "[Builder] KV Cache initialized: "
              << "capacity=" << kv_cache->allocated_capacity()
              << ", memory=" << kv_cache->memory_usage() / (1024.0 * 1024.0) << "MB"
              << std::endl;

    // Build computation graph
    std::cout << "[Builder] Building computation graph..." << std::endl;

    graph::GraphBuilder *builder = nullptr;
    if (model->model_type() == "qwen2") {
        builder = new graph::Qwen2GraphBuilder();
    } else {
        throw std::runtime_error("Unsupported model type: " + model->model_type());
    }

    auto graph = builder->build(model.get());

    std::cout << "[Builder] Computation graph built" << std::endl;

    // Create executor
    auto executor = graph::GraphExecutor::create(graph, kv_cache);

    std::cout << "[Builder] Graph executor created" << std::endl;

    // Create engine using unique_ptr and new
    auto engine = std::unique_ptr<InferenceEngine>(
        new InferenceEngine(
            std::move(model),
            std::move(tokenizer),
            graph,
            kv_cache,
            std::move(executor)));

    std::cout << "[Builder] Inference engine ready!" << std::endl;

    return engine;
}

} // namespace neollm