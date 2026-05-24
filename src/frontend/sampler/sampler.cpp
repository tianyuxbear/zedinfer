#include "frontend/sampler/sampler.hpp"
#include "backend/core/context/context.hpp"
#include "backend/ops/ops.hpp"
#include "zedinfer.h"
#include "zedinfer/activation.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <sstream>
#include <stdexcept>

namespace zedinfer::sampler {

// ============================================================================
// SamplerParams
// ============================================================================
bool SamplerParams::validate() const {
    if (temperature <= 0.0f) {
        return false;
    }
    if (top_k < 0) {
        return false;
    }
    if (top_p <= 0.0f || top_p > 1.0f) {
        return false;
    }
    return true;
}

std::string SamplerParams::info() const {
    std::ostringstream oss;
    oss << "\n=== SamplerParams ===\n"
        << "    temperature: " << temperature << "\n"
        << "    top_k: " << top_k << "\n"
        << "    top_p: " << top_p << "\n"
        << "    seed: " << seed;
    return oss.str();
}

// ============================================================================
// Sampler base class
// ============================================================================
tensor_t Sampler::getLastLogits(tensor_t logits) {
    auto shape = logits->shape();
    if (shape.size() == 1) {
        return logits; // Already [vocab_size]
    } else if (shape.size() == 2) {
        // Extract last position: [seq_len, vocab_size] -> [vocab_size]
        size_t seq_len = shape[0];
        return logits->slice(0, seq_len - 1, seq_len)->view({shape[1]});
    }
    throw std::runtime_error("Invalid logits shape: expected [vocab_size] or [seq_len, vocab_size]");
}

tensor_t Sampler::ensureCPU(tensor_t tensor) {
    if (tensor->deviceType() != ZEDINFER_DEVICE_CPU) {
        return tensor->to(ZEDINFER_DEVICE_CPU);
    }
    return tensor;
}

// ============================================================================
// ArgmaxSampler
// ============================================================================
int ArgmaxSampler::sample(tensor_t logits) {
    tensor_t last_logits = getLastLogits(logits);

    // Lazy-init pre-allocated buffers on first call
    if (!max_idx_dev_) {
        max_idx_dev_ = Tensor::create({1}, ZEDINFER_DTYPE_I64, last_logits->deviceType(), last_logits->deviceId());
        max_val_dev_ = Tensor::create({1}, exec_config_.data_type, last_logits->deviceType(), last_logits->deviceId());
        // Pre-allocate pinned host buffer once — avoids per-call
        // cudaMallocHost/cudaFreeHost which costs ~570us each when the
        // CUDA pinned memory subsystem is cold.
        if (last_logits->deviceType() != ZEDINFER_DEVICE_CPU) {
            max_idx_host_ = Tensor::create({1}, ZEDINFER_DTYPE_I64, ZEDINFER_DEVICE_CPU, 0);
        }
    }

    // Run argmax kernel (writes into pre-allocated device buffers)
    ops::argmax(max_idx_dev_, max_val_dev_, last_logits);

    // Copy result to CPU: reuse pinned host buffer, only memcpy
    if (max_idx_host_) {
        core::context().runtime().api()->memcpy_sync(max_idx_host_->data(), max_idx_dev_->data(), sizeof(int64_t),
                                                     ZEDINFER_MEMCPY_D2H);
        return static_cast<int>(*reinterpret_cast<const int64_t*>(max_idx_host_->data()));
    }
    // CPU path: read directly
    return static_cast<int>(*reinterpret_cast<const int64_t*>(max_idx_dev_->data()));
}

// ============================================================================
// GeneralSampler
// ============================================================================
GeneralSampler::GeneralSampler(const SamplerParams& params) : params_(params) {
    if (!params_.validate()) {
        throw std::invalid_argument("Invalid sampler parameters: " + params_.info());
    }

    // Initialize random number generator
    if (params_.seed == 0) {
        rng_.seed(std::random_device{}());
    } else {
        rng_.seed(params_.seed);
    }
}

void GeneralSampler::setSeed(unsigned int seed) {
    params_.seed = seed;
    if (seed == 0) {
        rng_.seed(std::random_device{}());
    } else {
        rng_.seed(seed);
    }
}

void GeneralSampler::setParams(const SamplerParams& params) {
    if (!params.validate()) {
        throw std::invalid_argument("Invalid sampler parameters: " + params.info());
    }
    params_ = params;
}

void GeneralSampler::setTemperature(float temperature) {
    if (temperature <= 0.0f) {
        throw std::invalid_argument("Temperature must be positive");
    }
    params_.temperature = temperature;
}

void GeneralSampler::setTopK(int top_k) {
    if (top_k < 0) {
        throw std::invalid_argument("top_k must be non-negative");
    }
    params_.top_k = top_k;
}

void GeneralSampler::setTopP(float top_p) {
    if (top_p <= 0.0f || top_p > 1.0f) {
        throw std::invalid_argument("top_p must be in (0, 1]");
    }
    params_.top_p = top_p;
}

int GeneralSampler::sample(tensor_t logits) {
    // Extract and ensure logits are on CPU
    tensor_t last_logits = getLastLogits(logits);
    last_logits = ensureCPU(last_logits);

    // Convert to F32 before raw-pointer read. Model logits live in the model's
    // working dtype (bf16 / f16 for the Qwen family); a bare reinterpret_cast
    // would treat 2-byte values as halves of a 4-byte float and produce garbage,
    // which makes the GeneralSampler emit token-id noise. ArgmaxSampler avoids
    // this by going through a dtype-aware ops::argmax kernel; we have no
    // equivalent CPU kernel here, so convert in place.
    if (last_logits->dtype() != ZEDINFER_DTYPE_F32) {
        last_logits = last_logits->to(ZEDINFER_DTYPE_F32);
    }

    size_t vocab_size = last_logits->numel();
    const float* logits_ptr = reinterpret_cast<const float*>(last_logits->data());

    // Apply temperature scaling
    std::vector<float> scaled_logits(logits_ptr, logits_ptr + vocab_size);
    applyTemperature(scaled_logits.data(), vocab_size);

    // Compute softmax probabilities
    std::vector<float> probs(vocab_size);
    applySoftmax(probs.data(), scaled_logits.data(), vocab_size);

    // Create (probability, index) pairs and sort descending
    std::vector<std::pair<float, int>> indexed_probs(vocab_size);
    for (size_t i = 0; i < vocab_size; ++i) { indexed_probs[i] = {probs[i], static_cast<int>(i)}; }
    std::sort(indexed_probs.begin(), indexed_probs.end(),
              [](const auto& a, const auto& b) { return a.first > b.first; });

    // Apply top-k filtering
    if (params_.top_k > 0 && params_.top_k < static_cast<int>(vocab_size)) {
        applyTopK(indexed_probs);
    }

    // Apply top-p (nucleus) filtering
    if (params_.top_p < 1.0f) {
        applyTopP(indexed_probs);
    }

    // Renormalize filtered probabilities
    float prob_sum = 0.0f;
    for (const auto& pair : indexed_probs) { prob_sum += pair.first; }

    std::vector<float> final_probs(indexed_probs.size());
    for (size_t i = 0; i < indexed_probs.size(); ++i) { final_probs[i] = indexed_probs[i].first / prob_sum; }

    // Sample from renormalized distribution
    int sampled_idx = sampleFromProbs(final_probs.data(), final_probs.size());
    return indexed_probs[sampled_idx].second;
}

void GeneralSampler::applyTemperature(float* logits, size_t size) {
    if (params_.temperature != 1.0f) {
        for (size_t i = 0; i < size; ++i) { logits[i] /= params_.temperature; }
    }
}

void GeneralSampler::applySoftmax(float* probs, const float* logits, size_t size) {
    // Numerical stability: subtract max before exp
    float max_logit = logits[0];
    for (size_t i = 1; i < size; ++i) { max_logit = std::max(max_logit, logits[i]); }

    // Compute exp and accumulate sum
    float sum = 0.0f;
    for (size_t i = 0; i < size; ++i) {
        probs[i] = std::exp(logits[i] - max_logit);
        sum += probs[i];
    }

    // Normalize to get probabilities
    for (size_t i = 0; i < size; ++i) { probs[i] /= sum; }
}

void GeneralSampler::applyTopK(std::vector<std::pair<float, int>>& indexed_probs) {
    // Keep only top-k highest probability tokens
    if (params_.top_k < static_cast<int>(indexed_probs.size())) {
        indexed_probs.resize(params_.top_k);
    }
}

void GeneralSampler::applyTopP(std::vector<std::pair<float, int>>& indexed_probs) {
    // Find nucleus: smallest set with cumulative probability >= top_p
    float cumsum = 0.0f;
    size_t cutoff = indexed_probs.size();

    for (size_t i = 0; i < indexed_probs.size(); ++i) {
        cumsum += indexed_probs[i].first;
        if (cumsum >= params_.top_p) {
            cutoff = i + 1;
            break;
        }
    }

    indexed_probs.resize(cutoff);
}

int GeneralSampler::sampleFromProbs(const float* probs, size_t size) {
    // Generate random value in [0, 1)
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    float rand_val = dist(rng_);

    // Find token using inverse CDF sampling
    float cumsum = 0.0f;
    for (size_t i = 0; i < size; ++i) {
        cumsum += probs[i];
        if (rand_val < cumsum) {
            return static_cast<int>(i);
        }
    }

    // Handle floating point precision: return last token
    return static_cast<int>(size - 1);
}

// ============================================================================
// Factory function
// ============================================================================
std::shared_ptr<Sampler> createSampler(ExecutorConfig exec_config, SamplerType type, const SamplerParams& params) {
    switch (type) {
        case SamplerType::ARGMAX:
            return std::make_shared<ArgmaxSampler>(exec_config);
        case SamplerType::GENERAL:
            return std::make_shared<GeneralSampler>(params);
        default:
            throw std::invalid_argument("Unknown sampler type");
    }
}

} // namespace zedinfer::sampler