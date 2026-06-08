#include "frontend/sampler/sampler.hpp"
#include "backend/core/context/context.hpp"
#include "backend/ops/ops.hpp"
#include "zedinfer.h"
#include "zedinfer/activation.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

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
    if (repetition_penalty <= 0.0f) {
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
        << "    repetition_penalty: " << repetition_penalty << "\n"
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
int ArgmaxSampler::sample(tensor_t logits, const std::vector<int>* /*recent_tokens*/) {
    // Debug toggle resolved once per process; keeps getenv() and the full-vocab
    // logit dump out of the per-token sampling path unless explicitly enabled.
    static const bool dump_top_logits = [] {
        const char* env = std::getenv("ZEDINFER_DUMP_TOP_LOGITS");
        return env && env[0] == '1';
    }();

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
    int chosen = -1;
    if (max_idx_host_) {
        core::context().runtime().api()->memcpy_sync(max_idx_host_->data(), max_idx_dev_->data(), sizeof(int64_t),
                                                     ZEDINFER_MEMCPY_D2H);
        chosen = static_cast<int>(*reinterpret_cast<const int64_t*>(max_idx_host_->data()));
    } else {
        chosen = static_cast<int>(*reinterpret_cast<const int64_t*>(max_idx_dev_->data()));
    }

    if (dump_top_logits) {
        // Copy full logits to host (fp32 path: getLastLogits may have already converted).
        const size_t vocab = static_cast<size_t>(last_logits->numel());
        std::vector<float> host(vocab);
        if (last_logits->dtype() == ZEDINFER_DTYPE_F32) {
            core::context().runtime().api()->memcpy_sync(host.data(), last_logits->data(), vocab * sizeof(float),
                                                         ZEDINFER_MEMCPY_D2H);
        } else if (last_logits->dtype() == ZEDINFER_DTYPE_BF16) {
            std::vector<uint16_t> bf16_host(vocab);
            core::context().runtime().api()->memcpy_sync(bf16_host.data(), last_logits->data(),
                                                         vocab * sizeof(uint16_t), ZEDINFER_MEMCPY_D2H);
            for (size_t i = 0; i < vocab; ++i) {
                uint32_t u = static_cast<uint32_t>(bf16_host[i]) << 16;
                std::memcpy(&host[i], &u, sizeof(float));
            }
        }
        std::vector<std::pair<int, float>> idx_val(vocab);
        for (size_t i = 0; i < vocab; ++i) { idx_val[i] = {static_cast<int>(i), host[i]}; }
        std::partial_sort(idx_val.begin(), idx_val.begin() + 5, idx_val.end(),
                          [](const auto& a, const auto& b) { return a.second > b.second; });
        fprintf(stderr, "[zedinfer-logits] chose=%d top5:", chosen);
        for (int k = 0; k < 5; ++k) { fprintf(stderr, " #%d=%d(logit=%.4f)", k, idx_val[k].first, idx_val[k].second); }
        fprintf(stderr, "\n");
    }

    return chosen;
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

std::vector<std::pair<float, int>> GeneralSampler::truncatedDist(tensor_t logits,
                                                                 const std::vector<int>* recent_tokens) {
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

    // Working copy. We modify in place: repetition penalty first (operates on
    // RAW logits so the multiply/divide stays on the original scale), then
    // temperature, then softmax. Order matters: HF's GenerationMixin applies
    // repetition_penalty BEFORE temperature for exactly this reason.
    std::vector<float> scaled_logits(logits_ptr, logits_ptr + vocab_size);
    applyRepetitionPenalty(scaled_logits.data(), vocab_size, recent_tokens);
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

    // Renormalize filtered probabilities so they sum to 1 over the kept support.
    float prob_sum = 0.0f;
    for (const auto& pair : indexed_probs) { prob_sum += pair.first; }
    if (prob_sum > 0.0f) {
        for (auto& pair : indexed_probs) { pair.first /= prob_sum; }
    }
    return indexed_probs;
}

int GeneralSampler::sampleFromDist(const std::vector<std::pair<float, int>>& dist) {
    if (dist.empty()) {
        return -1;
    }
    std::uniform_real_distribution<float> uni(0.0f, 1.0f);
    float rand_val = uni(rng_);
    float cumsum = 0.0f;
    for (const auto& pr : dist) {
        cumsum += pr.first;
        if (rand_val < cumsum) {
            return pr.second;
        }
    }
    return dist.back().second; // floating-point slack
}

int GeneralSampler::specRejectionSample(const std::vector<std::pair<float, int>>& p_dist,
                                        const std::vector<std::pair<float, int>>& q_dist, int draft, bool& accepted) {
    auto prob_of = [](const std::vector<std::pair<float, int>>& dist, int tok) -> float {
        for (const auto& pr : dist) {
            if (pr.second == tok) {
                return pr.first;
            }
        }
        return 0.0f;
    };
    const float qd = prob_of(q_dist, draft);
    const float pd = prob_of(p_dist, draft);

    // Accept the draft with probability min(1, p(draft)/q(draft)). qd > 0 since
    // the draft was drawn from q_dist; guard defensively anyway.
    std::uniform_real_distribution<float> uni(0.0f, 1.0f);
    const float u = uni(rng_);
    if (qd <= 0.0f || u <= pd / qd) {
        accepted = true;
        return draft;
    }

    // Reject: sample the corrected token from the residual normalize(max(0, p-q))
    // over the target support. This makes the committed token distributed
    // exactly as the target p.
    accepted = false;
    std::vector<std::pair<float, int>> residual;
    residual.reserve(p_dist.size());
    float sum = 0.0f;
    for (const auto& pr : p_dist) {
        const float r = pr.first - prob_of(q_dist, pr.second);
        if (r > 0.0f) {
            residual.emplace_back(r, pr.second);
            sum += r;
        }
    }
    if (residual.empty() || sum <= 0.0f) {
        // p's support is contained in q with no positive residual (rare on a
        // reject); fall back to a plain sample from the target.
        return sampleFromDist(p_dist);
    }
    for (auto& pr : residual) { pr.first /= sum; }
    return sampleFromDist(residual);
}

int GeneralSampler::sample(tensor_t logits, const std::vector<int>* recent_tokens) {
    tensor_t last_logits = getLastLogits(logits);
    if (canSampleOnGPU(last_logits)) {
        return sampleGPU(last_logits, recent_tokens);
    }
    return sampleFromDist(truncatedDist(logits, recent_tokens));
}

bool GeneralSampler::canSampleOnGPU(tensor_t last_logits) const {
    if (!last_logits || last_logits->deviceType() != ZEDINFER_DEVICE_NVIDIA || !last_logits->isContiguous()
        || params_.top_k < 0) {
        return false;
    }
    switch (last_logits->dtype()) {
        case ZEDINFER_DTYPE_F32:
        case ZEDINFER_DTYPE_F16:
        case ZEDINFER_DTYPE_BF16:
            return true;
        default:
            return false;
    }
}

bool GeneralSampler::needsGPUSort(tensor_t last_logits) const {
    const size_t vocab_size = last_logits->numel();
    const bool has_top_k_filter = params_.top_k > 0 && static_cast<size_t>(params_.top_k) < vocab_size;
    return (has_top_k_filter && params_.top_k > kGpuMaxTopK) || (!has_top_k_filter && params_.top_p < 1.0f);
}

void GeneralSampler::ensureGPUScratch(tensor_t last_logits, size_t recent_count) {
    const auto device_type = last_logits->deviceType();
    const int device_id = last_logits->deviceId();
    const size_t vocab_size = last_logits->numel();

    if (!gpu_sample_token_dev_ || gpu_sample_token_dev_->deviceType() != device_type
        || gpu_sample_token_dev_->deviceId() != device_id) {
        gpu_sample_token_dev_ = Tensor::create({1}, ZEDINFER_DTYPE_I64, device_type, device_id);
        gpu_sample_token_host_ = Tensor::create({1}, ZEDINFER_DTYPE_I64, ZEDINFER_DEVICE_CPU, 0);
    }

    if (!gpu_work_logits_ || gpu_work_logits_->deviceType() != device_type || gpu_work_logits_->deviceId() != device_id
        || gpu_work_logits_->numel() != vocab_size) {
        gpu_work_logits_ = Tensor::create({vocab_size}, ZEDINFER_DTYPE_F32, device_type, device_id);
    }

    if (needsGPUSort(last_logits)) {
        if (!gpu_work_ids_ || gpu_work_ids_->deviceType() != device_type || gpu_work_ids_->deviceId() != device_id
            || gpu_work_ids_->numel() != vocab_size) {
            gpu_work_ids_ = Tensor::create({vocab_size}, ZEDINFER_DTYPE_I32, device_type, device_id);
            gpu_sorted_logits_ = Tensor::create({vocab_size}, ZEDINFER_DTYPE_F32, device_type, device_id);
            gpu_sorted_ids_ = Tensor::create({vocab_size}, ZEDINFER_DTYPE_I32, device_type, device_id);
        }

        if (!gpu_sort_temp_ || gpu_sort_temp_->deviceType() != device_type || gpu_sort_temp_->deviceId() != device_id
            || gpu_sort_temp_vocab_size_ != vocab_size) {
            const size_t temp_bytes = ops::sample_sort_workspace_bytes(device_type, vocab_size);
            gpu_sort_temp_ = Tensor::create({temp_bytes}, ZEDINFER_DTYPE_U8, device_type, device_id);
            gpu_sort_temp_vocab_size_ = vocab_size;
        }
    }

    if (recent_count > 0
        && (!gpu_recent_tokens_ || gpu_recent_tokens_->deviceType() != device_type
            || gpu_recent_tokens_->deviceId() != device_id || gpu_recent_tokens_->numel() < recent_count)) {
        gpu_recent_tokens_ = Tensor::create({recent_count}, ZEDINFER_DTYPE_I32, device_type, device_id);
    }
}

int GeneralSampler::sampleGPU(tensor_t last_logits, const std::vector<int>* recent_tokens) {
    const size_t recent_count = recent_tokens ? recent_tokens->size() : 0;
    ensureGPUScratch(last_logits, recent_count);
    core::context().setDevice(last_logits->deviceType(), last_logits->deviceId());

    tensor_t recent_dev;
    if (recent_count > 0) {
        gpu_recent_tokens_host_.resize(recent_count);
        for (size_t i = 0; i < recent_count; ++i) {
            gpu_recent_tokens_host_[i] = static_cast<int32_t>((*recent_tokens)[i]);
        }
        core::context().runtime().api()->memcpy_sync(gpu_recent_tokens_->data(), gpu_recent_tokens_host_.data(),
                                                     recent_count * sizeof(int32_t), ZEDINFER_MEMCPY_H2D);
        recent_dev = gpu_recent_tokens_->slice(0, 0, recent_count);
    }

    std::uniform_real_distribution<float> uni(0.0f, 1.0f);
    float random = uni(rng_);
    if (random >= 1.0f) {
        random = std::nextafter(1.0f, 0.0f);
    }

    ops::sample_token(gpu_sample_token_dev_, last_logits, gpu_work_logits_, gpu_work_ids_, gpu_sorted_logits_,
                      gpu_sorted_ids_, gpu_sort_temp_, recent_dev, params_.temperature, params_.top_k, params_.top_p,
                      params_.repetition_penalty, random);
    core::context().runtime().api()->memcpy_sync(gpu_sample_token_host_->data(), gpu_sample_token_dev_->data(),
                                                 sizeof(int64_t), ZEDINFER_MEMCPY_D2H);
    return static_cast<int>(*reinterpret_cast<const int64_t*>(gpu_sample_token_host_->data()));
}

void GeneralSampler::applyTemperature(float* logits, size_t size) {
    if (params_.temperature != 1.0f) {
        for (size_t i = 0; i < size; ++i) { logits[i] /= params_.temperature; }
    }
}

// HF / CTRL (Keskar et al. 2019) repetition penalty.
//
// For each previously emitted token t:
//   logit[t] = logit[t] / penalty   if logit[t] > 0
//   logit[t] = logit[t] * penalty   if logit[t] < 0
//
// I.e., positive logits get pushed toward 0 (relatively less likely), negative
// logits get pushed further from 0 (also less likely). This is a multiplicative
// version of "make any token we already said less attractive next time".
//
// Why this matters: under temp+top_k+top_p sampling, when the model state
// drifts into a degenerate attractor the per-step distribution can become
// extremely peaked on a tiny rotating set (e.g. "Wait, the user is asking ...").
// Even sampling with top_p=0.95 picks essentially the same tokens because the
// nucleus is one token wide. Demoting the recently-seen tokens forces the
// nucleus to widen, breaks the loop, and keeps generation coherent under the
// model's recommended generation_config parameters.
//
// Apply to ALL recent tokens (prompt + previously generated) in the caller's
// list. For zedinfer the scheduler passes req->output_ids — i.e., only the
// generated tokens, not the prompt. Penalizing only generated tokens is a
// common middle ground that doesn't trigger on benign prompt repeats (e.g.
// quoted text in the user's question).
void GeneralSampler::applyRepetitionPenalty(float* logits, size_t size, const std::vector<int>* recent_tokens) {
    if (params_.repetition_penalty == 1.0f || !recent_tokens || recent_tokens->empty()) {
        return;
    }
    const float penalty = params_.repetition_penalty;
    for (int tid : *recent_tokens) {
        if (tid < 0 || static_cast<size_t>(tid) >= size) {
            continue;
        }
        const float lv = logits[tid];
        logits[tid] = lv > 0.0f ? (lv / penalty) : (lv * penalty);
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
