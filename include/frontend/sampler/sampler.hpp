#pragma once

#include "backend/tensor/tensor.hpp"
#include "zedinfer/activation.hpp"

#include <memory>
#include <random>
#include <string>

namespace zedinfer::sampler {

// Sampler type enumeration
enum class SamplerType {
    ARGMAX, // Greedy sampling: always pick max probability token
    GENERAL // General sampling: temperature + top_k + top_p filtering
};

constexpr const char* SamplerTypeNames[] = {"argmax", "general"};

constexpr const char* to_string(SamplerType type) {
    return SamplerTypeNames[static_cast<size_t>(type)];
}

// Sampling configuration parameters
struct SamplerParams {
    float temperature = 1.0f;        // Scale logits (1.0 = no scaling)
    int top_k = 0;                   // Keep top-k tokens (0 = disabled)
    float top_p = 1.0f;              // Nucleus sampling threshold (1.0 = disabled)
    float repetition_penalty = 1.0f; // HF CTRL-style penalty (1.0 = disabled)
    unsigned int seed = 0;           // Random seed (0 = random device)

    SamplerParams() = default;
    SamplerParams(float temp, int k = 0, float p = 1.0f, unsigned int s = 0)
        : temperature(temp), top_k(k), top_p(p), seed(s) {}

    bool validate() const;
    std::string info() const;
};

// Abstract base class for token sampling
class Sampler {
public:
    virtual ~Sampler() = default;

    // Sample next token from logits [seq_len, vocab_size] or [vocab_size].
    // `recent_tokens` (when non-null) is the list of previously-generated token
    // ids; samplers that support repetition penalty use it to suppress repeated
    // tokens. ArgmaxSampler ignores it.
    virtual int sample(tensor_t logits, const std::vector<int>* recent_tokens = nullptr) = 0;

    // Set random seed for reproducibility
    virtual void setSeed(unsigned int seed) = 0;

    // Get sampler type name
    virtual std::string name() const = 0;

protected:
    tensor_t getLastLogits(tensor_t logits);
    tensor_t ensureCPU(tensor_t tensor);
};

// Greedy sampling: always pick argmax
class ArgmaxSampler : public Sampler {
public:
    ArgmaxSampler(ExecutorConfig exec_config) : exec_config_(exec_config){};

    int sample(tensor_t logits, const std::vector<int>* recent_tokens = nullptr) override;
    void setSeed(unsigned int /*seed*/) override {} // No randomness needed
    std::string name() const override { return "Argmax"; }

private:
    ExecutorConfig exec_config_;

    // Pre-allocated buffers to avoid per-call cudaMallocHost/cudaFreeHost.
    // cudaMallocHost is expensive (~570us) when the pinned memory subsystem
    // is not warmed up, causing a 20%+ decode throughput regression.
    tensor_t max_idx_dev_;  // [1] I64 on device
    tensor_t max_val_dev_;  // [1] dtype on device
    tensor_t max_idx_host_; // [1] I64 on host (pinned)
};

// General sampler: supports temperature + top_k + top_p filtering
class GeneralSampler : public Sampler {
public:
    explicit GeneralSampler(const SamplerParams& params = SamplerParams());

    int sample(tensor_t logits, const std::vector<int>* recent_tokens = nullptr) override;
    void setSeed(unsigned int seed) override;
    std::string name() const override { return "General"; }

    // The truncated, renormalized distribution sample() draws from: applies
    // repetition penalty + temperature + softmax + top_k + top_p, returns the
    // surviving (probability, token_id) pairs sorted by probability descending
    // (probabilities sum to 1 over the kept support). Exposed so Qwen3.5 MTP
    // speculative decoding can run true rejection sampling against it.
    std::vector<std::pair<float, int>> truncatedDist(tensor_t logits,
                                                     const std::vector<int>* recent_tokens = nullptr);

    // Draw one token id from a precomputed truncatedDist() output (inverse-CDF
    // over the sampler's RNG stream).
    int sampleFromDist(const std::vector<std::pair<float, int>>& dist);

    // Speculative-decode rejection sampling at one position. p_dist is the
    // target (main model) truncatedDist and q_dist the draft (MTP head)
    // truncatedDist for the same position; `draft` was previously drawn from
    // q_dist. Accepts the draft with probability min(1, p(draft)/q(draft));
    // on reject samples the corrected token from the residual
    // normalize(max(0, p - q)) over the target support. Returns the committed
    // token and sets `accepted`. The committed token is distributed exactly as
    // the target p (standard speculative-decoding guarantee).
    int specRejectionSample(const std::vector<std::pair<float, int>>& p_dist,
                            const std::vector<std::pair<float, int>>& q_dist, int draft, bool& accepted);

    void setParams(const SamplerParams& params);
    const SamplerParams& getParams() const { return params_; }

    void setTemperature(float temperature);
    void setTopK(int top_k);
    void setTopP(float top_p);

private:
    SamplerParams params_;
    std::mt19937 rng_;

    void applyTemperature(float* logits, size_t size);
    void applyRepetitionPenalty(float* logits, size_t size, const std::vector<int>* recent_tokens);
    void applySoftmax(float* probs, const float* logits, size_t size);
    void applyTopK(std::vector<std::pair<float, int>>& indexed_probs);
    void applyTopP(std::vector<std::pair<float, int>>& indexed_probs);
    int sampleFromProbs(const float* probs, size_t size);
};

// Factory function to create sampler instances
std::shared_ptr<Sampler> createSampler(ExecutorConfig exec_config, SamplerType sampler_type,
                                       const SamplerParams& params = SamplerParams());

} // namespace zedinfer::sampler