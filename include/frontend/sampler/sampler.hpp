#pragma once

#include "backend/tensor/tensor.hpp"

#include <memory>
#include <random>
#include <string>

namespace zedinfer::sampler {

// Sampler type enumeration
enum class SamplerType {
    ARGMAX, // Greedy sampling: always pick max probability token
    GENERAL // General sampling: temperature + top_k + top_p filtering
};

constexpr const char *SamplerTypeNames[] = {
    "argmax",
    "general"};

constexpr const char *to_string(SamplerType type) {
    return SamplerTypeNames[static_cast<size_t>(type)];
}

// Sampling configuration parameters
struct SamplerParams {
    float temperature = 1.0f; // Scale logits (1.0 = no scaling)
    int top_k = 0;            // Keep top-k tokens (0 = disabled)
    float top_p = 1.0f;       // Nucleus sampling threshold (1.0 = disabled)
    unsigned int seed = 0;    // Random seed (0 = random device)

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

    // Sample next token from logits [seq_len, vocab_size] or [vocab_size]
    virtual int sample(tensor_t logits) = 0;

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
    ArgmaxSampler() = default;

    int sample(tensor_t logits) override;
    void setSeed(unsigned int /*seed*/) override {} // No randomness needed
    std::string name() const override { return "Argmax"; }
};

// General sampler: supports temperature + top_k + top_p filtering
class GeneralSampler : public Sampler {
public:
    explicit GeneralSampler(const SamplerParams &params = SamplerParams());

    int sample(tensor_t logits) override;
    void setSeed(unsigned int seed) override;
    std::string name() const override { return "General"; }

    void setParams(const SamplerParams &params);
    const SamplerParams &getParams() const { return params_; }

    void setTemperature(float temperature);
    void setTopK(int top_k);
    void setTopP(float top_p);

private:
    SamplerParams params_;
    std::mt19937 rng_;

    void applyTemperature(float *logits, size_t size);
    void applySoftmax(float *probs, const float *logits, size_t size);
    void applyTopK(std::vector<std::pair<float, int>> &indexed_probs);
    void applyTopP(std::vector<std::pair<float, int>> &indexed_probs);
    int sampleFromProbs(const float *probs, size_t size);
};

// Factory function to create sampler instances
std::shared_ptr<Sampler> createSampler(SamplerType sampler_type,
                                       const SamplerParams &params = SamplerParams());

} // namespace zedinfer::sampler