#pragma once

#include <cstddef>
#include <utility>

namespace zedinfer {

class InferenceEngine;

/**
 * Profiler: warmup and performance measurement.
 * Uses PagedForwardContext with temporary block table from the engine's block pool.
 *
 * Holds a non-owning pointer to the engine that owns this profiler (engine outlives
 * its own unique_ptr members by construction). Must not take a shared_ptr back, or
 * InferenceEngine would form a cycle with itself and never destruct.
 */
class Profiler {
public:
    explicit Profiler(InferenceEngine& engine);

    void warmup(size_t prefill_len = 128, size_t decode_steps = 128);
    std::pair<double, double> profile(size_t prefill_len = 128, size_t decode_steps = 128);

private:
    InferenceEngine* engine_;
};

} // namespace zedinfer
