#pragma once

#include <cstddef>
#include <memory>
#include <utility>

namespace zedinfer {

class InferenceEngine;

/**
 * Profiler: warmup and performance measurement.
 * Uses PagedForwardContext with temporary block table from the engine's block pool.
 */
class Profiler {
public:
    explicit Profiler(std::shared_ptr<InferenceEngine> engine);

    void warmup(size_t prefill_len = 128, size_t decode_steps = 128);
    std::pair<double, double> profile(size_t prefill_len = 128, size_t decode_steps = 128);

private:
    std::shared_ptr<InferenceEngine> engine_;
};

} // namespace zedinfer
