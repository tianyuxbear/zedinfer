#pragma once

#include "backend/tensor/tensor.hpp"
#include "frontend/models/base.hpp"

namespace zedinfer {
struct ExecutorConfig;
}

namespace zedinfer::model {

// M0 skeleton of the Qwen3.5 vision tower. The ctor only verifies the expected
// weight tensors are present in the loaded ModelWeights map (so config parse /
// weight load issues fail loudly at construction); the actual forward pass is
// implemented in M3 and currently throws.
class VisionTower {
public:
    VisionTower(const VisionConfig& cfg, const ModelWeights& w, const ExecutorConfig& exec);
    ~VisionTower();

    // Stub until M3 — throws std::runtime_error.
    tensor_t forward(tensor_t patches, tensor_t pos_ids_thw, const ExecutorConfig& exec);

private:
    VisionConfig cfg_;
};

} // namespace zedinfer::model
