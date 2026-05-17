#pragma once

#include "frontend/models/expert_pool.hpp"
#include "frontend/models/qwen3_5.hpp"
#include "frontend/models/qwen3_5_config.hpp"

#include <memory>
#include <string>

namespace zedinfer::model {

// Qwen3.5 MoE (35B-A3B) skeleton. Inherits the dense Qwen3.5 hybrid attention +
// SSM pool + optional vision tower from Qwen3_5Model, and adds the per-layer
// expert pool that v0.2.0's Qwen3MoEModel already established as the unit of
// CPU/GPU residency under PINNED_LRU.
//
// M0 only wires the load + dispatch path; the MoE forward path lands in M2.
class Qwen3_5MoeModel : public Qwen3_5Model {
public:
    Qwen3_5MoeModel(Qwen3_5MoEConfig config, std::unique_ptr<ModelWeights> weights, const ExecutorConfig& exec,
                    int max_concurrent, ExpertPoolConfig pool_cfg, const std::string& model_path);
    ~Qwen3_5MoeModel() override;

    std::string model_type() const override { return "qwen3_5_moe"; }
    size_t num_parameters() const override;

    const Qwen3_5MoEConfig& moe_config() const { return moe_config_; }
    ExpertPool& expert_pool() { return *expert_pool_; }
    const ExpertPool& expert_pool() const { return *expert_pool_; }

    void log_runtime_stats() const override {
        if (expert_pool_) {
            expert_pool_->log_stats();
        }
    }

private:
    Qwen3_5MoEConfig moe_config_;
    std::unique_ptr<ExpertPool> expert_pool_;
};

} // namespace zedinfer::model
