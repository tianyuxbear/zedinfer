#pragma once

#include "frontend/models/base.hpp"
#include "frontend/models/hybrid_forward_config.hpp"
#include "frontend/models/qwen3_5_config.hpp"
#include "frontend/models/ssm_state_pool.hpp"

#include <memory>
#include <string>

namespace zedinfer {
class ChatTemplateJinja;
} // namespace zedinfer

namespace zedinfer::model {

// Forward-declare the vision tower so this header does not have to pull in
// backend/tensor/tensor.hpp transitively for every TU that includes us.
class VisionTower;

// Dense Qwen3.5 model skeleton (M0). The ctor owns:
//   * Qwen3_5Config — parsed by Model::load_config
//   * ModelWeights — loaded by Model::load_weights inside Model::parse
//   * SSMStatePool — sized from config_.linear_attn + max_concurrent
//   * VisionTower (optional) — constructed when config_.has_vision is true
//   * ChatTemplateJinja (optional) — loaded when chat_template.jinja exists
//     in model_path. Held as shared_ptr so subclasses (Qwen3_5MoeModel) get
//     the same compiled template without re-parsing.
//
// forward_config() currently throws; the M1 forward path will populate it.
class Qwen3_5Model : public Model {
public:
    Qwen3_5Model(Qwen3_5Config config, std::unique_ptr<ModelWeights> weights, const ExecutorConfig& exec,
                 int max_concurrent, const std::string& model_path);
    ~Qwen3_5Model() override;

    const ModelConfig& config() const override { return config_; }
    const ModelWeights& weights() const override { return *weights_; }
    std::string model_type() const override { return "qwen3_5"; }
    size_t num_parameters() const override;

    ModelForwardConfig forward_config() const override;
    HybridForwardConfig hybrid_forward_config() const;

    SSMStatePool& ssm_state_pool() { return *ssm_pool_; }
    const SSMStatePool& ssm_state_pool() const { return *ssm_pool_; }
    const VisionTower* vision_tower() const { return vision_.get(); }

    // Nullable accessor: returns nullptr when chat_template.jinja was missing
    // from the model directory (e.g. unit-test fixtures without the file).
    const ChatTemplateJinja* chat_template() const { return chat_template_.get(); }

protected:
    Qwen3_5Config config_;
    std::unique_ptr<ModelWeights> weights_;
    std::unique_ptr<SSMStatePool> ssm_pool_;
    std::unique_ptr<VisionTower> vision_;
    std::shared_ptr<ChatTemplateJinja> chat_template_;
};

} // namespace zedinfer::model
