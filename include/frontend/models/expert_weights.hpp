#pragma once

#include "backend/tensor/tensor.hpp"

#include <string>
#include <vector>

namespace zedinfer::model {

// Which projection within an expert FFN.
enum class ExpertProj : int { Gate = 0, Up = 1, Down = 2 };

// One expert's FFN weights (gate_proj / up_proj / down_proj).
// Each projection is either quantized (packed + scale + optional g_idx) OR dense (weight).
// In Phase 2 (expert offloading), this struct becomes the unit of CPU/GPU migration.
struct ExpertFFN {
    tensor_t gate_packed = nullptr;
    tensor_t gate_scale = nullptr;
    tensor_t gate_g_idx = nullptr;
    tensor_t gate_weight = nullptr;

    tensor_t up_packed = nullptr;
    tensor_t up_scale = nullptr;
    tensor_t up_g_idx = nullptr;
    tensor_t up_weight = nullptr;

    tensor_t down_packed = nullptr;
    tensor_t down_scale = nullptr;
    tensor_t down_g_idx = nullptr;
    tensor_t down_weight = nullptr;

    bool is_quantized() const { return gate_packed != nullptr; }
    bool valid() const { return is_quantized() || gate_weight != nullptr; }
};

// Per-(layer, expert_id) storage of expert FFN weights.
// Provides O(1) indexed access without string-based weight lookups.
class ExpertWeights {
public:
    ExpertWeights(size_t num_layers, size_t num_experts_per_layer)
        : num_layers_(num_layers),
          num_experts_per_layer_(num_experts_per_layer),
          slots_(num_layers * num_experts_per_layer) {}

    ExpertFFN& at(size_t layer, size_t expert_id) { return slots_[layer * num_experts_per_layer_ + expert_id]; }
    const ExpertFFN& at(size_t layer, size_t expert_id) const {
        return slots_[layer * num_experts_per_layer_ + expert_id];
    }

    size_t num_layers() const { return num_layers_; }
    size_t num_experts_per_layer() const { return num_experts_per_layer_; }

private:
    size_t num_layers_;
    size_t num_experts_per_layer_;
    std::vector<ExpertFFN> slots_;
};

// Parse a per-expert tensor name of the form
//   layers.{L}.mlp.experts.{E}.{gate_proj|up_proj|down_proj}.{weight|weight_packed|weight_scale|weight_g_idx}
// into its components. Returns true on match. Shared by the Qwen3 and Qwen3.5
// MoE loaders, which use the identical per-expert naming convention (after the
// loader strips the "model." / "language_model." prefixes).
bool parse_expert_tensor_name(const std::string& name, size_t& layer, size_t& expert_id, std::string& proj,
                              std::string& suffix);

// Route a parsed (proj, suffix) tensor into the matching ExpertFFN slot.
// Unrecognized (proj, suffix) combinations are silently ignored.
void assign_expert_tensor(ExpertFFN& ffn, const std::string& proj, const std::string& suffix, tensor_t tensor);

} // namespace zedinfer::model
