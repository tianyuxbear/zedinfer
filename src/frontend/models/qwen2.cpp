#include "frontend/models/qwen2.hpp"

namespace neollm::model {

std::string Qwen2Model::get_embedding_weight_name() const {
    return "embed_tokens.weight";
}

std::string Qwen2Model::get_output_norm_weight_name() const {
    return "norm.weight";
}

std::string Qwen2Model::get_output_weight_name() const {
    return "lm_head.weight";
}

std::vector<std::string> Qwen2Model::get_layer_weight_names(int layer_idx) const {
    std::string prefix = "layers." + std::to_string(layer_idx) + ".";

    return {
        prefix + "self_attn.q_proj.weight",
        prefix + "self_attn.q_proj.bias",
        prefix + "self_attn.k_proj.weight",
        prefix + "self_attn.k_proj.bias",
        prefix + "self_attn.v_proj.weight",
        prefix + "self_attn.v_proj.bias",
        prefix + "self_attn.o_proj.weight",
        prefix + "mlp.gate_proj.weight",
        prefix + "mlp.up_proj.weight",
        prefix + "mlp.down_proj.weight",
        prefix + "input_layernorm.weight",
        prefix + "post_attention_layernorm.weight"};
}

size_t Qwen2Model::calculate_num_parameters() const {
    size_t total = 0;
    for (const auto &[name, tensor] : weights_->get_all_weights()) {
        total += tensor->numel();
    }
    return total;
}

} // namespace neollm::model