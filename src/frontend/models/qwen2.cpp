#include "frontend/models/qwen2.hpp"
#include "frontend/models/forward_config.hpp"
#include "frontend/models/forward_context.hpp"
#include "frontend/models/paged_forward_context.hpp"

#include <string>

namespace zedinfer::model {

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

tensor_t Qwen2Model::forward(
    const std::vector<int> &input_ids,
    int past_len,
    kvcache::KVCache &kvcache,
    const ExecutorConfig &exec_config) {

    ModelForwardConfig cfg{config_, *weights_, /*has_qkv_bias=*/true, /*has_qk_norm=*/false};
    ContiguousForwardContext ctx(input_ids, past_len, kvcache);
    return transformer_forward(cfg, ctx, exec_config);
}

tensor_t Qwen2Model::forward_batch(
    const BatchContext &batch,
    kvcache::BlockAllocator &allocator,
    const ExecutorConfig &exec_config) {

    ModelForwardConfig cfg{config_, *weights_, /*has_qkv_bias=*/true, /*has_qk_norm=*/false};
    PagedForwardContext ctx(batch, allocator);
    return transformer_forward(cfg, ctx, exec_config);
}

} // namespace zedinfer::model
