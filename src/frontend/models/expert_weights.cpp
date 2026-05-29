#include "frontend/models/expert_weights.hpp"

#include <utility>

namespace zedinfer::model {

bool parse_expert_tensor_name(const std::string& name, size_t& layer, size_t& expert_id, std::string& proj,
                              std::string& suffix) {
    static const std::string kPrefix = "layers.";
    static const std::string kMid = ".mlp.experts.";
    if (name.compare(0, kPrefix.size(), kPrefix) != 0) {
        return false;
    }
    size_t pos = kPrefix.size();
    size_t dot = name.find('.', pos);
    if (dot == std::string::npos) {
        return false;
    }
    try {
        layer = std::stoul(name.substr(pos, dot - pos));
    } catch (...) { return false; }
    if (name.compare(dot, kMid.size(), kMid) != 0) {
        return false;
    }
    pos = dot + kMid.size();
    dot = name.find('.', pos);
    if (dot == std::string::npos) {
        return false;
    }
    try {
        expert_id = std::stoul(name.substr(pos, dot - pos));
    } catch (...) { return false; }
    pos = dot + 1;
    dot = name.find('.', pos);
    if (dot == std::string::npos) {
        return false;
    }
    proj = name.substr(pos, dot - pos);
    if (proj != "gate_proj" && proj != "up_proj" && proj != "down_proj") {
        return false;
    }
    suffix = name.substr(dot + 1);
    return true;
}

void assign_expert_tensor(ExpertFFN& ffn, const std::string& proj, const std::string& suffix, tensor_t tensor) {
    tensor_t* target = nullptr;
    if (proj == "gate_proj") {
        if (suffix == "weight_packed") {
            target = &ffn.gate_packed;
        } else if (suffix == "weight_scale") {
            target = &ffn.gate_scale;
        } else if (suffix == "weight_g_idx") {
            target = &ffn.gate_g_idx;
        } else if (suffix == "weight") {
            target = &ffn.gate_weight;
        }
    } else if (proj == "up_proj") {
        if (suffix == "weight_packed") {
            target = &ffn.up_packed;
        } else if (suffix == "weight_scale") {
            target = &ffn.up_scale;
        } else if (suffix == "weight_g_idx") {
            target = &ffn.up_g_idx;
        } else if (suffix == "weight") {
            target = &ffn.up_weight;
        }
    } else if (proj == "down_proj") {
        if (suffix == "weight_packed") {
            target = &ffn.down_packed;
        } else if (suffix == "weight_scale") {
            target = &ffn.down_scale;
        } else if (suffix == "weight_g_idx") {
            target = &ffn.down_g_idx;
        } else if (suffix == "weight") {
            target = &ffn.down_weight;
        }
    }
    if (target) {
        *target = std::move(tensor);
    }
}

} // namespace zedinfer::model
