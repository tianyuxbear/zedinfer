#include "frontend/models/qwen2.hpp"

namespace zedinfer::model {

size_t Qwen2Model::calculate_num_parameters() const {
    size_t total = 0;
    for (const auto& [name, tensor] : weights_->get_all_weights()) { total += tensor->numel(); }
    return total;
}

} // namespace zedinfer::model
