#pragma once

#include "backend/tensor/tensor.hpp"

namespace neollm::ops {
void argmax(tensor_t max_idx, tensor_t max_val, tensor_t vals);
}
