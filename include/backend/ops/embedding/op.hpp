#pragma once

#include "backend/tensor/tensor.hpp"

namespace neollm::ops {
void embedding(tensor_t out, tensor_t index, tensor_t weight);
}
